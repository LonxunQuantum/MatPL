#if defined(MATPL_ENABLE_FUSED_FITTING) && defined(MATPL_FUSED_FITTING_HIP)

#include "../include/nep_fitting_jit.h"
#include "../include/nep_fitting_limits.h"
#include "../kernel/nep_fitting_jit_source.h"

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
// CUDA uses a 32-bit active mask. HIP's wave size differs; width still
// partitions each wave into the same independent TILE-wide atom rows.
const std::string& hip_source() {
    static const std::string source =
        std::string("#include <hip/hip_runtime.h>\n"
                    "#define __shfl_down_sync(mask, value, delta, width) "
                    "__shfl_down(value, delta, width)\n") + kNepFittingJitSource;
    return source;
}

struct JitModule {
    hipModule_t module = nullptr;
    hipFunction_t forward = nullptr;
    hipFunction_t backward = nullptr;
    hipFunction_t parameter_partials = nullptr;
    hipFunction_t parameter_reduce = nullptr;
};

constexpr int64_t kWorkspaceBytes = 16 * 1024 * 1024;

std::mutex module_mutex;
std::unordered_map<std::string, std::shared_ptr<JitModule>> modules;

void check_driver(hipError_t result, const char* operation) {
    if (result == hipSuccess) return;
    const char* name = hipGetErrorName(result);
    const char* message = hipGetErrorString(result);
    TORCH_CHECK(false, operation, " failed: ", name ? name : "HIP_ERROR",
                " (", message ? message : "unknown error", ")");
}

void check_hiprtc(hiprtcResult result, const char* operation) {
    TORCH_CHECK(result == HIPRTC_SUCCESS, operation, " failed: ",
                hiprtcGetErrorString(result));
}

uint64_t fnv1a(const std::string& text) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char byte : text) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

struct Specialization {
    int d, h, q, device;
    std::string arch;
    int hiprtc_major, hiprtc_minor;
    std::string key;
    std::string cache_dir;
    std::string cache_file;
};

std::string join_path(const std::string& left, const std::string& right) {
    if (left.empty()) return right;
    return left.back() == '/' ? left + right : left + '/' + right;
}

void ensure_directory(const std::string& path) {
    TORCH_CHECK(!path.empty(), "empty NEP fitting JIT cache directory");
    for (size_t pos = 1; pos <= path.size(); ++pos) {
        if (pos != path.size() && path[pos] != '/') continue;
        const std::string prefix = path.substr(0, pos);
        if (prefix.empty()) continue;
        if (::mkdir(prefix.c_str(), 0700) != 0 && errno != EEXIST) {
            TORCH_CHECK(false, "cannot create NEP fitting JIT cache directory ",
                        prefix, ": errno ", errno);
        }
    }
    struct stat info {};
    TORCH_CHECK(::stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode),
                "NEP fitting JIT cache path is not a directory: ", path);
}

bool is_regular_file(const std::string& path) {
    struct stat info {};
    return ::stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

class FileLock {
public:
    explicit FileLock(const std::string& path) : fd_(::open(path.c_str(), O_CREAT | O_RDWR, 0600)) {
        TORCH_CHECK(fd_ >= 0, "cannot open NEP fitting JIT cache lock ", path,
                    ": errno ", errno);
        if (::flock(fd_, LOCK_EX) != 0) {
            const int error = errno;
            ::close(fd_);
            fd_ = -1;
            TORCH_CHECK(false, "cannot lock NEP fitting JIT cache ", path,
                        ": errno ", error);
        }
    }

    ~FileLock() {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
    }

    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

private:
    int fd_;
};

void write_binary_atomic(const std::string& path, const std::vector<char>& bytes) {
    const std::string temporary = path + ".tmp." + std::to_string(::getpid());
    int fd = ::open(temporary.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    TORCH_CHECK(fd >= 0, "cannot create NEP fitting JIT cache ", temporary,
                ": errno ", errno);
    size_t written = 0;
    bool published = false;
    try {
        while (written < bytes.size()) {
            const ssize_t count = ::write(fd, bytes.data() + written, bytes.size() - written);
            TORCH_CHECK(count > 0, "cannot write NEP fitting JIT cache ", temporary,
                        ": errno ", errno);
            written += static_cast<size_t>(count);
        }
        TORCH_CHECK(::fsync(fd) == 0, "cannot sync NEP fitting JIT cache ",
                    temporary, ": errno ", errno);
        TORCH_CHECK(::close(fd) == 0, "cannot close NEP fitting JIT cache ",
                    temporary, ": errno ", errno);
        fd = -1;
        TORCH_CHECK(::rename(temporary.c_str(), path.c_str()) == 0,
                    "cannot publish NEP fitting JIT cache ", path,
                    ": errno ", errno);
        published = true;
    } catch (...) {
        if (fd >= 0) ::close(fd);
        if (!published) ::unlink(temporary.c_str());
        throw;
    }
}

Specialization specialization(const at::Tensor& reference, int d, int h, int q) {
    const c10::hip::HIPGuardMasqueradingAsCUDA guard(reference.device());
    hipDeviceProp_t properties{};
    C10_HIP_CHECK(hipGetDeviceProperties(&properties, reference.get_device()));
    int hiprtc_major = 0, hiprtc_minor = 0;
    check_hiprtc(hiprtcVersion(&hiprtc_major, &hiprtc_minor), "hiprtcVersion");
    const std::string arch = properties.gcnArchName;
    int runtime_version = 0, driver_version = 0;
    C10_HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    C10_HIP_CHECK(hipDriverGetVersion(&driver_version));
    std::ostringstream material;
    material << hip_source() << "|d=" << d << "|h=" << h << "|q=" << q
             << "|arch=" << arch << "|runtime=" << runtime_version << "|driver=" << driver_version << "|hiprtc=" << hiprtc_major << '.' << hiprtc_minor
             << "|std=c++17|-O3|hiprtc-v1";
    const std::string material_text = material.str();
    const uint64_t material_hash = fnv1a(material_text);
    std::ostringstream hash;
    hash << std::hex << std::setw(16) << std::setfill('0') << material_hash;
    std::ostringstream filename;
    filename << "nepfit-hiprtc-v1-" << arch << "-d" << d << "-h" << h << "-q" << q
             << '-' << hash.str() << ".hsaco";
    const char* override_dir = std::getenv("MATPL_NEP_JIT_CACHE");
    std::string root;
    if (override_dir && *override_dir) root = override_dir;
    else if (const char* xdg = std::getenv("XDG_CACHE_HOME")) root = xdg;
    else if (const char* home = std::getenv("HOME")) root = join_path(home, ".cache");
    else root = "/tmp";
    if (!(override_dir && *override_dir)) root = join_path(root, "matpl/nep_fitting");
    const std::string filename_text = filename.str();
    return {d, h, q, static_cast<int>(reference.get_device()), arch, hiprtc_major,
            hiprtc_minor, filename_text, root, join_path(root, filename_text)};
}

std::vector<char> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    TORCH_CHECK(input, "cannot open NEP fitting JIT cache ", path);
    const auto size = input.tellg();
    TORCH_CHECK(size > 0, "empty NEP fitting JIT cache ", path);
    std::vector<char> bytes(static_cast<size_t>(size));
    input.seekg(0);
    input.read(bytes.data(), size);
    TORCH_CHECK(input, "cannot read NEP fitting JIT cache ", path);
    return bytes;
}

std::vector<char> compile_code(const Specialization& spec) {
    struct Program {
        hiprtcProgram value = nullptr;
        ~Program() { if (value) hiprtcDestroyProgram(&value); }
    } owner;
    auto& program = owner.value;
    check_hiprtc(hiprtcCreateProgram(&program, hip_source().c_str(),
                                   "nep_fitting_hiprtc.hip", 0, nullptr, nullptr),
                "hiprtcCreateProgram");
    const std::string architecture = "--gpu-architecture=" + spec.arch;
    const std::string d = "-DJIT_D=" + std::to_string(spec.d);
    const std::string h = "-DJIT_H=" + std::to_string(spec.h);
    const std::string q = "-DJIT_Q=" + std::to_string(spec.q);
    const char* options[] = {"--std=c++17", "-O3", architecture.c_str(), d.c_str(), h.c_str(), q.c_str()};
    const hiprtcResult compile_result = hiprtcCompileProgram(program, 6, options);
    size_t log_size = 0;
    hiprtcGetProgramLogSize(program, &log_size);
    std::string log(log_size, '\0');
    if (log_size) hiprtcGetProgramLog(program, log.data());
    if (compile_result != HIPRTC_SUCCESS) {
        TORCH_CHECK(false, "HIPRTC failed for ", spec.key, ": ",
                    hiprtcGetErrorString(compile_result), "\n", log);
    }
    size_t code_size = 0;
    check_hiprtc(hiprtcGetCodeSize(program, &code_size), "hiprtcGetCodeSize");
    std::vector<char> code(code_size);
    check_hiprtc(hiprtcGetCode(program, code.data()), "hiprtcGetCode");
    return code;
}

std::shared_ptr<JitModule> load_module(const std::vector<char>& code) {
    auto result = std::make_shared<JitModule>();
    try {
        check_driver(hipModuleLoadData(&result->module, code.data()), "hipModuleLoadData");
        check_driver(hipModuleGetFunction(&result->forward, result->module,
                                         "fitting_atoms_forward"), "hipModuleGetFunction");
        check_driver(hipModuleGetFunction(&result->backward, result->module,
                                         "fitting_atoms_backward"), "hipModuleGetFunction");
        check_driver(hipModuleGetFunction(&result->parameter_partials, result->module,
                                         "fitting_parameter_partials"), "hipModuleGetFunction");
        check_driver(hipModuleGetFunction(&result->parameter_reduce, result->module,
                                         "fitting_parameter_reduce"), "hipModuleGetFunction");
    } catch (...) {
        if (result->module) hipModuleUnload(result->module);
        throw;
    }
    return result;
}

std::shared_ptr<JitModule> build_module(const at::Tensor& reference, int d, int h, int q) {
    const std::string process_key = std::to_string(reference.get_device()) + '|' +
        std::to_string(d) + '|' + std::to_string(h) + '|' + std::to_string(q);
    std::lock_guard<std::mutex> lock(module_mutex);
    if (auto found = modules.find(process_key); found != modules.end()) return found->second;
    const Specialization spec = specialization(reference, d, h, q);
    ensure_directory(spec.cache_dir);
    const c10::hip::HIPGuardMasqueradingAsCUDA guard(reference.device());
    const FileLock cache_lock(spec.cache_file + ".lock");
    std::shared_ptr<JitModule> result;
    bool cache_hit = false;
    if (is_regular_file(spec.cache_file)) {
        try {
            result = load_module(read_binary(spec.cache_file));
            cache_hit = true;
        } catch (const std::exception&) {
            TORCH_CHECK(::unlink(spec.cache_file.c_str()) == 0 || errno == ENOENT,
                        "cannot remove corrupt NEP fitting JIT cache ",
                        spec.cache_file, ": errno ", errno);
        }
    }
    if (!result) {
        const std::vector<char> code = compile_code(spec);
        result = load_module(code);
        write_binary_atomic(spec.cache_file, code);
    }
    if (const char* debug = std::getenv("MATPL_NEP_JIT_LOG")) {
        if (debug[0] == '1') std::fprintf(stderr, "[MatPL HIPRTC] %s %s device=%d\n",
            cache_hit ? "cache-hit" : "compiled", spec.cache_file.c_str(), spec.device);
    }
    modules.emplace(process_key, result);
    return result;
}

} // namespace

bool use_nep_fitting_hiprtc() {
    const char* enabled = std::getenv("MATPL_NEP_HIPRTC");
    const char* generic = std::getenv("MATPL_FITTING_GENERIC");
    const char* legacy = std::getenv("MATPL_FITTING_LEGACY");
    return !(enabled && enabled[0] == '0') &&
           !(generic && generic[0] == '1') &&
           !(legacy && legacy[0] == '1');
}

bool prepare_nep_fitting_jit(
    const at::Tensor& reference, int64_t d, int64_t h, int64_t q) {
    TORCH_CHECK(reference.is_cuda(), "NEP fitting JIT requires a HIP reference tensor");
    TORCH_CHECK(reference.scalar_type() == at::kDouble,
                "NEP fitting JIT requires an FP64 reference tensor");
    TORCH_CHECK(d >= 1 && d <= nep_fitting_limits::kMaxFeatures,
                "NEP fitting JIT requires 1 <= D <= ", nep_fitting_limits::kMaxFeatures);
    TORCH_CHECK(h >= 1 && h <= nep_fitting_limits::kMaxHiddenNeurons,
                "NEP fitting JIT requires 1 <= H <= ", nep_fitting_limits::kMaxHiddenNeurons);
    TORCH_CHECK(q == 1 || q == 2, "NEP fitting JIT requires Q=1 or Q=2");
    if (!use_nep_fitting_hiprtc()) return false;
    build_module(reference, d, h, q);
    return true;
}

void launch_nep_fitting_jit_forward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count, at::Tensor& y,
    at::Tensor& g) {
    auto module = build_module(x, x.size(1), w.size(2), v.size(2));
    const double* x_ptr = x.data_ptr<double>();
    const double* w_ptr = w.data_ptr<double>();
    const double* b_ptr = b.data_ptr<double>();
    const double* v_ptr = v.data_ptr<double>();
    const double* c_ptr = c.data_ptr<double>();
    const int64_t* ids_ptr = atom_ids.data_ptr<int64_t>();
    const int64_t* offsets_ptr = offsets.data_ptr<int64_t>();
    double* y_ptr = y.data_ptr<double>();
    double* g_ptr = g.data_ptr<double>();
    int n = static_cast<int>(x.size(0));
    void* args[] = {&x_ptr, &w_ptr, &b_ptr, &v_ptr, &c_ptr, &ids_ptr,
                    &offsets_ptr, &y_ptr, &g_ptr, &n};
    const auto stream = c10::hip::getCurrentHIPStream(x.get_device());
    check_driver(hipModuleLaunchKernel(module->forward,
        static_cast<unsigned int>((max_count + 7) / 8),
        static_cast<unsigned int>(w.size(0)), 1,
        static_cast<unsigned int>((x.size(1) > 96 ? 8 : 16) * 8), 1, 1, 0,
        reinterpret_cast<hipStream_t>(stream.stream()), args, nullptr),
        "hipModuleLaunchKernel(fitting_atoms_forward)");
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void launch_nep_fitting_jit_backward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count,
    const at::Tensor& grad_y, const at::Tensor& grad_g,
    std::vector<at::Tensor>& grads) {
    auto module = build_module(x, x.size(1), w.size(2), v.size(2));
    int d = static_cast<int>(x.size(1));
    int h = static_cast<int>(w.size(2));
    int q = static_cast<int>(v.size(2));
    int n = static_cast<int>(x.size(0));
    int types = static_cast<int>(w.size(0));
    const int tile = d > 96 ? 8 : 16;
    int tiles = (h + tile - 1) / tile;
    int stride = (d + 1 + q) * tile + q;
    int groups = std::min<int64_t>(
        types, kWorkspaceBytes / (tiles * stride * sizeof(double)));
    int slots = std::min<int64_t>(
        (max_count + 31) / 32,
        kWorkspaceBytes / (groups * tiles * stride * sizeof(double)));
    TORCH_CHECK(groups > 0 && slots > 0,
                "invalid NEP fitting JIT backward workspace dimensions");
    auto workspace = at::empty({groups, slots, tiles, stride}, x.options());

    const double* x_ptr = x.data_ptr<double>();
    const double* w_ptr = w.data_ptr<double>();
    const double* b_ptr = b.data_ptr<double>();
    const double* v_ptr = v.data_ptr<double>();
    const int64_t* ids_ptr = atom_ids.data_ptr<int64_t>();
    const int64_t* offsets_ptr = offsets.data_ptr<int64_t>();
    const double* a_ptr = grad_y.data_ptr<double>();
    const double* u_ptr = grad_g.data_ptr<double>();
    double* dx_ptr = grads[0].data_ptr<double>();
    void* backward_args[] = {&x_ptr, &w_ptr, &b_ptr, &v_ptr, &ids_ptr,
        &offsets_ptr, &a_ptr, &u_ptr, &dx_ptr, &n};
    const auto stream = c10::hip::getCurrentHIPStream(x.get_device());
    const auto cu_stream = reinterpret_cast<hipStream_t>(stream.stream());
    check_driver(hipModuleLaunchKernel(module->backward,
        static_cast<unsigned int>((max_count + 7) / 8),
        static_cast<unsigned int>(types), 1,
        static_cast<unsigned int>(tile * 8), 1, 1, 0, cu_stream,
        backward_args, nullptr), "hipModuleLaunchKernel(fitting_atoms_backward)");
    C10_HIP_KERNEL_LAUNCH_CHECK();

    double* partial_ptr = workspace.data_ptr<double>();
    double* dw_ptr = grads[1].data_ptr<double>();
    double* db_ptr = grads[2].data_ptr<double>();
    double* dv_ptr = grads[3].data_ptr<double>();
    double* dc_ptr = grads[4].data_ptr<double>();
    int parameter_size = d * h + h + h * q + q;
    for (int start = 0; start < types; start += groups) {
        const int batch = std::min(groups, types - start);
        void* partial_args[] = {&x_ptr, &w_ptr, &b_ptr, &v_ptr, &ids_ptr,
            &offsets_ptr, &a_ptr, &u_ptr, &partial_ptr, &n, &start, &slots};
        check_driver(hipModuleLaunchKernel(module->parameter_partials,
            static_cast<unsigned int>(slots), static_cast<unsigned int>(tiles),
            static_cast<unsigned int>(batch),
            static_cast<unsigned int>(tile * 8), 1, 1, 0, cu_stream,
            partial_args, nullptr),
            "hipModuleLaunchKernel(fitting_parameter_partials)");
        C10_HIP_KERNEL_LAUNCH_CHECK();

        void* reduce_args[] = {&partial_ptr, &dw_ptr, &db_ptr, &dv_ptr,
            &dc_ptr, &start, &slots};
        check_driver(hipModuleLaunchKernel(module->parameter_reduce,
            static_cast<unsigned int>((parameter_size + 255) / 256),
            static_cast<unsigned int>(batch), 1, 256, 1, 1, 0, cu_stream,
            reduce_args, nullptr),
            "hipModuleLaunchKernel(fitting_parameter_reduce)");
        C10_HIP_KERNEL_LAUNCH_CHECK();
    }
}

#endif
