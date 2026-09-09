#ifdef MATPL_ENABLE_FUSED_FITTING

#include "../include/nep_fitting_jit.h"
#include "../kernel/nep_fitting_jit_source.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <cctype>
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
#include <sys/stat.h>
#include <unistd.h>

namespace {
enum class JitMode { Disabled, Auto, Required };

struct JitModule {
    CUmodule module = nullptr;
    CUfunction forward = nullptr;
};

std::mutex module_mutex;
std::unordered_map<std::string, std::shared_ptr<JitModule>> modules;

JitMode jit_mode() {
    const char* raw = std::getenv("MATPL_NEP_FITTING_JIT");
    std::string value = raw ? raw : "auto";
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value.empty() || value == "auto") return JitMode::Auto;
    if (value == "0" || value == "off" || value == "false") return JitMode::Disabled;
    if (value == "1" || value == "on" || value == "true") return JitMode::Required;
    TORCH_CHECK(false, "MATPL_NEP_FITTING_JIT must be auto, 1, or 0; got ", value);
}

void check_driver(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    TORCH_CHECK(false, operation, " failed: ", name ? name : "CUDA_ERROR",
                " (", message ? message : "unknown error", ")");
}

void check_nvrtc(nvrtcResult result, const char* operation) {
    TORCH_CHECK(result == NVRTC_SUCCESS, operation, " failed: ",
                nvrtcGetErrorString(result));
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
    int d, h, q, sm, device;
    int nvrtc_major, nvrtc_minor;
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

Specialization specialization(const at::Tensor& reference, int d, int h, int q) {
    const c10::cuda::CUDAGuard guard(reference.device());
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, reference.get_device()));
    int nvrtc_major = 0, nvrtc_minor = 0;
    check_nvrtc(nvrtcVersion(&nvrtc_major, &nvrtc_minor), "nvrtcVersion");
    const int sm = properties.major * 10 + properties.minor;
    std::ostringstream material;
    material << kNepFittingJitSource << "|d=" << d << "|h=" << h << "|q=" << q
             << "|sm=" << sm << "|nvrtc=" << nvrtc_major << '.' << nvrtc_minor
             << "|std=c++14";
    const std::string material_text = material.str();
    const uint64_t material_hash = fnv1a(material_text);
    std::ostringstream hash;
    hash << std::hex << std::setw(16) << std::setfill('0') << material_hash;
    std::ostringstream filename;
    filename << "nepfit-v1-sm" << sm << "-d" << d << "-h" << h << "-q" << q
             << '-' << hash.str() << ".cubin";
    const char* override_dir = std::getenv("MATPL_NEP_JIT_CACHE");
    std::string root;
    if (override_dir && *override_dir) root = override_dir;
    else if (const char* xdg = std::getenv("XDG_CACHE_HOME")) root = xdg;
    else if (const char* home = std::getenv("HOME")) root = join_path(home, ".cache");
    else root = "/tmp";
    if (!(override_dir && *override_dir)) root = join_path(root, "matpl/nep_fitting");
    const std::string filename_text = filename.str();
    return {d, h, q, sm, static_cast<int>(reference.get_device()), nvrtc_major,
            nvrtc_minor, filename_text, root, join_path(root, filename_text)};
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

std::vector<char> compile_cubin(const Specialization& spec) {
    nvrtcProgram program = nullptr;
    check_nvrtc(nvrtcCreateProgram(&program, kNepFittingJitSource,
                                   "nep_fitting_jit.cu", 0, nullptr, nullptr),
                "nvrtcCreateProgram");
    const std::string architecture = "--gpu-architecture=sm_" + std::to_string(spec.sm);
    const std::string d = "-DJIT_D=" + std::to_string(spec.d);
    const std::string h = "-DJIT_H=" + std::to_string(spec.h);
    const std::string q = "-DJIT_Q=" + std::to_string(spec.q);
    const char* options[] = {"--std=c++14", architecture.c_str(), d.c_str(), h.c_str(), q.c_str()};
    const nvrtcResult compile_result = nvrtcCompileProgram(program, 5, options);
    size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    std::string log(log_size, '\0');
    if (log_size) nvrtcGetProgramLog(program, log.data());
    if (compile_result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&program);
        TORCH_CHECK(false, "NVRTC failed for ", spec.key, ": ",
                    nvrtcGetErrorString(compile_result), "\n", log);
    }
    size_t cubin_size = 0;
    check_nvrtc(nvrtcGetCUBINSize(program, &cubin_size), "nvrtcGetCUBINSize");
    std::vector<char> cubin(cubin_size);
    check_nvrtc(nvrtcGetCUBIN(program, cubin.data()), "nvrtcGetCUBIN");
    nvrtcDestroyProgram(&program);
    return cubin;
}

std::shared_ptr<JitModule> build_module(const at::Tensor& reference, int d, int h, int q) {
    const Specialization spec = specialization(reference, d, h, q);
    const std::string process_key = std::to_string(spec.device) + '|' + spec.key;
    std::lock_guard<std::mutex> lock(module_mutex);
    if (auto found = modules.find(process_key); found != modules.end()) return found->second;
    ensure_directory(spec.cache_dir);
    std::vector<char> cubin;
    if (is_regular_file(spec.cache_file)) cubin = read_binary(spec.cache_file);
    else {
        cubin = compile_cubin(spec);
        const auto temporary = spec.cache_file + ".tmp." + std::to_string(::getpid());
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        TORCH_CHECK(output, "cannot write NEP fitting JIT cache ", temporary);
        output.write(cubin.data(), static_cast<std::streamsize>(cubin.size()));
        output.close();
        TORCH_CHECK(std::rename(temporary.c_str(), spec.cache_file.c_str()) == 0,
                    "cannot publish NEP fitting JIT cache ", spec.cache_file,
                    ": errno ", errno);
    }
    const c10::cuda::CUDAGuard guard(reference.device());
    check_driver(cuInit(0), "cuInit");
    auto result = std::make_shared<JitModule>();
    check_driver(cuModuleLoadData(&result->module, cubin.data()), "cuModuleLoadData");
    check_driver(cuModuleGetFunction(&result->forward, result->module,
                                     "fitting_atoms_forward"), "cuModuleGetFunction");
    modules.emplace(process_key, result);
    return result;
}

std::shared_ptr<JitModule> module_for_mode(
    const at::Tensor& reference, int d, int h, int q) {
    const JitMode mode = jit_mode();
    if (mode == JitMode::Disabled) return nullptr;
    try {
        return build_module(reference, d, h, q);
    } catch (const std::exception& error) {
        if (mode == JitMode::Required) throw;
        TORCH_WARN_ONCE("NEP fitting JIT unavailable; using AOT kernel: ", error.what());
        return nullptr;
    }
}
} // namespace

bool prepare_nep_fitting_jit(
    const at::Tensor& reference, int64_t d, int64_t h, int64_t q) {
    TORCH_CHECK(reference.is_cuda(), "NEP fitting JIT requires a CUDA reference tensor");
    TORCH_CHECK(reference.scalar_type() == at::kDouble,
                "NEP fitting JIT requires an FP64 reference tensor");
    TORCH_CHECK(d >= 1 && d <= 96, "NEP fitting JIT requires 1 <= D <= 96");
    TORCH_CHECK(h >= 1 && h <= 100, "NEP fitting JIT requires 1 <= H <= 100");
    TORCH_CHECK(q == 1 || q == 2, "NEP fitting JIT requires Q=1 or Q=2");
    return module_for_mode(reference, d, h, q) != nullptr;
}

bool try_launch_nep_fitting_jit_forward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count, at::Tensor& y,
    at::Tensor& g) {
    auto module = module_for_mode(x, x.size(1), w.size(2), v.size(2));
    if (!module) return false;
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
    const auto stream = at::cuda::getCurrentCUDAStream();
    check_driver(cuLaunchKernel(module->forward,
        static_cast<unsigned int>((max_count + 7) / 8),
        static_cast<unsigned int>(w.size(0)), 1, 128, 1, 1, 0,
        reinterpret_cast<CUstream>(stream.stream()), args, nullptr),
        "cuLaunchKernel(fitting_atoms_forward)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

#endif
