#ifdef MATPL_ENABLE_FUSED_FITTING

#include "../include/nep_fitting_jit.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

namespace {
enum class JitMode { Disabled, Auto, Required };

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

uint64_t fnv1a(const std::string& text) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char byte : text) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::string specialization_key(const at::Tensor& reference, int d, int h, int q) {
    const c10::cuda::CUDAGuard guard(reference.device());
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, reference.get_device()));
    int nvrtc_major = 0, nvrtc_minor = 0;
    const nvrtcResult result = nvrtcVersion(&nvrtc_major, &nvrtc_minor);
    TORCH_CHECK(result == NVRTC_SUCCESS, "nvrtcVersion failed: ", nvrtcGetErrorString(result));
    std::ostringstream material;
    material << "nep-fitting-jit-v1|d=" << d << "|h=" << h << "|q=" << q
             << "|sm=" << properties.major << properties.minor
             << "|nvrtc=" << nvrtc_major << '.' << nvrtc_minor
             << "|std=c++14|tile=16|atoms=8";
    const std::string value = material.str();
    std::ostringstream key;
    key << value << "|src=" << std::hex << std::setw(16) << std::setfill('0')
        << fnv1a(value);
    return key.str();
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
    const JitMode mode = jit_mode();
    if (mode == JitMode::Disabled) return false;
    const std::string key = specialization_key(reference, d, h, q);
    if (mode == JitMode::Required) {
        TORCH_CHECK(false, "NEP fitting JIT compiler is not implemented yet for ", key);
    }
    TORCH_WARN_ONCE("NEP fitting JIT compiler is not implemented yet; using AOT kernel (", key, ")");
    return false;
}

#endif
