#include "./utilities/nep_mbgrad_opt.cuh"
#include "../include/nep_mb_secondgrad_jit.h"
#include "nep_mb_secondgrad_jit_source.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct ModuleState {
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
};

std::mutex module_mutex;
std::unordered_map<std::string, std::shared_ptr<ModuleState>> modules;

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

std::string read_jit_body() {
    std::istringstream input(kNepMbSecondgradJitSource);
    std::ostringstream body;
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("#include", 0) == 0 ||
            line.rfind("#pragma once", 0) == 0) continue;
        body << line << '\n';
    }
    return body.str();
}

const std::string& jit_body() {
    static const std::string body = read_jit_body();
    return body;
}

std::string specialization_name(int nmax, int nbasis, bool has_four,
                                bool has_five) {
    std::ostringstream name;
    name << "&nep_mb_secondgrad_fused<" << nmax << ", " << nbasis
         << ", 4, " << (has_four ? "true" : "false")
         << ", " << (has_five ? "true" : "false")
         << ", 4, false, 64>";
    return name.str();
}

std::shared_ptr<ModuleState> compile_module(
    int sm, int nmax, int nbasis, bool has_four, bool has_five) {
    const std::string expression = specialization_name(
        nmax, nbasis, has_four, has_five);
    const std::string source = jit_body()
        + "\ntemplate __global__ void " + expression.substr(1)
        + "(NepMbSecondGradArgs);\n";
    nvrtcProgram program = nullptr;
    check_nvrtc(nvrtcCreateProgram(&program, source.c_str(),
                                  "nep_mb_secondgrad_jit.cu", 0, nullptr, nullptr),
                "nvrtcCreateProgram");
    try {
        check_nvrtc(nvrtcAddNameExpression(program, expression.c_str()),
                    "nvrtcAddNameExpression");
        const std::string architecture = "--gpu-architecture=sm_" + std::to_string(sm);
        const char* options[] = {"--std=c++17", architecture.c_str()};
        const nvrtcResult compiled = nvrtcCompileProgram(program, 2, options);
        size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string log(log_size, '\0');
        if (log_size) nvrtcGetProgramLog(program, log.data());
        TORCH_CHECK(compiled == NVRTC_SUCCESS,
                    "NVRTC failed for ", expression, ": ",
                    nvrtcGetErrorString(compiled), "\n", log);
        const char* lowered = nullptr;
        check_nvrtc(nvrtcGetLoweredName(program, expression.c_str(), &lowered),
                    "nvrtcGetLoweredName");
        const std::string kernel_name(lowered);
        size_t cubin_size = 0;
        check_nvrtc(nvrtcGetCUBINSize(program, &cubin_size), "nvrtcGetCUBINSize");
        std::vector<char> cubin(cubin_size);
        check_nvrtc(nvrtcGetCUBIN(program, cubin.data()), "nvrtcGetCUBIN");
        check_nvrtc(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram");

        auto module = std::make_shared<ModuleState>();
        check_driver(cuModuleLoadData(&module->module, cubin.data()),
                     "cuModuleLoadData");
        try {
            check_driver(cuModuleGetFunction(&module->kernel, module->module,
                                             kernel_name.c_str()),
                         "cuModuleGetFunction");
        } catch (...) {
            cuModuleUnload(module->module);
            throw;
        }
        return module;
    } catch (...) {
        if (program != nullptr) nvrtcDestroyProgram(&program);
        throw;
    }
}

std::shared_ptr<ModuleState> module_for(
    int device, int nmax, int nbasis, bool has_four, bool has_five) {
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const int sm = prop.major * 10 + prop.minor;
    const std::string key = std::to_string(device) + ":" + std::to_string(sm)
        + ":" + specialization_name(nmax, nbasis, has_four, has_five);
    std::lock_guard<std::mutex> guard(module_mutex);
    const auto cached = modules.find(key);
    if (cached != modules.end()) return cached->second;
    check_driver(cuInit(0), "cuInit");
    auto module = compile_module(sm, nmax, nbasis, has_four, has_five);
    modules.emplace(key, module);
    return module;
}

} // namespace

void launch_nep_mb_secondgrad_jit(
    const NepMbSecondGradArgs& args, int nmax, int nbasis,
    bool has_four_body, bool has_five_body,
    int device, size_t shared_bytes) {
    if (args.atom_count == 0) return;
    C10_CUDA_CHECK(cudaSetDevice(device));
    const auto module = module_for(device, nmax, nbasis,
                                   has_four_body, has_five_body);
    NepMbSecondGradArgs launch_args = args;
    void* arguments[] = {&launch_args};
    const auto stream = c10::cuda::getCurrentCUDAStream(device);
    check_driver(cuLaunchKernel(module->kernel,
        static_cast<unsigned int>(args.atom_count), 1, 1,
        64, 1, 1, static_cast<unsigned int>(shared_bytes),
        reinterpret_cast<CUstream>(stream.stream()), arguments, nullptr),
        "cuLaunchKernel(nep_mb_secondgrad_fused)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
