cmake_minimum_required(VERSION 3.18)
project(NEP_GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 设置 CUDA 架构支持
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
endif()

# Kepler sm_30
# Maxwell, GTX980 sm_52
# Pascal, GTX1080, Titan X, Titan Xp sm_60
# Volta, Titan V sm_70
# Turing, RTX 2060, RTX 2070, RTX 2080, RTX 2080Ti, Titan RTX sm_75
# Ampere, A100 A800 sm_80
# Ampere, RTX 3070, RTX 3080, RTX 3090 sm_86 
# Ampere, RTX 4090 sm_89
# Hopper H100 H200 sm_90

# 引入 pybind11
find_package(pybind11 REQUIRED)

# 使用当前环境提供的 CUDA toolkit。NVIDIA 环境使用原生 toolkit，
# DCU 环境使用 DTK 提供的 CUDA 兼容层与 nvcc 包装器。
find_package(CUDAToolkit REQUIRED)

# 添加 pybind11 包装代码
pybind11_add_module(nep_gpu
    force/nep_pybind.cpp  # Pybind11 包装的 C++ 文件
    force/nep.cu          # CUDA 代码
    force/box.cu
    utilities/gpu_vector.cu # 其他 CUDA 实现
    utilities/main_common.cu
    utilities/error.cu
)

set_target_properties(nep_gpu PROPERTIES
    SUFFIX ".so"
    # Link the runtime explicitly below. This prevents CMake from adding the
    # NVIDIA-only cudadevrt archive, which DTK's CUDA compatibility layer does
    # not provide.
    CUDA_RUNTIME_LIBRARY None
)

# 包含 GPU toolkit 和本地目录
target_include_directories(nep_gpu PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/force
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities
)

# DTK 为这些标准 CUDA targets 提供兼容实现。
target_link_libraries(nep_gpu PRIVATE CUDA::cudart CUDA::cufft)
