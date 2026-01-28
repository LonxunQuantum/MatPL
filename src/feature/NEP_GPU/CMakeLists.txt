cmake_minimum_required(VERSION 3.15)
project(NEP_GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 设置 CUDA 架构支持
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)

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

# CUDA 相关配置
find_package(CUDA 11.8 REQUIRED)
enable_language(CUDA)

# 添加 pybind11 包装代码
pybind11_add_module(nep3_module
    force/nep3_pybind.cpp  # Pybind11 包装的 C++ 文件
    force/nep3.cu          # CUDA 代码
    force/box.cu
    utilities/gpu_vector.cu # 其他 CUDA 实现
    utilities/main_common.cu
    utilities/error.cu
)

# 包含 CUDA 和本地目录
target_include_directories(nep3_module PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/force
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities
)

# 链接 CUDA 库
target_link_libraries(nep3_module PRIVATE ${CUDA_LIBRARIES})

