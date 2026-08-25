# 1. 包含包
# PyTorch's Caffe2 CMake config opportunistically discovers external MKL.
# The PyTorch wheel already contains its CPU backend, so keep a separately
# installed pip MKL runtime from becoming a link dependency of CalcOps_cuda.
set(CMAKE_DISABLE_FIND_PACKAGE_MKL TRUE)
find_package(Torch REQUIRED)
enable_language(CUDA)  # 使用现代CMake方式启用CUDA

# 2. 设置 CUDA 架构支持
set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90") 
message(STATUS "Enabled CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# 3. 全局 CUDA 编译选项
set(CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS}"
    "--default-stream per-thread"     # 避免隐式同步
    "--ptxas-options=-v"             # 输出寄存器/共享内存使用
    "-Xcompiler -Wall"               # 启用警告
    "-O3"                           # 优化级别
    "-Wno-deprecated-gpu-targets"    # 抑制弃用警告
)

# 4. 收集所有源文件
file(GLOB_RECURSE GPU_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/kernel/*.cu
    ${CMAKE_SOURCE_DIR}/kernel/utilities/*.cu
)

# 移除绑定文件（绑定文件单独编译）
list(REMOVE_ITEM GPU_SOURCES ${CMAKE_SOURCE_DIR}/src/CalcOps_bind.cpp)

# 5. 创建核心计算库（把所有.cpp和.cu文件都编译到这里）
add_library(CalcOps_cuda SHARED ${GPU_SOURCES})

# 设置头文件包含路径
target_include_directories(CalcOps_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/kernel
    ${CMAKE_SOURCE_DIR}/kernel/utilities
)

# 设置CUDA架构属性
set_target_properties(CalcOps_cuda PROPERTIES
    CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90"
    POSITION_INDEPENDENT_CODE ON
)

# 添加编译选项
#target_compile_options(CalcOps_cuda PRIVATE
#    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
#    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
#)

# 链接Torch库
target_link_libraries(CalcOps_cuda PRIVATE ${TORCH_LIBRARIES})

# 6. 创建Python绑定库
add_library(CalcOps_bind SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps_bind.cpp)
target_compile_features(CalcOps_bind PRIVATE cxx_std_17)

# 设置绑定库的头文件路径
target_include_directories(CalcOps_bind PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# 链接核心库和Torch
target_link_libraries(CalcOps_bind PRIVATE CalcOps_cuda ${TORCH_LIBRARIES})
