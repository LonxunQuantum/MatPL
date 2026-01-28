enable_language(HIP)

# 1. 依赖项
# Torch（ROCm 版本）
find_package(Torch REQUIRED)

add_library(lib_thirdparty INTERFACE)
target_link_libraries(lib_thirdparty INTERFACE ${TORCH_LIBRARIES})
target_include_directories(lib_thirdparty INTERFACE ${TORCH_INCLUDE_DIRS})

# 2. 设置 DCU 架构（对应 CUDA 的 compute capability）
set(AMDGPU_TARGETS "gfx936" CACHE STRING "AMD GPU targets")
set(CMAKE_HIP_ARCHITECTURES "${AMDGPU_TARGETS}")
message(STATUS "Enabled AMDGPU targets: ${AMDGPU_TARGETS}")

# 3. NEP_TYPES 定义（保持不变）
if(NOT DEFINED NEP_TYPES)
    set(NEP_TYPES 20)
endif()
message(STATUS "NEP TYPES: ${NEP_TYPES}")
add_definitions(-DTYPESIN=${NEP_TYPES})

# 4. 收集源文件
file(GLOB_RECURSE GPU_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/kernel/*.hip
)

# 移除绑定文件
list(REMOVE_ITEM GPU_SOURCES ${CMAKE_SOURCE_DIR}/src/CalcOps_bind.cpp)

# 5. 核心计算库 (把所有 cpp 和 hip 都在这里编译)
add_library(CalcOps_cuda SHARED ${GPU_SOURCES})

# 设置头文件包含路径
target_include_directories(CalcOps_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIR} # 继承自顶层
)

# 编译选项
target_compile_options(CalcOps_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-return-type -Wno-unused-result -std=c++17>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result>
)

# 链接 Torch 和 Python 库
target_link_libraries(CalcOps_cuda PRIVATE lib_thirdparty)

# 6. Python 绑定库 (CalcOps_bind)
add_library(CalcOps_bind SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps_bind.cpp)

target_include_directories(CalcOps_bind PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIR}
)

# 链接核心库
target_link_libraries(CalcOps_bind PRIVATE CalcOps_cuda lib_thirdparty)

# 安装/输出设置 (确保库文件生成到顶层指定的位置)
set_target_properties(CalcOps_cuda PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set_target_properties(CalcOps_bind PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
