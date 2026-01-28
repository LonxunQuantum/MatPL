# include 
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

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
)

# 4. 显式指定多架构代码生成
#list(APPEND CUDA_NVCC_FLAGS
#    "-gencode arch=compute_60,code=sm_60"  # P100
#    "-gencode arch=compute_61,code=sm_61"  # 1080ti
#    "-gencode arch=compute_70,code=sm_70"  # V100
#    "-gencode arch=compute_75,code=sm_75"  # 2080
#    "-gencode arch=compute_80,code=sm_80"  # A100/A800
#    "-gencode arch=compute_86,code=sm_86"  # RTX 3090,3080ti
#    "-gencode arch=compute_89,code=sm_89"  # RTX 4090
#    "-gencode arch=compute_90,code=sm_90"  # H100 H200
#)

# 5. 添加 TYPES 定义，可以从外部传入
if(NOT DEFINED NEP_TYPES)
    set(NEP_TYPES 20)  # 默认值
endif()
message(STATUS "NEP TYPES: ${NEP_TYPES}")
add_definitions(-DTYPESIN=${NEP_TYPES})

# add_library
add_library(CalcOps_cuda SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps.cpp)
target_include_directories(CalcOps_cuda PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(CalcOps_cuda PRIVATE ${TORCH_LIBRARIES})

# register_op
file(GLOB CPP_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB CU_FILES ${CMAKE_SOURCE_DIR}/kernel/*.cu)
file(GLOB CU_FILES_UTILITIES ${CMAKE_SOURCE_DIR}/kernel/utilities/*.cu)
list(APPEND CU_FILES ${CU_FILES_UTILITIES})
file(GLOB CALC_FILES ${CMAKE_SOURCE_DIR}/src/CalcOp*.cpp)
list(REMOVE_ITEM CPP_FILES ${CALC_FILES})

foreach(CPP_FILE ${CPP_FILES})
	get_filename_component(CPP_LIB ${CPP_FILE} NAME)
	add_library(${CPP_LIB} SHARED ${CPP_FILE})
	set_target_properties(${CU_LIB} PROPERTIES
		CUDA_ARCHITECTURES "60;61;70;75;80;86;89"
		POSITION_INDEPENDENT_CODE ON
	)
	target_link_libraries(CalcOps_cuda PRIVATE ${CPP_LIB} ${TORCH_LIBRARIES})
endforeach(CPP_FILE)

foreach(CU_FILE ${CU_FILES})
	get_filename_component(CU_LIB ${CU_FILE} NAME)
	add_library(${CU_LIB} SHARED ${CU_FILE})
	target_link_libraries(CalcOps_cuda PRIVATE ${CU_LIB} ${TORCH_LIBRARIES})
endforeach(CU_FILE)

add_library(CalcOps_bind SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps_bind.cpp)
target_compile_features(CalcOps_bind PRIVATE cxx_std_17)
target_link_libraries(CalcOps_bind PRIVATE CalcOps_cuda ${TORCH_LIBRARIES})
