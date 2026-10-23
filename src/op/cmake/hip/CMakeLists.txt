enable_language(HIP)

if(NOT CMAKE_HIP_ARCHITECTURES AND DEFINED ENV{AMDGPU_TARGETS})
    set(CMAKE_HIP_ARCHITECTURES "$ENV{AMDGPU_TARGETS}")
endif()
if(NOT CMAKE_HIP_ARCHITECTURES)
    message(FATAL_ERROR
        "HIP backend requires CMAKE_HIP_ARCHITECTURES. Set it explicitly "
        "or export AMDGPU_TARGETS before configuring."
    )
endif()
message(STATUS "Enabled HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

file(GLOB_RECURSE HIP_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/kernel_hip/*.hip
    ${CMAKE_SOURCE_DIR}/kernel_hip/utilities/*.hip
)
list(REMOVE_ITEM HIP_SOURCES ${CMAKE_SOURCE_DIR}/src/CalcOps_bind.cpp)
list(REMOVE_ITEM HIP_SOURCES ${CMAKE_SOURCE_DIR}/src/nep_fitting_jit.cpp)

add_library(CalcOps_cuda SHARED ${HIP_SOURCES})
target_compile_definitions(CalcOps_cuda PRIVATE
    MATPL_HIP_ANGULAR_OUTPUT_PRUNE
    MATPL_ENABLE_FUSED_FITTING
    MATPL_FUSED_FITTING_HIP
)
target_include_directories(CalcOps_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/kernel_hip
    ${CMAKE_SOURCE_DIR}/kernel_hip/utilities
)
target_compile_options(CalcOps_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-return-type>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-result>
)
set_target_properties(CalcOps_cuda PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)
find_library(MATPL_HIPRTC_LIBRARY NAMES hiprtc
    HINTS "$ENV{ROCM_PATH}/lib" "$ENV{HIP_PATH}/lib" "${HIP_ROOT_DIR}/lib"
          "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib")
find_path(MATPL_HIPRTC_INCLUDE_DIR hip/hiprtc.h
    HINTS "$ENV{ROCM_PATH}/include" "$ENV{HIP_PATH}/include"
          "${CMAKE_HIP_COMPILER_ROCM_ROOT}/include")
if(NOT MATPL_HIPRTC_LIBRARY OR NOT MATPL_HIPRTC_INCLUDE_DIR)
    message(FATAL_ERROR "HIP fitting JIT requires libhiprtc and hip/hiprtc.h")
endif()
target_include_directories(CalcOps_cuda PRIVATE ${MATPL_HIPRTC_INCLUDE_DIR})
target_link_libraries(CalcOps_cuda PRIVATE ${TORCH_LIBRARIES} ${MATPL_HIPRTC_LIBRARY})

add_library(CalcOps_bind SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps_bind.cpp)
target_compile_definitions(CalcOps_bind PRIVATE
    MATPL_ENABLE_FUSED_FITTING
    MATPL_FUSED_FITTING_HIP
)
target_compile_features(CalcOps_bind PRIVATE cxx_std_17)
target_include_directories(CalcOps_bind PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(CalcOps_bind PRIVATE CalcOps_cuda ${TORCH_LIBRARIES})
