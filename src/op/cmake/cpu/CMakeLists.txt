# 4. add_library
add_library(CalcOps_cpu SHARED 
    ${CMAKE_SOURCE_DIR}/src/CalcOps_cpu.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_calculate_nepneighbor.cpp
    ${CMAKE_SOURCE_DIR}/src/nep_cpu.cpp
)
target_include_directories(CalcOps_cpu PRIVATE ${CMAKE_SOURCE_DIR})
# Link only the CPU runtime. CUDA-enabled Torch's imported target also pulls
# in libcuda, which prevents loading this fallback on driver-free CPU nodes.
target_link_libraries(CalcOps_cpu PUBLIC
    "${MATPL_TORCH_PATH}/lib/libtorch_cpu.so"
    "${MATPL_TORCH_PATH}/lib/libc10.so"
)

add_library(CalcOps_bind_cpu SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps_bind_cpu.cpp)
target_compile_features(CalcOps_bind_cpu PRIVATE cxx_std_17)
target_link_libraries(CalcOps_bind_cpu PRIVATE CalcOps_cpu)

# 5. Post-build steps: Copy CPU libraries as GPU-compatible libraries
if(MATPL_RESOLVED_BACKEND STREQUAL "CPU")
    add_custom_command(TARGET CalcOps_bind_cpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libCalcOps_bind_cpu.so
            ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libCalcOps_bind.so
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libCalcOps_bind_cpu.so
            ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libCalcOps_cuda.so
        COMMENT "Copying CPU libraries to GPU-compatible libraries."
    )
endif()
