# 4. add_library
add_library(CalcOps_cpu SHARED 
    ${CMAKE_SOURCE_DIR}/src/CalcOps_cpu.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_calculate_nepneighbor.cpp
    ${CMAKE_SOURCE_DIR}/src/nep_cpu.cpp
)
target_include_directories(CalcOps_cpu PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(CalcOps_cpu PRIVATE ${TORCH_LIBRARIES})
# target_link_libraries(CalcOps_cpu PRIVATE "/data/home/hfhuang/software/anaconda3/envs/pytorch2/lib/python3.11/site-packages/torch/lib/libtorch.so")
# target_link_libraries(CalcOps_cpu PRIVATE "/data/home/hfhuang/software/anaconda3/envs/pytorch2/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so")
target_link_libraries(CalcOps_cpu PRIVATE ${TORCH_PATH}/lib/libtorch.so)
target_link_libraries(CalcOps_cpu PRIVATE ${TORCH_PATH}/lib/libtorch_cpu.so)

add_library(CalcOps_bind_cpu SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps_bind_cpu.cpp)
target_compile_features(CalcOps_bind_cpu PRIVATE cxx_std_17)
target_link_libraries(CalcOps_bind_cpu PRIVATE CalcOps_cpu ${TORCH_LIBRARIES})

# 5. Post-build steps: Copy CPU libraries as GPU-compatible libraries
if (NOT HAS_CUDA_SUPPORT)
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
