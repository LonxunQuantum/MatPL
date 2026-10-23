cmake_minimum_required(VERSION 3.21)
include("${CMAKE_CURRENT_LIST_DIR}/../ResolveCudaArchitectures.cmake")

if(MATPL_TEST_INVALID_ARCH)
    set(ENV{MATPL_CUDA_ARCHITECTURES} "sm_70")
    matpl_resolve_cuda_architectures(ignored_arch ignored_torch)
    return()
endif()

function(assert_architectures expected expected_torch)
    matpl_resolve_cuda_architectures(actual actual_torch)
    if(NOT actual STREQUAL expected OR NOT actual_torch STREQUAL expected_torch)
        message(FATAL_ERROR
            "Expected ${expected} / ${expected_torch}, got ${actual} / ${actual_torch}"
        )
    endif()
endfunction()

set(ENV{MATPL_CUDA_ARCHITECTURES} "70")
set(CMAKE_CUDA_ARCHITECTURES "86")
set(ENV{CMAKE_CUDA_ARCHITECTURES} "80")
set(ENV{TORCH_CUDA_ARCH_LIST} "8.0")
assert_architectures("70" "7.0")

set(ENV{MATPL_CUDA_ARCHITECTURES} "70;86;70")
assert_architectures("70;86" "7.0;8.6")
set(ENV{MATPL_CUDA_ARCHITECTURES} "100")
assert_architectures("100" "10.0")

unset(ENV{MATPL_CUDA_ARCHITECTURES})
assert_architectures("86" "8.6")
unset(CMAKE_CUDA_ARCHITECTURES)
assert_architectures("80" "8.0")
unset(ENV{CMAKE_CUDA_ARCHITECTURES})
assert_architectures("60;70;75;80;86;89;90" "6.0;7.0;7.5;8.0;8.6;8.9;9.0")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -DMATPL_TEST_INVALID_ARCH=ON -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE invalid_result OUTPUT_QUIET ERROR_QUIET
)
if(invalid_result EQUAL 0)
    message(FATAL_ERROR "Invalid architecture was accepted")
endif()
message(STATUS "CUDA architecture resolution tests passed")
