cmake_minimum_required(VERSION 3.21)

include("${CMAKE_CURRENT_LIST_DIR}/../ResolveBackend.cmake")

if(MATPL_TEST_INVALID_BACKEND)
    matpl_resolve_backend("metal" "" "" ignored_result)
    return()
endif()

function(assert_backend requested cuda_version hip_version expected)
    matpl_resolve_backend(
        "${requested}"
        "${cuda_version}"
        "${hip_version}"
        actual
    )
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR
            "Backend resolution failed: request=${requested}, "
            "cuda=${cuda_version}, hip=${hip_version}; "
            "expected=${expected}, actual=${actual}"
        )
    endif()
endfunction()

assert_backend("CPU" "12.4" "6.3" "CPU")
assert_backend("CUDA" "" "" "CUDA")
assert_backend("HIP" "" "" "HIP")
assert_backend("AUTO" "12.4" "" "CUDA")
assert_backend("AUTO" "12.4" "6.3" "HIP")
assert_backend("AUTO" "" "" "CPU")
assert_backend("hip" "" "" "HIP")

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -DMATPL_TEST_INVALID_BACKEND=ON
        -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE invalid_result
    OUTPUT_QUIET
    ERROR_QUIET
)
if(invalid_result EQUAL 0)
    message(FATAL_ERROR "Unsupported backend was accepted")
endif()

message(STATUS "ResolveBackend tests passed")
