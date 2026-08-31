cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(CalcOps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MATPL_GPU_BACKEND AUTO CACHE STRING "AUTO, CUDA, HIP, or CPU")
set_property(CACHE MATPL_GPU_BACKEND PROPERTY STRINGS AUTO CUDA HIP CPU)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import torch; print('|'.join([str(int(torch._C._GLIBCXX_USE_CXX11_ABI)), torch.utils.cmake_prefix_path, torch.version.cuda or 'NONE', getattr(torch.version, 'hip', None) or 'NONE', torch.__path__[0]]))"
    OUTPUT_VARIABLE MATPL_TORCH_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE MATPL_TORCH_INFO_RESULT
    ERROR_VARIABLE MATPL_TORCH_INFO_ERROR
)
if(NOT MATPL_TORCH_INFO_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Unable to query PyTorch with ${Python3_EXECUTABLE}: "
        "${MATPL_TORCH_INFO_ERROR}"
    )
endif()

string(REPLACE "|" ";" MATPL_TORCH_INFO_LIST "${MATPL_TORCH_INFO}")
list(LENGTH MATPL_TORCH_INFO_LIST MATPL_TORCH_INFO_LENGTH)
if(NOT MATPL_TORCH_INFO_LENGTH EQUAL 5)
    message(FATAL_ERROR "Unexpected PyTorch probe output: ${MATPL_TORCH_INFO}")
endif()
list(GET MATPL_TORCH_INFO_LIST 0 MATPL_TORCH_ABI)
list(GET MATPL_TORCH_INFO_LIST 1 MATPL_TORCH_CMAKE_PREFIX_PATH)
list(GET MATPL_TORCH_INFO_LIST 2 MATPL_TORCH_CUDA_VERSION)
list(GET MATPL_TORCH_INFO_LIST 3 MATPL_TORCH_HIP_VERSION)
list(GET MATPL_TORCH_INFO_LIST 4 MATPL_TORCH_PATH)
if(MATPL_TORCH_CUDA_VERSION STREQUAL "NONE")
    set(MATPL_TORCH_CUDA_VERSION "")
endif()
if(MATPL_TORCH_HIP_VERSION STREQUAL "NONE")
    set(MATPL_TORCH_HIP_VERSION "")
endif()

if(MATPL_TORCH_ABI STREQUAL "0")
    message(STATUS "Using _GLIBCXX_USE_CXX11_ABI=0 from PyTorch")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
else()
    message(STATUS "Using _GLIBCXX_USE_CXX11_ABI=1 from PyTorch")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
endif()

list(APPEND CMAKE_PREFIX_PATH "${MATPL_TORCH_CMAKE_PREFIX_PATH}")
set(CMAKE_DISABLE_FIND_PACKAGE_MKL TRUE)
find_package(Torch REQUIRED)

include_directories("${Python3_INCLUDE_DIRS}")
include_directories("${MATPL_TORCH_PATH}/include")
include_directories("${MATPL_TORCH_PATH}/include/torch/csrc/api/include")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/obj")
file(MAKE_DIRECTORY
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResolveBackend.cmake")
matpl_resolve_backend(
    "${MATPL_GPU_BACKEND}"
    "${MATPL_TORCH_CUDA_VERSION}"
    "${MATPL_TORCH_HIP_VERSION}"
    MATPL_RESOLVED_BACKEND
)

if(MATPL_RESOLVED_BACKEND STREQUAL "CUDA")
    if(MATPL_TORCH_CUDA_VERSION STREQUAL "")
        message(FATAL_ERROR
            "MATPL_GPU_BACKEND=CUDA requires a CUDA PyTorch build. "
            "Current PyTorch reports CUDA='${MATPL_TORCH_CUDA_VERSION}', "
            "HIP='${MATPL_TORCH_HIP_VERSION}'. Use a CUDA PyTorch environment "
            "or configure -DMATPL_GPU_BACKEND=CPU."
        )
    endif()
    find_program(MATPL_NVCC_EXECUTABLE nvcc)
    if(NOT MATPL_NVCC_EXECUTABLE)
        message(FATAL_ERROR
            "CUDA backend selected, but nvcc was not found. Load the CUDA "
            "toolkit or configure -DMATPL_GPU_BACKEND=CPU."
        )
    endif()
    set(CMAKE_CUDA_COMPILER "${MATPL_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler")
elseif(MATPL_RESOLVED_BACKEND STREQUAL "HIP")
    if(MATPL_TORCH_HIP_VERSION STREQUAL "")
        message(FATAL_ERROR
            "MATPL_GPU_BACKEND=HIP requires a HIP PyTorch build. "
            "Current PyTorch reports CUDA='${MATPL_TORCH_CUDA_VERSION}', "
            "HIP='${MATPL_TORCH_HIP_VERSION}'. Use the DTK/HIP PyTorch environment "
            "or configure -DMATPL_GPU_BACKEND=CPU."
        )
    endif()
    find_program(MATPL_HIPCC_EXECUTABLE hipcc)
    if(NOT MATPL_HIPCC_EXECUTABLE)
        message(FATAL_ERROR
            "HIP backend selected, but hipcc was not found. Source the DTK "
            "environment or configure -DMATPL_GPU_BACKEND=CPU."
        )
    endif()
endif()

string(TOLOWER "${MATPL_RESOLVED_BACKEND}" MATPL_RESOLVED_BACKEND_LOWER)
file(WRITE "${CMAKE_BINARY_DIR}/matpl_backend.txt"
    "${MATPL_RESOLVED_BACKEND_LOWER}\n"
)
message(STATUS
    "MatPL backend: requested=${MATPL_GPU_BACKEND}, "
    "resolved=${MATPL_RESOLVED_BACKEND}, "
    "torch.cuda=${MATPL_TORCH_CUDA_VERSION}, torch.hip=${MATPL_TORCH_HIP_VERSION}"
)

add_subdirectory("cmake/${MATPL_RESOLVED_BACKEND_LOWER}")
