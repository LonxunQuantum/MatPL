# Resolve targets before loading Torch: its CMake otherwise probes visible GPUs
# and can add unrelated architectures to CMAKE_CUDA_FLAGS on build/login nodes.
function(matpl_resolve_cuda_architectures architectures_output torch_output)
    if(NOT "$ENV{MATPL_CUDA_ARCHITECTURES}" STREQUAL "")
        set(architectures "$ENV{MATPL_CUDA_ARCHITECTURES}")
    elseif(NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(architectures "${CMAKE_CUDA_ARCHITECTURES}")
    elseif(NOT "$ENV{CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(architectures "$ENV{CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(architectures "60;70;75;80;86;89;90")
    endif()

    set(torch_architectures "")
    foreach(arch IN LISTS architectures)
        # MatPL accepts numeric CMake targets (70;86); Torch expects 7.0;8.6.
        if(NOT arch MATCHES "^([1-9][0-9]*)([0-9])$")
            message(FATAL_ERROR
                "Invalid CUDA architecture '${arch}'. Use numeric targets, "
                "for example MATPL_CUDA_ARCHITECTURES='70;86'."
            )
        endif()
        list(APPEND torch_architectures "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    endforeach()
    list(REMOVE_DUPLICATES architectures)
    list(REMOVE_DUPLICATES torch_architectures)
    set(${architectures_output} "${architectures}" PARENT_SCOPE)
    set(${torch_output} "${torch_architectures}" PARENT_SCOPE)
endfunction()
