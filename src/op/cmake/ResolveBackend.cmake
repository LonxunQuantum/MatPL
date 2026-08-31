function(matpl_resolve_backend requested cuda_version hip_version output_variable)
    string(TOUPPER "${requested}" normalized_request)
    set(supported_backends AUTO CUDA HIP CPU)
    if(NOT normalized_request IN_LIST supported_backends)
        message(FATAL_ERROR
            "Unsupported MATPL_GPU_BACKEND=${requested}. "
            "Expected one of AUTO, CUDA, HIP, or CPU."
        )
    endif()

    if(normalized_request STREQUAL "AUTO")
        if(NOT "${hip_version}" STREQUAL "")
            set(resolved_backend HIP)
        elseif(NOT "${cuda_version}" STREQUAL "")
            set(resolved_backend CUDA)
        else()
            set(resolved_backend CPU)
        endif()
    else()
        set(resolved_backend "${normalized_request}")
    endif()

    set(${output_variable} "${resolved_backend}" PARENT_SCOPE)
endfunction()
