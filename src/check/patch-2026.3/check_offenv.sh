#!/bin/bash

# pwact>=0.3 pwdata>=0.4.7
# MatPL-2025.3

# 检查 ifort 编译器版本是否不小于19.1 和 MKL 库是否存在
check_ifort_mkl() {
    if command -v ifort &> /dev/null; then
        ifort_version=$(ifort --version | grep "ifort" | awk '{print $3}' | cut -d'.' -f1-2)
        major_version=$(echo $ifort_version | cut -d'.' -f1)
        minor_version=$(echo $ifort_version | cut -d'.' -f2)
        
        if [ "$major_version" -gt 19 ] || ([ "$major_version" -eq 19 ] && [ "$minor_version" -ge 1 ]); then
            echo "ifort version is no less than 19.1, current version is $ifort_version."
        else
            echo "ifort version is not greater than 19.1, current version is $ifort_version."
        fi
    else
        echo "ifort compiler not found."
    fi
    
    # 检查 MKL 库是否存在
    if [ -d "/opt/intel/mkl" ] || [ -d "$MKLROOT" ]; then
        echo "MKL library is installed."
    else
        echo "MKL library is not installed."
    fi
}

# 检查 GCC 版本是否为 8.x
check_gcc_version() {
    if command -v gcc &> /dev/null; then
        gcc_version=$(gcc -dumpversion | cut -d. -f1)
        if [ "$gcc_version" -eq 8 ]; then
            echo "GCC version is exactly 8, current version is $gcc_version."
        else
            echo "GCC version is not 8, current version is $gcc_version."
        fi
    else
        echo "GCC not found."
    fi
}

# 检查 CUDA 版本是否大于等于 11.8
check_cuda_version() {
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d"," -f1 | cut -c2-)
        major_version=$(echo $cuda_version | cut -d'.' -f1)
        minor_version=$(echo $cuda_version | cut -d'.' -f2)

        if [ "$major_version" -gt 11 ] || ([ "$major_version" -eq 11 ] && [ "$minor_version" -ge 8 ]); then
            echo "CUDA version is 11.8 or higher, current version is $cuda_version."
        else
            echo "CUDA version is lower than 11.8, current version is $cuda_version."
        fi
    else
        echo "nvcc command not found, CUDA might not be installed."
    fi
}

# 检查是否存在 nvcc 命令
check_nvcc() {
    if command -v nvcc &> /dev/null; then
        echo "nvcc command exists."
    else
        echo "nvcc command does not exist."
    fi
}

# 执行检查
check_ifort_mkl
check_gcc_version

check_cuda_version
check_nvcc

