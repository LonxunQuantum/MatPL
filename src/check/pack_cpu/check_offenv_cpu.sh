#!/bin/bash

# 环境检查结果标志
ENV_PASS=true
FORTRAN_COMPATIBLE=true
GCC_COMPATIBLE=true

# 检查 ifort 编译器版本是否不小于19.1 和 MKL 库是否存在
check_ifort_mkl() {
    local ifort_ok=true
    local mkl_ok=true
    
    echo "=== Checking ifort compiler and MKL library ==="
    
    if command -v ifort &> /dev/null; then
        ifort_version=$(ifort --version | grep "ifort" | awk '{print $3}' | cut -d'.' -f1-2)
        major_version=$(echo $ifort_version | cut -d'.' -f1)
        minor_version=$(echo $ifort_version | cut -d'.' -f2)
        
        if [ "$major_version" -gt 19 ] || ([ "$major_version" -eq 19 ] && [ "$minor_version" -ge 1 ]); then
            echo "✓ ifort version: $ifort_version (>= 19.1)"
        else
            echo "✗ ifort version: $ifort_version (< 19.1)"
            ifort_ok=false
            FORTRAN_COMPATIBLE=false
            # ifort版本不通过不影响整体环境检查，只显示警告
        fi
    else
        echo "✗ ifort compiler not found"
        ifort_ok=false
        FORTRAN_COMPATIBLE=false
        # ifort不存在不影响整体环境检查，只显示警告
    fi
    
    # 检查 MKL 库是否存在
    if [ -d "/opt/intel/mkl" ] || [ -d "$MKLROOT" ] || [ ! -z "$MKLROOT" ]; then
        echo "✓ MKL library is installed"
    else
        echo "✗ MKL library is not installed"
        mkl_ok=false
        FORTRAN_COMPATIBLE=false
        # MKL不存在不影响整体环境检查，只显示警告
    fi
}

# 检查 GCC 版本是否为 8.x 或更高
check_gcc_version() {
    echo "=== Checking GCC version ==="
    
    if command -v gcc &> /dev/null; then
        gcc_version=$(gcc -dumpversion | cut -d. -f1)
        if [ "$gcc_version" -ge 8 ]; then
            echo "✓ GCC version: $(gcc -dumpversion) (>= 8.0)"
        else
            echo "✗ GCC version: $(gcc -dumpversion) (< 8.0)"
            GCC_COMPATIBLE=false
            ENV_PASS=false
        fi
    else
        echo "✗ GCC not found"
        GCC_COMPATIBLE=false
        ENV_PASS=false
    fi
}

# 执行检查
echo "========================================"
echo "      CPU Environment Check Starting"
echo "========================================"
echo ""

check_ifort_mkl
echo ""
check_gcc_version
echo ""

echo "========================================"
echo "        Environment Summary"
echo "========================================"

# 总结输出
if [ "$ENV_PASS" = true ]; then
    echo "✓ Environment check completed. All requirements are satisfied."
    
    if [ "$FORTRAN_COMPATIBLE" = false ]; then
        echo ""
        echo "⚠ Warning: ifort compiler or MKL library is missing or does not meet version requirements."
        echo "  This will affect the compilation of Linear and NN models, but does not affect the use of DP and NEP models."
    fi
else
    echo "✗ Environment check failed. GCC check failed. Please check your GCC (>= 8.0)."
    
    if [ "$FORTRAN_COMPATIBLE" = false ]; then
        echo ""
        echo "⚠ Warning: ifort compiler or MKL library is missing or does not meet version requirements."
        echo "  This will affect the compilation of Linear and NN models, but does not affect the use of DP and NEP models."
    fi
fi

echo "========================================"
