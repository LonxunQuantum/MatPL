#!/bin/bash

# Default make command (single core)
MAKE_CMD="make"

PATCH_DIR=$(pwd)
BASE_DIR="$1"
VERSION="$2"
ENV_DIR="${BASE_DIR}/matpl-${VERSION}"
MATPL_DIR="${BASE_DIR}/$3"
CLEAN_ALL="$4"

echo "patch file dir is $PATCH_DIR"
echo "MatPL root dir is $BASE_DIR"
echo "ENV   root dir is $ENV_DIR"

# 检查目录是否存在
if [ -d "$BASE_DIR" ]; then
  ls "$BASE_DIR"
else
  echo "Warning: Base directory $BASE_DIR does not exist"
fi

# 激活虚拟环境
if [ -f "$ENV_DIR/bin/activate" ]; then
  . "$ENV_DIR/bin/activate"
  echo "Virtual environment activated"
else
  echo "Warning: Virtual environment not found at $ENV_DIR/bin/activate"
fi

# 保存前4个参数，然后处理剩余的选项参数
ARG1="$1"
ARG2="$2"
ARG3="$3"
ARG4="$4"

# 移除前4个位置参数，处理剩余的参数
shift 4

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -j*)
            MAKE_CMD="make $1"
            shift
            ;;
        *)
            # 忽略其他参数
            shift
            ;;
    esac
done

echo "Using MAKE_CMD = $MAKE_CMD"

# 检查CUDA是否可用
NEP_GPU_DIR="${MATPL_DIR}/src/feature/NEP_GPU"

if [ ! -d "$NEP_GPU_DIR" ]; then
    echo "Error: NEP-GPU directory not found: $NEP_GPU_DIR"
    exit 1
fi

# 进入feature目录
cd "${MATPL_DIR}/src/feature" || {
    echo "Error: Cannot change to directory ${MATPL_DIR}/src/feature"
    exit 1
}

# make nep-gpu interface
cd "NEP_GPU" || {
    echo "Error: Cannot change to NEP_GPU directory"
    exit 1
}

# 清理旧文件
if [ "$CLEAN_ALL" = "1" ]; then
    echo "Cleaning previous build..."
    rm -rf build/
fi

# 创建构建目录
mkdir -p build
cd build || {
    echo "Error: Cannot change to build directory"
    exit 1
}

# 获取pybind11目录
PYBIND11_DIR=$(python -m pybind11 --cmakedir 2>/dev/null)
if [ -z "$PYBIND11_DIR" ]; then
    echo "Warning: Cannot find pybind11 directory, trying alternative..."
    # 尝试其他方法
    PYBIND11_DIR=$(python3 -m pybind11 --cmakedir 2>/dev/null)
fi

# 检查CUDA是否可用
CUDA_AVAILABLE=0
if command -v nvcc >/dev/null 2>&1 || [ -n "$CUDA_HOME" ] || [ -n "$CUDA_PATH" ]; then
    CUDA_AVAILABLE=1
    echo "CUDA detected, building with GPU support"
else
    echo "Warning: CUDA not detected, building without GPU support"
    echo "         To enable GPU support, please install CUDA toolkit"
fi

# 构建
if [ -n "$PYBIND11_DIR" ]; then
    echo "Using pybind11 directory: $PYBIND11_DIR"
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        cmake -Dpybind11_DIR="$PYBIND11_DIR" ..
    else
        echo "Building without CUDA support"
        cmake -Dpybind11_DIR="$PYBIND11_DIR" -DENABLE_CUDA=OFF ..
    fi
else
    echo "Warning: pybind11 not found via python -m pybind11, using default cmake"
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        cmake ..
    else
        cmake -DENABLE_CUDA=OFF ..
    fi
fi

# 编译
if ! $MAKE_CMD; then
    echo "Error: Build failed"
    # 如果失败，尝试不带CUDA构建
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        echo "Retrying without CUDA support..."
        cmake -DENABLE_CUDA=OFF ..
        if $MAKE_CMD; then
            echo "Build successful without CUDA support"
        else
            echo "Error: Build failed even without CUDA"
            exit 1
        fi
    else
        exit 1
    fi
fi

# 复制生成的库文件
NEP_FILES=$(ls nep3_module*.so 2>/dev/null | head -1)
if [ -n "$NEP_FILES" ]; then
    cp nep3_module*.so nep_gpu.so 2>/dev/null
    echo "Copied nep3_module*.so to nep_gpu.so"
else
    echo "Warning: No nep3_module*.so files found in build directory"
    echo "Checking for other library files..."
    ls -la *.so 2>/dev/null || echo "No .so files found"
fi

cd "$PATCH_DIR" || {
    echo "Error: Cannot return to patch directory $PATCH_DIR"
    exit 1
}

echo "Feature dir compilation successful!"
