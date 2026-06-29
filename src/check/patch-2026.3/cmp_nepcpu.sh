#!/bin/bash

# Default make command (single core)
MAKE_CMD="make"

PATCH_DIR=$(pwd)
BASE_DIR="$1"
VERSION="$2"
MATPL_DIR="${BASE_DIR}/$3"
CLEAN_ALL="$4"
CPU_ONLY="$5"

# 设置环境目录
if [ "$CPU_ONLY" = "0" ]; then
  ENV_DIR="${BASE_DIR}/matpl-${VERSION}"
else
  ENV_DIR="${BASE_DIR}/matpl_cpu-${VERSION}"
fi

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

# 保存前5个参数，然后处理剩余的选项参数
ARG1="$1"
ARG2="$2"
ARG3="$3"
ARG4="$4"
ARG5="$5"

# 移除前5个位置参数，处理剩余的参数
shift 5

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

# make nep-cpu interface
NEP_CPU_DIR="${MATPL_DIR}/src/feature/nep_find_neigh"

if [ ! -d "$NEP_CPU_DIR" ]; then
    echo "Error: NEP-CPU directory not found: $NEP_CPU_DIR"
    exit 1
fi

cd "$NEP_CPU_DIR" || {
    echo "Error: Cannot change to directory $NEP_CPU_DIR"
    exit 1
}

# 清理旧文件
if [ "$CLEAN_ALL" = "1" ]; then
    echo "Cleaning previous build..."
    rm -rf build/*
    rm -f findneigh.so 2>/dev/null
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

if [ -n "$PYBIND11_DIR" ]; then
    echo "Using pybind11 directory: $PYBIND11_DIR"
    cmake -Dpybind11_DIR="$PYBIND11_DIR" ..
else
    echo "Warning: pybind11 not found via python -m pybind11, using default cmake"
    cmake ..
fi

# 编译
if ! $MAKE_CMD; then
    echo "Error: Build failed"
    exit 1
fi

# 复制生成的库文件
cp findneigh.* ../findneigh.so 2>/dev/null

echo "Feature dir compilation successful!"

# 返回原始目录
cd "$PATCH_DIR" || {
    echo "Error: Cannot return to patch directory $PATCH_DIR"
    exit 1
}
