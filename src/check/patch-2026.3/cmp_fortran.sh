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

# 激活虚拟环境
if [ -f "$ENV_DIR/bin/activate" ]; then
  . "$ENV_DIR/bin/activate"
  echo "Virtual environment activated"
else
  echo "Warning: Virtual environment not found at $ENV_DIR/bin/activate"
fi

# 进入MatPL源码目录
if [ ! -d "$MATPL_DIR/src" ]; then
  echo "Error: Directory $MATPL_DIR/src does not exist"
  exit 1
fi

cd "$MATPL_DIR/src" || {
  echo "Error: Cannot change to directory $MATPL_DIR/src"
  exit 1
}

echo "Current directory: $(pwd)"
ls -la

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

# 创建必要的目录
mkdir -p bin
mkdir -p lib

# 编译Fortran代码
echo "Compiling Fortran codes..."

# 编译各个目录
for dir in pre_data/gen_feature pre_data/fit pre_data/fortran_code md/fortran_code; do
  if [ -d "$dir" ]; then
    echo "Compiling in $dir..."
    if ! $MAKE_CMD -C "$dir"; then
      echo "Error: Compilation failed in $dir"
      exit 1
    fi
  else
    echo "Warning: Directory $dir not found"
  fi
done

# 进入bin目录
cd bin || {
  echo "Error: Cannot change to bin directory"
  exit 1
}

# 创建符号链接
echo "Creating symbolic links..."

# 检查main_MD.x是否存在
if [ -f "../md/fortran_code/main_MD.x" ]; then
  ln -sf ../md/fortran_code/main_MD.x .
  echo "Created link for main_MD.x"
else
  echo "Warning: main_MD.x not found at ../md/fortran_code/main_MD.x"
fi

# 创建主程序链接
if [ -f "../../main.py" ]; then
  ln -sf ../../main.py ./MATPL
  ln -sf ../../main.py ./matpl
  ln -sf ../../main.py ./MatPL
  ln -sf ../../main.py ./PWMLFF
  ln -sf ../../main.py ./pwmlff
  echo "Created main program links"
else
  echo "Warning: ../../main.py not found"
fi

# 返回原始目录
cd "$PATCH_DIR" || {
  echo "Error: Cannot return to patch directory $PATCH_DIR"
  exit 1
}

echo "Fortran compilation successful!"
echo "All tasks completed!"
