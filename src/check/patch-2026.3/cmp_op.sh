#!/bin/bash

# Default make command (single core)
MAKE_CMD="make"

PATCH_DIR=$(pwd)
BASE_DIR=$1
VERSION=$2
MATPL_DIR=${BASE_DIR}/$3
CLEAN_ALL=$4
CPU_ONLY=$5

# 设置环境目录
if [ "$CPU_ONLY" = "0" ]; then
  ENV_DIR=${BASE_DIR}/matpl-${VERSION}
else
  ENV_DIR=${BASE_DIR}/matpl_cpu-${VERSION}
fi

# 检查环境目录是否存在
if [ -f "$ENV_DIR/bin/activate" ]; then
  . "$ENV_DIR/bin/activate"
else
  echo "Warning: Virtual environment not found at $ENV_DIR/bin/activate"
fi

echo "patch file dir is $PATCH_DIR"
echo "MatPL root dir is $BASE_DIR"
echo "ENV   root dir is $ENV_DIR"

# 检查目录是否存在
if [ -d "$BASE_DIR" ]; then
  ls "$BASE_DIR"
else
  echo "Error: Base directory $BASE_DIR does not exist"
  exit 1
fi

cd "$MATPL_DIR/src" || {
  echo "Error: Cannot change to directory $MATPL_DIR/src"
  exit 1
}

# Parse command line arguments（仅保留 -j 参数，已移除 -n）
shift 5

while [ $# -gt 0 ]; do
    case $1 in
        -j*)
            MAKE_CMD="make $1"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Using MAKE_CMD = $MAKE_CMD"

# 检查目录是否存在
if [ ! -d "op" ]; then
    echo "Error: 'op' directory not found in $(pwd)"
    exit 1
fi

cd op || {
    echo "Error: Cannot change to 'op' directory"
    exit 1
}

if [ "$CLEAN_ALL" = "1" ]; then
    rm -rf build
fi

mkdir -p build
cd build || {
    echo "Error: Cannot change to 'build' directory"
    exit 1
}

# 固定使用默认值 20（不再支持 -n 参数）
cmake -DNEP_TYPES=20 ..
if ! $MAKE_CMD; then
    echo "Error: OP operator compilation failed"
    exit 1
fi

cd "$PATCH_DIR" || {
    echo "Error: Cannot return to patch directory $PATCH_DIR"
    exit 1
}

echo "OP operator compilation successful!"
echo ""
