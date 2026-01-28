#!/bin/bash

# Default make command (single core) and NEP types

MAKE_CMD="make"
NEP_TYPES=20  # 默认值

PATCH_DIR=$(pwd)
BASE_DIR=$1
VERSION=$2
ENV_DIR=${BASE_DIR}/matpl-${VERSION}
MATPL_DIR=${BASE_DIR}/$3
CLEAN_ALL=$4
source $ENV_DIR/bin/activate

echo "patch file dir is $PATCH_DIR"
echo "MatPL root dir is $BASE_DIR"
echo "ENV   root dir is $ENV_DIR"
ls $BASE_DIR

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -j*)
            MAKE_CMD="make $1"
            shift
            ;;
        -n*)
            # 提取 -n 后面的数字
            NEP_TYPES="${1#-n}"
            # 如果没有数字，则检查下一个参数是否为数字
            if [[ -z "$NEP_TYPES" ]]; then
                if [[ "$2" =~ ^[0-9]+$ ]]; then
                    NEP_TYPES="$2"
                    shift
                else
                    echo "Error: -n requires a numeric argument"
                    exit 1
                fi
            fi
            shift
            ;;
        *)
            # 忽略其他参数
            shift
            ;;
    esac
done

echo "Using NEP_TYPES = $NEP_TYPES"
echo "Using MAKE_CMD = $MAKE_CMD"

cd $MATPL_DIR/src/feature
# make nep-gpu interface
cd NEP_GPU/
if [ $CLEAN_ALL -eq 1 ]; then
    rm -rf build/
fi
mkdir -p build
cd build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && $MAKE_CMD
cp nep3_module*.so nep_gpu.so
cd $PATCH_DIR
echo "Feature dir compilation successful!"
