#!/bin/bash

# Default make command (single core) and NEP types

MAKE_CMD="make"
NEP_TYPES=20  # 默认值

PATCH_DIR=$(pwd)
BASE_DIR=$1
VERSION=$2
MATPL_DIR=${BASE_DIR}/$3
CLEAN_ALL=$4
CPU_ONLY=$5
if [ $CPU_ONLY -eq 0 ]; then
  ENV_DIR=${BASE_DIR}/matpl-${VERSION}
else
  ENV_DIR=${BASE_DIR}/matpl_cpu-${VERSION}
fi
echo "patch file dir is $PATCH_DIR"
echo "MatPL root dir is $BASE_DIR"
echo "ENV   root dir is $ENV_DIR"
ls $BASE_DIR

source $ENV_DIR/bin/activate

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

# make nep-cpu interface
cd $MATPL_DIR/src/feature/nep_find_neigh
if [ $CLEAN_ALL -eq 1 ]; then
    rm -rf build/*
    rm -f findneigh.so
fi
#rm -f findneigh.so
mkdir -p build
cd build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && $MAKE_CMD
cp findneigh.* ../findneigh.so
echo "Feature dir compilation successful!"
cd $PATCH_DIR

