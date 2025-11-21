#!/bin/bash

# Default make command (single core) and NEP types
MAKE_CMD="make"
NEP_TYPES=20  # 默认值

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

mkdir -p bin
mkdir -p lib

# make nep-cpu interface
cd feature/nep_find_neigh
rm -rf build/*
rm -f findneigh.so
mkdir -p build
cd build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && $MAKE_CMD
cp findneigh.* ../findneigh.so
cd ../../
# make nep-gpu interface
mkdir NEP_GPU/build
cd NEP_GPU/build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && $MAKE_CMD
cp nep3_module*.so nep_gpu.so
cd ../../../

cd bin
ln -s ../../main.py ./MATPL
ln -s ../../main.py ./matpl
ln -s ../../main.py ./MatPL
ln -s ../../main.py ./PWMLFF
ln -s ../../main.py ./pwmlff
ln -s ../../main_mnode.py ./MNEP

cd ..            # back to src dir

cd op
rm build -r
# python setup.py install --user
mkdir build
cd build
# for bigmodel the types should be 100
cmake -DNEP_TYPES=$NEP_TYPES ..
$MAKE_CMD
cd ..
cd ..

current_path=$(pwd)
parent_path=$(dirname "$current_path")

# write enviromenet to env.sh
cat <<EOF > ../env.sh
# Load for MatPL
export PYTHONPATH=$parent_path:\$PYTHONPATH

export PATH=$current_path/bin:\$PATH
EOF

echo ""
echo ""
echo "================================="
echo "MatPL has been successfully installed. Please load the MatPL environment variables before use."
echo "You can load the environment variables by running (recommended):"
echo ""
echo "  source $parent_path/env.sh"
echo ""
echo "Or by executing the following commands:"
echo ""
echo "  export PYTHONPATH=$parent_path:\$PYTHONPATH"
echo "  export PATH=$current_path/bin:\$PATH"
echo ""
echo "=================================="
