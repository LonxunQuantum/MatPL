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
#ls $BASE_DIR

source $ENV_DIR/bin/activate
cd $MATPL_DIR/src

ls $MATPL_DIR/src

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

$MAKE_CMD -C pre_data/gen_feature
$MAKE_CMD -C pre_data/fit
$MAKE_CMD -C pre_data/fortran_code  # spack load gcc@7.5.0
$MAKE_CMD -C md/fortran_code

cd bin

#ln -s ../pre_data/mlff.py .
#ln -s ../pre_data/seper.py .
#ln -s ../pre_data/gen_data.py .
#ln -s ../pre_data/data_loader_2type.py .
#ln -s ../../utils/read_torch_wij.py . 
#ln -s ../../utils/plot_nn_test.py . 
#ln -s ../../utils/plot_mlff_inference.py .
#ln -s ../../utils/read_torch_wij_dp.py . 
ln -s ../md/fortran_code/main_MD.x .

ln -s ../../main.py ./MATPL
ln -s ../../main.py ./matpl
ln -s ../../main.py ./MatPL
ln -s ../../main.py ./PWMLFF
ln -s ../../main.py ./pwmlff
#ln -s ../../main_mnode.py ./MNEP

#chmod +x ./mlff.py
#chmod +x ./seper.py
#chmod +x ./gen_data.py
#chmod +x ./data_loader_2type.py
#chmod +x ./train.py
#chmod +x ./test.py
#chmod +x ./predict.py
#chmod +x ./read_torch_wij.py
#chmod +x ./read_torch_wij_dp.py
#chmod +x ./plot_nn_test.py
#chmod +x ./plot_mlff_inference.py 


echo "fortran compilation successful!"
cd $PATCH_DIR
