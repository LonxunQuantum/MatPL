#!/bin/bash

# 默认使用单核编译
JOB_COUNT=1
NEP_TYPES=""  # 初始化为空


PATCH_DIR=$(pwd)
BASE_DIR=$1
CPU_ONLY=$2
VERSION=$3
CLEAN_ALL=$4
if [ $CPU_ONLY -eq 1 ]; then
  ENV_DIR=${BASE_DIR}/matpl_cpu-${VERSION}
  MATPL_DIR=${BASE_DIR}/MatPL_cpu-${VERSION}
else
  ENV_DIR=${BASE_DIR}/matpl-${VERSION}
  MATPL_DIR=${BASE_DIR}/MatPL-${VERSION}
fi

source $ENV_DIR/bin/activate
echo "ENV_DIR/bin/activate path for lammps-fortran is: $ENV_DIR/bin/activate"

LAMMPS_FORTRAN=$BASE_DIR/lammps-fortran

echo "patch file dir is $PATCH_DIR"
echo "MatPL root dir is $BASE_DIR"
echo "ENV   root dir is $ENV_DIR"
echo "LMPS  root dir is $LAMMPS_FORTRAN"
ls $BASE_DIR

# 解析命令行参数
while getopts "j:n:" opt; do
  case $opt in
    j)
      JOB_COUNT=$OPTARG
      echo "Using $JOB_COUNT CPU cores for compilation"
      ;;
    n)
      NEP_TYPES=$OPTARG
      echo "Using NEP_TYPES = $NEP_TYPES"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# 编译 lammps-fortran
cd $LAMMPS_FORTRAN/src/MATPL/fortran_code
  if [ $CLEAN_ALL -eq 1 ]; then
      make clean
  fi
  if [ $JOB_COUNT -gt 1 ]; then
      make -j1 # $JOB_COUNT
  else
      make
  fi
cd ../../
echo "The compilation of the lammps-fortran MATPL library has been completed. Start compiling LAMMPS ..."

# 继续编译 LAMMPS 模块 to src
if [ $CLEAN_ALL -eq 1 ]; then
    make clean-all
fi
rm lmp_mpi -rf
make yes-KSPACE
make yes-MANYBODY
make yes-REAXFF
make yes-MOLECULE
make yes-QEQ
make yes-REPLICA
make yes-RIGID
make yes-MEAM
make yes-MC
make yes-MATPL
make yes-SHOCK

if [ $JOB_COUNT -gt 1 ]; then
    make mpi -j$JOB_COUNT mode=shared
else
    make mpi mode=shared
fi

echo "The compilation of the lammps-fortran has been completed."

# 检查文件是否生成成功
if [ -f "lmp_mpi" ]; then
    echo "Lammps fortran version completed successfully!"
else
    echo "Lammps fortran version compilation  errors. Please check the installation logs!"
fi

cd $PATCH_DIR

