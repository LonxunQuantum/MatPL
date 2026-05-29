#!/bin/bash

JOB_COUNT=1
GPU_ARCH=${GPU_ARCH:-AMPERE86}
DPREC=${DPREC:-}
DPREC_ENABLED=false

PATCH_DIR=$(pwd)
BASE_DIR=$1
CPU_ONLY=$2
VERSION=$3
CLEAN_ALL=$4
shift 4

if [ "$CPU_ONLY" -eq 1 ]; then
  ENV_DIR=${BASE_DIR}/matpl_cpu-${VERSION}
else
  ENV_DIR=${BASE_DIR}/matpl-${VERSION}
fi

source $ENV_DIR/bin/activate
echo "ENV_DIR/bin/activate path for lammps install is: $ENV_DIR/bin/activate"

LAMMPS_NEPKK=${BASE_DIR}/lammps-${VERSION}

echo "patch file dir is $PATCH_DIR"
echo "MatPL root dir is $BASE_DIR"
echo "ENV   root dir is $ENV_DIR"
echo "LMPS  root dir is $LAMMPS_NEPKK"
ls $BASE_DIR

while [ $# -gt 0 ]; do
  case "$1" in
    -j)
      if [ -z "$2" ]; then
        echo "Option -j requires an argument." >&2
        exit 1
      fi
      JOB_COUNT=$2
      echo "Using $JOB_COUNT CPU cores for compilation"
      shift 2
      ;;
    -j*)
      JOB_COUNT=${1#-j}
      echo "Using $JOB_COUNT CPU cores for compilation"
      shift
      ;;
    -a|--gpu-arch)
      if [ -z "$2" ]; then
        echo "Option $1 requires an argument." >&2
        exit 1
      fi
      GPU_ARCH=$2
      shift 2
      ;;
    --gpu-arch=*)
      GPU_ARCH=${1#--gpu-arch=}
      shift
      ;;
    -d|--dprec)
      DPREC=ON
      shift
      ;;
    *)
      echo "Invalid option: $1" >&2
      exit 1
      ;;
  esac
done

case "${DPREC,,}" in
  1|true|yes|on)
    DPREC_ENABLED=true
    ;;
  ""|0|false|no|off)
    DPREC_ENABLED=false
    ;;
  *)
    echo "Invalid DPREC value: $DPREC. Use ON/OFF, true/false, or 1/0." >&2
    exit 1
    ;;
esac

if [ "$CPU_ONLY" -eq 1 ]; then
  echo "CPU-only installation detected, skipping lammps-${VERSION} GPU Kokkos compilation."
  cd $PATCH_DIR
  exit 0
fi

if [ ! -d "$LAMMPS_NEPKK" ]; then
  echo "Error: LAMMPS directory does not exist: $LAMMPS_NEPKK"
  cd $PATCH_DIR
  exit 1
fi

cd $LAMMPS_NEPKK

LAMMPS_BUILD_DIR=build
DPREC_CMAKE_OPTION=
if [ "$DPREC_ENABLED" = true ]; then
    LAMMPS_BUILD_DIR=build-64
    DPREC_CMAKE_OPTION="-DPREC_NEPINFER=ON"
fi

if [ "$CLEAN_ALL" -eq 1 ]; then
    echo "Cleaning LAMMPS build directory: $LAMMPS_BUILD_DIR"
    rm -rf "$LAMMPS_BUILD_DIR"
fi

mkdir -p "$LAMMPS_BUILD_DIR"
cd "$LAMMPS_BUILD_DIR"
echo "Using Kokkos CUDA architecture: $GPU_ARCH"
if [ "$DPREC_ENABLED" = true ]; then
    echo "DPREC_NEPINFER enabled; lammps-${VERSION} will be built in build-64"
fi

cmake -C ../cmake/presets/basic.cmake \
   -DPKG_MESONT=no \
   -DPKG_JPEG=no \
   -DPKG_KOKKOS=yes \
   -DPKG_NEP_KK=yes \
   -DKokkos_ENABLE_CUDA=yes \
   -DKokkos_ENABLE_OPENMP=yes \
   -DKokkos_ENABLE_CUDA_LAMBDA=yes \
   -DFFT_KOKKOS=CUFFT \
   -DKokkos_ARCH_${GPU_ARCH}=ON \
   $DPREC_CMAKE_OPTION \
   -DTEST_TIME=ON \
   ../cmake

if [ $? -ne 0 ]; then
    echo "Error: LAMMPS CMake configuration failed"
    cd $PATCH_DIR
    exit 1
fi

cmake --build . -j$JOB_COUNT
if [ $? -ne 0 ]; then
    echo "Error: LAMMPS build failed"
    cd $PATCH_DIR
    exit 1
fi

cat <<EOF > "$LAMMPS_NEPKK/env.sh"
# Environment for MatPL-${VERSION} lammps path
export PATH=$LAMMPS_NEPKK/$LAMMPS_BUILD_DIR:\$PATH
EOF

echo "The compilation of the lammps-${VERSION} has been completed."

cd $PATCH_DIR
echo ""
