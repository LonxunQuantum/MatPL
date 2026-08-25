#!/bin/bash
set -euo pipefail
# 解析必选项 -nN (N为整数)
if [ $# -eq 0 ] || [[ "$1" != -n* ]]; then
  echo "Usage: $0 -nN  (例如: -n5)"
  exit 1
fi

UPDATE_N="${1#-n}"
if ! [[ "$UPDATE_N" =~ ^[0-9]+$ ]]; then
  echo "Error: N必须是整数"
  exit 1
fi

SCRIPT_BASE="matpl-2026.3"
SCRIPT_NAME="${SCRIPT_BASE}-update${UPDATE_N}"

rm -f "${SCRIPT_NAME}.sh.tar.gz" "${SCRIPT_NAME}.sh" MatPL-2026.3.tar.gz MatPL-2026.3.tar.gz.base64
rm -f "${SCRIPT_NAME}.sh.tar.gz.part_"*

# 打包环境
rm matpl-2026.3.tar.gz -rf
conda pack -n matpl-2026.3
cp matpl-2026.3.tar.gz bk/

# 将打包好的环境和 MatPL 目录打包成 tar.gz 文件
for dep_pkg in \
  lammps-offline-deps/OpenBLAS-0.3.23.tar.gz \
  lammps-offline-deps/gsl-2.7.1.tar.gz \
  lammps-offline-deps/plumed-src-2.9.4.tgz \
  lammps-offline-deps/voro++-0.4.6.tar.gz; do
  if [ ! -f "$dep_pkg" ]; then
    echo "Error: missing offline dependency package: $dep_pkg"
    exit 1
  fi
done
PACK_JOB_COUNT=${PACK_JOB_COUNT:-8} bash ./build-offline-deps.sh
for dep_lib in \
  lammps-offline-deps/install/lib/libopenblas.so \
  lammps-offline-deps/install/lib/libgsl.so \
  lammps-offline-deps/install/lib/libgslcblas.so \
  lammps-offline-deps/install/lib/libgfortran.so.3 \
  lammps-offline-deps/install/lib/libquadmath.so.0 \
  lammps-offline-deps/install/lib/libvoro++.a \
  lammps-offline-deps/install/include/voro++/voro++.hh \
  lammps-offline-deps/install/bin/offline-deps-smoke; do
  if [ ! -f "$dep_lib" ]; then
    echo "Error: missing prebuilt offline dependency: $dep_lib"
    exit 1
  fi
done
LD_LIBRARY_PATH="$PWD/lammps-offline-deps/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  lammps-offline-deps/install/bin/offline-deps-smoke
rm -rf lammps-offline-deps/build lammps-offline-deps/*.bad*
tar -czf MatPL-2026.3.tar.gz matpl-2026.3.tar.gz MatPL-2026.3 lammps-25-4 lammps-offline-deps

# 将 tar.gz 文件编码成 base64
base64 MatPL-2026.3.tar.gz > MatPL-2026.3.tar.gz.base64

# 复制模板脚本并添加 base64 编码的 tar.gz 数据
cp matpl-2026.3.sh.template ${SCRIPT_NAME}.sh
cat MatPL-2026.3.tar.gz.base64 >> ${SCRIPT_NAME}.sh

# 创建时间戳文件
timestamp=$(date +"%Y-%m-%d-%H:%M")
echo "Package created at: $timestamp" > packtime-$timestamp
echo "This file indicates the packaging time of the installation package." >> packtime-$timestamp

# 打包最终的脚本
tar -czvf ${SCRIPT_NAME}.sh.tar.gz ${SCRIPT_NAME}.sh check_offenv.sh packtime-$timestamp
rm -f packtime-$timestamp

# 分割
split -b 800M ${SCRIPT_NAME}.sh.tar.gz ${SCRIPT_NAME}.sh.tar.gz.part_

md5sum "${SCRIPT_NAME}.sh.tar.gz" "${SCRIPT_NAME}.sh.tar.gz.part_"* > md5.txt

# copy file
TARGET_DIR="/share/public/PWMLFF_test_data/matpl-pack/2026.3-gpu"
UPDATE_DIR="${TARGET_DIR}/update${UPDATE_N}"
if [ -d "${UPDATE_DIR}" ]; then
  mv "${UPDATE_DIR}" "${UPDATE_DIR}-bk"
  echo "已存在目录 ${UPDATE_DIR}，已重命名为 ${UPDATE_DIR}-bk"
fi
mkdir -p "${UPDATE_DIR}"
cp -r ${SCRIPT_NAME}.sh.tar.gz.part_* "${UPDATE_DIR}/"
(
  cd "${UPDATE_DIR}"
  md5sum "${SCRIPT_NAME}.sh.tar.gz.part_"* > md5.txt
)
echo "文件已复制到 ${UPDATE_DIR}"
