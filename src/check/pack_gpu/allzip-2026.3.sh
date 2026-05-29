#!/bin/bash
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

rm -rf *part_a* ${SCRIPT_NAME}.sh.tar.gz MatPL-2026.3.tar.gz MatPL-2026.3.tar.gz.base64 ${SCRIPT_NAME}.sh

# 打包环境
# rm matpl-2026.3.tar.gz -rf
# conda pack -n matpl-2026.3
# cp matpl-2026.3.tar.gz bk/

# 将打包好的环境和 MatPL 目录打包成 tar.gz 文件
tar -czf MatPL-2026.3.tar.gz matpl-2026.3.tar.gz MatPL-2026.3 lammps-23-4

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

md5sum ${SCRIPT_NAME}.sh.tar.gz > md5.txt
md5sum ${SCRIPT_NAME}.sh.tar.gz.part_aa >> md5.txt
md5sum ${SCRIPT_NAME}.sh.tar.gz.part_ab >> md5.txt
md5sum ${SCRIPT_NAME}.sh.tar.gz.part_ac >> md5.txt
md5sum ${SCRIPT_NAME}.sh.tar.gz.part_ad >> md5.txt
md5sum ${SCRIPT_NAME}.sh.tar.gz.part_ae >> md5.txt

# copy file
TARGET_DIR="/share/public/PWMLFF_test_data/matpl-pack/2026.3-gpu"
UPDATE_DIR="${TARGET_DIR}/update${UPDATE_N}"
if [ -d "${UPDATE_DIR}" ]; then
  mv "${UPDATE_DIR}" "${UPDATE_DIR}-bk"
  echo "已存在目录 ${UPDATE_DIR}，已重命名为 ${UPDATE_DIR}-bk"
fi
mkdir -p "${UPDATE_DIR}"
cp -r md5.txt ${SCRIPT_NAME}.sh.tar.gz.part_* "${UPDATE_DIR}/"
echo "文件已复制到 ${UPDATE_DIR}"
