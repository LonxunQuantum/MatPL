#!/bin/bash
# =============================================================================
# MatPL-2025.3 CPU 
# =============================================================================
# 解析必选项 -nN (N为整数)
if [ $# -eq 0 ] || [[ "$1" != -n* ]]; then
  echo "Usage: $0 -nN (例如: -n5)"
  exit 1
fi

UPDATE_N="${1#-n}"
if ! [[ "$UPDATE_N" =~ ^[0-9]+$ ]]; then
  echo "Error: N必须是整数"
  exit 1
fi

SCRIPT_BASE="matpl_cpu-2025.3"
SCRIPT_NAME="${SCRIPT_BASE}-update${UPDATE_N}"

echo "=========================================="
echo "正在创建 MatPL CPU 更新包: ${SCRIPT_NAME}"
echo "=========================================="

# 清理旧文件
rm -rf *part_a* ${SCRIPT_NAME}.sh.tar.gz MatPL_cpu-2025.3.tar.gz.base64 ${SCRIPT_NAME}.sh MatPL_cpu-2025.3.tar.gz

# 打包环境（如果需要每次重新打包可取消注释下面3行）
# rm -rf matpl_cpu-2025.3.tar.gz
# conda pack -n matpl_cpu-2025.3
# cp matpl_cpu-2025.3.tar.gz bk/

# 将打包好的环境和 MatPL 目录打包成 tar.gz 文件
tar -czf MatPL_cpu-2025.3.tar.gz matpl_cpu-2025.3.tar.gz MatPL_cpu-2025.3 lammps-stable

# 将 tar.gz 文件编码成 base64
base64 MatPL_cpu-2025.3.tar.gz > MatPL_cpu-2025.3.tar.gz.base64

# 复制模板脚本并添加 base64 编码的 tar.gz 数据
cp matpl_cpu-2025.3.sh.template ${SCRIPT_NAME}.sh
cat MatPL_cpu-2025.3.tar.gz.base64 >> ${SCRIPT_NAME}.sh

# 创建时间戳文件
timestamp=$(date +"%Y-%m-%d-%H:%M")
echo "Package created at: $timestamp" > packtime-$timestamp
echo "This file indicates the packaging time of the installation package." >> packtime-$timestamp

# 打包最终的脚本
tar -czvf ${SCRIPT_NAME}.sh.tar.gz ${SCRIPT_NAME}.sh check_offenv_cpu.sh packtime-$timestamp
rm -f packtime-$timestamp

# 分割（CPU版本使用600M分割，与原脚本保持一致）
split -b 600M ${SCRIPT_NAME}.sh.tar.gz ${SCRIPT_NAME}.sh.tar.gz.part_

# 生成 md5
md5sum ${SCRIPT_NAME}.sh.tar.gz > md5.txt
md5sum ${SCRIPT_NAME}.sh.tar.gz.part_aa >> md5.txt
md5sum ${SCRIPT_NAME}.sh.tar.gz.part_ab >> md5.txt
md5sum ${SCRIPT_NAME}.sh.tar.gz.part_ac >> md5.txt
md5sum ${SCRIPT_NAME}.sh.tar.gz.part_ad >> md5.txt
md5sum ${SCRIPT_NAME}.sh.tar.gz.part_ae >> md5.txt

# 复制到目标目录（仿照GPU脚本，使用 updateN 子目录）
TARGET_DIR="/share/public/PWMLFF_test_data/matpl-pack/2025.3-cpu"
UPDATE_DIR="${TARGET_DIR}/update${UPDATE_N}"

if [ -d "${UPDATE_DIR}" ]; then
  mv "${UPDATE_DIR}" "${UPDATE_DIR}-bk"
  echo "已存在目录 ${UPDATE_DIR}，已重命名为 ${UPDATE_DIR}-bk"
fi

mkdir -p "${UPDATE_DIR}"
cp -r md5.txt ${SCRIPT_NAME}.sh.tar.gz.part_* "${UPDATE_DIR}/"

echo "=========================================="
echo "打包完成！"
echo "文件已复制到: ${UPDATE_DIR}"
echo "脚本名称: ${SCRIPT_NAME}.sh"
echo "分割文件: ${SCRIPT_NAME}.sh.tar.gz.part_*"
echo "MD5文件: md5.txt"
echo "=========================================="
echo "打包结束时间: $(date)"

