#!/bin/bash

# ====================== 参数解析 ======================
UPDATE_N=""
PULL_CODE=false
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -n*)
      UPDATE_N="${1#-n}"
      ;;
    -p)
      PULL_CODE=true
      ;;
    -h|--help)
      SHOW_HELP=true
      ;;
    *)
      echo "未知参数: $1"
      SHOW_HELP=true
      ;;
  esac
  shift
done

if [ "$SHOW_HELP" = true ]; then
  echo "用法: $0 -nN [-p] [-h]"
  echo ""
  echo "选项说明:"
  echo "  -nN    必选参数：设置更新编号 N（正整数）"
  echo "         例如 -n5 将生成 matpl-patch-update5-YYYY.MM.DD.sh"
  echo "         用于在安装包文件名中标记本次 updateN 版本"
  echo "  -p     可选参数：进入 MatPL-2026.3 目录执行 git pull origin 2026.3"
  echo "         用于拉取最新的代码后再进行打包"
  echo "  -h, --help  显示此帮助信息并退出"
  echo ""
  exit 0
fi

if [ -z "$UPDATE_N" ] || ! [[ "$UPDATE_N" =~ ^[0-9]+$ ]]; then
  echo "错误: 必须提供 -nN 参数，且 N 必须是正整数！"
  echo "使用 $0 -h 查看帮助"
  exit 1
fi

echo "=== 开始打包 matpl-patch-update${UPDATE_N} ==="

# ==================== git pull (如果指定了 -p) ====================
SOURCE_DIR="MatPL-2026.3"

if [ "$PULL_CODE" = true ]; then
  if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 目录 $SOURCE_DIR 不存在，无法执行 git pull!"
    exit 1
  fi
  echo "正在进入 $SOURCE_DIR 目录执行 git pull origin 2026.3 ..."
  cd "$SOURCE_DIR" || exit 1
  git pull origin 2026.3
  if [ $? -eq 0 ]; then
    echo "git pull origin 2026.3 执行成功。"
  else
    echo "警告: git pull 执行可能出现问题，请检查以上输出。"
  fi
  cd .. || exit 1
fi

# ==================== 查找并复制最新 pwact 和 pwdata 包 ====================
PYPIPACKAGES_DIR="$HOME/pack/pypipackages"

echo "=== 正在从 $PYPIPACKAGES_DIR 查找最新版本包 ==="

# 查找最新 pwact
LATEST_PWACT=$(ls -1 "$PYPIPACKAGES_DIR"/pwact-*.tar.gz 2>/dev/null | sort -V | tail -n 1)
if [ -z "$LATEST_PWACT" ]; then
    echo "错误: 在 $PYPIPACKAGES_DIR 下未找到任何 pwact-*.tar.gz 文件!"
    exit 1
fi

# 查找最新 pwdata
LATEST_PWDATA=$(ls -1 "$PYPIPACKAGES_DIR"/pwdata-*.tar.gz 2>/dev/null | sort -V | tail -n 1)
if [ -z "$LATEST_PWDATA" ]; then
    echo "错误: 在 $PYPIPACKAGES_DIR 下未找到任何 pwdata-*.tar.gz 文件!"
    exit 1
fi

echo "找到最新 pwact 版本: $(basename "$LATEST_PWACT")"
echo "找到最新 pwdata 版本: $(basename "$LATEST_PWDATA")"

# 复制到当前目录
cp "$LATEST_PWACT" .
cp "$LATEST_PWDATA" .

PWACT_DIR=$(basename "$LATEST_PWACT")
PWDATA_DIR=$(basename "$LATEST_PWDATA")

echo "已复制最新包到当前目录"

# 获取当前日期
CURRENT_DATE=$(date +%Y.%m.%d)

# 输出文件名
OUTPUT_TARFILE="matpl-patch-update${UPDATE_N}-${CURRENT_DATE}.tar.gz"
OUTPUT_BASE64="matpl-patch-update${UPDATE_N}-${CURRENT_DATE}.tar.gz.base64"
OUTPUT_SHFILE="matpl-patch-update${UPDATE_N}-${CURRENT_DATE}.sh"

# 检查必要文件
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 目录 $SOURCE_DIR 不存在!"
    exit 1
fi
if [ ! -f "$PWACT_DIR" ]; then
    echo "错误: 文件 $PWACT_DIR 不存在!"
    exit 1
fi
if [ ! -f "$PWDATA_DIR" ]; then
    echo "错误: 文件 $PWDATA_DIR 不存在!"
    exit 1
fi

# 删除旧的临时文件
rm -rf matpl-patch-*.tar.gz matpl-patch-*.tar.gz.base64 matpl-patch-*.sh

# 打包
tar --exclude="$SOURCE_DIR/.git" \
    --exclude="$SOURCE_DIR/.gitignore" \
    --exclude="$SOURCE_DIR/example" \
    -czf "$OUTPUT_TARFILE" "$SOURCE_DIR" "$PWACT_DIR" "$PWDATA_DIR" \
    check_offenv.sh check_diff.sh cmp_pip.sh cmp_nepcpu.sh cmp_nepgpu.sh \
    cmp_op.sh cmp_fortran.sh cmp_lmps.sh

# base64 编码
base64 "$OUTPUT_TARFILE" > "$OUTPUT_BASE64"

# 生成最终 sh 脚本
cp matpl-2026.3-patch.template "$OUTPUT_SHFILE"
cat "$OUTPUT_BASE64" >> "$OUTPUT_SHFILE"

# 复制最终 sh 脚本到共享目录
TARGET_DIR="/share/public/PWMLFF_test_data/matpl-pack/2026.3-gpu"
UPDATE_DIR="${TARGET_DIR}/update${UPDATE_N}"
mkdir -p "$UPDATE_DIR"
cp -f "$OUTPUT_SHFILE" "$UPDATE_DIR/"

# 完成提示
echo "=========================================="
echo "压缩完成！"
echo "生成的 sh 安装脚本: $(pwd)/$OUTPUT_SHFILE"
echo "sh 脚本文件大小: $(du -h "$OUTPUT_SHFILE" | cut -f1)"
echo "sh 脚本已复制到: ${UPDATE_DIR}/${OUTPUT_SHFILE}"
echo "=========================================="
