#!/bin/bash

# 获取当前日期（年月日格式）
CURRENT_DATE=$(date +%Y.%m.%d)

# 定义源目录和输出文件名
SOURCE_DIR="Patch-MatPL-2025.3"
PWACT_DIR="pwact-0.4.2.tar.gz"
PWDATA_DIR="pwdata-0.5.4.tar.gz"
OUTPUT_TARFILE="matpl-patch-${CURRENT_DATE}.tar.gz"
OUTPUT_BASE64="matpl-patch-${CURRENT_DATE}.tar.gz.base64"
OUTPUT_SHFILE="matpl-patch-${CURRENT_DATE}.sh" 

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 目录 $SOURCE_DIR 不存在!"
    exit 1
fi

# 检查源目录是否存在
if [ ! -f "$PWACT_DIR" ]; then
    echo "错误: 文件 $PWACT_DIR 不存在!"
    exit 1
fi

# 检查源文件是否存在
if [ ! -f "$PWDATA_DIR" ]; then
    echo "错误: 文件 $PWDATA_DIR 不存在!"
    exit 1
fi

# 删除临时文件
rm -rf matpl-patch-*.tar.gz  matpl-patch-*.tar.gz.base64 matpl-patch-*.sh

# 使用tar命令压缩，排除指定的目录和文件
tar --exclude="$SOURCE_DIR/.git" \
    --exclude="$SOURCE_DIR/.gitignore" \
    --exclude="$SOURCE_DIR/example" \
    -czf "$OUTPUT_TARFILE" "$SOURCE_DIR" "$PWACT_DIR" "$PWDATA_DIR" check_offenv.sh check_diff.sh cmp_pip.sh cmp_nepcpu.sh cmp_nepgpu.sh cmp_op.sh cmp_fortran.sh cmp_lmps.sh cmp_lmps_fortran.sh

# 将 tar.gz 文件编码成 base64
base64 $OUTPUT_TARFILE > $OUTPUT_BASE64

# 复制模板脚本并添加 base64 编码的 tar.gz 数据
cp matpl-2025.3-patch.template $OUTPUT_SHFILE
cat $OUTPUT_BASE64 >> $OUTPUT_SHFILE

# 输出完成信息
echo "压缩完成！"
echo "生成的 sh 安装脚本: $(pwd)/$OUTPUT_SHFILE"
echo "sh 脚本文件大小: $(du -h "$OUTPUT_SHFILE" | cut -f1)"

