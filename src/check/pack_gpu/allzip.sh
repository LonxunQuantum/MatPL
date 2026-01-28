#!/bin/bash

rm -rf *part_a*  matpl-2025.3.sh.tar.gz MatPL-2025.3.tar.gz MatPL-2025.3.tar.gz.base64 matpl-2025.3.sh

# 打包环境
#rm matpl-2025.3.tar.gz -rf
#conda pack -n matpl-2025.3
#cp matpl-2025.3.tar.gz bk/

# 将打包好的环境和 MatPL 目录打包成 tar.gz 文件
tar -czf MatPL-2025.3.tar.gz matpl-2025.3.tar.gz MatPL-2025.3 lammps-stable

# 将 tar.gz 文件编码成 base64
base64 MatPL-2025.3.tar.gz > MatPL-2025.3.tar.gz.base64

# 复制模板脚本并添加 base64 编码的 tar.gz 数据
cp matpl-2025.3.sh.template matpl-2025.3.sh
cat MatPL-2025.3.tar.gz.base64 >> matpl-2025.3.sh

# 打包最终的脚本
tar -czvf matpl-2025.3.sh.tar.gz matpl-2025.3.sh check_offenv.sh

# 分割
split -b 800M matpl-2025.3.sh.tar.gz matpl-2025.3.sh.tar.gz.part_

md5sum matpl-2025.3.sh.tar.gz > md5.txt
md5sum matpl-2025.3.sh.tar.gz.part_aa >> md5.txt
md5sum matpl-2025.3.sh.tar.gz.part_ab >> md5.txt
md5sum matpl-2025.3.sh.tar.gz.part_ac >> md5.txt
md5sum matpl-2025.3.sh.tar.gz.part_ad >> md5.txt
md5sum matpl-2025.3.sh.tar.gz.part_ae >> md5.txt

# copy file
cp -r md5.txt matpl-2025.3.sh.tar.gz.part_* /share/public/PWMLFF_test_data/matpl-pack/2025.3-gpu/

