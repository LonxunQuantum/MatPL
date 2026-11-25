#!/usr/bin/env bash
# MatPL 自动测试的站点配置模板。在仓库根目录复制一份，再按你的集群环境修改。
# cp src/auto_test/site.example.sh /data/home/wuxingxing/tmp/matpl-site.sh
#
# run 和 base build 都可以读取这份配置。例如先预览作业命令：
# python3 -B src/auto_test/auto_test.py run \
#   --work-dir /data/home/wuxingxing/tmp \
#   --suite test \
#   --site-config /data/home/wuxingxing/tmp/matpl-site.sh \
#   --dry-run
#
# 配置优先级从高到低为命令行、此文件、框架内置默认值。

# GPU/DCU 用例的候选 Slurm 分区。多个分区用逗号分隔，只用 3090 时写 "3090"。
# 对应 --partition。
MATPL_QA_PARTITION="3090,q4"

# CPU 用例的候选分区。例如写 "cpu"，CPU 作业就只提交到 cpu 分区。
# 留空时沿用 MATPL_QA_PARTITION。对应 --cpu-partition。
MATPL_QA_CPU_PARTITION=""

# 集群登记的加速卡资源名。NVIDIA 集群通常用 "gpu"，部分 DCU 集群用 "dcu"。
# 这里只填资源名，卡数由用例决定。例如双卡用例生成 --gres=gpu:2。
MATPL_QA_GRES="gpu"

# 每个 Slurm 作业的最长运行时间，格式为 HH:MM:SS。对应 --time。
MATPL_QA_TIME="00:30:00"

# 框架最多同时提交并等待的用例数，包括正在排队和运行的作业。
# 改成 1 会逐个提交。这个值不影响每个用例申请的 CPU 或 GPU 数量。对应 --jobs。
MATPL_QA_JOBS=4

# 计算节点上用于初始化 Conda 的脚本路径。请指向实际安装位置。对应 --conda-sh。
MATPL_QA_CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"

# 作业启动时通过 conda activate 激活的环境。这里填已安装的 MatPL 环境名。
# 对应 --conda-env。
MATPL_QA_CONDA_ENV="matpl-2026.3"

# 作业启动时用 module load 加载的 CUDA 模块。工具链已经预加载，或使用 HIP 环境时，
# 可设为 ""，框架就会跳过这一步。对应 --cuda-module。
MATPL_QA_CUDA_MODULE="cuda/11.8-share"

# 要测试另一份 MatPL 代码时，取消下面的注释并填写仓库根目录。
# 该目录须包含 env.sh。不设置时，使用当前测试框架所在的 MatPL 仓库。对应 --code。
# MATPL_QA_CODE="/path/to/MatPL"
