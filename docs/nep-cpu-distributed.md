# NEP CPU 多节点、多进程训练

CPU/GPU 资源分配、`workers` 与计算线程的区别，以及完整 Slurm/计算节点直接运行命令，统一见 [训练资源与启动说明](nep-training-resources.md)。本文保留 CPU 构建与功能范围说明。

CPU 使用 Gloo 同步梯度，复用现有 NEP CPU 特征计算与 PyTorch 求导；支持能量、力、virial 训练。GPU 默认仍使用 NCCL。LKF/GKF 仍只支持单进程。

在 `nep.json` 顶层设置：

```json
"device": "cpu"
```

`device` 可选 `auto`（默认）、`cpu`、`cuda`。`dist_backend` 默认自动选择，CPU 无需额外填写；若显式填写，应使用 `gloo`。`workers` 控制每个训练进程的数据加载子进程数，不是训练进程数。

## 构建

需要与当前 Python/PyTorch 环境匹配的 CPU 算子库。在计算节点执行：

```bash
conda activate matpl-2026.3
module load cmake/3.31.6
source /opt/rh/devtoolset-8/enable
cd /data/home/wuxingxing/xcode/MatPL-dcu-dev/src
sh build.sh -j4
```

构建脚本按当前 PyTorch 后端和编译环境自动选择：CUDA 版 PyTorch 且 PATH 中的 `nvcc` 能正常运行时，编译 CUDA 和 CPU；HIP 版 PyTorch 且 `hipcc` 能正常运行时，编译 HIP 和 CPU；否则只编译 CPU。只安装 CUDA/HIP 版 PyTorch、没有对应编译器也可以构建 CPU。可先执行 `sh build.sh --dry-run -j4` 查看选择结果。

判断不依赖 GPU/DCU 是否可见，屏蔽 GPU 不会关闭已检测到的加速后端编译。加载编译环境后发生的算子编译错误仍会报错，不会自动退回 CPU。HIP 算子的编译器为 `hipcc`；DTK 的 `nvcc` 兼容层仅用于额外的 NEP-GPU 接口。

CPU 算子只链接 PyTorch CPU 运行库；CPU 构建不再要求 CUDA/HIP toolkit 或 NVIDIA 驱动。使用 CUDA 版 PyTorch 时，其自身随包提供的运行库仍需完整安装。

## Slurm：两节点、每节点两个训练进程

仓库示例为 `example/nep_cpu/run.slurm`。将脚本中的工作目录改为自己的训练目录，确保所有节点都能访问代码、数据和输出目录，再提交 `sbatch run.slurm`。

测试示例使用 `3080ti,new3080ti,q3` 分区并屏蔽 GPU；实际使用 CPU 分区时修改 partition 即可，不需要申请 GPU。进程总数由 Slurm 的节点数和 `ntasks-per-node` 决定。每个 rank 的 CPU 线程数、DataLoader workers 与 `cpus-per-task` 应配套设置，避免超量创建线程。

所有 rank 必须使用相同的 `MASTER_ADDR` 和 `MASTER_PORT`。多节点不使用 `localhost`；并行作业在同一主节点上要使用不同端口。

## 单节点与 torchrun

在计算节点、训练目录中运行：

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    /data/home/wuxingxing/xcode/MatPL-dcu-dev/main.py train nep.json
```

也支持多节点 torchrun 的 `RANK/WORLD_SIZE/LOCAL_RANK` 环境。直接使用 `python main.py train nep.json` 时，CPU 默认单进程；GPU 保留按可见卡数自动启动进程的行为。

batch_size 是每个 rank 的设置；例如四个 rank 各取 8 帧，一次全局更新共处理 32 帧。`mix` 同样是每个 rank 的原子预算。自动学习率缩放继续沿用现有配置，做单/多进程数值对照时应固定有效学习率和全局 batch。

checkpoint 仍由 rank 0 保存。需要恢复优化器状态并接着原 epoch 训练时，设置顶层 `recover_train=true` 和 `optimizer.reset_epoch=false`。原有默认值 `reset_epoch=true` 只继承模型权重，重新开始优化器和训练轮数；两者含义没有改变。

续训时配置统一使用当前 `nep.json`，ckpt 只恢复模型权重、优化器动量和已完成步数。学习率按当前配置与恢复步数重新计算；`warm_epochs` 按当前每轮步数换算，旧 scheduler 和 `warmup_updates` 不会覆盖新配置。详见 [checkpoint 恢复规则](nep-checkpoint-resume.md)。

本次使用 FP64 验证了两节点四进程、单节点两进程、CPU 单进程，以及能量/力/virial 损失的梯度和 Adam 更新一致性。测试数据为现有 MPtrj 小样本，包含 LMDB 和 extxyz 输入。

本功能只涉及 NEP 训练入口，不表示其他模型也已支持 CPU 分布式训练。编译和训练测试均应提交到计算节点。
