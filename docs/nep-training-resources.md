# NEP CPU/GPU 训练资源与启动说明

适用范围：`nep-dcu/dev` 当前的 NEP 训练入口，更新于 2026-09-17。支持 CPU/GPU 单节点、多节点、多进程训练；本文以 Adam/AdamW 为例，LKF/GKF 仅支持单进程。

**训练和编译必须在计算节点执行，禁止在登录节点进行大规模测试。** 登录节点可以编辑配置和提交 `sbatch`。直接登录计算节点运行的示例，要求该节点资源已分配给你，或集群明确允许直接使用。

## 1. 先区分三种数量

| 设置 | 控制什么 | 不控制什么 |
| --- | --- | --- |
| Slurm `--ntasks-per-node` / torchrun `--nproc-per-node` | 每节点的训练进程数，每个进程称为一个 rank | 数据加载进程数、计算线程数 |
| Slurm `--cpus-per-task` | 分配给一个 rank 及其子进程共享的 CPU 资源 | 不会自动启动这么多个训练进程或设置 `workers` |
| `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` | 对应计算库的线程数量上限 | 不负责申请 CPU，也不保证所有算子都使用这么多线程 |
| `nep.json` 顶层 `workers` | 每个 rank 的 DataLoader 数据加载子进程数，默认 **1** | 不控制训练 rank 数，不自动读取 `cpus-per-task` |
| Slurm `--gres=gpu:4` | 每节点申请 4 张 GPU | 不自动启动 4 个训练进程 |

Slurm 中的 CPU 数可能按物理核或硬件线程计数，取决于集群配置。直接运行时也应以实际允许使用的 CPU 为准，而不是整台机器的标称核数。

### CPU：多个 rank 分数据，线程协作计算一个 rank 的任务

例如两节点、每节点 24 CPU，设置 `ntasks-per-node=4`、`cpus-per-task=6`：

- 每节点 4 个训练 rank，每个 rank 获得 6 CPU；全局 8 个 rank、48 CPU。
- 各 rank 处理自己的 batch，计算梯度后通过 Gloo 同步，再更新模型。
- 若计算线程上限设为 6，支持多线程的算子可在这 6 CPU 上协作计算本 rank 的任务；这不表示再拆成 6 个独立 batch。
- 若 `workers=1`，每个 rank 的训练 DataLoader 再有 1 个数据加载子进程，与训练进程共享这 6 CPU，不额外获得 CPU。

CPU 路径中仍有串行操作，因此不能用“线程数 × 单线程速度”估算性能。训练和验证若都保留 persistent workers，存活的数据加载进程还可能多于单个 loader 的数量。

### GPU：通常一张卡对应一个 rank

每节点 4 卡、4 rank、`cpus-per-task=6`，表示每张卡对应的训练进程及其数据加载子进程共享 6 CPU。CPU 负责读取、解码、组装 batch、Python 调度和提交 GPU 工作，主要模型计算在 GPU 上执行。

因此，GPU 训练分配了 6 CPU，也可以把计算库线程上限设为 1，并让数据加载使用部分 CPU。`workers` 不必等于 6；盲目增加它可能增加内存、I/O 竞争和上下文切换。

资源申请参数的含义见 [Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html#OPT_cpus-per-task)，线程变量见 [PyTorch 线程配置](https://docs.pytorch.org/docs/2.14/threading_environment_variables.html)。

## 2. nep.json 与环境

把下面字段合入已有 `nep.json` 的**顶层**，保留模型、数据、优化器等其他配置；以下片段不是完整训练配置。

CPU：

```json
{
  "device": "cpu",
  "workers": 1
}
```

GPU：

```json
{
  "device": "cuda",
  "workers": 1
}
```

`device` 默认 `auto`：有可用 GPU 时使用 GPU，否则使用 CPU。正式实验建议显式设置，避免设备选择与预期不同。`dist_backend` 默认自动选择 CPU/Gloo、GPU/NCCL，一般无需填写。当前 extxyz 等非 LMDB 加载路径应使用 `workers>=1`；LMDB 路径支持 `workers=0`，表示在训练进程内加载数据。

下文使用本集群路径；其他环境需替换 Conda、代码和训练目录。算子应预先构建，见 [CPU 构建说明](nep-cpu-distributed.md#构建) 和 [CUDA/HIP 构建说明](building-cuda-hip.md)。标准命令仍是在源码 `src/` 中执行 `sh build.sh -j4`：检测到匹配的 GPU/DCU 编译环境时编译加速后端与 CPU，否则只编译 CPU。

不要在 JSON 中保留与启动命令冲突的 `master_addr`、`master_port`；当前代码优先使用 JSON 中这两个字段。下面示例由启动器或环境变量提供地址和端口，JSON 中省略它们即可。

## 3. Slurm 提交运行

这里统一使用 **srun 启动每个训练 rank**。`sbatch` 申请资源，`srun` 才启动多个训练进程；不要把下面的 `srun python ...` 改成每个 task 再运行一次 `torchrun`。

### 3.1 CPU：两节点，每节点 24 CPU、4 个训练进程

在含有 `nep.json` 的训练目录保存为 `cpu.slurm`：

```bash
#!/bin/bash
#SBATCH --job-name=nep-cpu
#SBATCH --partition=3080ti,new3080ti,q3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --output=cpu-%j.out

set -euo pipefail
source /data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh
conda activate matpl-2026.3
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
export MASTER_ADDR MASTER_PORT

REPO=/data/home/wuxingxing/xcode/MatPL-dcu-dev
cd "$SLURM_SUBMIT_DIR"
srun --cpu-bind=cores --kill-on-bad-exit=1 --wait=30 \
    python "$REPO/main.py" train nep.json
```

提交：`sbatch cpu.slurm`。无需申请 GPU，脚本也显式屏蔽了 GPU。CPU 分区可用时替换 `partition` 即可；单节点改 `--nodes=1`，其余设置不变时为 4 rank、24 CPU。

这里把计算线程上限设为 6，数据加载也共享这些 CPU。如果发生明显竞争，可在保持 `cpus-per-task=6` 的同时降低计算线程数，再比较稳定阶段耗时；资源配额与线程上限可以不同。

### 3.2 GPU：两节点，每节点 4 卡、24 CPU

在训练目录保存为 `gpu.slurm`：

```bash
#!/bin/bash
#SBATCH --job-name=nep-gpu
#SBATCH --partition=3090,q4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --output=gpu-%j.out

set -euo pipefail
module load cuda/11.8-share
source /data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh
conda activate matpl-2026.3
export PYTHONUTF8=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
export MASTER_ADDR MASTER_PORT

REPO=/data/home/wuxingxing/xcode/MatPL-dcu-dev
cd "$SLURM_SUBMIT_DIR"
srun --cpu-bind=cores --gpu-bind=single:1 --kill-on-bad-exit=1 --wait=30 \
    python "$REPO/main.py" train nep.json
```

提交：`sbatch gpu.slurm`。全局 8 个训练 rank、8 张卡、48 CPU；单节点四卡改 `--nodes=1`。使用 3080Ti 时可改为 `--partition=3080ti,new3080ti,q3`，并按节点资源调整 CPU 数。

GPU 可见性和卡绑定由 Slurm 管理，不要在脚本内把 `CUDA_VISIBLE_DEVICES` 固定成物理卡号。当前代码兼容每个 task 只看见其绑定卡，也兼容各 rank 看见节点全部已分配卡的方式。

两个脚本都只在 batch 主进程选一次端口，再传给全部 rank；不能让每个 rank 单独生成端口。多节点必须能互相访问 `MASTER_ADDR:MASTER_PORT`。

## 4. 在计算节点直接运行

以下命令在**计算节点**执行。没有 Slurm 时，`torchrun` 负责启动进程，但不分配 CPU、不绑定 CPU 核，也不申请 GPU。资源边界由已有分配、容器或系统管理规则决定。

每个节点先执行公共环境设置，把 `/path/to/training` 换成实际目录：

```bash
source /data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh
conda activate matpl-2026.3
export PYTHONUTF8=1
REPO=/data/home/wuxingxing/xcode/MatPL-dcu-dev
cd /path/to/training
```

### 4.1 单节点 CPU：64 个可用 CPU，8 rank × 8 计算线程

JSON 设置 `device=cpu`、`workers=1`，执行：

```bash
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    "$REPO/main.py" train nep.json
```

`--nproc-per-node=8` 才是训练进程数。8×8 是计算线程的容量估算，此外还有数据加载、通信等线程/进程，并非整个作业恰好只有 64 个线程。这是测试起点，不是已验证的最优配置；可以与 16 rank × 4 线程比较，必要时减少计算线程为加载和通信留出余量。

只执行 `python "$REPO/main.py" train nep.json` 且没有外部启动器的 rank 环境时，CPU 默认只有 **1 个训练进程**，不会因为机器有 64 CPU 就自动启动 64 个进程。

### 4.2 单节点 GPU：4 张可用卡，4 个训练进程

JSON 设置 `device=cuda`、`workers=1`，执行：

```bash
module load cuda/11.8-share
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    "$REPO/main.py" train nep.json
```

确保当前会话可见 4 张已分配 GPU；如果之前执行过 CPU 示例的屏蔽命令，应先恢复本会话获分配的 GPU 可见性。在无调度器且四张卡均归自己使用的机器上，可显式设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`；在 Slurm 分配内保留调度器给出的设置。只有 2 张可见卡时，将进程数改为 2。

当前 GPU 入口也支持不经启动器直接执行 `python main.py train nep.json`，它会按可见 GPU 数启动本机训练进程。为明确进程数量，本文统一采用 `torchrun`。

### 4.3 多节点：每个节点各运行一次 torchrun

以两个 CPU 节点、每节点 8 rank 为例：先在两节点分别执行公共环境与 CPU 线程/屏蔽设置，再执行下面各自的命令。两个节点必须使用相同代码、配置、可访问的数据和同一个输出目录。

节点 0，假设它的可达地址是 `10.0.0.10`：

```bash
torchrun --nnodes=2 --node-rank=0 --nproc-per-node=8 \
    --master-addr=10.0.0.10 --master-port=29531 \
    "$REPO/main.py" train nep.json
```

节点 1：

```bash
torchrun --nnodes=2 --node-rank=1 --nproc-per-node=8 \
    --master-addr=10.0.0.10 --master-port=29531 \
    "$REPO/main.py" train nep.json
```

替换示例 IP 和空闲端口；同一作业必须一致，同一主节点上的并行作业应使用不同端口。两个命令需要在各自终端同时保持运行，`torchrun` 不会自动通过 SSH 启动另一台节点。多节点不使用 `--standalone` 或 `localhost`。

两节点各 4 张 GPU 时，改用 GPU 环境和 `device=cuda`，将两个命令的 `--nproc-per-node` 都改成 4，其余节点编号和共享地址规则相同。启动参数见 [PyTorch torchrun 文档](https://docs.pytorch.org/docs/2.14/elastic/run.html)；以上只使用当前 PyTorch 2.2 环境支持的参数。

## 5. 改进程数时，也要检查 batch 和学习率

整数 `optimizer.batch_size=B` 是**每个 rank** 的 batch；常规每步全局结构数为 `B × world_size`。例如每 rank 16 个结构，8 rank 为 128 个，64 rank 为 1024 个。增加 rank 会改变训练设置，不只是换一种 CPU 调度方式。

LMDB 的 `optimizer.batch_size="mix:10240"` 是每个 rank 的目标原子预算，不是全局结构数或严格显存上限。各步结构数量会变化；单个超预算结构仍会独占一个 batch。比较资源布局时，应同时记录实际全局 batch、`scale_lr` 和有效学习率，避免把训练设置差异误认为性能差异。

`workers=1` 可作为 CPU/GPU 的共同起点；加载确实成为瓶颈后，再增加到 2 或 4 做对照。增加 workers 不改变训练 rank 数或全局 batch。分别比较预热后的 step/epoch 耗时、CPU/GPU 利用率和主机内存，不应只根据核数决定 workers。

## 6. 启动检查与常见问题

| 现象 | 检查位置 |
| --- | --- |
| CPU 只启动一个训练进程 | 是否用了 `torchrun --nproc-per-node=N` 或 `srun`；仅改 `workers` 不会增加训练进程 |
| 申请了多个 task，但只在一台机器运行 | `sbatch` 脚本是否真正调用 `srun`；申请资源不会自动分发普通 `python` 命令 |
| GPU rank 数超过可用卡数 | 单机 torchrun 的进程数、当前 GPU 可见性、Slurm 卡绑定是否匹配 |
| 卡在分布式初始化 | 两节点命令是否都已启动，共享地址/端口是否可达，JSON 是否覆盖了地址/端口 |
| CPU 忙但训练变慢 | rank、计算线程和 workers 是否过多；子进程共享 CPU 配额，不是额外获得资源 |
| extxyz 设置 `workers=0` 报错 | 当前非 LMDB 路径保留预取/persistent 设置，先使用 `workers>=1` |
| 纯 CPU 测试仍要求 GPU | JSON 是否显式为 `device=cpu`，是否已构建匹配当前 PyTorch 的 CPU 算子库 |

分布式启动日志会输出每个 rank 的 `LocalRank`、`device` 和 `backend`。CPU 应显示 `device cpu, backend gloo`；GPU 应显示 CUDA 设备和 NCCL。Slurm 为每个 rank 隔离一张卡时，各 rank 都显示 `cuda:0` 是正常现象，它们使用的是各自可见的第 0 张卡。

续训配置与进程资源设置分开处理，见 [checkpoint 恢复规则](nep-checkpoint-resume.md)。本文只说明 NEP，不代表其他模型已有相同的 CPU 分布式能力。
