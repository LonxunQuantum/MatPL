# NEP 多节点多卡 LMDB 训练：数据加载、负载均衡与 DDP 汇总

本文说明 MatPL NEP 在顶层配置为 `"format": "lmdb"` 时，多节点、多 GPU 训练的数据加载和结果汇总逻辑。内容对应源码提交 `0fcfca9`。

## 1. 结论先行

1. 数据集不会把 OMat24 的全部 frame 解码到内存。内存中主要保存 shard 路径、`nextid`、`deleted_ids`、累计长度和 sampler 索引；真正的 frame 在 DataLoader worker 收到索引后，才从 LMDB 读取、解压、解析并转成 tensor。
2. 每个 rank 都根据相同的 `seed + epoch` 独立重建同一条全局 shuffle 序列，然后只取得属于自己的、不与其他 rank 重叠的 batch。rank 对应的是全局顺序中的不同位置，不是固定的连续索引区间，也不是固定 LMDB shard。
3. 整数 `batch_size=B` 表示每个 rank 每步读取 B 个 frame。`batch_size="mix:N"` 表示每个 rank 每步的目标原子预算是 N；除单 frame 本身已超过 N 的 singleton 例外，普通 batch 不超过 N。它的目的是降低不同 frame 原子数差异造成的显存和耗时波动。
4. `mix:N` 只按原子数做贪心打包，不是严格的跨卡负载均衡器。同一 iteration 中各卡的原子数仍可能不同；单个 frame 若已超过 N，会独占一个 batch 并超过预算；相同原子数的结构也可能因近邻数不同而具有不同计算量。
5. 各 rank 的 iteration 数严格相同。sampler 只输出能够组成完整 `world_size` 组的 batch，末尾不足一组的数据会被丢弃，不通过复制数据进行 padding。
6. `workers>0` 使用的是每个 rank 各自的 PyTorch DataLoader 多进程，不是一个跨 rank 的共享线程池。训练 loader 会启用 `prefetch_factor=2`、persistent worker 和 pinned memory。
7. 每个 iteration 中，每个 rank 先计算本地 loss。执行 `loss.backward()` 时，PyTorch DDP 对参数梯度做跨 rank all-reduce，并除以 `world_size`；因此模型更新使用的是各 rank 本地梯度的等权平均。每个 rank 随后执行相同的 optimizer step，模型参数保持一致。
8. `reduce_loss` 只影响日志指标是否在打印时跨 rank 求平均，不参与反向传播。epoch 结束时指标会统一归约，但当前日志归约是 rank 均值，并非严格按全局 frame 数或原子数加权的总体指标。

## 2. 多节点进程和 rank 的建立

多节点训练由 Slurm `srun` 启动，预期资源配置是每个 GPU 对应一个独立进程。入口只在 `SLURM_NNODES > 1` 时从 Slurm 环境读取：

- `SLURM_NTASKS` → 全局 `world_size`；
- `SLURM_PROCID` → 全局 `rank`；
- `SLURM_LOCALID` → 节点内 `local_rank`。

随后每个进程执行：

```text
dist.init_process_group(
    backend="nccl",
    init_method="tcp://MASTER_ADDR:MASTER_PORT",
    rank=rank,
    world_size=world_size,
)
torch.cuda.set_device(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank], ...)
```

这里不是通用的 `torchrun` / `RANK` / `LOCAL_RANK` 启动路径。单节点即使使用 `srun` 多 task，也不会走上述 Slurm 多节点分支，而是按 `torch.cuda.device_count()` 由 `mp.spawn` 启动本机进程。

相关实现：

- [`src/user/nep_work.py`](../src/user/nep_work.py#L59-L114)
- [`src/PWMLFF/nep_network.py`](../src/PWMLFF/nep_network.py#L500-L517)
- [`src/PWMLFF/nep_network.py`](../src/PWMLFF/nep_network.py#L932-L936)

多节点运行必须满足：

- 每个节点的 LMDB 路径都可见，并且相同绝对路径指向相同数据；
- 所有 rank 使用相同的 `MASTER_ADDR` 和 `MASTER_PORT`；
- Slurm 必须为每个 task 正确绑定一个 GPU，并使 `SLURM_LOCALID` 对应进程可见的本地 CUDA ordinal；代码本身不会校验资源绑定；
- `LKF` 和 `GKF` 不支持多卡；多卡 NEP 使用支持 DDP 的优化器，例如 Adam；
- 当前进程组后端在代码中固定为 NCCL。

一个双节点、每节点四卡的启动骨架如下。分区必须按实际集群选择 `q4` 或 `3090`：

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=3090

source /path/to/MatPL/env.sh

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29501  # 所有 rank 必须使用同一个可用端口

srun MATPL train nep.json
```

> 不要让每个 rank 分别选择随机端口。若没有显式设置统一的 `MASTER_PORT`，各进程可能得到不同端口，导致进程组无法建立。

## 3. 从配置到 DataLoader 的总体数据流

```text
train_data 目录/文件列表
        │
        ▼
递归发现并排序 .aselmdb shard
        │
        ▼
读取每个 shard 的 nextid/deleted_ids 元数据
        │                 不读取全部 frame
        ▼
NepLmdbDataset + 全局确定性 sampler
        │
        ├── 整数 B ──> DistributedFrameBatchSampler
        │
        └── mix:N ──> natoms sidecar ──> DistributedAtomBatchSampler
                                      │
                                      ▼
                     每个 rank 获得不同的索引 batch
                                      │
                                      ▼
                 DataLoader worker 按索引读取 LMDB frame
                                      │
                                      ▼
                    zlib + JSON/orjson + tensor 转换
                                      │
                                      ▼
                       collate 后送入对应本地 GPU
```

`NepLmdbDataset` 初始化时，每个 shard 只读取 `nextid` 和可选的 `deleted_ids`，并保存：

- shard 的绝对路径；
- 逻辑 frame 数；
- 删除行的 ID；
- 各 shard 的累计结束位置；
- 原子类型到模型类型编号的映射；
- 少量初始化统计字段。

它不保存完整 frame 列表。参见 [`AseLmdbShard`](../src/pre_data/nep_lmdb_dataset.py#L79-L139) 和 [`NepLmdbDataset`](../src/pre_data/nep_lmdb_dataset.py#L741-L802)。

## 4. 块级全局 shuffle 和每个 rank 的数据差异

### 4.1 shuffle 不是为每个 rank 独立随机

所有 rank 使用相同的：

```text
dataset size + seed + epoch + shuffle 设置
```

在 `mix:N` 模式下，各 rank 还必须使用相同的 natoms 数据。因此每个 rank 都能独立推导出相同的全局顺序，不需要 rank 0 先生成巨大索引数组再广播。

`BlockShuffleIndices` 使用 `Random(seed + epoch)`：

1. 将完整索引域划成默认最多 65536 个索引的 block；
2. 打乱 block 的顺序；
3. 每次只构造一个 block 的索引并在 block 内打乱；
4. 流式输出索引，然后释放该 block。

这是块级 shuffle：不同 block 的元素不会互相交错，不是对全部索引的均匀随机 permutation。它避免创建 `O(dataset_size)` 大小的全量索引置换；额外索引内存主要是一个 block、约 `ceil(dataset_size / block_size)` 个 block ID，以及 sampler 的跨-rank batch 缓冲。参见 [`BlockShuffleIndices`](../src/pre_data/nep_lmdb_dataset.py#L141-L183)。

同一 epoch 内，一个进入完整 batch 组的 frame 只会分给一个 rank。不同 rank 的索引互不重叠，但它们可能来自同一个 `.aselmdb` shard。下一 epoch 使用新的 `seed + epoch` 顺序，rank 对应的 frame 会变化。

### 4.2 整数 frame batch 的分配

若：

```json
"batch_size": 2
```

则每个 rank 每步取得 2 个 frame。设 `world_size=2`、数据集有 10 个 frame 且关闭 shuffle：

```text
全局顺序: 0 1 2 3 | 4 5 6 7 | 8 9

iteration 1:
  rank 0 -> [0, 1]
  rank 1 -> [2, 3]

iteration 2:
  rank 0 -> [4, 5]
  rank 1 -> [6, 7]

尾部 [8, 9] 不足 batch_size * world_size，丢弃。
```

每个 rank 的 step 数为：

```text
floor(dataset_size / (batch_size * world_size))
```

相关实现：[`DistributedFrameBatchSampler`](../src/pre_data/nep_lmdb_dataset.py#L186-L265)。

整数 batch 保证每卡 frame 数相同，但当 frame 的原子数差异很大时，显存、邻居表规模和计算时间可能严重不均衡。

## 5. `mix:N` 的精确定义和作用

### 5.1 N 是每个 rank 的原子预算

配置：

```json
"batch_size": "mix:4096"
```

表示每个 rank 的每个本地 batch 目标原子预算为 4096；除单 frame 本身超过 N 时形成的 singleton 例外，普通本地 batch 不超过 4096。它不是：

- 每个 rank 4096 个 frame；
- 全部 GPU 合计 4096 个原子；
- 固定恰好 4096 个原子。

若有 W 个 rank，则一个理想全局 iteration 的原子规模大约不超过 `W * N`，但实际值由 frame 边界和超预算 frame 决定。

语法解析参见 [`parse_lmdb_batch_size`](../src/pre_data/nep_lmdb_dataset.py#L29-L39)。

### 5.2 贪心打包

sampler 按全局 shuffle 顺序读取 natoms：

```text
当前 batch 为空
逐个查看下一个 frame 的 natoms
  ├── 加入后不超过 N：加入当前 batch
  ├── 加入后会超过 N：先结束当前 batch，再处理该 frame
  └── 单 frame 已超过 N：让它单独成为一个超预算 batch
```

生成全局 atom-batch 序列后，每连续 `world_size` 个 batch 组成一个 iteration；rank `r` 取得其中第 `r` 个 batch。参见 [`DistributedAtomBatchSampler`](../src/pre_data/nep_lmdb_dataset.py#L586-L686)。

示例：

```text
natoms = [3, 4, 6, 2, 5, 10, 4]
mix:8
world_size = 2
shuffle = false

全局贪心 batch:
  A = [0, 1] -> 7 atoms
  B = [2, 3] -> 8 atoms
  C = [4]    -> 5 atoms
  D = [5]    -> 10 atoms，单 frame 已超过预算
  E = [6]    -> 4 atoms

iteration 1:
  rank 0 -> A，7 atoms
  rank 1 -> B，8 atoms

iteration 2:
  rank 0 -> C，5 atoms
  rank 1 -> D，10 atoms

E 不足以组成两个 rank 的完整组，因此丢弃。
```

### 5.3 `mix:N` 解决了什么

`mix:N` 的主要作用是把“固定 frame 数”改成“近似固定原子数”：

- 大结构的一个 batch 会包含较少 frame；
- 小结构的一个 batch 会包含较多 frame；
- 正常 batch 的原子数不会超过 N；
- 显存占用和计算时间通常比固定 frame batch 更稳定；
- 较少出现某个 rank 因拿到多个大结构而远慢于其他 rank 的情况。

若 OMat24 训练子集中的 frame 原子数跨度较大，这种方式尤其有用。

### 5.4 `mix:N` 不保证什么

当前实现不能保证同一个 iteration 中各 rank 的工作量完全相等，原因包括：

1. 贪心 batch 只是按顺序分组，没有在一个 rank 组内再次做最小最大配对或动态调度。例如同一步可能是 5、7、8、10 个原子。
2. 单个 frame 若 `natoms > N`，仍会独占 batch 并超过 N，可能造成显存峰值和明显的 straggler。
3. NEP 计算量不仅取决于原子数，还取决于 cutoff 内的邻居数、结构密度、cell 扩展和角向组合数量。原子数相同并不代表 FLOPs 相同。
4. 若设置了 `batch_max_types`，collate 可能过滤某些 frame，导致实际进入模型的 batch 比 sampler 预算更小。默认 `-1` 不启用该过滤。
5. DDP 每一步都需要所有 rank 进入梯度归约，因此快卡会等待最慢卡；代码没有运行时 work stealing。

所以，“防止某个卡某次 batch 不均衡”的准确说法是：

> 当前实现通过每 rank 原子预算限制最常见的 frame-size 不均衡，但只降低波动，不提供严格的计算量均衡保证。

### 5.5 实际配置建议

- 尽量让 N 不小于训练集中最大的单 frame 原子数。否则超预算 singleton 不受 N 限制，N 失去显存硬上限作用。
- 在显存允许范围内，根据 natoms 分布选择 N。N 太小会增加 singleton、iteration 数和通信占比；N 太大又会增加显存峰值及 rank 间绝对耗时差。
- 先从能够稳定运行的较小 N 开始，观察日志中的 `Time`、GPU 利用率和显存峰值，再逐步增大。
- 对密度或邻居数差异特别大的数据，仅按 natoms 无法完全均衡。若仍有明显 straggler，需要后续实现基于近邻数或实测成本的 cost-aware sampler；当前代码尚未实现。
- sampler 在每个 epoch 设置新顺序后会重新检查是否至少能组成一个完整的跨-rank batch；不能组成时会尽早报错，而不是让部分 rank 进入 DDP 后挂起。

## 6. 为什么各 rank 的 iteration 数相同

DDP 要求所有 rank 以一致顺序参与 collective。若某个 rank 少一个 batch，其他 rank 会在下一次 backward 的 collective 中等待，最终挂起。

为避免这种情况，两种 sampler 都以完整的跨-rank 组为单位输出：

- frame 模式：收集 `batch_size * world_size` 个 frame 后，每 rank 分一段；
- mix 模式：收集 `world_size` 个已经打包好的 atom batch 后，每 rank 分一个；
- 尾部不足完整组时直接丢弃；
- 不复制 frame，不用 padding 补齐；
- 每个训练 epoch 对训练 loader 调用 `set_epoch(epoch)` 后，重新验证当前顺序下的 batch 数不为零。验证 loader 不随训练 epoch 调用 `set_epoch`，因而保持构造时的 epoch 0 顺序；它的 sampler 仍保证各 rank step 数一致。

因此，对任意保留下来的 iteration：

```text
rank 0 step count == rank 1 step count == ... == rank W-1 step count
```

但是，“step 数相同”不等于“每一步耗时相同”。

相关实现：

- [`_set_training_loader_epoch`](../src/PWMLFF/nep_network.py#L470-L494)
- [`_require_lmdb_training_batches`](../src/PWMLFF/nep_network.py#L54-L70)

## 7. `mix:N` 为什么需要 natoms sidecar

只保存 LMDB frame 索引还不足以在读取前组成 atom-budget batch，因为 sampler 必须预先知道每个 frame 的原子数。

第一次使用 `mix:N` 时：

1. 创建 `json_dir/.matpl_lmdb_cache`；
2. 以 `shard_index % world_size` 把 shard 分配给不同 rank；
3. 每个负责该 shard 的 rank 顺序扫描其中全部 frame；
4. 解压并解析 frame，读取 `numbers` 长度；
5. 为每个 shard 写入一个 `int32` natoms 文件和一个 JSON manifest；
6. 临时文件经过 flush/fsync 后用 `os.replace` 原子替换；
7. 全部 rank 在 barrier 等待；
8. 每个 rank 使用只读 `np.memmap` 映射所有 natoms 文件。

sidecar 的指纹包含源路径、文件大小、mtime、`nextid`、`deleted_ids` 和逻辑长度。指纹不匹配时才会重建。参见 [`LmdbNatomsCache`](../src/pre_data/nep_lmdb_dataset.py#L390-L583) 和 [`_lmdb_natoms`](../src/PWMLFF/nep_network.py#L608-L624)。

这有三个效果：

- 第一次扫描较慢，但后续训练无需为 sampler 再次读取完整 LMDB frame；
- natoms 使用 `int32`，1,077,382 个 frame 的原始计数约占 4.1 MiB，不需要 Python 整数列表；
- `np.memmap` 只映射文件，实际页由操作系统按需调入。

cache 根目录实际是配置 JSON 所在目录下的 `.matpl_lmdb_cache`，不是 LMDB 数据目录的子目录。多节点时，这个 cache 必须位于所有节点可见的共享路径。所有 rank 还必须用相同的 resolved absolute LMDB 路径和相同的 shard 顺序访问数据，因为 sidecar 文件名和指纹包含 shard 的绝对路径。不同节点若使用不同挂载前缀，即使文件内容相同，也会破坏 cache 协作。当前实现按全局 rank 分工构建 shard；若 cache 是各节点互不相同的本地目录，barrier 后某些节点会看不到由其他节点创建的 sidecar。

也可以使用节点本地 cache，但必须在启动训练前给每个节点预置全部 shard 的、指纹有效的 sidecar。不能依赖一次多节点首跑在各节点本地目录中分别补齐，因为每个 global rank 只负责部分 shard。

## 8. 实际 frame 如何按索引从磁盘加载

DataLoader worker 收到索引后，`NepLmdbDataset.__getitem__` 执行：

1. 通过累计 shard 长度定位 `.aselmdb` 文件；
2. 把逻辑索引转换为跳过 `deleted_ids` 后的物理 row ID；
3. 从当前 worker 的 environment LRU 获取或打开 LMDB；
4. 建立短只读 transaction；
5. 用 row ID 键读取压缩 bytes；
6. zlib 解压；
7. 优先使用 `orjson.loads`，不可用时回退到标准 `json.loads`；
8. 校验原子类型、位置、cell、PBC、energy、force、stress 等；
9. 转换为 NumPy/Torch tensor；
10. collate 将变长的原子级 tensor 拼接，其他字段 stack，并生成累计 `num_atom_sum`。

相关实现：

- [`NepLmdbDataset.__getitem__`](../src/pre_data/nep_lmdb_dataset.py#L861-L889)
- [`NepLmdbDataset._convert_frame`](../src/pre_data/nep_lmdb_dataset.py#L891-L1010)
- [`variable_length_collate_fn`](../src/pre_data/nep_data_loader.py#L39-L93)

压缩 bytes、解压后的 JSON 对象和单 frame tensor 只在当前取样/batch 生命周期内存在。训练不会把已解码的全部数据缓存在 Python 中。

## 9. 是否使用多线程或多进程加载

LMDB Dataset 自身没有创建读取线程。并发来自 PyTorch DataLoader：

```json
"workers": 2
```

该参数直接传给每个 rank 的 `DataLoader(num_workers=workers)`：

- `workers=0`：rank 主进程同步执行读取、解压、解析和 collate；
- `workers>0`：每个 rank 启动 `workers` 个独立子进程；每个子进程负责收到的整个索引 batch；
- 总 worker 数约为“每节点 rank 数 × workers”，而不是整个集群只有 `workers` 个；
- `pin_memory=True` 时，PyTorch 还会管理 pinned-memory 搬运流程，但这不是 LMDB Dataset 自己的线程池。

例如双节点、每节点 4 个 rank，配置 `workers=2`，仅计算一个正在工作的 DataLoader：

```text
每节点 DataLoader worker 数 = 4 * 2 = 8
全作业该 DataLoader 的 worker 数 = 2 * 4 * 2 = 16
```

train、validation 和临时 stat loader 是不同的 worker pool。若多个 persistent loader 已经开始迭代并同时存活，总进程数会高于上面的单-loader 数量。

默认 `workers=1`。相关实现：

- [`InputParam.workers`](../src/user/input_param.py#L184-L188)
- [`_lmdb_loader`](../src/PWMLFF/nep_network.py#L529-L540)

### 9.1 worker 与 LMDB environment 的关系

Python LMDB Environment 对象和句柄不跨进程复用；同一节点上的不同进程仍可以共享内核 page cache：

- Dataset 被 pickle 时，`__getstate__` 清空 environment LRU；
- fork 后首次访问会检测 PID 变化并关闭继承状态；
- 每个 worker 按需打开自己的只读 environment；
- 每个 worker 默认最多保留 8 个 environment，超出时关闭最久未使用的句柄；
- 每次 `__getitem__` 只创建短 read transaction。

参见 [`__getstate__`、`_environment` 和 `close`](../src/pre_data/nep_lmdb_dataset.py#L807-L846)。

“最多 8 个 environment”是每个 Dataset 进程的限制，不是每个节点或整个任务的总上限。若一台节点有 4 个 rank、每 rank 4 个 worker，理论上活跃训练 loader 最多可持有约 `4 * 4 * 8 = 128` 个 environment；实际数量取决于访问到的 shard。

## 10. 如何避免磁盘读取成为瓶颈

当前实现采用以下措施。

### 10.1 只按 sampler 选中的索引读取

训练进程不会先把完整 OMat24 解码。每个 rank 只读取当前 epoch 中分配给自己的 frame，尾部丢弃 frame 也不会被加载。

### 10.2 DataLoader 预取

当 `workers>0` 时：

```text
prefetch_factor = 2
persistent_workers = true  # train/valid/test
pin_memory = true
```

DataLoader 配置的预取因子是每 worker 2 个 batch，使磁盘读取、zlib/JSON CPU 解析和 GPU 计算有机会重叠；实际队列状态还取决于消费速度、worker 状态和 PyTorch 实现。worker 在 epoch 间保持存活，避免反复启动进程和重建 LMDB environment LRU。

用于一次性初始化统计的 stat loader 使用 `persistent_workers=false`，统计完成后释放 worker 和句柄。

注意：预取单位是 batch，不是 frame。`mix:N` 的 frame 数可变，因此 host RAM 大小取决于实际原子数、worker 数和预取队列。粗略的在途预取上限是每个 rank `workers * 2` 个 batch，另有正在消费和 pinned 的 batch。

当前训练循环使用普通 `.to(device)`，没有显式设置 `non_blocking=True`，因此不要把 `pin_memory=True` 理解为已经保证 CPU→GPU 传输完全异步重叠。

### 10.3 随机训练读取关闭 readahead

按 frame 读取和 shard 元数据读取使用：

```text
readonly=True
lock=False
readahead=False
meminit=False
```

训练顺序经过 block shuffle，属于随机读取。关闭 readahead 可以避免内核把大量相邻但本 epoch/本 rank 未必使用的页提前读入，减少共享文件系统流量和 OS page cache 污染。

`readahead=False` 不等于“完全不用缓存”。LMDB 仍通过 mmap 工作，实际访问过的页仍可能留在当前节点的 OS page cache。相同节点上的 rank/worker 可以受益于同一内核页缓存；不同节点不共享物理 page cache。由于 environment 使用 `readonly=True, lock=False`，训练不会与外部 writer 协调，训练期间必须把 LMDB shard 视为不可变数据，不能原地写入。

natoms sidecar 首次构建是顺序全 shard 扫描，因此该路径反而使用 `readahead=True`。

### 10.4 environment LRU

每个 worker 最多打开 8 个 shard，避免同时为大量 shard 保留文件描述符和 mmap。重复访问近期 shard 时则复用 environment，减少频繁 open/close。

### 10.5 可选的 orjson

如果环境已安装 `orjson`，frame 解码优先使用它；否则使用标准 JSON，不要求升级 LMDB 或额外依赖。

### 10.6 有界索引 shuffle

sampler 只保留 block ID、一个索引 block，以及一个跨 rank batch 组，而不创建百万级 Python frame 对象。它主要解决 Python 内存问题；相对完全随机的全局 permutation 有一定块内局部性，但 block 顺序和 block 内索引都被打乱，跨 shard/页访问仍可能是随机 I/O。

## 11. I/O 调参建议和剩余瓶颈

即使使用懒加载，以下工作仍会对每个实际 frame 执行：

- LMDB B-tree/页访问和短 transaction；
- zlib 解压；
- JSON/orjson 解析；
- NumPy shape/finite 校验；
- NumPy 到 Torch tensor 转换；
- collate 的 concat/stack 拷贝；
- pinned-memory 和 CPU→GPU 搬运。

因此 worker 不是越多越好。总 worker 数会随节点内 GPU 数放大，过多 worker 可能造成：

- 共享 SSD/NFS/并行文件系统的随机读争用；
- CPU 解压和 JSON 解析争用；
- 文件描述符和 mmap 数增加；
- `workers * prefetch_factor` 带来的 host/pinned RAM 增长；
- 进程调度开销增加。

建议按以下顺序调优：

1. 从 `workers=1` 或 `2` 每 rank 开始。
2. 观察训练日志中的 `Data` 与 `Time`，同时查看 GPU 利用率、CPU 利用率、`iostat` 和共享存储吞吐。
3. 若 GPU 经常等待且存储/CPU 尚有余量，逐步增加 workers。
4. 若存储已满负荷、Data 时间没有下降或内存压力上升，减少 workers。
5. 让 `#SBATCH --cpus-per-task` 至少覆盖每 rank 的 DataLoader workers，并为主训练进程留出余量。
6. 条件允许时，将只读 shard 放到节点本地 NVMe；但所有节点必须保持相同的 resolved absolute 路径。`mix:N` 的 sidecar 最稳妥的做法仍是放在所有 rank 可见的共享目录；若也放到节点本地，则必须在启动前为每个节点预置全部 shard 的有效 sidecar，不能让当前的全局-rank 分工逻辑在首跑时分别生成。
7. 首次 `mix:N` 运行会构建 natoms sidecar，耗时不能代表后续 epoch/后续作业的稳定加载性能。

当前还存在一个 CPU 开销：`mix:N` 的 `__len__` 需要遍历完整 natoms 流以计算当前 epoch 的全局 batch 数；结果会按 epoch 缓存，但每个 rank 仍需独立完成一次 O(dataset size) 的计数。随后真正迭代时还会再次遍历该流。该过程读取的是紧凑的 int32 sidecar，不会解码 LMDB frame，但超大数据集上仍需要计入 epoch 启动开销。

## 12. 初始化统计如何跨 rank 汇总

正式训练前需要 energy shift、平均/最大原子数、descriptor 范围和最大近邻数。它们不会通过每个 rank 重复扫描完整 LMDB 得到。

### 12.1 全局统计抽样

全局样本数为：

```text
min(dataset_size, lmdb_stat_frames, 32768 * world_size)
```

所有 rank 用同一 seed 生成同一组全局随机索引，然后 rank `r` 处理：

```text
global_indices[r::world_size]
```

本地索引在读取前排序，以改善统计阶段的访问局部性。各 rank 样本互不重叠，每 rank 最多 32768 个 frame。参见 [`select_stat_indices`](../src/pre_data/nep_lmdb_dataset.py#L267-L304)。

例如 `lmdb_stat_frames=32768`、`world_size=8` 且数据足够大时，每个 rank 大约读取 4096 个统计 frame，不是每个 rank 都读取 32768 个。

### 12.2 归约类型

energy shift 与原子数统计先使用上述随机样本，由每个 rank 计算本地累计量，再执行：

| 统计量 | collective | 结果 |
|---|---|---|
| `AᵀA`、`AᵀE` | `SUM` | 合并能量平移正规方程 |
| frame count、atom count sum | `SUM` | 计算全局样本平均原子数 |
| max atoms | `MAX` | 得到样本中的最大 frame 原子数 |

随后 descriptor scaler 和邻居容量通过 `forscaler_loader` 的各 rank 分片计算，再执行：

| 统计量 | collective | 结果 |
|---|---|---|
| descriptor max/min | `MAX` / `MIN` | 得到统一 scaler 范围 |
| radial/angular max neighbor | `MAX` | 得到统一近邻容量 |

`forscaler_loader` 覆盖的是 sampler 实际保留的 batch，不一定等于 energy-shift 随机样本，也不包含该 sampler 丢弃的尾部。每个 rank 从相同的归约结果独立求解相同 energy shift 和 `q_scaler`，初始化结束后再同步 barrier；不是由 rank 0 计算后广播。

energy 统计为空的 rank 使用零值 `AᵀA`、`AᵀE` 和计数；scaler 分片为空的 rank 使用 descriptor `-inf/+inf` 与邻居数 0 等归约单位元。所有 rank 仍参加 collective，不会提前退出导致其他 rank 挂起。

相关实现：

- [`_reduce_lmdb_statistics`](../src/PWMLFF/nep_network.py#L542-L584)
- [`_prepare_lmdb_statistics`](../src/PWMLFF/nep_network.py#L586-L605)
- [`_calculate_lmdb_neighbor_scaler`](../src/PWMLFF/nep_network.py#L73-L109)

## 13. 一个训练 iteration 中发生了什么

设 `world_size=W`。每个 iteration 的逻辑如下：

```text
rank 0:   batch 0 ─> forward ─> local loss ─> backward（bucket all-reduce）─┐
rank 1:   batch 1 ─> forward ─> local loss ─> backward（bucket all-reduce）─┤
...                                                                       ├─> backward 返回时梯度已同步
rank W-1: batch W-1 ─> forward ─> local loss ─> backward（bucket all-reduce）┘
                                                                               │
                                                                               ▼
                                    每个 rank 执行相同 optimizer.step()
                                                                               │
                                                                               ▼
                                          各 rank 模型参数继续保持一致
```

图中的通信不是等所有 rank 完成全部 backward 后才开始。DDP 会在反向传播中按 gradient bucket 就绪顺序归约，通信可以与剩余反传重叠；`loss.backward()` 返回时，才可把梯度视为已经完成同步。

单个 rank 内部的顺序为：

1. DataLoader 提供本地 batch，并搬到 `local_rank` 对应 GPU；
2. 计算该 batch 的径向/角向最大近邻数；
3. 构建邻居表；
4. 执行 DDP 包装模型的 forward；
5. 用本地标签计算 energy、force、virial 等 MSE 和组合 loss；
6. `optimizer.zero_grad()`；
7. `loss.backward()`；
8. DDP reducer 在 backward 中对梯度 bucket 做 all-reduce；
9. 可选执行梯度裁剪；
10. 每个 rank 调用相同的 `optimizer.step()` 和 LR scheduler step；
11. 更新本地日志 meter。

相关实现：[`train`](../src/PWMLFF/nep_mods/nep_trainer.py#L305-L566)。

## 14. 每个卡的结果怎么汇总：是取平均吗

需要区分三类“结果”，并另外说明 rank 0 的职责。

### 14.1 用于更新模型的梯度：取 rank 平均

每个 rank 的 loss 不会先 gather 到 rank 0，也不会先求一个全局 loss 标量。每个 rank 对自己的 batch 计算本地 loss 和本地梯度。

PyTorch DDP 在 `loss.backward()` 期间对每个参数的梯度执行等价于：

```text
global_grad = (grad_rank0 + grad_rank1 + ... + grad_rankW-1) / W
```

随后每个 rank 都持有相同的 `global_grad`，并各自在本地执行相同 optimizer step。因此没有一个“主卡”负责合并模型，也不需要每步把参数广播回其他卡。

### 14.2 这不是严格的全局原子加权平均

本地 loss 使用 `nn.MSELoss` 的默认 mean；组合 loss 中的 energy 和 virial 项还使用该 rank 的本地 `avg_atom_number` 做归一化。若各 rank 的本地 batch 包含不同 frame 数、原子数或有效标签数，则 DDP 得到的是：

```text
各 rank 已在本地归一化后的组合 loss 梯度的等权平均
```

而不是：

```text
把所有 rank 的全部原子/坐标误差放在一起后的严格全局 mean 梯度
```

例如 rank 0 有 3000 个原子、rank 1 有 4000 个原子时，两者的本地梯度在 DDP 中仍各占 1/2 权重。`mix:N` 让原子数尽量接近，从而减小这种权重偏差，但不保证完全消除。

force、charge、BEC、缺失 virial mask 等项的有效元素数也可能不同，因此这一点不仅与 frame 数有关。

### 14.3 日志 loss/指标：也是 rank 平均，但只用于显示

`AverageMeter.all_reduce()` 对每个 rank 已计算出的 `root`、`val` 和 `avg` 使用 `ReduceOp.AVG`。它不归约各 rank 的 `sum` 和 `count`，因此：

- 显示的是各 rank 指标的简单平均；
- 不是按各 rank frame 数加权的全局平均；
- RMSE 类指标是“先在每 rank 开根，再对 rank 求平均”，不是“先合并全局 MSE，再开根”。

`reduce_loss=true` 只决定训练过程中按 `print_freq` 打印时是否提前同步这些 meter。无论其值如何，多卡训练在 epoch 结束时都会同步 meter。这个设置不改变 backward、梯度或 optimizer。

### 14.4 rank 0 的职责

rank 0 不是梯度汇总主卡，也不是唯一执行 optimizer 的进程。所有 rank 都参与 collective、backward 和 optimizer step。rank 0 只承担面向外部的 I/O 和显示职责，包括打印输入与进度、创建模型目录、写 train/valid 日志、保存 step/epoch checkpoint，以及转换 GPUMD 文件；需要时其他 rank 会在这些操作前后参加 barrier。

相关实现：

- [`nep_trainer.py`](../src/PWMLFF/nep_mods/nep_trainer.py#L470-L548)
- [`loss.py`](../src/loss/loss.py#L83-L107)
- [`AverageMeter.all_reduce`](../src/utils/train_log.py#L45-L52)
- [`nep_network.py`](../src/PWMLFF/nep_network.py#L1241-L1351)
- [`nep_work.py`](../src/user/nep_work.py#L109-L110)

## 15. 配置示例

下面只列出与 LMDB 加载和分布式训练直接相关的字段，应将它们合并进完整的 NEP JSON 配置；`atom_type`、模型结构和其他训练参数按实际任务补充。这里使用已知存在的 OMat24 `valid` 目录演示目录自动发现，正式训练时请替换为实际训练集目录或 `.aselmdb` 文件列表。

```json
{
  "model_type": "NEP",
  "format": "lmdb",
  "train_data": [
    "/data/public/wuxingxing/metadata/decompress/Omat24/valid"
  ],
  "lmdb_stat_frames": 32768,
  "workers": 2,
  "precision": "float64",
  "reduce_loss": false,
  "optimizer": {
    "optimizer": "ADAM",
    "batch_size": "mix:4096"
  }
}
```

参数含义：

| 参数 | 含义 |
|---|---|
| `format: lmdb` | 只在该值下启用 `NepLmdbDataset` 大数据路径 |
| `train_data` / `valid_data` | 可为 `.aselmdb` 文件、文件列表或递归目录 |
| `lmdb_stat_frames` | 全局请求的初始化统计 frame 数，不是每 rank 数量 |
| `workers` | 每个 rank 的 DataLoader worker 进程数 |
| `precision: float64` | 当前 CalcOps descriptor kernel 对 LMDB NEP 训练的要求 |
| `batch_size: mix:4096` | 每个 rank 的目标原子预算为 4096 |
| `reduce_loss` | 是否在中间打印时跨 rank 平均日志 meter；不影响梯度 |

## 16. 当前保证与当前不保证

### 已保证

- 不将全量 frame 解码并驻留在 Python 内存；
- 同一 epoch 内，保留数据在 rank 间不重叠；
- 所有 rank 的 iteration 数一致；
- shuffle 对给定 seed/epoch 可复现；
- `mix:N` 正常 batch 不超过每 rank N 个原子；
- cache 路径共享且版本、源绝对路径、大小、mtime、`nextid`、`deleted_ids`、逻辑长度等指纹未变化时，natoms cache 可跨作业复用；该指纹不包含文件内容 hash，LMDB 应保持不可变；
- DataLoader environment 对 worker 进程安全；
- 梯度由 DDP 跨 rank 平均，参数保持同步；
- 初始化统计量通过明确的 SUM/MAX/MIN collective 汇总。

### 未保证

- 一个 epoch 覆盖所有 frame；不足完整 rank 组的尾部会丢弃；
- 同一步各 rank 的原子数完全相同；
- 相同原子数对应相同计算时间；
- 单 frame 超过 `mix:N` 时仍遵守 N 的上限；
- 训练梯度严格按全局 frame、原子或有效标签数加权；DDP 使用的是各 rank 本地归一化梯度的等权平均；
- 日志指标是严格按全局样本/原子加权的统计值；
- workers 越多加载一定越快；
- 不共享的节点本地 sidecar cache 能自动协同；
- 多节点未显式统一 `MASTER_ADDR/MASTER_PORT` 时能够正常启动。

## 17. 排查加载瓶颈时应观察什么

| 现象 | 更可能的原因 | 优先动作 |
|---|---|---|
| `Data` 时间高、GPU 利用率低、磁盘未打满 | workers 不足或 JSON/解压 CPU 不足 | 逐步增加每 rank workers |
| `Data` 时间高、磁盘已打满 | 共享存储随机读争用 | 降低 workers，使用节点本地 NVMe，检查 shard 分布 |
| host/pinned RAM 持续升高 | workers × 2 个预取 batch 过多 | 降低 workers 或减小 `mix:N` |
| 某些 iteration 明显更慢 | 超预算大 frame 或高近邻密度结构 | 检查 natoms 最大值，增大 N 到覆盖单 frame，必要时开发 cost-aware sampler |
| 首次启动很慢，后续正常 | 正在构建 natoms sidecar | 保留并共享 `.matpl_lmdb_cache` |
| 多节点在初始化时挂起 | MASTER 地址/端口不一致或网络/NCCL 问题 | 显式统一 `MASTER_ADDR/MASTER_PORT`，检查 rank/GPU 映射 |
| 某 epoch 报无法形成完整 batch | 数据或 `mix:N` 打包后不足 `world_size` 个 batch | 减小 world size 或 batch/atom budget |

## 18. 关键源码索引

- 路径发现、LMDB shard 和 frame 解码：[`src/pre_data/nep_lmdb_dataset.py`](../src/pre_data/nep_lmdb_dataset.py)
- frame/mix sampler、统计抽样和 natoms cache：[`src/pre_data/nep_lmdb_dataset.py`](../src/pre_data/nep_lmdb_dataset.py)
- LMDB DataLoader、统计 collective、DDP 包装：[`src/PWMLFF/nep_network.py`](../src/PWMLFF/nep_network.py)
- 单 iteration forward/backward/optimizer：[`src/PWMLFF/nep_mods/nep_trainer.py`](../src/PWMLFF/nep_mods/nep_trainer.py)
- 日志指标归约：[`src/utils/train_log.py`](../src/utils/train_log.py)
- Slurm rank/local-rank/world-size 建立：[`src/user/nep_work.py`](../src/user/nep_work.py)
