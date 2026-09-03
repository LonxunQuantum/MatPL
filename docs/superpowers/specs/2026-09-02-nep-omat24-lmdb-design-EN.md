# NEP OMat24 LMDB 训练设计

## 背景

MatPL 现有的 NEP `UniDataset` 会急切地要求 `pwdata.Config` 将每一帧具体化为 Python 图像对象。随后，它会遍历完整图像列表以推导原子计数和能量平移。在分布式训练中，每个 rank 都会重复保留这份状态，因此即使优化器一次只消耗一个批次，OMat24 规模的输入也不具备可行性。

OMat24 `.aselmdb` 文件使用 ASE LMDB 布局：整数帧键从 `1` 开始，`nextid` 和可选的 `deleted_ids` 记录描述逻辑索引域，并且每一帧都是 zlib 压缩的 JSON 对象。这并非 DPA4C 使用的 msgpack 模式，因此 MatPL 在采用 DPA4C 的延迟读取、有界预取、确定性洗牌和 rank 级批次分区的同时，需要一个 ASE-LMDB 解码器。

## 目标

- 仅当顶层 JSON 包含 `"format": "lmdb"` 时选择大数据路径。
- 接受一个 `.aselmdb` 文件、文件列表、目录或其混合；递归搜索目录输入以查找全部 `.aselmdb` 文件。
- 使结构数据不驻留在内存中，并且仅解码为当前 rank 选择的帧。
- 在单节点和多节点 DDP 中保持确定性的 epoch 洗牌和相等的优化器步数。
- 支持整数帧计数批次和 DPA4C 风格的 `"mix:N"` 原子预算批次。
- 从分布在各个 rank 上的有界、确定性全局样本中估算 NEP 初始化统计量。
- 保持现有 `UniDataset` 和所有非 LMDB 格式不变。

## 非目标

- 读取 DPA4C 的 msgpack LMDB 模式。
- 改变 NEP 模型、损失或优化器的语义。
- 将 OMat24 转换为另一种磁盘数据集格式。
- 安装或升级 LMDB；现有的 `lmdb==1.5.1` 已足够。

## 用户界面

LMDB 路径由现有顶层字段选择：

```json
{
  "format": "lmdb",
  "train_data": [
    "/data/public/wuxingxing/metadata/decompress/Omat24/valid"
  ],
  "lmdb_stat_frames": 32768,
  "optimizer": {
    "batch_size": 32
  }
}
```

`train_data`、`valid_data` 和 `test_data` 使用相同的发现规则。输入会扩展为已解析的绝对文件路径、排序并去重。不存在的路径、非 `.aselmdb` 文件，或不含 `.aselmdb` 后代的目录，都会在 DDP worker 启动前失败。

`optimizer.batch_size` 有两种 LMDB 模式：

- 正整数：每个 rank 在每个优化器步骤中处理的帧数；
- 字符串 `"mix:N"`：每个 rank 的正原子预算。

对于非 LMDB 格式，字符串批次大小仍然无效。

`lmdb_stat_frames` 是所需的全局统计样本数，默认值为 32768。实际样本大小为：

```text
min(dataset_frames, lmdb_stat_frames, 32768 * world_size)
```

因此没有任何 rank 会处理超过 `4096 * 8 == 32768` 个统计帧。

## 组件

### 路径发现与配置

除 `lmdb` 外，`WorkFileStructure` 会保留每种格式原有的路径行为。LMDB 路径通过专用的发现辅助函数解析。`InputParam` 解析 `lmdb_stat_frames`，将其验证为正整数，并将其序列化到 checkpoint JSON 中。

### 延迟 ASE-LMDB 数据集

`src/pre_data/nep_lmdb_dataset.py` 负责新的路径。在构造时，它仅短暂打开每个分片以读取 `nextid` 和可选的 `deleted_ids`，随后关闭它。驻留的索引状态由分片路径、计数、累积偏移量和已删除 ID 构成；它不会将 `range(nextid)` 构造为列表。

`__getitem__` 通过二分查找将全局逻辑索引映射为 `(shard, ASE row id)`。每个 DataLoader 进程都会延迟打开只读环境，参数如下：

```python
lmdb.open(
    path,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)
```

每进程八条目的 LRU 会限制已打开环境的数量。数据集的 pickle 序列化和进程 fork 绝不会携带存活的 LMDB 句柄。环境会在 LRU 驱逐时以及显式数据集清理时关闭。

帧值使用 `zlib` 解压，并在可用时使用 `orjson` 解码，否则回退到标准 `json` 模块。不使用 pickle 反序列化。

### 帧转换

解码器输出由现有 `variable_length_collate_fn` 消费的相同映射：

- `numbers` 映射到已配置的 NEP 类型索引；
- `positions`、`cell`、`energy` 和 `forces` 转为带类型的张量；
- 六分量 ASE 应力 `[xx, yy, zz, yz, xz, xy]` 通过 `virial = -stress * abs(det(cell))` 转为对称的 3x3 维里；
- 缺失的维里使用现有的 `-1e6` 掩码；
- 当 `train_ei` 为 false 时，缺失的原子能量使用有限占位符；而 `train_ei=true` 会因标签错误而失败；
- 电荷、片段和 BEC 字段使用与 `UniDataset` 相同的默认值。

必需标签缺失、损坏的 JSON/zlib 记录、非周期性晶胞、无效形状以及 `atom_type` 中缺少的原子类型，都会包含分片路径和行键。

### 有界内存分布式采样器

整数采样器无需完整的 `torch.randperm` 即可生成确定性的全局排列。它会洗牌块 ID，然后在一个有界块内洗牌索引。种子和 epoch 决定顺序。

对于每 rank 帧批次 `B`，采样器消耗 `B * world_size` 个全局索引，并给每个 rank 分配其不重叠的 `B` 帧切片，最后丢弃不完整的全局超级批次。因此，每个 rank 都有相同数量的优化器步骤，且不会重复训练帧。Rank 数据既不是连续的源区间，也不与另一 rank 共享。

验证使用升序全局索引和相同的全局超级批次拆分。

### 原子预算批次

ASE LMDB 没有逐帧原子计数元数据。因此，严格的 `mix:N` 打包使用紧凑的 `int32` 原子计数侧车文件。首次使用时，rank 会划分 LMDB 分片，使用启用 readahead 的方式顺序扫描分配给它们的分片，并在 `<json_dir>/.matpl_lmdb_cache` 下原子地写入每个分片一个缓存文件。已解析路径、大小、修改时间、`nextid` 和已删除 ID 的指纹会使过期缓存条目失效。分布式屏障会确保完整缓存对采样前可见。后续运行会对缓存进行内存映射。

mix 采样器遍历确定性的全局索引流，在添加一帧会超过 `N` 前关闭一个批次，并允许大于 `N` 的帧形成单帧批次。完成的批次会按 `world_size` 分组缓冲；rank `r` 接收第 `r` 项，最终不完整的 rank 组被丢弃。采样器内存受限于一个洗牌块和一组 rank 批次。

### 全局采样统计

所有 rank 独立构造相同的确定性全局样本，并取用 `sample_indices[rank::world_size]`。本地索引在读取前排序，以改善存储局部性。

每个 rank 累积：

- 原子计数总和、最大值和帧数；
- 用于 NEP 能量平移的逐帧组成正规方程；
- 描述符分量最小值和最大值；
- 径向和角向邻居数的最大值。

分布式归约会在每个 rank 上产生相同的能量平移、平均/最大原子数、q 缩放器和邻居容量。绝不会为这些初始化统计量扫描完整数据集。

### 集成

`nep_network.load_data()` 会根据 `format == "lmdb"` 分支。它在保留现有返回约定的同时，构造新的数据集、采样器、DataLoader 和采样统计加载器。对于 LMDB，epoch 更新目标为 `loader.batch_sampler.set_epoch`；对于旧版数据集，则保留 `DistributedSampler.set_epoch`。

DataLoader 预取保持有界。只有当 `workers > 0` 时才启用 `persistent_workers` 和 `prefetch_factor`，因此 `workers: 0` 也有效。

## 测试与验收

测试使用标准库 `unittest`，因为 `matpl-2026.3` 不包含 pytest。

- 临时 ASE-LMDB 测试夹具验证发现、已删除行、延迟句柄、转换、应力符号/顺序、标签错误和损坏行。
- 纯采样器测试验证可复现性、epoch 变化、完整覆盖、不重叠 rank、相等步骤计数、有界块和 mix 预算。
- 统计测试验证全局样本上限和模拟多 rank 聚合。
- 一个真实数据只读测试验证 11 个分片、1,077,382 帧、随机读取和非线性 RSS 行为。
- 通过复制可信的压缩行导出的一个小型 smoke LMDB，被用于一次简短的单 GPU 运行，以及一次在 `q4` 或 `3090` 上进行的单节点四 GPU Slurm 运行。
- 整数和 `mix:N` 路径都必须进入训练、保持 loss/gradients 有限，并写入 checkpoint。

不得将任何源代码或测试结果推送到 GitHub remote。
