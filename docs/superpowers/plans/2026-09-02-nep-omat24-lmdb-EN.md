# NEP OMat24 LMDB 训练实施计划

> **供智能体执行者使用：** 必须使用子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，逐项实施本计划。各步骤使用复选框（`- [ ]`）语法跟踪状态。

**目标：** 让 MatPL NEP 能够直接从 OMat24 ASE-LMDB 文件训练，并实现按帧懒读取、内存有界的 shuffle、分布式统计，以及整数批次或按原子数预算的批次。

**架构：** 仅当 `format == "lmdb"` 时选择专用的 `NepLmdbDataset` 路径；旧有 `UniDataset` 保持不变。数据集每次只解码一个被选中的 zlib-JSON frame，自定义 sampler 构造确定性的全局 rank 批次，并用有界的全局样本计算 NEP 初始化统计量。

**技术栈：** Python 3.11、PyTorch 2.2 distributed/DataLoader API、python-lmdb 1.5.1、NumPy、zlib、JSON/orjson、Python 标准库 unittest、Slurm/NCCL。

**设计规格：** `docs/superpowers/specs/2026-09-02-nep-omat24-lmdb-design.md`

## 全局约束

- 仅当顶层配置为 `"format": "lmdb"` 时启用大数据加载路径。
- 不得改变任何现有格式下 `UniDataset` 的行为。
- 不安装或升级 LMDB；使用已有的 `lmdb==1.5.1`。
- `train_data`、`valid_data` 和 `test_data` 均接受递归目录和 `.aselmdb` 文件列表。
- 默认全局统计样本数为 32768 帧；每个 rank 的硬上限为 32768 帧。
- 整数 `batch_size` 表示每个 rank 的 frame 数；`"mix:N"` 表示每个 rank 的原子数。
- 各分布式 rank 的优化器 step 数必须相等，且不得通过填充造成 frame 重复。
- Slurm 集成测试只能使用 `q4` 或 `3090` 分区。
- 绝不向 GitHub 远端推送更改。

---

### 任务 1：LMDB 路径和配置契约

**文件：**
- 新建：`src/pre_data/nep_lmdb_dataset.py`
- 修改：`src/user/work_file_param.py:161-205`
- 修改：`src/user/input_param.py:186-214,258-340`
- 测试：`src/test/test_nep_lmdb_config.py`

**接口：**
- 产出：`discover_aselmdb_files(paths: Sequence[str]) -> list[str]`。
- 产出：`InputParam.lmdb_stat_frames: int`，默认值为 `32768`。
- 产出：`WorkFileStructure.*_data_path` 中规范化后的 LMDB 路径。

- [ ] **步骤 1：编写失败的配置测试**

创建标准库测试，构建嵌套临时目录并断言：

```python
files = discover_aselmdb_files([root, explicit_file, root])
self.assertEqual(files, sorted({str(nested_file.resolve()), str(explicit.resolve())}))
```

同时断言：空目录、错误的文件后缀和不存在的路径会抛出 `ValueError`；`lmdb_stat_frames` 会拒绝零、布尔值、字符串和负整数。

- [ ] **步骤 2：运行测试并确认 RED**

运行：

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_config -v
```

预期：导入 `discover_aselmdb_files` 失败。

- [ ] **步骤 3：实现最小化的发现与解析逻辑**

实现一个无副作用的辅助函数：

```python
def discover_aselmdb_files(paths):
    discovered = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            matches = list(path.rglob("*.aselmdb"))
            if not matches:
                raise ValueError(f"No .aselmdb files found under {path}")
            discovered.update(item.resolve() for item in matches if item.is_file())
        elif path.is_file() and path.suffix == ".aselmdb":
            discovered.add(path)
        else:
            raise ValueError(f"Expected an .aselmdb file or directory: {path}")
    return sorted(str(path) for path in discovered)
```

仅在 `self.format == "lmdb"` 时进入 `set_train_valid_file()` 的对应分支。在 `InputParam` 中解析并序列化 `lmdb_stat_frames`。

- [ ] **步骤 4：运行聚焦测试并确认 GREEN**

运行步骤 2 中的 unittest 命令。预期：所有测试通过。

- [ ] **步骤 5：提交到本地仓库**

```bash
git add -f src/test/test_nep_lmdb_config.py
git add src/pre_data/nep_lmdb_dataset.py src/user/work_file_param.py src/user/input_param.py
git commit -m "feat(nep): add LMDB input configuration"
```

### 任务 2：ASE-LMDB 元数据与 frame 懒解码器

**文件：**
- 修改：`src/pre_data/nep_lmdb_dataset.py`
- 测试：`src/test/test_nep_lmdb_dataset.py`

**接口：**
- 产出：带有逻辑长度和行 ID 映射的 `AseLmdbShard(path: str)`。
- 产出：`NepLmdbDataset(...).__getitem__(index) -> dict[str, torch.Tensor]`。
- 产出：`NepLmdbDataset.close() -> None`。
- 使用：任务 1 产生的规范化路径。

- [ ] **步骤 1：在测试模块中编写可复用且安全的 LMDB fixture**

该 fixture 将行写成 zlib 压缩的 JSON，并用 `nextid` 以及可选的 `deleted_ids` 键写入元数据。包含两个原子数不同、stress 值已知的周期性结构。不要使用 pickle。

- [ ] **步骤 2：编写失败的元数据/懒加载测试**

断言构造过程只读取计数，不保留 environment 或 frame 列表；删除的行 ID 映射正确；负索引和越界索引遵循 Dataset 语义；`len(dataset)` 等于所有 shard 逻辑长度之和。

- [ ] **步骤 3：运行解码器测试并确认 RED**

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_dataset -v
```

预期：缺少 `AseLmdbShard` 和 `NepLmdbDataset`。

- [ ] **步骤 4：实现 shard 元数据和 LRU environment 所有权管理**

将活动 environment 保存在容量上限为 8 的 `OrderedDict` 中。确保 `__getstate__()` 返回 environment 缓存为空的状态，并让 `close()` 关闭全部 environment。映射已删除 ID 时，不构造完整的整数 ID 列表。

- [ ] **步骤 5：实现 frame 转换**

解码一行，并输出与 `variable_length_collate_fn` 匹配的键。使用以下方式转换 ASE Voigt stress：

```python
scaled = -np.asarray(stress, dtype=float) * abs(np.linalg.det(cell))
virial = np.array([
    [scaled[0], scaled[5], scaled[4]],
    [scaled[5], scaled[1], scaled[3]],
    [scaled[4], scaled[3], scaled[2]],
])
```

验证 shape、PBC、必需标签、原子类型和有限数值。异常信息以 `<path>: frame key <id>` 为前缀。

- [ ] **步骤 6：扩充 tensor 和错误测试，然后确认 GREEN**

断言每个输出 tensor 的 shape/dtype、collate 后的累积行为、stress 的符号与顺序、缺失 virial 的 mask、缺失原子能策略、未知元素，以及损坏的 zlib/JSON 错误。持续运行步骤 3 的命令，直到测试通过。

- [ ] **步骤 7：提交到本地仓库**

```bash
git add -f src/test/test_nep_lmdb_dataset.py
git add src/pre_data/nep_lmdb_dataset.py
git commit -m "feat(nep): lazily decode ASE LMDB frames"
```

### 任务 3：内存有界的分布式 frame sampler

**文件：**
- 修改：`src/pre_data/nep_lmdb_dataset.py`
- 测试：`src/test/test_nep_lmdb_sampler.py`

**接口：**
- 产出：`BlockShuffleIndices(size, block_size, seed, epoch, shuffle)` 可迭代对象。
- 产出：`DistributedFrameBatchSampler(dataset_size, batch_size, rank, world_size, seed, shuffle)`。
- 产出：`set_epoch(epoch: int)`、`__iter__()` 和精确的 `__len__()`。

- [ ] **步骤 1：编写失败的单 rank sampler 测试**

断言固定的 seed/epoch 会复现相同顺序，新 epoch 会改变顺序；除被丢弃的末尾部分外，每个完整索引恰好出现一次；内部缓冲区绝不超过配置的 block 加一个 super-batch。

- [ ] **步骤 2：编写失败的多 rank sampler 测试**

对 world size 2 和 4 实例化小规模测试输出，并断言各 rank 长度相等、每个 rank 的 batch size 正确、frame 集合两两不相交、所有保留 frame 的并集覆盖完整，以及验证集顺序为升序。

- [ ] **步骤 3：运行测试并确认 RED**

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_sampler -v
```

预期：sampler 导入失败。

- [ ] **步骤 4：实现 block shuffle 和全局 super-batch**

打乱 block ID 列表；每次只为一个 block 分配索引并打乱该 block，然后以流式方式输出索引。每 `batch_size * world_size` 个索引分为一组，输出切片 `[rank * batch_size:(rank + 1) * batch_size]`。丢弃最后不足一组的部分。

- [ ] **步骤 5：运行测试并确认 GREEN**

运行步骤 3 的命令。预期：所有 sampler 测试通过。

- [ ] **步骤 6：提交到本地仓库**

```bash
git add -f src/test/test_nep_lmdb_sampler.py
git add src/pre_data/nep_lmdb_dataset.py
git commit -m "feat(nep): add bounded distributed LMDB sampler"
```

### 任务 4：全局抽样的 NEP 统计量

**文件：**
- 修改：`src/pre_data/nep_lmdb_dataset.py`
- 修改：`src/PWMLFF/nep_network.py:434-553,774-839`
- 测试：`src/test/test_nep_lmdb_stats.py`

**接口：**
- 产出：`select_stat_indices(size, requested, rank, world_size, seed, per_rank_cap=32768) -> list[int]`。
- 产出：带有正规方程、原子计数和 merge 方法的 `LmdbEnergyStatistics`。
- 使用：`calculate_neighbor_scaler`，用于计算本地 descriptor/近邻极值。

- [ ] **步骤 1：编写失败的样本选择测试**

断言全局选择具有确定性、各 rank 切片互不相交、全局默认值为 32768、受数据集大小限制、受请求大小限制，并且每个 rank 存在 32768 的硬上限。

- [ ] **步骤 2：编写失败的聚合统计测试**

使用组成向量和能量已知的合成 frame。合并模拟的 rank 累加器，并将 `energy_shift`、平均原子数和最大原子数与 NumPy 直接计算结果比较。

- [ ] **步骤 3：运行测试并确认 RED**

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_stats -v
```

预期：样本选择/统计 API 尚不存在。

- [ ] **步骤 4：实现有界样本选择和 CPU 累加器**

使用 `random.Random(seed).sample(range(size), count)`，再按 rank 跨步切片，并在 I/O 前对本地索引排序。累加 `A.T @ A` 和 `A.T @ energy`，不保留 frame 对象。

- [ ] **步骤 5：集成分布式归约**

在 LMDB 分支中，对正规方程、原子数的 count/sum/max、descriptor 的 min/max，以及径向/角向近邻最大值进行归约。每个 rank 都根据完全相同的已归约正规方程独立求解 energy shift。使用归约运算的单位元处理本地样本切片为空的情况。

- [ ] **步骤 6：运行统计测试和现有配置测试**

```bash
PYTHONPATH=. python -m unittest \
  src.test.test_nep_lmdb_stats \
  src.test.test_nep_lmdb_config \
  src.test.test_nep_lmdb_dataset -v
```

预期：所有测试通过。

- [ ] **步骤 7：提交到本地仓库**

```bash
git add -f src/test/test_nep_lmdb_stats.py
git add src/pre_data/nep_lmdb_dataset.py src/PWMLFF/nep_network.py
git commit -m "feat(nep): distribute sampled LMDB statistics"
```

### 任务 5：原子数缓存与 `mix:N` sampler

**文件：**
- 修改：`src/pre_data/nep_lmdb_dataset.py`
- 修改：`src/PWMLFF/nep_network.py:434-553,912-930`
- 测试：`src/test/test_nep_lmdb_mix_sampler.py`

**接口：**
- 产出：带有 `build_assigned(rank, world_size)` 和随机访问能力的 `LmdbNatomsCache(dataset, cache_dir)`。
- 产出：`DistributedAtomBatchSampler(natoms, atom_budget, rank, world_size, seed, shuffle)`。
- 使用：任务 3 的 block-shuffled 索引流。

- [ ] **步骤 1：编写失败的缓存测试**

构建原子数已知的 fixture。断言第一次构建通过原子 rename 为每个 shard 写入一个 `int32` 文件，第二次加载会复用该文件，而文件大小/mtime 或 `nextid` 的变化会让缓存失效。断言缓存使用内存映射，而不是将数组加载进内存。

- [ ] **步骤 2：编写失败的 mix sampler 测试**

断言每个输出 batch 的原子总数均 `<= N`，但单个超预算 frame 可以独占一个 batch；各 rank 的 step 数相等且 frame 集合互不相交；epoch 变化具有确定性；最后不足 `world_size` 个的 batch 会被丢弃。

- [ ] **步骤 3：运行测试并确认 RED**

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_mix_sampler -v
```

预期：缓存和 mix sampler 导入失败。

- [ ] **步骤 4：实现逐 shard 缓存**

使用 `readahead=True` 顺序打开各 shard，仅解码 JSON 中足以统计 `numbers` 的部分；写入临时数组，执行 `fsync`，然后以原子方式替换最终缓存文件。使用包含 shard 指纹的 JSON manifest。在 DDP 中，将 shard 索引 `i` 分配给 rank `i % world_size`，随后执行 barrier，再对所有 shard 进行内存映射。

- [ ] **步骤 5：实现在线 mix 打包**

持续累积 frame，直到下一个 frame 会导致超出预算。将已完成的 batch 缓存在长度为 `world_size` 的列表中，输出属于当前 rank 的条目，然后清空该组。使用相同的打包规则，为当前 epoch 实现并缓存精确长度。

- [ ] **步骤 6：运行测试并确认 GREEN**

运行步骤 3 的命令。预期：所有缓存和 mix 测试通过。

- [ ] **步骤 7：提交到本地仓库**

```bash
git add -f src/test/test_nep_lmdb_mix_sampler.py
git add src/pre_data/nep_lmdb_dataset.py src/PWMLFF/nep_network.py
git commit -m "feat(nep): support atom-budget LMDB batches"
```

### 任务 6：NEP 加载器端到端集成

**文件：**
- 修改：`src/PWMLFF/nep_network.py:434-553,774-930`
- 修改：`src/user/optimizer_param.py:1-35,150-175`
- 测试：`src/test/test_nep_lmdb_integration.py`

**接口：**
- 使用：`NepLmdbDataset`、两种分布式 batch sampler、抽样统计量和 `variable_length_collate_fn`。
- 产出：现有的 `load_data()` tuple，以及现有 trainer 的 batch 映射。

- [ ] **步骤 1：编写失败的 DataLoader 集成测试**

构建一个小型双 shard LMDB 和轻量级 `InputParam` stub。断言 LMDB 分支返回的训练/验证 loader，其 batch 包含 `nep_trainer.train` 所需的精确 tensor 键和不等长原子拼接格式。

- [ ] **步骤 2：运行集成测试并确认 RED**

因为导入 `nep_network` 会加载 CalcOps，所以需要在 GPU 节点上运行：

```bash
PYTHONPATH=. python -m unittest src.test.test_nep_lmdb_integration -v
```

预期：`load_data()` 仍会为 LMDB 构造 `UniDataset`。

- [ ] **步骤 3：添加隔离的 `format == "lmdb"` 分支**

使用 `batch_sampler` 构造 LMDB dataset 和 DataLoader。仅当 worker 数大于零时设置 `persistent_workers` 和 `prefetch_factor`。让旧有分支在文本层面保持独立且不变。

- [ ] **步骤 4：将 epoch 转发给正确的 sampler**

添加辅助函数：存在 `loader.batch_sampler.set_epoch(epoch)` 时调用它，否则回退到现有的 `DistributedSampler` 逻辑。验证整数和 `mix:N` 两种 batch-size 语法，并给出包含上下文的错误信息。

- [ ] **步骤 5：运行所有纯测试及 GPU 集成测试**

```bash
PYTHONPATH=. python -m unittest \
  src.test.test_nep_lmdb_config \
  src.test.test_nep_lmdb_dataset \
  src.test.test_nep_lmdb_sampler \
  src.test.test_nep_lmdb_stats \
  src.test.test_nep_lmdb_mix_sampler \
  src.test.test_nep_lmdb_integration -v
```

预期：所有测试通过。

- [ ] **步骤 6：提交到本地仓库**

```bash
git add -f src/test/test_nep_lmdb_integration.py
git add src/PWMLFF/nep_network.py src/user/optimizer_param.py
git commit -m "feat(nep): train directly from OMat24 LMDB"
```

### 任务 7：真实 OMat24 与四 GPU 验收

**文件：**
- 在 Git 外修改：`/data/home/wuxingxing/datas/training_test/mgpus/omat24/nep.json`
- 在 Git 外修改：`/data/home/wuxingxing/datas/training_test/mgpus/omat24/train.job`
- 在 Git 外新建：`/data/home/wuxingxing/datas/training_test/mgpus/omat24/smoke.aselmdb`

**接口：**
- 使用：任务 1–6 完成的 LMDB 训练路径。
- 产出：日志、checkpoint，以及可复现的 Slurm smoke 配置。

- [ ] **步骤 1：运行真实数据的元数据和内存检查**

从目录实例化数据集，并断言共有 11 个 shard、1,077,382 个 frame。读取固定索引和随机索引，记录构造前后的 RSS，并验证不存在与 frame 数量成正比的 Python 列表。

- [ ] **步骤 2：创建小型可信 smoke LMDB**

从真实的只读 LMDB 中复制数量有界的压缩 frame 值到新的测试 LMDB，并写入匹配的压缩 `nextid`。不要用 pickle 反序列化，也不要修改公共数据集。

- [ ] **步骤 3：运行一个简短的单 GPU 整数批次作业**

设置一个 epoch、较小的统计样本数和整数 batch size。确认模型初始化成功、首尾 loss 均为有限值，并创建 checkpoint。

- [ ] **步骤 4：运行一个简短的单节点四 GPU 整数批次作业**

仅使用 `#SBATCH --partition=q4,3090`。确认四个 rank 均完成初始化、执行相同数量的 step，并且没有 collective 不匹配地完成训练。

- [ ] **步骤 5：运行四 GPU `mix:N` smoke 作业**

只将 `optimizer.batch_size` 改为较小的 `mix:N`。确认原子数缓存的创建/复用、batch 遵守预算、各 rank step 数相等，并输出 checkpoint。

- [ ] **步骤 6：运行回归和语法检查**

```bash
python -m compileall -q src/pre_data/nep_lmdb_dataset.py \
  src/user/work_file_param.py src/user/input_param.py \
  src/user/optimizer_param.py src/PWMLFF/nep_network.py
git diff --check
git status --short
```

再次运行可直接执行的现有 NEP 配置测试和所有新 unittest 模块。预期：零失败，且没有非预期的已跟踪产物。

- [ ] **步骤 7：记录结果但不推送**

在最终交接中记录精确命令、Slurm 作业 ID、通过/失败结果、RSS 峰值和已知限制。保持本地分支与 worktree 不变。不要运行 `git push`。
