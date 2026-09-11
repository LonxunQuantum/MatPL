# OMat24 训练示例

本目录提供 OMat24 ASE-LMDB 数据的 FP64 NEP 多卡训练配置，以及 `mix:N` 原子预算对应的 frame batch-size 分布。

## 配置文件

- `nep.json`：6 epoch、固定 `batch_size=128` 的基础配置。
- `nep_batchset.json`：2 epoch、固定 `batch_size=256` 的对照配置。
- `nep_mixset.json`：2 epoch、`batch_size="mix:20480"` 的原子预算配置。

`lmdb_stat_frames=32768` 表示每张卡最多抽取 32768 个 frame 计算 feature scaler。固定整数 batch 按 frame 数组 batch；`mix:N` 按每卡每步目标原子数打包，可降低结构原子数不均衡造成的显存和耗时波动。

示例中的训练数据路径是本项目集群上的 OMat24 路径。其他环境需要修改三个 JSON 文件的 `train_data`。

## 提交训练

`dev.job` 默认使用：

```text
/data/home/wuxingxing/xcode/MatPL-dcu-dev
```

如代码位于其他目录，通过 `MATPL_ROOT` 覆盖，不需要修改脚本：

```bash
MATPL_ROOT=/path/to/MatPL sbatch dev.job
```

脚本申请 2 个节点、每节点 4 张 GPU，并使用 `3090,q4` 分区。Python 环境、CUDA module 和分区应按实际集群调整。

## `mix:N` 分布数据

`cout/` 保存 mix 值为 1024、2048、4096、6144、8192、10240、12288、16384、20480、40960 时的：

- 每个 mix 值独立的分布图；
- 对应绘图频率 CSV；
- 汇总 CSV 和 JSON；
- 2×5 汇总图；
- 可复现统计的 Python 和 Slurm 脚本。

统计任务默认读取本项目集群上的 OMat24 路径。可通过环境变量覆盖：

```bash
OMAT24_SOURCE=/path/to/omat24/train sbatch cout/run_omat24_mix_batchsize.slurm
```

结果默认写入 `cout/`。脚本采用与 MatPL epoch 0 相同的 block shuffle 和 greedy `mix:N` 打包规则。
