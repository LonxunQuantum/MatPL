# MatPL NEP 测试

在集群登录节点进入 MatPL 仓库根目录。下面的命令统一把日志、临时文件和报告写到 `/data/home/wuxingxing/tmp`。

```bash
cd /data/home/wuxingxing/xcode/MatPL-dcu-dev
```

## 完整测试

运行全部 46 项，包括 12 项六体系数值回归和 34 项 BaTiO 功能测试。数值回归与历史基线比较，BaTiO 功能测试按下文各组的通过条件验收。

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --profile full

# 执行结果展示
# Run 20261123-181809-d95365c8  /data/home/wuxingxing/tmp/runs/20261123-181809-d95365c8
# Base 20261120-132417-832ab19c
# PASS regression/16_metal/batch32 
# PASS regression/MPtraj/batch32 
# ...
# PASS lmdb/BaTiO/gpu2-frames 
# PASS lmdb/BaTiO/gpu2-mix 
# 46/46 passed  /data/home/wuxingxing/tmp/runs/20261123-181809-d95365c8/summary.json

```

## 冒烟测试

确认 5 个常用入口可用，包括 BaTiO 单卡训练、CPU 和 GPU 的 CLI Ewald 推理，以及 CPU 和 GPU 的 ASE 公开计算器。训练检查更新和产物，推理检查输出，并用本次 CLI 结果对照相同设备上的 ASE 结果。内置基线没有这 5 项的历史条目。

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --profile smoke
```

查看完整用例及所需的 CPU 和 GPU 资源。

```bash
python3 -B src/auto_test/auto_test.py list
```

## 六体系训练回归

`16_metal`、`MPtraj`、`AuAg`、`LiSiC`、`BaTiO` 和 `Si-SiO2-La2O3-HfO2-TiN` 各用 batch 1 和 batch 32 训练一轮。

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite regression

# 执行结果展示
# python auto_test.py run --work-dir ~/tmp/ --suite regression
# Run 20261123-161751-4146ba06  /data/home/wuxingxing/tmp/runs/20261123-161751-4146ba06
# Base 20261120-132417-832ab19c
# PASS regression/16_metal/batch32 
# PASS regression/MPtraj/batch32 
# PASS regression/MPtraj/batch1 
# PASS regression/AuAg/batch32 
# PASS regression/16_metal/batch1 
# PASS regression/LiSiC/batch1 
# PASS regression/AuAg/batch1 
# PASS regression/LiSiC/batch32 
# PASS regression/Si-SiO2-La2O3-HfO2-TiN/batch32 
# PASS regression/BaTiO/batch32 
# PASS regression/BaTiO/batch1 
# PASS regression/Si-SiO2-La2O3-HfO2-TiN/batch1 
# 12/12 passed  /data/home/wuxingxing/tmp/runs/20261123-161751-4146ba06/summary.json
```


只跑 BaTiO 的 batch 32。

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite regression \
  --case BaTiO \
  --variant batch32

# 执行结果展示
# Run 20261123-162838-237d5e72  /data/home/wuxingxing/tmp/runs/20261123-162838-237d5e72
# Base 20261120-132417-832ab19c
# PASS regression/BaTiO/batch32 
# 1/1 passed  /data/home/wuxingxing/tmp/runs/20261123-162838-237d5e72/summary.json
```

**怎么判定 PASS**。训练完成后，框架将第 1 轮的 loss、启用的 RMSE、最后一次使用的学习率，以及 checkpoint 中的优化器更新次数，与历史基线中同一体系、同一 batch 的条目逐项比较。数值指标须在容差内，更新次数须相同。

默认读取 `src/auto_test/quick_test` 的固定输入，以及 `src/auto_test/reference/manifest.json` 的基线 `20261120-132417-832ab19c`。工作目录已有发布的基线时，优先用该版本。要固定使用内置基线，可追加 `--base 20261120-132417-832ab19c`。要改用外部数据，可追加 `--data-root /path/to/quick_test`。

默认容差为 `1e-9 + 1e-10 × |基线值|`，可用 `--atol` 和 `--rtol` 调整。输入哈希不匹配时，回归用例标记为 `BLOCKED`。设备、后端或 PyTorch 版本的差异会记录到报告中，数值仍按设定的容差比较。

## BaTiO 功能测试

一次运行全部 34 项 BaTiO 功能测试。命令开头打印的 `Base` 是当前选中的版本。内置基线只有六体系训练回归的 12 个条目，下面五组采用各自的通过条件。如果另行发布的基线包含某个同名用例，框架还会对该用例比较历史结果。

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite train,optimizer,test,ase,lmdb
```

完整 BaTiO 功能测试的输出示例。

```text
Run 20261123-172447-21ea6d07  /data/home/wuxingxing/tmp/runs/20261123-172447-21ea6d07
Base 20261120-132417-832ab19c
PASS train/BaTiO/cpu1
PASS train/BaTiO/autograd
PASS train/BaTiO/gpu2
...
PASS lmdb/BaTiO/gpu1-frames
PASS lmdb/BaTiO/gpu2-mix
34/34 passed  /data/home/wuxingxing/tmp/runs/20261123-172447-21ea6d07/summary.json
```

用 `--suite` 选择测试组，用 `--variant` 选择组内用例。选多个组或多个用例时用逗号分隔。例如，单独比较 GPU 单卡与双卡的 PPPM 推理结果。

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite test \
  --variant gpu1-pppm,gpu2-pppm
```

### 训练与验证

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite train
```

训练使用 48 帧，验证使用 8 帧。两组样本各半包含 BEC 标签和应力标签。

| `--variant` | 检查内容 |
|---|---|
| `cpu1` | CPU 单进程训练与验证 |
| `gpu1` | GPU 单卡训练，启用能量、力、virial、charge 和 BEC 损失 |
| `gpu2` | GPU 双卡 DDP 训练与验证 |
| `autograd` | 单卡关闭解析梯度，使用 autograd |
| `resume` | 续训、更新步数，以及 checkpoint 与导出文本模型的推理对照 |
| `charge0-bec0` | 关闭 charge 和 BEC 损失 |
| `charge0-bec1` | 关闭 charge 损失，启用 BEC 损失 |
| `charge1-bec0` | 启用 charge 损失，关闭 BEC 损失 |

三个 charge/BEC 开关用例持续启用能量、力和 virial 损失。带 BEC 标签和缺 BEC 标签的帧用于检查混合标签掩码。

**怎么判定 PASS**。实际设备和进程数须符合设置，训练和验证须完成预期轮数。框架检查对应损失开关的日志指标及其有限性，确认发生了优化器更新，checkpoint 可读且权重有限，并检查导出的 qNEP 文本模型。`resume` 还要求续训推进更新次数和学习率进度，并在相同输入上逐项比较续训后的 checkpoint 与导出文本模型的推理结果。

### 学习率与周期重启

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite optimizer
```

四个用例使用 BaTiO 单卡 ADAM 和 batch size 4。开启 `scale_lr` 时，峰值学习率为 `0.001 × √4 = 0.002`。

| `--variant` | 周期重启 | `scale_lr` | 预期峰值 LR |
|---|---|---|---:|
| `restart0-scale0` | 关 | 关 | `0.001` |
| `restart0-scale1` | 关 | 开 | `0.002` |
| `restart1-scale0` | 开 | 关 | `0.001` |
| `restart1-scale1` | 开 | 开 | `0.002` |

**怎么判定 PASS**。先通过训练与验证的常规检查，再把日志和 checkpoint 中的学习率状态与配置算出的预期值比较。

- 关闭重启时跑 1 轮。checkpoint 中的 scheduler 须为空，最后一次实际学习率和优化器保存的学习率都须等于预期峰值。
- 开启重启时以 `t_0=1`、`t_mult=2` 跑 4 轮。checkpoint 须保存正确的峰值和 `T_mult=2`，保存时的周期长度须为初始周期的 4 倍。检查针对峰值和最终调度状态。

### CLI 推理

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite test

# 测试运行展示
# Run 20261123-181346-81734020  /data/home/wuxingxing/tmp/runs/20261123-181346-81734020
# Base 20261120-132417-832ab19c
# PASS test/BaTiO/cpu1-pppm 
# PASS test/BaTiO/cpu1-ewald 
# PASS test/BaTiO/cpu4-ewald 
# PASS test/BaTiO/cpu4-pppm 
# PASS test/BaTiO/gpu2-ewald 
# PASS test/BaTiO/gpu1-pppm 
# PASS test/BaTiO/gpu1-ewald 
# PASS test/BaTiO/gpu2-pppm 
# PASS test/BaTiO/cpu1-default 
# PASS test/BaTiO/cpu1-invalid 
# 10/10 passed  /data/home/wuxingxing/tmp/runs/20261123-181346-81734020/summary.json

```

用例调用 `MATPL test`，用同一个模型和同一批 BaTiO 结构比较不同执行方式。

| 设备 | Ewald 的 `--variant` | PPPM 的 `--variant` |
|---|---|---|
| CPU 单进程 | `cpu1-ewald` | `cpu1-pppm` |
| CPU 四进程 | `cpu4-ewald` | `cpu4-pppm` |
| GPU 单卡 | `gpu1-ewald` | `gpu1-pppm` |
| GPU 双卡 | `gpu2-ewald` | `gpu2-pppm` |

**怎么判定 PASS**。先核对实际设备、worker 数、逐帧原子数和输出形状，检查预测值有限、原子能求和一致和总电荷接近零。数值对照分三类。

- 同一次运行选中同一算法的 CPU 单进程与四进程，或 GPU 单卡与双卡时，逐项比较总能、原子能、力、virial、charge 和 BEC。每个分量容差为 `5e-5 + 1e-6 × |单进程或单卡参考值|`。
- 每个 PPPM 用例另跑一次 Ewald，要求两种算法的 virial 最大差值超过 `1e-5`，以检测意外回退。
- `cpu1-default` 和 `cpu1-invalid` 与显式 Ewald 的结果逐项对照。非法值用例还要求输出回退警告。

### ASE 推理

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite ase

# 测试运行展示
# Base 20261120-132417-832ab19c
# PASS ase/BaTiO/cpu-calculator 
# PASS ase/BaTiO/cpu-pppm 
# PASS ase/BaTiO/cpu-charged 
# PASS ase/BaTiO/cpu-ewald 
# PASS ase/BaTiO/gpu-pppm 
# PASS ase/BaTiO/gpu-ewald 
# PASS ase/BaTiO/gpu-calculator 
# PASS ase/BaTiO/gpu-charged 
# 8/8 passed  /data/home/wuxingxing/tmp/runs/20261123-181020-377bcd30/summary.json

```

| 检查内容 | CPU 的 `--variant` | GPU 的 `--variant` |
|---|---|---|
| 公开计算器，检查能量、力、应力和结构改变后重算 | `cpu-calculator` | `gpu-calculator` |
| 底层接口，Ewald、零总电荷 | `cpu-ewald` | `gpu-ewald` |
| 底层接口，PPPM、零总电荷 | `cpu-pppm` | `gpu-pppm` |
| 底层接口，PPPM、总电荷 1 | `cpu-charged` | `gpu-charged` |

**怎么判定 PASS**。公开计算器检查原子能求和及输出形状，并在移动原子后验证力发生变化。底层接口检查 charge 和 BEC 的形状，以及逐原子电荷求和是否符合 `total_charge / sqrt_epsilon_inf`。PPPM 用例还与同一输入下的 Ewald 比较 virial，要求最大差值超过 `1e-5`。

用 `--suite test,ase` 同时选中对应的用例时，框架比较相同设备、相同算法下的 CLI 和 ASE 输出。公开计算器与 CLI Ewald 对照总能、原子能、力和 virial。零总电荷的底层用例还比较 charge 和 BEC。非零总电荷用例采用前述电荷求和检查。

### LMDB 训练

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite lmdb

# 执行结果展示
# Run 20261123-174522-166cc43c  /data/home/wuxingxing/tmp/runs/20261123-174522-166cc43c
# Base 20261120-132417-832ab19c
# PASS lmdb/BaTiO/gpu1-frames 
# PASS lmdb/BaTiO/gpu1-mix 
# PASS lmdb/BaTiO/gpu2-mix 
# PASS lmdb/BaTiO/gpu2-frames 
# 4/4 passed  /data/home/wuxingxing/tmp/runs/20261123-174522-166cc43c/summary.json

```

用例从 BaTiO 固定子集构建小型 `.aselmdb`，使用 FP64 训练。真实 BEC 标签监督由 extxyz 的训练组覆盖。

| 批次类型 | 单卡的 `--variant` | 双卡的 `--variant` |
|---|---|---|
| 整数 batch size 4 | `gpu1-frames` | `gpu2-frames` |
| `mix:64` 原子预算 | `gpu1-mix` | `gpu2-mix` |

**怎么判定 PASS**。四个组合分别验收实际 GPU 和 DDP rank、训练和验证是否跑完第 1 轮、日志中的指标是否齐全且有限，以及 checkpoint、参数更新和 qNEP 模型导出。优化器更新次数须大于 0。这一组的单卡、双卡和两种批次配置各自验收，不互相对比数值。

## 切换被测代码

默认激活 `matpl-2026.3`，再加载当前 checkout 根目录下的 `env.sh`。换一份被测代码时，将 `/path/to/MatPL` 换成另一份 checkout 的目录。它的 `env.sh` 应让 `MATPL` 指向那份 checkout 的 `main.py`。

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite regression \
  --code /path/to/MatPL
```

## 重建、发布与复用基线

内置基线来自提交 `941d15a`。下面用当前干净 checkout 的提交生成 12 项候选基线。只重建 BaTiO batch 32 时，追加 `--case BaTiO --variant batch32`。

```bash
python3 -B src/auto_test/auto_test.py base build \
  --work-dir /data/home/wuxingxing/tmp \
  --suite regression \
  --reference "$(git rev-parse HEAD)"
```

构建成功后，在同一个工作目录发布候选基线。把 `CANDIDATE_ID` 换成上一步输出的候选 ID。局部发布会继承其余基线条目。

```bash
python3 -B src/auto_test/auto_test.py base promote CANDIDATE_ID \
  --work-dir /data/home/wuxingxing/tmp
```

用内置基线重新比较已有结果。把 `RUN_ID` 换成先前运行的 ID，此命令复用训练产物。

```bash
python3 -B src/auto_test/auto_test.py compare RUN_ID \
  --work-dir /data/home/wuxingxing/tmp \
  --base 20261120-132417-832ab19c
```

## 设置 Slurm 分区和运行环境

默认候选分区为 `3090,q4`。逗号分隔的列表是可选分区。下面的命令把 GPU 和 CPU 作业都限制在 `3090`。要增加分区，把 `--partition 3090` 改为 `--partition 3090,q4`。如果 CPU 用例应单独使用 `cpu` 分区，再加 `--cpu-partition cpu`。

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite test \
  --partition 3090
```

经常切换环境时，把示例脚本复制到工作目录，再编辑副本。

```bash
cp src/auto_test/site.example.sh /data/home/wuxingxing/tmp/matpl-site.sh
```

例如，把脚本里的分区设为下面两项。CPU 分区留空时沿用 `MATPL_QA_PARTITION` 的列表。

```bash
MATPL_QA_PARTITION="3090,q4"
MATPL_QA_CPU_PARTITION="cpu"
```

运行时用 `--site-config` 指向该脚本。先加 `--dry-run` 检查实际生成的 `sbatch` 命令，再移除该选项提交作业。

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /data/home/wuxingxing/tmp \
  --suite test \
  --site-config /data/home/wuxingxing/tmp/matpl-site.sh \
  --dry-run
```

`run` 和 `base build` 均支持站点配置。框架用 Bash 读取脚本中的 `MATPL_QA_*` 设置。示例还提供 GRES 名称、作业时限、并发数、Conda 初始化脚本、Conda 环境、CUDA 模块和 MatPL checkout 的设置。命令行显式传入的值优先，其次是脚本值，最后是框架默认值。

## 作业资源与报告

`run`、`base build`、`base promote` 和 `compare` 需要提供 `--work-dir`，也接受 `--work_dir`。目录须位于 Git 仓库外，并可从登录节点和计算节点访问。`list` 可直接执行。给 `run` 或 `base build` 加上 `--dry-run` 可以预览选中的作业和资源申请。

默认 Slurm 分区为 `3090,q4`，每项限时 30 分钟，最多并发 4 项。可用 `--jobs`、`--partition`、`--cpu-partition` 调整。已有计算节点资源时，可加 `--executor local --jobs 1`。DCU 可配合 HIP 环境和分区使用 `--gres dcu --cuda-module ''`。

| 工作目录下的位置 | 内容 |
|---|---|
| `runs/RUN_ID/summary.json` | 本次运行的总结果 |
| `runs/RUN_ID/` 下的各用例目录 | 配置、Slurm 日志、`result.json`、临时文件和缓存 |
| 用例目录内的 `model_record/` | 训练日志、checkpoint 和导出模型 |
| `candidates/CANDIDATE_ID/` | 待发布基线的作业记录和结果 |
| `baselines/BASE_ID/` | 在该工作目录发布的基线 |

终端中间的单项 `PASS` 表示该作业的检查已通过。历史基线或跨用例对比在最后汇总，最终结果以通过总数和 `summary.json` 为准。

单独验证测试框架本身。

```bash
python3 -B -m unittest discover -s src/auto_test -v
```
