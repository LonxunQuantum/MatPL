# MatPL DCU 编译与快速回归测试

本文档适用于华中一区A区（核心节点）上的 MatPL DCU（BW1000) 版本。示例仓库路径为：

`dcu-deploy/scnet` 提供 DCU 编译和环境初始化脚本，`dcu-deploy/quick_test` 用于在代码修改或算子优化后快速检查训练结果是否出现明显回归。

## 1. 编译 DCU/HIP 版本

进入仓库根目录：

```bash
cd /public/home/pwmat/wuxing/MatPL-main-2026.3
```

如需先查看构建命令而不实际编译：

```bash
MATPL_BUILD_JOBS=4 ./dcu-deploy/scnet/install-dcu.sh --dry-run
```

编译 NEP 所需的 CPU、DCU/HIP 算子和 NEP-GPU 接口：

```bash
MATPL_BUILD_JOBS=4 ./dcu-deploy/scnet/install-dcu.sh
```

`MATPL_BUILD_JOBS` 控制并行编译任务数，默认为 4

主要 DCU 构建产物包括：

```text
src/op/build/hip/lib/libCalcOps_bind.so
src/feature/NEP_GPU/build/hip/nep_gpu.so
```

如果修改了 C++、HIP、CUDA、CMake 或算子接口，应重新编译。仅修改 Python 代码时不需要重新编译。

## 2. 加载运行环境

每次打开新的 shell 后，在仓库根目录执行：

```bash
# 替换为自己的安装目录
cd /public/home/pwmat/wuxing/MatPL-main-2026.3
source dcu-deploy/scnet/setup-dcu-env.sh
source env.sh
```

`setup-dcu-env.sh` 必须使用 `source`，它负责加载 GCC、Conda 环境和 DTK 编译/运行环境。`env.sh` 由编译脚本生成，用于把当前 MatPL 源码和命令加入 `PYTHONPATH`、`PATH`。

DTK 初始化时可能提示 `rocm_smi` 路径不存在并回退到默认库。只要后续编译和计算节点测试正常，这个提示不代表构建失败。

## 3. 优化代码后的快速测试

### 3.1 提交全部测试

进入快速测试目录：

```bash
cd /public/home/pwmat/wuxing/MatPL-main-2026.3/dcu-deploy/quick_test
```

使用默认输出目录 `./quick_train`，并行提交所有测试：

```bash
./run_quick_train.sh
```

也可以指定其他输出目录：

```bash
./run_quick_train.sh /public/home/pwmat/wuxing/training_test/my_quick_train
```

脚本会执行以下操作：

1. 为 5 个测试体系分别准备 `batch_size=1` 和 `batch_size=32`，共 10 个任务；
2. 将每个案例复制到独立工作目录；
3. 自动修改复制后的 `nep.json`；
4. 通过 `sbatch` 提交全部任务；
5. 将任务编号、batch size 和工作目录写入 `jobs.tsv`。

**注意：如果指定的输出目录已经存在，脚本会先将其完整删除再重新创建。不要把源码目录、用户主目录或需要保留数据的目录作为输出目录。**

### 3.2 任务结束后比较结果

等待全部任务结束后，由用户手动运行：

```bash
cd /public/home/pwmat/wuxing/MatPL-main-2026.3/dcu-deploy/quick_test
python3 compare_quick_train.py quick_train
```

比较规则如下：

- 只比较第 1 个 epoch，完全忽略后续 epoch；
- 分别比较 `batch_size=1` 和 `batch_size=32`；
- 比较 `loss`、`RMSE_Etot(eV/atom)`、`RMSE_F(eV/Å)`、`RMSE_virial(eV/atom)` 四列；
- 不比较学习率和运行时间；
- 使用绝对误差阈值 `1e-10`；
- 任一结果文件缺少 epoch 1 时，该案例判定失败。

全部通过时会输出：

```text
Summary: 10/10 passed, 0/10 failed
```

脚本返回码为 0。出现 `[FAIL]` 时，输出会列出具体案例、指标、实际值、参考值和绝对误差。

该测试用于快速发现数据读取、前向计算、反向传播或算子修改造成的明显回归。它只验证第一个 epoch，不能替代更长时间训练、断点恢复、学习率重启和完整精度验证。

## 4. 推荐工作流程

优化代码后按以下顺序执行：

```bash
cd /public/home/pwmat/wuxing/MatPL-main-2026.3

# 修改了原生算子或构建文件时重新编译
MATPL_BUILD_JOBS=4 ./dcu-deploy/scnet/install-dcu.sh

# 加载本次构建
source dcu-deploy/scnet/setup-dcu-env.sh
source env.sh

# 提交快速测试
cd dcu-deploy/quick_test
./run_quick_train.sh

# 等待 squeue 中的任务全部结束后比较
python3 compare_quick_train.py quick_train
```

只有 `Summary: 10/10 passed, 0/10 failed` 才表示本轮快速回归全部通过。

## 5. 仓库路径变化

`quick_test/*/run.sh` 中使用了当前仓库的绝对路径，以确保 Slurm 计算节点加载正确源码。如果仓库被移动或复制到其他位置，需要同步修改这些作业脚本中的两行：

```bash
source <新的仓库路径>/dcu-deploy/scnet/setup-dcu-env.sh
source <新的仓库路径>/env.sh
```

可使用以下命令检查所有案例加载的路径：

```bash
cd /public/home/pwmat/wuxing/MatPL-main-2026.3/dcu-deploy
grep -H "^source " quick_test/*/run.sh
```
