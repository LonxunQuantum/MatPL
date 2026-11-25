# MatPL DCU 编译与快速回归测试

本文档适用于华中一区 A 区（核心节点）上的 MatPL DCU（BW1000）版本。

`dcu-deploy/scnet` 提供 DCU 编译和环境初始化脚本。训练回归由 `src/auto_test` 执行。

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

## 3. 运行 DCU 训练回归

完成编译后，在仓库根目录按第 2 节加载 DCU 环境，再在同一个 shell 中运行。示例沿用华中一区 A 区的 `hx1hdnormal` 分区，`--work-dir` 请指定仓库外的可写目录。

```bash
python3 -B src/auto_test/auto_test.py run \
  --work-dir /public/home/pwmat/wuxing/matpl-auto-test \
  --suite regression \
  --partition hx1hdnormal \
  --gres dcu \
  --time 01:00:00 \
  --cuda-module '' \
  --conda-sh "$(conda info --base)/etc/profile.d/conda.sh" \
  --conda-env "$CONDA_PREFIX"
```

`--cuda-module ''` 让作业沿用提交环境中的 DTK。作业在计算节点重新激活当前 Conda 环境，并加载本仓库的 `env.sh`。追加 `--dry-run` 可先查看生成的 `sbatch` 命令。

回归覆盖 6 个体系，每个体系以 batch 1 和 batch 32 各训练 1 个 epoch，共 12 项。框架提交并等待作业，然后将训练指标、学习率和优化器更新次数与选定基线逐项比较。工作目录有已发布的基线时优先使用该基线，否则使用仓库内置的基线。

框架为每次运行创建独立目录。日志和产物保存在 `--work-dir` 下的 `runs/RUN_ID/`，最终通过情况和数值比较见该目录的 `summary.json`。

单项运行、BaTiO 功能测试和基线重建命令见 [auto_test 使用说明](../src/auto_test/README.md)。
