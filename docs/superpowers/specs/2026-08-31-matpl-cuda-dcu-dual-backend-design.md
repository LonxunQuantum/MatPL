# MatPL CUDA/DCU 双后端统一设计

- 日期：2026-08-31
- 状态：已确认设计，待实施计划
- 目标分支：`nep-dcu/merge`
- 目标仓库：`/public/home/pwmat/wuxing/MatPL-main-2026.3`
- 最新 main 基线：`a8e2bc5`
- 旧 DCU 移植基线：`b9cc30c8dbfea617a6b8cabadab8c476c556b56e`

## 1. 背景

现有代码分为两套：

- `MatPL-main-2026.3` 是最新 main 代码，GPU 算子面向 NVIDIA CUDA；当前工作分支为 `nep-dcu/merge`。
- `MatPL-DCU-2026.3` 基于 2026-05-08 的 main 提交 `b9cc30c`，在 `src/op` 中将 CUDA 算子移植为 HIP，并加入曙光 DCU/DTK 专用设置。

旧 DCU 仓库中的 Python 文件除 `src/PWMLFF/nep_network.py` 删除一行打印信息外，没有业务代码变化；其余 Python 状态变化均为 `100755` 到 `100644` 的权限变化。旧 DCU 的 `src/op` 则包含真实的 HIP 移植和部分 DCU 专用算法重写。

最新 main 相比旧基线前进了 68 个提交。其中 `src/op` 相关变化涉及 descriptor VJP、charge/BEC/PPPM 相关接口，以及 multi-batch virial 和 NaN 修复。直接用旧 DCU 的 `src/op` 覆盖最新 main 会丢失这些功能和修复。

## 2. 目标

本设计的目标是：

1. 只维护一套 Python 业务代码和一套 C++/Torch 算子接口。
2. 在同一分支中同时保存 NVIDIA CUDA 和 DCU HIP 两套 GPU kernel 实现。
3. 构建时只选择 CUDA、HIP 或 CPU 中的一种，避免编译缓存、目标文件和动态库相互污染。
4. 保留现有 DCU 专用 kernel 重写及其性能特征。
5. 保持最新 main 的 CUDA 文件路径和业务代码结构，降低以后同步 main 的冲突成本。
6. 对 CUDA 和 HIP 暴露完全一致的 Torch operator 接口，使 Python 代码无需区分算子语义。

## 3. 非目标

本阶段不做以下工作：

- 不立即把所有 CUDA/HIP kernel 重构为完全相同的单一源文件。
- 不要求 CUDA 与 DCU 的绝对执行性能相同。
- 不在同一个 Python 进程中同时加载 CUDA 和 HIP 动态库。
- 不把集群账户、绝对路径、module 命令或 conda 安装动作写入通用构建系统。
- 不借本次兼容改造重构无关的 Python、Fortran 或训练业务代码。

## 4. 方案选择

### 4.1 采用方案：共享接口，双 kernel 后端

共享 Python、C++/Torch 绑定、算子声明和 CPU 实现；CUDA 与 HIP 仅在 GPU kernel、runtime utilities 和后端 CMake 配置上分离。构建系统在一次配置中只启用一个 GPU 后端。

该方案能保留 DCU 专用优化，同时让 main 新增的共享接口只维护一次。

### 4.2 未采用：复制完整 `src/op`

分别维护完整的 CUDA 和 DCU `src/op` 初期迁移简单，但 `CalcOps.cpp`、`CalcOps.h`、绑定代码和算子声明也会重复。main 后续增加接口时仍需改两套，不符合减少维护成本的目标。

### 4.3 暂缓：所有 kernel 使用单一源码

大多数 kernel 可以通过 CUDA/HIP API 映射或 `hipify-perl` 转换，但 `calculateNepFeat*`、`calculateNepMbFeat*` 等 DCU kernel 已包含 wavefront 规约、small-box 实现和同步策略等人工重写。立即统一会增加正确性和性能风险，可在双后端稳定后逐个评估。

## 5. 总体架构

```text
Python 业务代码（共享）
        |
统一算子加载器（共享）
        |
CalcOps C++/Torch API 与绑定（共享）
        |
        +--------------------+
        |                    |
CUDA kernel 后端       HIP kernel 后端
*.cu + CUDA utilities  *.hip + HIP utilities
        |                    |
NVIDIA CUDA/PyTorch     DCU DTK/HIP PyTorch
```

建议目录结构：

```text
src/op/
├── CMakeLists.txt
├── include/                         # 共享算子声明
├── src/                             # 共享 C++/Torch 实现和绑定
├── kernel/                          # 保持 main 现有 CUDA 路径
│   ├── *.cu
│   └── utilities/
├── kernel_hip/                      # DCU/HIP 实现
│   ├── *.hip
│   └── utilities/
└── cmake/
    ├── cpu/CMakeLists.txt
    ├── cuda/CMakeLists.txt
    └── hip/CMakeLists.txt
```

保留 main 的 CUDA 路径，不把现有 `.cu` 文件整体移动到新目录，以便后续合并 main 时继续获得较清晰的 Git 历史和较少的路径冲突。

## 6. 文件所有权与迁移规则

### 6.1 始终采用最新 main 的共享文件

以下内容以 `MatPL-main-2026.3` 为唯一来源：

- 全部 Python 业务代码；
- `src/op/include/*.h`；
- `src/op/src/*.cpp`；
- `src/op/cmake/cpu/CMakeLists.txt`；
- `src/build.sh` 和 `src/clean.sh` 的现有 main 功能；
- `src/feature/NEP_GPU` 的业务源码。

`CalcOps.cpp`、`CalcOps.h`、`CalcOps_bind.cpp`、`op_declare.h`、`calculate_nepfeat.h` 和 `calculate_nepvirial.cpp` 只保存一份，CUDA 与 HIP 后端共同使用。

### 6.2 CUDA 后端

以下内容继续由 main 的 CUDA 代码维护：

- `src/op/kernel/*.cu`；
- `src/op/kernel/utilities/*`；
- `src/op/cmake/cuda/CMakeLists.txt`。

CUDA kernel 的算法、参数、张量布局和共享接口变化需要评估是否同步到对应 HIP kernel。

### 6.3 HIP 后端

从旧 DCU 仓库迁入：

- 全部有效 `.hip` kernel；
- HIP `error`、`gpu_vector`、stream/runtime 处理；
- `common.cuh`、`nep3_small_box*.cuh`；
- 修改后的 `nep_utilities*.cuh`；
- DCU wavefront 规约和专用同步实现；
- 基于 `gfx936` 的 HIP 构建参数。

HIP utilities 放入 `src/op/kernel_hip/utilities/`，不得覆盖 CUDA 的同名 utilities。DCU 版本中的 `src/op/kernel/1` 是临时文件，不迁移。

### 6.4 不迁移的旧 DCU 变化

以下变化不带入新分支：

- Python 和脚本的纯权限变化；
- `src/build.sh` 中删除 Fortran 支持的修改；
- `src/clean.sh` 中删除原有清理逻辑的修改；
- `cpu_calculate_nepneighbor.cpp` 中 `MAX_TYPES` 到 `TYPES` 的无关改名；
- `nep_cpu.h` 中重复且未实际使用的 `NUM_ELEMENTS`；
- 写死用户目录的安装、作业和环境脚本；
- 构建产物、CMake cache、目标文件和临时脚本。

`src/PWMLFF/nep_network.py` 删除打印信息的变化不随旧补丁整体迁移；如最新 main 仍存在同一打印且确认不需要，应作为独立、可审查的修改处理。

## 7. CUDA/HIP 接口契约

共享 C++ 层只依赖后端无关的 `launch_*` 函数声明。每个 CUDA/HIP kernel 对必须满足：

1. 导出函数名称相同；
2. 参数顺序、参数类型和 const 属性相同；
3. 输入输出张量形状和内存布局相同；
4. forward、backward、second-grad/VJP 语义相同；
5. 当前 PyTorch stream 行为一致；
6. 错误处理不静默吞掉 kernel 或 runtime 错误。

后端专用类型不得泄漏到共享 `src/op/include`。CUDA/HIP runtime、stream 和 error 类型应仅出现在各自 kernel 目录中。

## 8. 必须人工同步的现有变化

### 8.1 `calculateNepVirial`

最新 main 已经修改：

- `calculateNepVirial.cu`；
- `calculate_nepvirial.cpp`；
- `calculate_nepfeat.h`；
- `op_declare.h`；
- `CalcOps.cpp`。

共享 C++ 和头文件直接采用最新 main。旧 `calculateNepVirial.hip` 必须人工加入：

- `num_atom` 参数；
- `batch_num` 参数；
- 每个 image 对应的 `net_grad` 选择；
- 在提前 return 之前完成共享内存加载的 NaN 修复；
- 最新 main 的 kernel launch 参数顺序。

### 8.2 descriptor VJP

最新 main 新增：

- `calculateNepFeatWithGradContext`；
- `calculateNepFeatInputGrad`；
- `calculateNepMbFeatWithGradContext`；
- `calculateNepMbFeatInputGrad`。

这些 API 和 Torch 注册属于共享层。HIP 后端需要通过既有的 feat、mbfeat、grad 和 second-grad kernel 提供相同结果，不能复制或回退共享 C++ 实现。

## 9. 构建系统

### 9.1 后端选项

顶层 CMake 提供：

```text
MATPL_GPU_BACKEND=AUTO|CUDA|HIP|CPU
```

默认值为 `AUTO`。自动选择依据 PyTorch 的编译后端，而不是运行节点是否有可见 GPU：

```text
torch.version.hip  != None  -> HIP
torch.version.cuda != None  -> CUDA
否则                        -> CPU
```

不得使用 `torch.cuda.is_available()` 作为编译能力判断，因为登录节点可能没有可见 GPU，但安装的是 CUDA/HIP 版 PyTorch。

### 9.2 严格校验

显式选择后端时：

- CUDA 要求 CUDA 版 PyTorch 和可用 `nvcc`；
- HIP 要求 HIP 版 PyTorch 和可用 `hipcc`；
- 条件不满足时使用 `FATAL_ERROR`；
- 不允许静默回退到 CPU。

`AUTO` 无法找到可用 GPU 工具链时可以选择 CPU，但必须输出明确的配置摘要和警告。

### 9.3 CMake target

共享 C++ 源码先形成公共对象目标，例如 `CalcOps_common`。一次配置只进入一个 GPU 子目录：

- CUDA：`enable_language(CUDA)`，编译 `kernel/*.cu`；
- HIP：`enable_language(HIP)`，编译 `kernel_hip/*.hip`；
- CPU：不启用 GPU language。

后端内部 target 可使用中性名称 `CalcOps_gpu_backend`。为了兼容现有 Python，最终动态库和 Torch namespace 暂时保持：

```text
libCalcOps_bind.so
torch.ops.CalcOps_cuda
```

### 9.4 架构参数

HIP 默认目标为 `gfx936`，同时允许覆盖：

```bash
cmake -DMATPL_GPU_BACKEND=HIP \
      -DMATPL_HIP_ARCHITECTURES=gfx936 ..
```

CUDA 架构保留 main 当前默认集合，并允许通过 CMake cache 变量覆盖。默认架构不应硬编码在多个脚本中。

## 10. 构建产物隔离

后端分别使用：

```text
src/op/build/cuda/
src/op/build/hip/
src/op/build/cpu/
```

不同后端不得共用 CMake cache、目标文件或生成目录。`src/build.sh` 接受统一选项，例如：

```bash
bash build.sh --gpu-backend cuda -j4
bash build.sh --gpu-backend hip -j4
bash build.sh --gpu-backend cpu -j4
bash build.sh --gpu-backend auto -j4
```

构建脚本只负责选择构建目录、传递 CMake 变量和执行构建，不修改源文件或第三方安装目录。

## 11. Python 算子加载

新增一个共享算子加载模块，例如：

```text
src/op_loader.py
```

它集中负责：

- 判断当前 PyTorch 为 CUDA、HIP 或 CPU；
- 选择 `src/op/build/<backend>/lib/`；
- 加载对应 `libCalcOps_bind.so`；
- 防止同一进程重复加载；
- 校验动态库后端与 PyTorch 后端一致；
- 返回现有 `torch.ops.CalcOps_cuda` namespace。

当前散落在 `nep_net.py`、`dp_dp.py`、`dp_dp_typ_emb.py`、`cheby_net.py`、trainer 和 data loader 中的重复加载逻辑，统一改为调用该模块。Python 业务逻辑仍保持一套。

CUDA 和 HIP 实现注册相同的 Torch operator，因此一个 Python 进程只加载当前环境对应的一个后端，这是预期约束。

## 12. `feature/NEP_GPU` 处理

第一阶段不复制 `src/feature/NEP_GPU`。旧 DCU 版本仅将：

```cmake
find_package(CUDA 11.8 REQUIRED)
```

放宽为：

```cmake
find_package(CUDA REQUIRED)
```

这表明 DTK CUDA 兼容层能够承担当前编译。版本和工具链差异应通过 CMake 变量表达，而不是由安装脚本对源码执行 `sed -i`。

如果实际验证证明 DTK 兼容层不足，再把 `feature/NEP_GPU` 作为独立的第二阶段后端拆分，不扩大本阶段 `src/op` 改造范围。

## 13. 环境和部署边界

通用构建系统不得包含：

- `/public/home/pwmat/...` 等账户绝对路径；
- 自动 `conda install` 或 `pip install`；
- 对 PyTorch `Caffe2Targets.cmake` 的原地修改；
- 自动修改 `.bashrc`；
- 集群专用 module 和 Slurm 参数。

DCU 环境设置放入独立部署目录，例如：

```text
deploy/scnet/setup-dcu-env.sh
deploy/scnet/install-dcu.sh
deploy/scnet/examples/*.job
```

部署脚本可以加载 DTK 26.04、GCC 9.3、conda 环境以及设置 `ROCM_PATH`、`CMAKE_PREFIX_PATH` 等，但不得改变核心源码。缺失依赖时应打印可操作的错误信息并退出。

## 14. 错误处理

构建和运行必须提供可诊断的失败信息：

- 配置阶段输出选定后端、PyTorch 版本、编译器路径和目标架构；
- PyTorch 后端与所选编译器不匹配时立即失败；
- 动态库不存在或后端不匹配时，加载器报告期望路径和检测到的 PyTorch 后端；
- CUDA/HIP kernel launch 和 runtime 调用检查最后错误；
- 明确区分“没有 GPU 设备”和“没有对应编译工具链”；
- 显式 GPU 构建失败时不生成或复制伪装成 GPU 库的 CPU 动态库。

## 15. 验证策略

### 15.1 构建矩阵

| 环境 | 配置 | 通过标准 |
|---|---|---|
| CPU | `MATPL_GPU_BACKEND=CPU` | 仅构建 CPU 算子 |
| NVIDIA | `MATPL_GPU_BACKEND=CUDA` | 仅编译 `.cu` |
| DCU/DTK 26.04 | `MATPL_GPU_BACKEND=HIP` | 仅编译 `.hip`，目标为 `gfx936` |
| 自动选择 | `MATPL_GPU_BACKEND=AUTO` | 与 PyTorch 编译后端一致 |

检查 `compile_commands.json`，确保 CUDA 构建不包含 `kernel_hip`，HIP 构建不包含 `.cu` kernel。

### 15.2 接口和加载验证

- 两个 GPU 后端成功加载 `libCalcOps_bind.so`；
- `torch.ops.CalcOps_cuda` 中 operator 集合一致；
- 所有共享 `launch_*` 声明均能在对应后端链接；
- CPU fallback 加载行为保持可用；
- 不允许后端动态库与 PyTorch 类型不匹配。

### 15.3 数值一致性

使用相同的小规模确定性输入，在 CUDA、HIP 和可用的 CPU/参考实现之间比较：

- forward 输出；
- 一阶梯度；
- second-grad/VJP；
- neighbor list；
- force 和 virial；
- multi-batch virial；
- 小于 9 个有效邻居和 `neigh_num % 128` 为 1 到 8 的 NaN 回归场景。

浮点容差按数据类型和归约顺序设置，测试不得要求不同 GPU 后端逐位一致。

### 15.4 必测算子

- `calculateNepFeat`；
- `calculateNepFeatInputGrad`；
- `calculateNepMbFeat`；
- `calculateNepMbFeatInputGrad`；
- `calculateNepVirial`；
- `calculateNepNeighbor`；
- 一阶及二阶梯度相关 kernel；
- DCU wavefront 规约和 small-box 路径。

### 15.5 端到端验证

- CUDA 单卡 NEP 训练与推理；
- CUDA 多卡 NEP 训练；
- DCU 单卡 NEP 训练与推理；
- DCU 多卡/多节点训练；
- main 新增 descriptor VJP 测试；
- 典型数据集上的能量、力和 virial 结果对照。

性能测试与正确性测试分开。正确性是合并门槛；DCU 专用 kernel 不应出现明显性能退化，但不要求 CUDA 与 DCU 的绝对速度相同。

## 16. main 后续同步规则

以后合并 main 时按以下规则处理：

- Python、`src/op/src` 和 `src/op/include`：只合并共享版本；
- CUDA runtime、CUDA 架构或纯 CUDA 性能修改：可只更新 CUDA 后端；
- kernel 数学逻辑、参数、张量形状、索引或输出语义变化：必须同步检查 HIP；
- 新增 CUDA kernel：同时新增同签名 HIP 实现或明确禁用对应 HIP 功能；
- 修改算子注册：CUDA/HIP operator 集合必须保持一致；
- 修改 `calculateNepVirial`、descriptor grad、second-grad 等高风险路径时，必须运行双后端数值测试。

维护一份 CUDA/HIP kernel 对照清单，并通过测试检查 operator/schema 集合，避免依靠人工记忆发现接口漂移。

## 17. 实施顺序

1. 在最新 main 上建立后端选择和独立构建目录，但暂不迁入 HIP kernel。
2. 保持 CUDA 构建和测试通过，证明结构调整没有破坏 main。
3. 增加统一 Python 算子加载器，并验证 CUDA/CPU 行为。
4. 迁入 HIP kernel 和 HIP utilities，不覆盖 CUDA 文件。
5. 接入 HIP CMake，完成 DCU 编译和动态库加载。
6. 人工同步 `calculateNepVirial` 最新修复和 descriptor VJP 接口。
7. 建立接口、数值、梯度和端到端验证矩阵。
8. 整理 DCU 部署脚本，移除通用构建中的账户和集群硬编码。
9. CUDA 与 DCU 双后端验证通过后，再清理旧 `MatPL-DCU-2026.3` 的维护职责。

## 18. 验收标准

设计实施完成需同时满足：

1. 最新 main 的 Python 功能和 CUDA 算子测试通过。
2. 同一分支能分别在 NVIDIA CUDA 和 DCU DTK 环境完成构建。
3. CUDA/HIP 构建目录、CMake cache、目标文件和动态库互不混用。
4. Python 业务代码不包含散落的 CUDA/DCU 分支判断。
5. CUDA 与 HIP 注册相同的 Torch operator 接口。
6. `calculateNepVirial` multi-batch 和 NaN 修复在 HIP 中生效。
7. descriptor VJP、一阶梯度和二阶梯度在两个后端通过数值验证。
8. DCU 专用重写被保留，并完成至少一次性能回归检查。
9. 通用构建不修改 PyTorch/conda 安装目录，不包含账户绝对路径。
10. 后续 main 更新只需维护一套 Python/共享 C++ 代码，并通过明确的 kernel 对照规则同步 HIP。
