# MatPL CUDA/DCU 双后端 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox checkoffs and must be completed in order.

**目标：** 在 `MatPL-main-2026.3` 的 `nep-dcu/merge` 分支上实现一套 Python/公共 C++ 接口、两套互不干扰的 CUDA 与 DCU/HIP 算子后端，并保留 CPU 回退能力。

**架构：** Python 只通过一个加载器选择运行后端；公共算子注册和头文件继续位于 `src/op/src`、`src/op/include`；NVIDIA 源保留在 `src/op/kernel`，HIP 源放入 `src/op/kernel_hip`。顶层 CMake 解析 `MATPL_GPU_BACKEND=AUTO|CUDA|HIP|CPU`，分别进入 `cmake/cuda`、`cmake/hip`、`cmake/cpu`，构建产物分别写入 `src/op/build/<backend>`。CUDA/HIP 都继续导出 `libCalcOps_bind.so` 和 `torch.ops.CalcOps_cuda`，避免改动模型层算子调用。

**技术栈：** Python 3、PyTorch C++ Extension、CMake 3.21+、CUDA/NVCC、DTK/HIPCC、Bash、pytest、CTest/CMake script mode。

**设计依据：** `docs/superpowers/specs/2026-08-31-matpl-cuda-dcu-dual-backend-design.md`

**全局约束：**

- 以最新 main 代码为唯一公共代码基线，不反向覆盖 Python 或公共 C++ 文件。
- CUDA 路径尽量保持 main 的文件位置和实现，降低后续合并成本。
- HIP 迁移以旧 DCU 仓库为来源，但必须补齐 main 在基线之后的接口和正确性修复。
- 禁止在部署脚本中写死用户账号、私钥或 `/public/home/pwmat`；禁止用 `sed -i` 临时修改源码。
- 每个任务先写失败测试，再实现，再运行目标验证，然后独立提交。

## Task 1：统一 Python 算子加载器

**Files:**

- Create: `src/utils/op_loader.py`
- Create: `src/test/test_op_loader.py`
- Modify: `src/model/cheby_net.py`
- Modify: `src/model/dp_dp.py`
- Modify: `src/model/dp_dp_typ_emb.py`
- Modify: `src/model/nep_net.py`
- Modify: `src/pre_data/dpuni_data_loader.py`
- Modify: `src/pre_data/find_maxneighbor.py`
- Modify: `src/pre_data/nep_data_loader.py`
- Modify: `src/PWMLFF/nep_mods/nep_trainer.py`
- Modify: `src/test/test_nep_electric/test_descriptor_vjp.py`

- [ ] **Step 1：为后端检测和路径选择写失败测试**

  在 `src/test/test_op_loader.py` 用伪造的 `torch.version` 和 `torch.cuda.is_available()` 覆盖：HIP 编译、CUDA 编译、纯 CPU、GPU 不可用时回退 CPU，以及三种库路径。期望路径分别为：

  ```text
  src/op/build/cuda/lib/libCalcOps_bind.so
  src/op/build/hip/lib/libCalcOps_bind.so
  src/op/build/cpu/lib/libCalcOps_bind_cpu.so
  ```

- [ ] **Step 2：确认测试按预期失败**

  Run: `python -m pytest -q src/test/test_op_loader.py`

  Expected: 因 `src.utils.op_loader` 尚不存在而失败。

- [ ] **Step 3：实现最小加载器 API**

  在 `src/utils/op_loader.py` 实现并加类型注解：

  ```python
  def detect_compiled_backend(torch_module=torch) -> str: ...
  def select_runtime_backend(torch_module=torch) -> str: ...
  def get_library_path(backend: str, src_root: Path | None = None) -> Path: ...
  def load_calc_ops(torch_module=torch, src_root: Path | None = None): ...
  ```

  检测顺序必须是 `torch.version.hip`、`torch.version.cuda`、CPU；运行时无可用 GPU 时选择 CPU。`load_calc_ops` 只加载一次共享库并返回 `CalcOps_cuda` 或 `CalcOps_cpu` 命名空间；不存在的后端和库文件给出包含后端、路径和重建提示的异常。

- [ ] **Step 4：替换所有模型和数据代码中的重复加载逻辑**

  上述 8 个生产 Python 文件统一改成：

  ```python
  from src.utils.op_loader import load_calc_ops
  CalcOps = load_calc_ops()
  ```

  删除各自对 `op/build/lib` 的硬编码；测试文件也通过加载器获得算子，不再自行拼路径。

- [ ] **Step 5：验证加载器及 Python 语法**

  Run: `python -m pytest -q src/test/test_op_loader.py`

  Expected: 全部通过。

  Run: `python -m py_compile src/utils/op_loader.py src/model/cheby_net.py src/model/dp_dp.py src/model/dp_dp_typ_emb.py src/model/nep_net.py src/pre_data/dpuni_data_loader.py src/pre_data/find_maxneighbor.py src/pre_data/nep_data_loader.py src/PWMLFF/nep_mods/nep_trainer.py`

  Expected: 无输出，退出码 0。

- [ ] **Step 6：提交**

  ```bash
  git add src/utils/op_loader.py src/model src/pre_data src/PWMLFF/nep_mods/nep_trainer.py
  git add -f src/test/test_op_loader.py src/test/test_nep_electric/test_descriptor_vjp.py
  git commit -m "refactor: centralize CalcOps backend loading"
  ```

## Task 2：建立 CMake 后端路由和隔离构建目录

**Files:**

- Create: `src/op/cmake/ResolveBackend.cmake`
- Create: `src/op/cmake/tests/test_resolve_backend.cmake`
- Rename: `src/op/cmake/gpu/CMakeLists.txt` → `src/op/cmake/cuda/CMakeLists.txt`
- Modify: `src/op/CMakeLists.txt`
- Modify: `src/op/cmake/cuda/CMakeLists.txt`
- Modify: `src/op/cmake/cpu/CMakeLists.txt`

- [ ] **Step 1：写后端解析的 CMake script 测试**

  测试 `CPU`、`CUDA`、`HIP`、`AUTO+torch.version.cuda`、`AUTO+torch.version.hip`、非法值六种输入。解析模块仅负责把显式输入或 PyTorch 探测结果规范化为 `MATPL_RESOLVED_BACKEND`，编译器与 PyTorch 能力校验由顶层配置完成。

- [ ] **Step 2：运行并观察失败**

  Run: `cmake -P src/op/cmake/tests/test_resolve_backend.cmake`

  Expected: 因 `ResolveBackend.cmake` 尚不存在而失败。

- [ ] **Step 3：实现后端解析模块和顶层路由**

  将 `src/op/CMakeLists.txt` 最低版本提高到 3.21，新增缓存变量：

  ```cmake
  set(MATPL_GPU_BACKEND AUTO CACHE STRING "AUTO, CUDA, HIP, or CPU")
  set_property(CACHE MATPL_GPU_BACKEND PROPERTY STRINGS AUTO CUDA HIP CPU)
  ```

  顶层执行一次 Python 查询同时获得 PyTorch ABI、`torch.version.cuda`、`torch.version.hip`，调用 `ResolveBackend.cmake` 后只 `add_subdirectory(cmake/${resolved})`。显式 CUDA/HIP 与当前 PyTorch 不匹配时 `FATAL_ERROR`，消息必须包含当前 PyTorch 能力和建议命令。生成 `${CMAKE_BINARY_DIR}/matpl_backend.txt`，内容为规范化的小写后端名，并打开 `CMAKE_EXPORT_COMPILE_COMMANDS`。

- [ ] **Step 4：将现有 GPU CMake 重命名为 CUDA 专用**

  使用 `git mv src/op/cmake/gpu src/op/cmake/cuda`。CUDA 子目录继续只 glob `kernel/*.cu` 和 `kernel/utilities/*.cu`；CPU 子目录只编译 CPU 文件，并移除依赖旧 `COMPILE_WITH_GPU` 的路径判断，改由已解析后端决定是否生成兼容别名。

- [ ] **Step 5：验证路由**

  Run: `cmake -P src/op/cmake/tests/test_resolve_backend.cmake`

  Expected: 输出 `ResolveBackend tests passed`，退出码 0。

  Run: `cmake -S src/op -B src/op/build/cpu -DMATPL_GPU_BACKEND=CPU`

  Expected: 配置成功，`src/op/build/cpu/matpl_backend.txt` 内容为 `cpu`。

- [ ] **Step 6：提交**

  ```bash
  git add src/op/CMakeLists.txt src/op/cmake
  git commit -m "build: route CalcOps through explicit backends"
  ```

## Task 3：扩展 build.sh 的后端参数

**Files:**

- Modify: `src/build.sh`
- Create: `src/test/test_build_backend_cli.sh`

- [ ] **Step 1：写命令行路由失败测试**

  测试 `--gpu-backend auto|cuda|hip|cpu`、大小写规范化、非法值、旧 `-jN` 和 `-m nn` 参数兼容。为避免真正编译，脚本测试使用新增的 `--dry-run`，断言 CMake 的 `-B` 目录和 `-DMATPL_GPU_BACKEND` 参数。

- [ ] **Step 2：运行并观察失败**

  Run: `bash src/test/test_build_backend_cli.sh`

  Expected: 当前脚本不认识新参数，测试失败。

- [ ] **Step 3：实现安全的分后端构建**

  `src/build.sh` 新增 `--gpu-backend` 和 `--dry-run`，默认 `auto`。每次只清理/配置 `src/op/build/<resolved-or-requested>`；不得删除整个 `src/op/build`。原 Fortran、`-jN`、`-m nn` 行为保持不变。AUTO 配置完成后从 `matpl_backend.txt` 得到实际目录，必要时先在临时 `auto` 配置目录探测，再构建解析后的目录。

- [ ] **Step 4：验证 CLI**

  Run: `bash src/test/test_build_backend_cli.sh`

  Expected: 全部断言通过。

  Run: `bash -n src/build.sh`

  Expected: 退出码 0。

- [ ] **Step 5：提交**

  ```bash
  git add src/build.sh
  git add -f src/test/test_build_backend_cli.sh
  git commit -m "build: isolate CUDA HIP and CPU build trees"
  ```

## Task 4：导入 HIP 算子树并增加 HIP CMake

**Files:**

- Create: `src/op/cmake/hip/CMakeLists.txt`
- Create: `src/op/kernel_hip/calculateCompress.hip`
- Create: `src/op/kernel_hip/calculateCompress_grad.hip`
- Create: `src/op/kernel_hip/calculateForce.hip`
- Create: `src/op/kernel_hip/calculateForceGrad.hip`
- Create: `src/op/kernel_hip/calculateNepFeat.hip`
- Create: `src/op/kernel_hip/calculateNepFeat_grad.hip`
- Create: `src/op/kernel_hip/calculateNepFeat_secondgradout.hip`
- Create: `src/op/kernel_hip/calculateNepForce.hip`
- Create: `src/op/kernel_hip/calculateNepMbFeat.hip`
- Create: `src/op/kernel_hip/calculateNepMbFeat_grad.hip`
- Create: `src/op/kernel_hip/calculateNepMbFeat_secondgradout.hip`
- Create: `src/op/kernel_hip/calculateNepNeighbor.hip`
- Create: `src/op/kernel_hip/calculateNepVirial.hip`
- Create: `src/op/kernel_hip/calculateVirial.hip`
- Create: `src/op/kernel_hip/calculateVirialGrad.hip`
- Create: `src/op/kernel_hip/utilities/common.cuh`
- Create: `src/op/kernel_hip/utilities/error.cuh`
- Create: `src/op/kernel_hip/utilities/error.hip`
- Create: `src/op/kernel_hip/utilities/gpu_vector.cuh`
- Create: `src/op/kernel_hip/utilities/gpu_vector.hip`
- Create: `src/op/kernel_hip/utilities/main_common.cuh`
- Create: `src/op/kernel_hip/utilities/main_common.hip`
- Create: `src/op/kernel_hip/utilities/nep3_small_box.cuh`
- Create: `src/op/kernel_hip/utilities/nep3_small_box_mbgrad.cuh`
- Create: `src/op/kernel_hip/utilities/nep_utilities.cuh`
- Create: `src/op/kernel_hip/utilities/nep_utilities_mb_secondc.cuh`
- Create: `src/test/test_hip_source_manifest.py`

- [ ] **Step 1：写 HIP 文件清单和隔离测试**

  测试每个 `kernel/*.cu` 都有同名 `kernel_hip/*.hip`，HIP CMake 只引用 `kernel_hip`，CUDA CMake 只引用 `kernel`。扫描 HIP 源中未迁移的运行时 API（`cudaSetDevice`、`cudaGetLastError`、`cudaDeviceSynchronize`、`cudaMalloc`、`cudaMemcpy`），允许兼容宏名称但不允许直接函数调用。

- [ ] **Step 2：运行并观察失败**

  Run: `python -m pytest -q src/test/test_hip_source_manifest.py`

  Expected: 因 `kernel_hip` 与 HIP CMake 尚不存在而失败。

- [ ] **Step 3：从旧 DCU 仓库复制经过验证的 HIP 文件**

  来源固定为 `/public/home/pwmat/wuxing/MatPL-DCU-2026.3/src/op/kernel` 中上述 `.hip`/utility 文件；不复制临时文件 `src/op/kernel/1`，不覆盖 main 的 `src/op/src`、`src/op/include`、CPU 或 CUDA 文件，也不复制旧仓库的 `nep_cpu.h` 改动。

- [ ] **Step 4：实现 HIP CMake**

  `enable_language(HIP)`，使用当前 PyTorch 的 include/library/ABI 信息，只 glob `kernel_hip/*.hip` 和 `kernel_hip/utilities/*.hip`。目标名、输出库名和注册命名空间与 CUDA 保持一致；HIP 架构允许通过 `CMAKE_HIP_ARCHITECTURES` 或环境设置传入，不写死某一 DCU 型号。

- [ ] **Step 5：修复移植残留并通过静态测试**

  将 `error.cuh` 强调试路径中的 `cudaDeviceSynchronize()` 改为 `hipDeviceSynchronize()`，清除其他直接 CUDA runtime 调用。

  Run: `python -m pytest -q src/test/test_hip_source_manifest.py`

  Expected: 全部通过。

- [ ] **Step 6：在 DTK 环境配置 HIP 后端**

  ```bash
  source /public/software/apps/anaconda3/2023.09/bin/activate
  conda activate matpl-2026.3
  source /public/software/compiler/dtk-26.04/env.sh
  source /public/software/compiler/dtk-26.04/cuda/env.sh
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  cmake -S src/op -B src/op/build/hip -DMATPL_GPU_BACKEND=HIP
  cmake --build src/op/build/hip -j4
  ```

  Expected: 生成 `src/op/build/hip/lib/libCalcOps_bind.so`，编译命令不包含 `src/op/kernel/*.cu`。

- [ ] **Step 7：提交**

  ```bash
  git add src/op/cmake/hip src/op/kernel_hip
  git add -f src/test/test_hip_source_manifest.py
  git commit -m "feat: add isolated HIP CalcOps backend"
  ```

## Task 5：同步最新 main 算子接口和正确性修复

**Files:**

- Modify: `src/op/kernel_hip/calculateNepVirial.hip`
- Modify: `src/op/kernel_hip/utilities/error.cuh`
- Create: `src/test/test_cuda_hip_launch_signatures.py`
- Modify: `src/test/test_nep_electric/test_force_virial_paths.py`
- Modify: `src/test/test_nep_electric/test_descriptor_vjp.py`

- [ ] **Step 1：写 CUDA/HIP launch 接口一致性测试**

  解析两棵源目录中的 `void launch_*` 声明/定义，规范化空白和 `.cu/.hip` 差异后比较函数名、参数数量和参数类型。特别断言 `launch_calculate_nep_virial` 同时接收 `num_atom` 和 `batch_num`。

- [ ] **Step 2：运行并观察失败**

  Run: `python -m pytest -q src/test/test_cuda_hip_launch_signatures.py`

  Expected: HIP `calculateNepVirial` 与最新 CUDA 接口不一致，测试失败。

- [ ] **Step 3：把最新 virial 语义移植到 HIP**

  以 main 的 `calculateNepVirial.cu` 为语义基准，在 HIP 版本中同步：

  - `num_atom` 和 `batch_num` 参数；
  - 每个 image 选择各自的 `net_grad`；
  - 线程提前返回之前完成需要的 shared-memory 梯度加载与同步；
  - 保留 HIP 的 wavefront、kernel launch 和 runtime API 写法。

- [ ] **Step 4：将运行测试改为后端中立**

  `test_nep_virial_cuda_backward_uses_each_batch_gradient` 重命名为 GPU 后端中立名称；测试两个 image、`num_atom=[1,1]`，只给第二个 image 上游梯度，并验证能量梯度来自第二个 image。descriptor VJP 测试继续覆盖共享 `CalcOps.cpp` 新接口。

- [ ] **Step 5：运行接口和 DCU 正确性测试**

  Run: `python -m pytest -q src/test/test_cuda_hip_launch_signatures.py src/test/test_hip_source_manifest.py`

  Expected: 全部通过。

  在上一个任务的 DTK/conda 环境中运行：

  Run: `MATPL_GPU_BACKEND=HIP python -m pytest -q src/test/test_nep_electric/test_descriptor_vjp.py src/test/test_nep_electric/test_force_virial_paths.py`

  Expected: HIP 设备测试通过；无设备的 CPU 用例正常执行，硬件相关用例只允许基于明确的设备检测跳过。

- [ ] **Step 6：提交**

  ```bash
  git add src/op/kernel_hip
  git add -f src/test/test_cuda_hip_launch_signatures.py src/test/test_nep_electric/test_force_virial_paths.py src/test/test_nep_electric/test_descriptor_vjp.py
  git commit -m "fix: align HIP CalcOps with current CUDA interfaces"
  ```

## Task 6：整理 NEP_GPU 与超算部署入口

**Files:**

- Modify: `src/feature/NEP_GPU/CMakeLists.txt`
- Create: `deploy/scnet/setup-dcu-env.sh`
- Create: `deploy/scnet/install-dcu.sh`
- Create: `src/test/test_dcu_deploy_scripts.sh`

- [ ] **Step 1：写部署脚本约束测试**

  测试两个脚本 `bash -n` 通过、不含 `/public/home/pwmat`、不含私钥文件名、不执行 `sed -i` 修改 Caffe/源码，并断言安装脚本调用 `src/build.sh --gpu-backend hip`。

- [ ] **Step 2：运行并观察失败**

  Run: `bash src/test/test_dcu_deploy_scripts.sh`

  Expected: 部署脚本尚不存在，测试失败。

- [ ] **Step 3：去掉 CUDA 11.8 硬版本要求**

  `src/feature/NEP_GPU/CMakeLists.txt` 改为从当前 PyTorch/工具链发现 CUDA 兼容接口，不再 `find_package(CUDA 11.8 REQUIRED)`。保留该功能当前源树，不在本次迁移中复制第二套 feature 代码；DTK 的 CUDA 兼容层作为 DCU 构建入口。

- [ ] **Step 4：实现参数化部署脚本**

  `setup-dcu-env.sh` 只负责 module/conda/DTK 环境，所有路径可由环境变量覆盖；`install-dcu.sh` 接收源码目录、并行度和可选安装前缀，调用统一构建脚本。失败立即退出，输出解析后的后端和产物路径。

- [ ] **Step 5：验证部署入口**

  Run: `bash src/test/test_dcu_deploy_scripts.sh`

  Expected: 全部通过。

  Run: `bash -n deploy/scnet/setup-dcu-env.sh deploy/scnet/install-dcu.sh`

  Expected: 退出码 0。

- [ ] **Step 6：提交**

  ```bash
  git add src/feature/NEP_GPU/CMakeLists.txt deploy/scnet
  git add -f src/test/test_dcu_deploy_scripts.sh
  git commit -m "build: add reproducible DCU deployment entrypoint"
  ```

## Task 7：全量分层验证和维护文档

**Files:**

- Create: `docs/building-cuda-hip.md`
- Modify: `README.md`（仅增加指向双后端构建文档的入口）

- [ ] **Step 1：补充用户构建文档**

  中文说明 AUTO/CUDA/HIP/CPU 的选择规则、三个构建目录、DTK 环境、CUDA 环境、库名兼容策略、故障信息和测试矩阵。明确没有对应硬件时哪些验证未执行，不能把静态检查写成硬件通过。

- [ ] **Step 2：运行 Python/静态回归**

  ```bash
  python -m pytest -q \
    src/test/test_op_loader.py \
    src/test/test_hip_source_manifest.py \
    src/test/test_cuda_hip_launch_signatures.py
  bash src/test/test_build_backend_cli.sh
  bash src/test/test_dcu_deploy_scripts.sh
  python -m compileall -q src
  ```

  Expected: 全部通过。

- [ ] **Step 3：构建和测试 CPU**

  ```bash
  cmake -S src/op -B src/op/build/cpu -DMATPL_GPU_BACKEND=CPU
  cmake --build src/op/build/cpu -j4
  MATPL_GPU_BACKEND=CPU python -m pytest -q src/test/test_nep_electric/test_force_virial_paths.py
  ```

  Expected: CPU 库生成，CPU 用例通过。

- [ ] **Step 4：构建和测试 HIP/DCU**

  在 Task 4 的 DTK/conda 环境中重新从干净的 `src/op/build/hip` 配置并编译，然后运行 descriptor VJP 与 force/virial 测试。

  Expected: HIP 库生成且 DCU 测试通过。

- [ ] **Step 5：验证 CUDA 隔离**

  在 NVIDIA CUDA + CUDA PyTorch 环境执行：

  ```bash
  cmake -S src/op -B src/op/build/cuda -DMATPL_GPU_BACKEND=CUDA
  cmake --build src/op/build/cuda -j4
  MATPL_GPU_BACKEND=CUDA python -m pytest -q src/test/test_nep_electric/test_descriptor_vjp.py src/test/test_nep_electric/test_force_virial_paths.py
  ```

  Expected: CUDA 库生成且测试通过。若当前 DCU 集群没有 NVIDIA 工具链/硬件，只记录“未在本机执行”，同时保留 CMake 路由、源清单、接口一致性和 `compile_commands.json` 隔离证据。

- [ ] **Step 6：检查差异和提交最终文档**

  Run: `git diff --check && git status --short && git log --oneline --decorate -8`

  Expected: 无空白错误；只有预期文档待提交。

  ```bash
  git add README.md
  git add -f docs/building-cuda-hip.md
  git commit -m "docs: explain CUDA and DCU backend builds"
  ```

- [ ] **Step 7：最终验收**

  Run: `git status --short`

  Expected: 工作树干净。汇报每一层实际执行的命令、通过数量、生成库路径、提交列表，以及任何因硬件缺失而未执行的项目。

## 自审清单

- [ ] 设计文档中的单套 Python、公共 C++、双 kernel 树、独立构建目录均有对应任务。
- [ ] 保留了 `libCalcOps_bind.so` 与 `CalcOps_cuda` 外部接口，避免模型层分叉。
- [ ] 包含 latest main 的 virial batch 梯度修复和 descriptor VJP 回归。
- [ ] CUDA 与 HIP 的编译输入由静态清单和 `compile_commands.json` 双重验证。
- [ ] 部署脚本无账户/私钥硬编码，不在构建时修改源码。
- [ ] 每个任务都有先失败测试、最小实现、验证命令和独立提交。
- [ ] 不包含占位符、伪造通过结果或要求当前 DCU 主机具备 NVIDIA 硬件的假设。
