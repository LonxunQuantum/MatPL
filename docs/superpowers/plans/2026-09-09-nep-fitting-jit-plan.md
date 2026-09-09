# NEP Fitting NVRTC JIT 实施计划

> **执行要求：** 使用 `superpowers:executing-plans`，按任务逐项执行；每项都遵循测试先行、先红后绿，并在独立验证通过后提交。

**目标：** 为 FP64 单隐藏层 NEP fused fitting 实现按实际 `(D,H,Q,SM)` 编译和缓存的 NVRTC 专用核函数，同时保留现有 AOT 回退路径。

**架构：** 预编译的 `CalcOps_cuda` 内新增 JIT 管理器，使用 NVRTC 生成当前 GPU 的 CUBIN，并通过 CUDA Driver API 在 PyTorch 当前 stream 上启动。模型移至 GPU 后、DDP 包装前主动预热；直接算子调用可在首次 forward 延迟准备。

**技术栈：** C++17、CUDA 11.8/12.4、NVRTC、CUDA Driver API、PyTorch C++ Extension、Python 3.11、pytest、Slurm。

**规格：** `docs/superpowers/specs/2026-09-09-nep-fitting-jit-design.md`

## 全局约束

- 只支持 FP64、单隐藏层、`1 <= D <= 96`、`1 <= H <= 100`、`Q in {1,2}`。
- 不改变张量布局、autograd 返回值、checkpoint 格式、原子分组和训练损失。
- `auto` 模式失败时警告一次并回退 AOT；严格模式报错；关闭模式不接触 NVRTC。
- JIT 编译只允许发生在显式预热或第一次 fitting 调用中，不得发生在稳定训练 step 内。
- 所有 Driver API kernel 都必须使用 PyTorch 当前 CUDA stream。
- 不完全展开 `D x H`，继续使用 16 神经元 tile 和有界线程局部累加器。
- CPU 和 HIP 构建行为保持不变。

---

### 任务 1：JIT 模式、缓存键与构建依赖

**文件：**

- 新建：`src/op/include/nep_fitting_jit.h`
- 新建：`src/op/src/nep_fitting_jit.cpp`
- 修改：`src/op/cmake/cuda/CMakeLists.txt`
- 修改：`src/op/src/CalcOps_bind.cpp`
- 修改：`src/op/include/calculate_nepfitting.h`
- 修改：`src/op/src/calculate_nepfitting.cpp`
- 新建：`tests/test_nep_fitting_jit.py`

**接口：**

- `bool prepare_nep_fitting_jit(const at::Tensor& reference, int64_t d, int64_t h, int64_t q)`
- `torch.ops.CalcOps_cuda.nep_fitting_jit_prepare(reference, d, h, q) -> bool`
- `JitMode` 解析 `MATPL_NEP_FITTING_JIT=auto|1|0`
- 缓存键由源码版本、`D/H/Q`、SM、NVRTC 版本和编译选项生成。

- [ ] **步骤 1：编写失败测试**

在 `tests/test_nep_fitting_jit.py` 中用子进程加载真实 CalcOps，验证关闭模式返回
`False` 且空缓存目录没有文件；严格模式调用新增 raw op。此时测试应因 op 不存在而失败。

```python
def test_disabled_mode_does_not_touch_cache(tmp_path):
    result = run_probe(tmp_path, mode="0", d=35, h=60, q=1)
    assert result["prepared"] is False
    assert result["files"] == []
```

- [ ] **步骤 2：确认 RED**

运行：

```bash
python -m pytest tests/test_nep_fitting_jit.py::test_disabled_mode_does_not_touch_cache -q
```

预期：失败，错误明确指出 `nep_fitting_jit_prepare` 未注册。

- [ ] **步骤 3：实现最小模式和缓存骨架**

在 JIT 管理器中实现环境变量解析、尺寸检查、GPU compute capability 查询、稳定
FNV-1a 源码版本哈希、缓存路径生成。关闭模式直接返回 `false`；其他模式暂时报告
“编译器尚未实现”。在 CUDA CMake 中使用 `find_package(CUDAToolkit REQUIRED)`，链接
`CUDA::nvrtc` 和 `CUDA::cuda_driver`。注册 prepare raw op。

- [ ] **步骤 4：确认 GREEN 并检查 CPU/HIP**

运行关闭模式测试、CUDA 构建，以及已有 op-loader CPU 测试。确认关闭模式不会创建目录。

- [ ] **步骤 5：提交**

```bash
git add src/op tests/test_nep_fitting_jit.py
git commit -m "feat: add NEP fitting JIT runtime contract"
```

---

### 任务 2：NVRTC 编译、原子前向和 feature 一阶导

**文件：**

- 新建：`src/op/kernel/nep_fitting_jit_source.h`
- 修改：`src/op/include/nep_fitting_jit.h`
- 修改：`src/op/src/nep_fitting_jit.cpp`
- 修改：`src/op/kernel/nep_fitting.cu`
- 修改：`tests/test_nep_fitting_jit.py`

**接口：**

- `bool try_launch_nep_fitting_jit_forward(...)`
- JIT module 导出 `extern "C" __global__ void fitting_atoms_forward(...)`。
- 返回 `false` 只表示启动前无法获得专用 module；`cuLaunchKernel` 失败必须抛错。

- [ ] **步骤 1：编写前向一致性失败测试**

为 `(D,H,Q)=(35,60,1)`、`(96,100,1)`、`(31,33,2)` 构造 ragged type groups，
在独立子进程中分别运行 `MATPL_NEP_FITTING_JIT=0` 和 `=1`，保存并比较 `Y` 与
`G=dY/dfeature`。严格模式当前应因无法编译而失败。

- [ ] **步骤 2：确认 RED**

```bash
python -m pytest tests/test_nep_fitting_jit.py -k forward_matches_aot -q
```

预期：严格 JIT probe 报出尚未实现的编译错误。

- [ ] **步骤 3：实现 NVRTC 编译和 module 加载**

内置纯 CUDA 源码，使用以下编译选项生成当前 SM 的 CUBIN：

```text
--std=c++14 --gpu-architecture=sm_<major><minor> --device-as-default-execution-space
```

通过 `nvrtcGetCUBINSize/nvrtcGetCUBIN` 取得二进制，调用 `cuModuleLoadData` 和
`cuModuleGetFunction`。编译日志必须附在严格模式异常中。

- [ ] **步骤 4：实现专用前向 kernel**

用编译期 `D/H/Q` 替换动态循环边界，保留 `kHiddenTile=16`、`kAtoms=8`。完整 tile
不执行 `j < H` 判断，最后一个 tail tile单独处理。使用 `cuLaunchKernel` 在
`at::cuda::getCurrentCUDAStream()` 上启动，并立即检查 Driver API 和 CUDA 错误。

- [ ] **步骤 5：确认 GREEN**

运行前向一致性测试和现有 `tests/test_nep_fused_fitting.py`。误差标准沿用现有 FP64
测试，不放宽容差。

- [ ] **步骤 6：提交**

```bash
git add src/op tests/test_nep_fitting_jit.py
git commit -m "feat: JIT compile fused NEP fitting forward"
```

---

### 任务 3：feature 反向梯度和 fitting 参数梯度

**文件：**

- 修改：`src/op/kernel/nep_fitting_jit_source.h`
- 修改：`src/op/include/nep_fitting_jit.h`
- 修改：`src/op/src/nep_fitting_jit.cpp`
- 修改：`src/op/kernel/nep_fitting.cu`
- 修改：`tests/test_nep_fitting_jit.py`

**接口：**

- `bool try_launch_nep_fitting_jit_backward(...)`
- JIT module 新增 `fitting_atoms_backward`、`fitting_parameter_partials`、
  `fitting_parameter_reduce` 三个固定符号。

- [ ] **步骤 1：编写反向失败测试**

对相同的三个尺寸组合使用任意 `gradY/gradG`，比较 JIT 与 AOT 的 `gradX`、`gradW`、
`gradb`、`gradV`、`gradc`；增加仅 `Y`、仅 `G` 和 `Q=2` 两个 head 使用情况。

- [ ] **步骤 2：确认 RED**

```bash
python -m pytest tests/test_nep_fitting_jit.py -k backward_matches_aot -q
```

预期：JIT 前向可运行，但 backward 尚未命中专用实现。

- [ ] **步骤 3：实现三个反向 kernel**

将现有 workspace 上限保持为 16 MiB，复用相同的 `groups/slots/tiles/stride` 计算。
局部数组根据编译期尺寸缩小；tile 内循环有限展开。JIT 管理器负责临时 workspace
分配和三次 Driver API launch，任何启动错误直接抛出。

- [ ] **步骤 4：确认 GREEN 和优化器一致性**

运行 JIT 反向测试，并运行现有训练测试，确认一次 Adam 更新后全部 fitting 参数与
AOT 路径一致。

- [ ] **步骤 5：提交**

```bash
git add src/op tests/test_nep_fitting_jit.py
git commit -m "feat: JIT compile fused NEP fitting backward"
```

---

### 任务 4：持久缓存、文件锁和损坏恢复

**文件：**

- 修改：`src/op/src/nep_fitting_jit.cpp`
- 修改：`tests/test_nep_fitting_jit.py`

**接口：**

- 每个缓存键对应一个 `.cubin` 和一个 `.lock`。
- 文件内容先写入同目录唯一临时文件，`fsync` 后原子 `rename`。
- 进程内 module map 以 `(device,D,H,Q,key)` 为键并由 mutex 保护。

- [ ] **步骤 1：编写缓存失败测试**

增加三个子进程测试：第二个进程复用文件且 mtime 不变；四个并发进程最终只有一个
CUBIN；截断 CUBIN 后下一次 prepare 能够重新生成并成功运行前向。

- [ ] **步骤 2：确认 RED**

```bash
python -m pytest tests/test_nep_fitting_jit.py -k "reuse or concurrent or corrupt" -q
```

预期：当前实现没有持久缓存或无法恢复损坏文件。

- [ ] **步骤 3：实现缓存协议**

使用 `flock(LOCK_EX)` 串行化同 key 编译。持锁后再次检查缓存；加载失败时删除 CUBIN
并只重编译一次。自动模式下缓存权限错误回退 AOT；严格模式抛出包含缓存路径的异常。

- [ ] **步骤 4：确认 GREEN**

运行完整 `tests/test_nep_fitting_jit.py`，并确认测试结束后没有遗留临时文件。

- [ ] **步骤 5：提交**

```bash
git add src/op/src/nep_fitting_jit.cpp tests/test_nep_fitting_jit.py
git commit -m "feat: cache NEP fitting JIT modules safely"
```

---

### 任务 5：模型加载阶段预热

**文件：**

- 修改：`src/model/nep_fused_fitting.py`
- 修改：`src/PWMLFF/nep_network.py`
- 修改：`tests/test_nep_fused_training.py`
- 修改：`tests/test_nep_fitting_jit.py`

**接口：**

- `prepare_fitting_jit(model: torch.nn.Module) -> bool`
- CUDA FP64 单隐藏层模型调用 raw prepare；CPU、HIP、非支持网络直接返回 `False`。

- [ ] **步骤 1：编写预热时序失败测试**

通过真实小型 NEP 模型验证 helper 能得到正确 `D/H/Q`；在 `load_model_optimizer()`
中记录调用顺序，断言 prepare 发生于 `.to(device)` 之后、DDP 构造之前。

- [ ] **步骤 2：确认 RED**

```bash
python -m pytest tests/test_nep_fitting_jit.py -k prepare_model -q
```

预期：`prepare_fitting_jit` 尚不存在。

- [ ] **步骤 3：实现 Python helper 和训练接入**

从第一个 fitting network 的隐藏层权重读取 `D/H`，从 charge mode 得到 `Q`。仅在
CUDA、FP64、固定单隐藏层约束满足时调用 raw op。在模型移至 GPU 后、DDP 包装前调用。

- [ ] **步骤 4：确认 GREEN 与 fallback**

运行 CPU fallback、单 GPU、charge、DDP 测试；关闭模式下训练路径必须继续命中 AOT。

- [ ] **步骤 5：提交**

```bash
git add src/model/nep_fused_fitting.py src/PWMLFF/nep_network.py tests
git commit -m "feat: prepare fitting JIT before NEP training"
```

---

### 任务 6：跨架构构建、资源检查和 OMat24 基准

**文件：**

- 修改：`tests/benchmark_nep_fused_fitting.py`
- 新建：`tests/run_nep_fitting_jit_checks.slurm`
- 新建：`tests/check_nep_fitting_jit_cubin.py`
- 修改：`tests/benchmarks/nep-fused-fitting-stage1.md`

**接口：**

- 基准 JSON 新增 `jit_prepare_ms`、`jit_cache_hit`、稳态 step 指标和显存峰值。
- CUBIN 检查脚本读取 `cuobjdump --dump-resource-usage`，报告寄存器、stack 和 spill。

- [ ] **步骤 1：编写基准字段失败测试**

新增测试，断言 JIT 基准输出包含准备耗时、缓存命中状态、稳态样本以及 peak allocated/
reserved memory。当前基准因字段缺失而失败。

- [ ] **步骤 2：确认 RED**

```bash
python -m pytest tests/test_nep_fitting_jit.py -k benchmark_schema -q
```

- [ ] **步骤 3：实现基准和资源检查**

预热计时独立于 warmup/steps；首次编译和缓存命中各运行一次。资源脚本拒绝任何 spill，
并为每个 kernel 输出寄存器和共享内存。Slurm 脚本加载 CUDA 11.8；4090 环境允许改为
CUDA 12.4，但不改变测试内容。

- [ ] **步骤 4：运行完整验证**

在 3090/q4 上运行：CUDA 构建、JIT/AOT 全部 pytest、双卡 DDP、compute-sanitizer 的
memcheck/racecheck/synccheck、冷缓存和热缓存 OMat24 基准。在可用节点分别生成
SM60、SM70、SM86、SM89 CUBIN；至少实际运行 SM86，其余必须完成 NVRTC 编译和
资源解析。

- [ ] **步骤 5：比较性能和显存**

使用相同 mini_data_test、batch size、warmup 和 step 数比较 AOT/JIT。报告准备耗时、
稳态中位数/P95、吞吐量、peak allocated/reserved 和最大数值误差；JIT 若无稳态收益，
保留功能但将默认模式调整为 `0` 并记录证据。

- [ ] **步骤 6：最终回归并提交**

```bash
python -m pytest tests/test_nep_fitting_jit.py tests/test_nep_fused_fitting.py \
  tests/test_nep_fused_runtime.py tests/test_nep_fused_training.py \
  tests/test_nep_fused_ddp.py src/test/test_nep_electric/test_mb_secondgrad_kernel.py -q
git add tests
git commit -m "test: validate NEP fitting JIT across CUDA architectures"
```
