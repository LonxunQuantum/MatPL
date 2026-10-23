# MatPL CUDA 与 DCU/HIP 双后端构建指南

## 1. 代码组织

MatPL 的 Python 代码和公共 C++ 接口只维护一套，GPU 算子按后端隔离：

```text
src/op/kernel/          NVIDIA CUDA 算子（.cu）
src/op/kernel_hip/      DCU/HIP 算子（.hip）
src/op/cmake/cuda/      CUDA 构建目标
src/op/cmake/hip/       HIP 构建目标
src/op/cmake/cpu/       CPU 构建目标
src/op/build/cuda/      CUDA 构建产物
src/op/build/hip/       HIP 构建产物
src/op/build/cpu/       CPU 构建产物
```

`src/utils/op_loader.py` 根据当前 PyTorch 的编译后端选择动态库：

- `torch.version.hip` 非空时加载 `src/op/build/hip/lib/libCalcOps_bind.so`；
- 否则 `torch.version.cuda` 非空时加载 `src/op/build/cuda/lib/libCalcOps_bind.so`；
- 两者均为空时加载 `src/op/build/cpu/lib/libCalcOps_bind_cpu.so`。

CUDA 与 HIP 构建目录互不复用，切换后端不会覆盖另一后端的产物。

## 2. 通用编译入口

在仓库根目录执行统一构建命令：

```bash
src/build.sh -j4
```

脚本根据当前 PyTorch 自动识别后端，并始终构建 CPU 算子作为运行时托底：

- CUDA PyTorch：构建 CUDA 与 CPU 算子；
- HIP PyTorch：构建 HIP 与 CPU 算子；
- CPU PyTorch：只构建 CPU 算子。

CPU 算子是必需的 fallback，CPU 构建失败会使整个构建失败。构建脚本不再提供手动选择 CUDA、HIP 或 CPU 的参数。

可以先查看将要执行的命令而不产生构建文件：

```bash
src/build.sh --dry-run -j4
```

需要 NN/Linear 的 Fortran 程序时，继续使用原有参数：

```bash
src/build.sh -j8 -m nn
```

## 3. NVIDIA CUDA 环境

确认当前 PyTorch 是 CUDA 版本，并且 `nvcc` 在 `PATH` 中：

```bash
python -c "import torch; print(torch.version.cuda)"
command -v nvcc
src/build.sh -j4
```

主要产物为：

```text
src/op/build/cuda/lib/libCalcOps_bind.so
src/feature/NEP_GPU/build/cuda/nep_gpu.so
```

如果 CUDA toolkit 不在标准位置，可在编译前设置 `CUDAToolkit_ROOT` 或
`CUDA_HOME`。

可用 `MATPL_CUDA_ARCHITECTURES` 指定目标显卡架构，统一作用于 `src/build.sh`
构建的 NEP-GPU 接口、CUDA 算子以及算子使用的 PyTorch CMake 配置：

```bash
export MATPL_CUDA_ARCHITECTURES=70       # V100，只生成架构 70
sh src/build.sh -j4
# 多种显卡共用：export MATPL_CUDA_ARCHITECTURES="70;80;86;89;90"
```

使用数字和分号列表（例如 `70;86`），无需另设 `TORCH_CUDA_ARCH_LIST`。
MatPL 会将 `70` 转为 PyTorch 的 `7.0` 格式，并覆盖冲突的 PyTorch 架构设置。
未设置时，使用 `CMAKE_CUDA_ARCHITECTURES` 环境变量，或默认列表
`60;70;75;80;86;89;90`。目标架构必须在当前 CUDA toolkit 与 PyTorch 支持范围内。

编译只要求对应版本的 PyTorch 和可用的 `nvcc`，不要求节点存在可见 GPU；
上述目标列表也不依赖 GPU 自动探测。HIP/CPU 后端不使用该 NVIDIA 架构参数。
直接通过 CMake 构建 `src/op` 时，也可以使用 `MATPL_CUDA_ARCHITECTURES`
环境变量；未设置它时，接受 `-DCMAKE_CUDA_ARCHITECTURES=70`。

## 4. 神威超算 DCU/DTK 环境

仓库提供了参数化环境脚本；必须使用 `source`，以便环境保留在当前 shell：

```bash
source dcu-deploy/scnet/setup-dcu-env.sh
src/build.sh -j4
```

也可以使用一体化入口：

```bash
MATPL_BUILD_JOBS=4 deploy/scnet/install-dcu.sh
```

脚本的默认值对应当前神威超算软件栈，但都可以由环境变量覆盖：

| 变量 | 用途 |
| --- | --- |
| `MATPL_GCC_MODULE` | GCC module 名称 |
| `MATPL_CONDA_SH` | Conda 初始化脚本 |
| `MATPL_CONDA_ENV` | Conda 环境名称 |
| `MATPL_DTK_ROOT` | DTK 安装根目录 |
| `MATPL_DTK_ENV` | DTK 环境脚本 |
| `MATPL_DTK_NVCC` | DTK 的 `nvcc` 兼容包装器 |
| `MATPL_DTK_CUDA_ROOT` | DTK CUDA 兼容层根目录 |
| `MATPL_BUILD_JOBS` | 并行编译任务数 |

主要产物为：

```text
src/op/build/hip/lib/libCalcOps_bind.so
src/feature/NEP_GPU/build/hip/nep_gpu.so
```

`NEP_GPU` feature 与 `src/op` 的策略不同：它继续只维护一套 `.cu` 源码，
DCU 环境通过 DTK 的 CUDA 兼容包装器编译；其 CUDA/HIP 构建缓存仍分别写入
`build/cuda` 和 `build/hip`，避免缓存串用。

DTK 初始化时可能提示 `rocm_smi` 路径不存在并回退到默认
`rocm_smi_lib`。只要后续编译、模块加载和计算节点测试通过，此提示不代表
MatPL 构建失败。

## 5. 算子修改规则

修改或新增 `src/op` 算子时：

1. CUDA 实现在 `src/op/kernel/*.cu` 中维护；
2. 对应 HIP 实现在 `src/op/kernel_hip/*.hip` 中维护；
3. 公共函数声明、参数顺序、数据类型和 batch 语义必须一致；
4. CUDA CMake 不得引用 `kernel_hip`，HIP CMake 不得引用 `kernel/*.cu`；
5. 分别在对应硬件上运行数值回归，不能只以“编译通过”代替运行验证。

接口和目录隔离的静态回归命令：

```bash
python -m unittest \
  src.test.test_hip_source_manifest \
  src.test.test_cuda_hip_launch_signatures
```

构建入口与部署脚本回归命令：

```bash
bash src/test/test_build_backend_cli.sh
bash src/test/test_dcu_deploy_scripts.sh
cmake -P src/op/cmake/tests/test_resolve_backend.cmake
cmake -P src/op/cmake/tests/test_resolve_cuda_architectures.cmake
```

在加载 CUDA 工具链的计算节点上验证实际编译参数（测试主动屏蔽 GPU，模拟
登录节点的设备可见性；临时构建目录不会覆盖现有产物）：

```bash
bash tests/test_cuda_architectures.sh
# 同时完整编译两个 CUDA 库，并检查产物中的实际目标架构：
MATPL_TEST_BUILD=1 MATPL_TEST_JOBS=4 bash tests/test_cuda_architectures.sh
```

## 6. 验证边界

DCU/HIP 的最终验收应在申请了 DCU 资源的计算节点上完成。NVIDIA CUDA 的
最终验收同样需要 NVIDIA GPU 节点；在只有 DCU 的集群上只能检查 CUDA 源码
完整性、接口一致性和 CMake 路由，不能声称已经完成 CUDA 运行时验证。
