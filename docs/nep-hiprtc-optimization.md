# NEP DCU/HIP 优化与动态特化

本版本同步自 2026-10-22 完成实装的 `matpl-dcu` 优化源码，基于 OneScience `develop` 中的 MatPL（来源提交 `cb6ce026587f70983add4eb247caba60f11ff493`）。包含拟合网络 HIPRTC、角向一阶反向输出裁剪与重新融合，以及多体二阶反向 HIPRTC。相关公共接口、数据加载、训练日志和构建依赖一并同步。

## 构建与运行

使用 HIP 版 PyTorch、兼容的 DTK/HIP 工具链以及 HIPRTC 头文件和动态库。从新的构建目录配置，不复用其他源码树的 CMakeCache 或对象文件。构建入口会根据当前 PyTorch 自动选择后端，并构建 CPU 托底算子：

```bash
# 先激活自己的 HIP PyTorch 环境，并加载 DTK。
export AMDGPU_TARGETS=gfx936  # 按实际 DCU 型号调整
bash src/build.sh --dry-run -j4
bash src/build.sh -j4
source env.sh
cd /path/to/training-case
matpl train nep.json
```

`env.sh` 由本机构建生成。完整 CUDA/HIP/CPU 构建说明见 [双后端构建指南](building-cuda-hip.md)。仓库提供的是源码，不包含本机预编译库、训练集或 HIPRTC 缓存。

## 拟合网络

HIP 默认使用 HIPRTC 为 FP64、单隐藏层 tanh 网络的前向、输入梯度、参数 partials 和 reduce 四阶段动态特化。设备源码与 CUDA NVRTC 路径共用，支持：

- 输入维度 `1 <= D <= 103`。
- 隐藏层宽度 `1 <= H <= 200`。
- 输出维度 `Q=1/2`。

参数反向采用分块 partials + reduce，避免少数 block 遍历所有原子。元素类型数和原子数保留为运行时参数，工作区有容量上限。

默认缓存位于 `~/.cache/matpl/nep_fitting`。`MATPL_NEP_JIT_CACHE` 可覆盖目录；`MATPL_NEP_HIPRTC=0` 可切回既有 AOT/通用实现。已有对照开关 `MATPL_FITTING_GENERIC=1`、`MATPL_FITTING_LEGACY=1` 也会关闭拟合网络 HIPRTC。

## 角向一阶反向

HIP 根据下游输出需求裁剪中间计算和全局写入，并使用 HIPRTC 对融合路径动态特化。默认不再只匹配 16_metal。几何、basis 和球谐项在融合实现中复用，临时变量的存活范围缩短；不需要的稠密二阶上下文在二阶融合路径中重算。

`MATPL_NEP_ANGULAR_PRUNE=0` 可关闭该输出裁剪优化。具体输出组合由算子上下文与梯度需求决定，不按算例目录名判断。

## 多体二阶反向

默认执行 `nep_mb_secondgrad_precompute_angular` + `nep_mb_secondgrad_fused` 两阶段，包含 HfO2。CUDA 使用 NVRTC，HIP 使用 HIPRTC，数学参数范围对齐：

| 参数 | CUDA 与 HIP 支持范围 |
|---|---|
| 底层 `NMAX` | 5–20，包含端点 |
| 底层 `NBASIS` | 5–20，包含端点 |
| JSON 对应的角向 `n_max`、`basis_size` | 4–19，底层计数等于配置值加一 |
| 元素类型数 | 1–118 |
| `l_max` | `[4,0,0]`、`[4,2,0]`、`[4,2,1]` |
| 共享内存 | 所需容量不超过设备的 `sharedMemPerBlock` |

特化维度为 `NMAX`、`NBASIS`、`HAS4`、`HAS5`，模块及缓存还区分设备/架构。类型数是运行时参数，不产生额外二阶特化。七个小算例均在范围内，共对应四组二阶特化参数；HfO2 不再有默认回退例外。HIP 保留自身 precompute + fused 算法及 CTA128，不照搬 CUDA 的硬件线程块设置。

```bash
# 默认：支持范围内走融合，范围外走原路径。
export MATPL_NEP_MB_SECONDGRAD_MODE=auto
# 可选：范围外报错。
# export MATPL_NEP_MB_SECONDGRAD_MODE=optimized
# 可选：使用原二阶路径。
# export MATPL_NEP_MB_SECONDGRAD_MODE=legacy
```

MODE 未设置时，旧 `MATPL_NEP_MB_SECONDGRAD_FUSED=0` 兼容强制原路径；显式 MODE 优先。默认二阶缓存目录为 `~/.cache/matpl/nep_mb_secondgrad`，可用 `MATPL_NEP_MB_SECONDGRAD_JIT_CACHE` 覆盖。

## 动态编译与性能口径

设置 `MATPL_NEP_JIT_LOG=1` 可查看首次编译和磁盘缓存命中。缓存包含设备源码、特化参数、架构、HIPRTC/runtime/driver 版本及编译选项，并通过文件锁与临时文件 rename 避免并发发布冲突。默认缓存位置遵守 `XDG_CACHE_HOME`。

首次运行新参数组合有编译开销。训练性能继续按三个 epoch 中第二、第三个 epoch 的耗时平均值统计；首次编译与首轮耗时单列。二阶阶段的性能应合计 precompute 和 fused 两个 kernel。

## 已有验证与适用边界

在 gfx936、onescience311、PyTorch 2.5.1/HIP 6.3.25405、DTK25.04.4 环境中：

- 拟合网络此前完成 72 组代表性尺寸/原子分布检查，使用 `atol=1e-10, rtol=0`。
- 最新二阶 HIPRTC 已完成七个算例参数的小张量实际二阶调用及有限值检查；参数下界、非原 AOT 组合和 `(NMAX=20, NBASIS=20, TYPES=118)` 上界也完成调用。
- 二阶 `auto` 回退、`optimized` 拒绝范围外参数、跨进程缓存和实际加载路径已检查。
- 二阶 HIPRTC 版本未重新跑完整训练精度回归和三个 epoch 性能。此前 AOT 的训练性能不能直接作为这一版本的实测。

已知数值差异继续保留：HfO2 的完整训练指标此前出现明显分叉，尚未解决；16_metal 此前最大 loss 差约 `1e-9`。HfO2 仍按本次要求默认启用融合，不能将有限值和启动检查解读为完整训练精度已通过。

发布同步只做源码一致性、Python/脚本语法、构建命令预览和 Git 差异检查，不重新运行训练或回归套件。
