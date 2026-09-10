# NEP 第一阶段实现与实测结果

融合 CUDA fitting 已接入训练：联合计算输出和 feature 一阶导，反向包含力训练所需的混合二阶导。支持 float64、单隐藏层、D≤96/H≤100 及 energy/charge。原子顺序保持不变，保留原 Parameter、checkpoint、优化器和旧路径兼容性。

代码已合入 `nep-dcu/dev`。

**主要结果：fitting 约加速 15 倍；256 结构整步吞吐提升约 7.2%；最大 batch 未增加。**

| RTX 3090，float64 | 原实现 | 融合实现 |
|---|---:|---:|
| 完整 step 中位耗时 | 969.54 ms | 904.70 ms |
| 完整 step p95 | 977.33 ms | 909.06 ms |
| PyTorch allocated 峰值 | 601.94 MiB | 600.20 MiB |
| 设备进程显存采样峰值 | 3687 MiB | 3675 MiB |
| 最大通过 batch | 1793 结构 / 15352 原子 | 1793 结构 / 15352 原子 |

真实案例为 `mini_data_test` 的原 checkpoint 和 D=35/H=40、89 元素配置。固定 256 结构 / 2045 原子，预热 20 步、测量 100 步；整步包含分组、上传、邻居、模型、loss/backward、梯度裁剪、empty_cache 和 Adam，不含 LMDB 读盘。真实 batch 的输出最大绝对差为 `2.84e-14`，全部参数梯度和一次 Adam 更新通过对照。

设备显存用独立进程、每 20 ms 采样，包含 CUDA 上下文和原生分配，可能漏掉短暂峰值。PyTorch 统计不包含现有算子的部分原生分配。容量测试使用 seed=2023 的固定结构序列，每次在新进程中预热 2 步、测量 3 步；两版均在 1794 结构 OOM。该容量边界依赖结构大小、元素和邻居数量。

| Fitting 单项：前向 + 反向中位耗时 | 原实现 | 融合实现 |
|---|---:|---:|
| D=35 / H=40 | 80.80 ms | 5.33 ms |
| D=35 / H=60 | 81.05 ms | 5.35 ms |
| D=96 / H=100 | 83.60 ms | 5.57 ms |

单项使用真实 batch 的元素数量分布及固定随机 feature，包含参数打包。融合版不保存全 batch 隐层，参数归约空间最多 16 MiB；但打包和归约仍占显存，小 batch 的 fitting 单项峰值可能高于原实现。因此第一阶段没有获得可观的整步显存节省。

验证覆盖全部 27 项测试，单卡和双卡分别运行，包括完整力训练、charge/BEC、未使用输出头的 None 梯度及 Adam 动量、DDP、非默认 stream、分片尾部和极不均衡的 89 元素输入。CUDA memcheck、racecheck、synccheck 均通过，零错误、零竞争告警。RTX 3090 已实测；P100、V100、3080 Ti 仅有对应架构的编译覆盖，尚无这些型号的实机结果。

第二阶段应优先检查 `src/op/kernel/calculateNepMbFeat_secondgradout.cu` 中的 `dfeat_c3` 临时数组，其大小随原子数、最大邻居数、出现元素数及描述符阶数相乘增长。此次 OOM 日志指向现有 `GPU_Vector::resize` 原生分配；该大数组是下一步定位显存瓶颈的重点。

复现入口：在上述工作目录加载 `matpl-2026.3`、CUDA 11.8、CMake 3.31.6 和 devtoolset-8，构建当前 CUDA 扩展后执行：

```bash
mkdir -p .validation
MAX_BATCH=8192 sbatch tests/run_nep_fused_checks.slurm
```

原始结果已保存到 `tests/benchmarks/nep-fused-fitting-stage1.json`；设备显存为同目录的 `nep-fused-fitting-memory-{original,fused}.json`。详细日志保留在工作目录 `.validation/`。对应作业：基准 4668714，双卡 4668531，设备显存 4668723，CUDA 检查 4668721。

## NVRTC JIT 专用化结果

在固定 FP64、单隐藏层和 `D≤96/H≤100/Q∈{1,2}` 的边界内增加了 NVRTC
运行时专用化。JIT 同时生成 forward、`dE/dfeature`、feature 反向梯度和 fitting
参数梯度 kernel，CUBIN 使用进程内缓存及带 `flock` 的持久缓存；损坏文件会在锁内
重建。模型加载时可在移入 CUDA 后、DDP 包装前预热。

RTX 3090 上使用相同的 OMat24 `mini_data_test`、256 结构/2045 原子、10 次预热和
30 次测量进行了 AOT、JIT 冷缓存和 JIT 热缓存对照：

| FP64 指标 | AOT 融合核 | JIT 冷缓存 | JIT 热缓存 |
|---|---:|---:|---:|
| JIT prepare | 关闭 | 772.43 ms | 1.56 ms |
| D=35/H=40 fitting 前向+反向 | 6.18 ms | 7.24 ms | 7.36 ms |
| D=35/H=60 fitting 前向+反向 | 6.26 ms | 7.39 ms | 7.38 ms |
| 完整 step p50 | 116.39 ms | 118.09 ms | 118.38 ms |
| 完整 step p95 | 118.47 ms | 120.12 ms | 120.36 ms |
| 完整 step peak allocated | 599.34 MiB | 599.34 MiB | 599.34 MiB |
| 完整 step peak reserved | 626.00 MiB | 626.00 MiB | 626.00 MiB |

JIT 在该实际网络上没有稳态收益：常见 D=35/H=40 fitting 慢约 17%–19%，完整 step
慢约 1.5%–1.7%，显存峰值相同。表中的 AOT 数据作为历史性能基线保留。为统一后续
优化路径并消除重复 CUDA 实现，当前 CUDA fused fitting 已改为始终使用 JIT；不再
提供运行模式开关或 AOT 回退。后续性能优化只修改 JIT kernel。

CUDA 11.8 NVRTC 已为最大 `D=96/H=100/Q=2` 编译 SM60、SM70、SM86、SM89
CUBIN。最大双 head forward 的专用共享内存累加方案将寄存器从 255 降至
SM60/70 的 86/91 和 SM86/89 的 72，stack 为 0；四个 kernel 在四种架构上
均为 0 字节 LOCAL/spill。3090 上的 memcheck、racecheck、synccheck 均为零错误。

原始对照保存在 `.validation/nep-fitting-{aot,jit-cold,jit-hot}-20260909-201901.json`，
验证作业为 4671744，sanitizer 作业为 4671746。
