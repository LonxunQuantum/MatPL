# NEP 多体二阶导 CUDA 核函数优化设计

## 目标与边界

优化 `launch_calculate_nepmbfeat_secondgradout_c3` 的 FP64 训练路径。OMat24 中的主要目标是缩短 `find_angular_gardc_neigh`，并消除其后 `aggregate_dfeat_c3` 所需的大型中间张量。

本阶段：

- 所有输入、运算、共享内存暂存和累加均为 FP64；
- 保持 PyTorch 接口、张量布局、参数梯度语义和原子物理顺序；
- 支持 SM60、SM70、SM86，覆盖 P100、V100、RTX 3080 Ti/3090；
- 不引入 Tensor Core、混合精度、原子排序或其他算子优化。

## 实现结构

保留旧实现作为通用回退，并新增模板特化的融合核。首个特化覆盖 OMat24：`n_max_3b=5`、`n_base_3b=9`、`lmax_3=4`、包含四体和五体项。运行时计划器只在尺寸、共享内存和 GPU 架构均满足时选择新核；其他配置继续调用旧核。环境开关支持 `auto`、`optimized` 和 `legacy`，用于同一二进制的正确性与性能对照。

新核使用一个 CTA 处理一个中心原子，32 或 64 个线程以步进方式处理有效角向邻居。CTA 先在共享内存中构造邻域元素类型位图，再由一个线程按元素编号生成稳定的局部类型列表。该列表完整覆盖 `dsnlm_dc` 的非零类型范围，因此无需遍历整个 batch 的元素类型。

局部类型每 4 种一组处理。每组将所需的 `dsnlm_dc`、当前中心类型对应的系数以及输出累加区放入共享内存。邻居线程计算直接项及交叉项，通过共享内存 FP64 `atomicAdd` 合并；CTA 完成后把每个 `[中心类型, 邻居类型, n, basis]` 结果一次写入最终 `gradsecond_c3`。跨中心原子的合并仍使用全局 FP64 `atomicAdd`。

这样可删除：

- `[N, max_neigh, n, batch_types, basis]` 的 `dfeat_c3`；
- `[N, all_types, n, basis]` 的 `tmp_dfeat_c3`；
- `aggregate_dfeat_c3` 和 `aggregate_features` 两个独立 kernel；
- batch 元素种类数的 CPU 回读及三个显式 `cudaDeviceSynchronize`。

## 模板与寄存器控制

`n_max_3b`、`n_base_3b`、三体最大角动量以及四体/五体开关作为编译期参数，使短循环和地址计算能够展开。只生成经过选择的常用组合，避免实例数量和编译时间失控；未生成组合使用旧核。

当前核同时保存六组 24 元素 FP64 数组，ncu 测得 253 个寄存器／线程。新实现将 L=1、2、3、4 的代数拆成独立的内联模板作用域，每次只保留当前 L 所需的 `2L+1` 个分量。`fn/fnp` 使用精确的编译期长度 9，不再按 `MAX_NUM_N` 分配。共享内存分阶段复用，避免通过 `--maxrregcount` 强行产生 local-memory spill。

模板展开、类型分块和线程数分别做单变量对照；若完全展开导致寄存器数或耗时上升，则保留分段循环，不以展开数量作为优化目标。

## 正确性与回退

先添加旧核与新核的 A/B 测试，覆盖：单一邻域类型、重复类型、超过 4 种局部类型、超过一个 warp 的邻居数、空槽邻居，以及三体/四体/五体组合。比较 `gradsecond_c3`、完整 loss 和所有可训练参数梯度；FP64 目标误差为 `rtol <= 1e-9`、`atol <= 1e-11`，并记录最大绝对与相对误差。

运行 `compute-sanitizer` 的 memcheck、racecheck 和 synccheck。分别编译 SM60、SM70、SM86；有可用节点时在相应 GPU 上运行最小数值测试。任何不支持的尺寸、共享内存不足或显式 `legacy` 请求均走旧核。

## 性能验收

在同一张 RTX 3090 上使用 `mini_data_test` 的相同 8 个稳态 batch 对照：

- 新核必须快于旧的 `find_angular_gardc_neigh + aggregate_dfeat_c3 + aggregate_features` 总和；
- nsys 中不再出现大型 `dfeat_c3` 原生分配，当前最大 2.625 GiB 临时张量应消失；
- 记录整步时间、主核时间、核启动数和原生 CUDA 分配峰值；
- 用 ncu 检查寄存器数、occupancy、warp stall 和 FP64 管线利用率；
- 最后运行实际 epoch，确认训练指标与基线一致并报告整体收益。

只有同时通过梯度一致性、sanitizer 和整步性能测试后，`auto` 才默认选择新核。
