# NEP Fitting NVRTC JIT 设计

## 目标

针对模型实际使用的 `(D, H, Q, SM)`，为现有 FP64 单隐藏层 fused fitting
算子生成专用核函数，避免预编译所有特征数和隐藏层宽度组合。支持范围保持为：

- 特征数：`1 <= D <= 96`
- 隐藏层神经元数：`1 <= H <= 100`
- 输出头数量：`Q = 1` 或 `Q = 2`

## 第一阶段范围

JIT 只覆盖当前 `nep_fitting.cu` 中的四类核函数：

- 原子能量和 feature 一阶导前向计算；
- 对 feature 的反向梯度计算；
- fitting 参数梯度的分块累加；
- fitting 参数梯度的归并。

原子分组、参数排列、PyTorch autograd 行为、checkpoint 格式和 FP64 精度均不改变。
NEP descriptor、force、virial 和 descriptor 二阶导核函数不属于本阶段。

## 运行时架构

`CalcOps_cuda` 仍然作为普通 PyTorch 扩展预编译。扩展内部新增一个 JIT 管理器，
链接 NVRTC 和 CUDA Driver API。JIT 管理器将 CUDA 源码内置在动态库中，把实际的
`D`、`H`、`Q` 设置为编译期常量，并为当前 GPU 计算能力生成一个专用模块。

每个模块导出固定名称的四个 fitting kernel。现有 launcher 首先查找匹配的 JIT
模块：找到时在 PyTorch 当前 CUDA stream 上启动专用 kernel；不可用时调用现有
AOT 通用 kernel。输入、输出和参数张量的布局不发生变化。

专用 kernel 继续采用每 16 个隐藏层神经元一个 tile，以及有界的线程局部累加器。
完整 tile 和尾部 tile 分开处理，feature 循环只进行有限展开，不把完整的 `D x H`
网络完全展开，以控制指令体积和寄存器压力。

## 预热与运行模式

新增一个原始算子，输入 CUDA 参考张量以及 `(D, H, Q)`，返回对应 JIT 模块是否准备
成功。Python 辅助函数从 NEP 模型读取这些参数。

`nep_network.load_model_optimizer()` 在模型移动到目标 GPU 后、DDP 包装前调用预热。
直接调用 fused fitting 算子的测试或其他代码路径，可在第一次 forward 时延迟准备。
训练 epoch 内不会重复编译。

环境变量 `MATPL_NEP_FITTING_JIT` 控制行为：

- 未设置：关闭 JIT，直接使用 AOT；RTX 3090 的 OMat24 稳态基准表明当前 JIT
  比 AOT 慢，因此默认不承担编译与运行开销；
- 设为 `auto`：尝试 JIT；失败时只警告一次并回退 AOT；
- 设为 `1`：强制使用 JIT，编译或加载失败直接报错；
- 设为 `0`：关闭 JIT，不加载 NVRTC，直接使用 AOT。

JIT 只对 CUDA 构建生效，CPU 和 HIP 路径保持不变。

## 缓存与多进程并发

缓存键包含：

- kernel 源码哈希；
- `D`、`H`、`Q`；
- GPU 计算能力；
- NVRTC 版本；
- 影响生成代码的编译选项。

默认缓存目录为 `${XDG_CACHE_HOME:-$HOME/.cache}/matpl/nep_fitting`。可以通过
`MATPL_NEP_JIT_CACHE` 指定节点本地磁盘目录。

编译进程对每个缓存键加文件锁，先写临时文件，再原子重命名。在 DDP 环境下，首个
取得锁的进程负责生成缓存，其他进程等待后加载完整文件。每个进程分别加载自己的
CUDA module。缓存损坏或与当前环境不兼容时，删除对应文件并重新编译一次。

CUDA module 保留到进程结束。JIT 预热必须在 CUDA Graph capture 之前完成。缓存只
包含生成的 CUDA 二进制，不保存模型参数或训练数据。

## 错误处理

不支持的维度继续沿用现有输入检查错误。在 `auto` 模式下，NVRTC 缺失、编译失败、
缓存目录不可写或 module 加载失败都会产生一次简短警告，然后回退到 AOT kernel。
严格模式下，这些错误直接抛出。

kernel 启动错误必须立即检查。由于错误发生前可能已经写入部分输出，启动失败后不在
同一次调用中重新执行 AOT kernel。

## 验证标准

测试需要证明：

1. 运行模式解析正确，缓存键稳定；
2. 关闭 JIT 时不会生成或加载缓存；
3. 严格模式可以为当前 SM 生成有效缓存文件；
4. 第二个进程可以复用已有缓存；
5. 多进程并发准备只产生一个有效缓存文件；
6. 边界尺寸和非 16 对齐尺寸下，JIT 的输出、feature 一阶导、feature 反向梯度、
   参数梯度和 Adam 更新与 AOT/参考实现一致；
7. `Q=2` charge 模式、非默认 stream、DDP，以及 P100、V100、3080 Ti、3090、4090
   对应架构的编译均受支持；
8. `auto` 模式的失败回退能够正常训练；
9. OMat24 基准将编译/加载耗时与稳态 step 耗时分开统计，并记录显存峰值。

现有 AOT 测试套件继续作为回归验证门槛。
