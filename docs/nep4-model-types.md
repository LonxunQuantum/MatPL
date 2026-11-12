# NEP4 与带电荷模型类型设计

## 目标与范围

在现有 NEP 训练实现中，以顶层 `model_type` 唯一确定 NEP4/NEP5 和是否启用电荷。NEP4 与现有 NEP5 的拟合网络区别仅为**所有元素共享能量输出层的最后一个 bias**；各元素的其他网络参数仍独立。带电荷模型沿用现有 `charge2` 的能量、电荷计算及训练路径。

本设计只调整 JSON、checkpoint 和训练模型内部的类型表达及对应拟合网络行为。`nep.txt` 的头名称、字段顺序和已有读写格式保持不变；本轮不改变描述符、CUDA/HIP 算子的计算定义，也不涉及电荷模式 2 以外的模式。

## 四种模型类型

| JSON `model_type` | 文本力场版本 | 能量输出层 bias | 电荷输出 |
| --- | --- | --- | --- |
| `NEP` | NEP5 | 每种元素独立 | 无 |
| `NEP4` | NEP4 | 所有元素共用一个 | 无 |
| `NEP_charge2` | NEP5 charge2 | 每种元素独立 | 有 |
| `NEP4_charge2` | NEP4 charge2 | 所有元素共用一个 | 有 |

大小写按现有入口统一规范化；checkpoint 保存规范化后的 `model_type`。JSON 的 `model.descriptor` 不再包含 `charge_mode`、`gpumd_nep4`，checkpoint 的配置快照也不保存这两个字段。实现中删除这两个配置字段的读取与序列化，不专门为旧字段添加校验、兼容或迁移代码。内部计算需要的电荷标记可以从 `model_type` 推导。

例如普通 NEP4 配置为：

```json
{
  "model_type": "NEP4",
  "model": {
    "descriptor": { "cutoff": [6.0, 5.0], "n_max": [4, 4], "basis_size": [8, 8], "l_max": [4, 2, 1] },
    "fitting_net": { "network_size": [80, 1] }
  }
}
```

带电荷时仅将 `model_type` 改为 `NEP4_charge2` 或 `NEP_charge2`；其他电荷训练参数沿用现有设置。

## 网络参数与计算

设元素类型为 \(t\)，描述符为 \(x\)，其子网络隐层及能量头输出（不含最后一层 bias）为 \(g_t(x)\)：

- NEP5：\(E_i=g_{t_i}(x_i)+b_{t_i}\)，各元素各有一个 \(b_t\)。
- NEP4：\(E_i=g_{t_i}(x_i)+b_{\mathrm{common}}\)，整个模型只有一个可训练的 \(b_{\mathrm{common}}\)。

`NEP4_charge2` 使用现有带电荷路径的共享能量 bias；普通 `NEP4` 复用同一个设计。带电荷模型的 charge head、`sqrt_epsilon_inf` 及后续静电计算仍按现有逻辑执行；**共享的只是能量头 bias**。由于 bias 与描述符无关，\(\partial E_i/\partial x_i\) 的解析表达式不变，但它必须处在能量的 autograd 计算图中，使能量和力/维里损失能够更新该参数。

普通 `NEP4` 和 `NEP4_charge2` 的各元素拟合子网络不创建最后一层能量 bias，模型顶层创建一次 `common_bias`，并在普通与融合 fitting 路径的能量输出上加上它。`NEP` 与 `NEP_charge2` 保留各元素独立 bias；不为它们创建闲置的 `common_bias` 参数。由此，优化器和 checkpoint 只包含当前类型实际使用的参数。

## 配置、checkpoint 与文本力场

1. **配置入口**：在读取顶层 `model_type` 时解析为内部的 NEP 版本（4/5）及是否带电荷（否/2），其余训练流程使用这一解析结果。训练、测试、分布式后端选择及路径处理均应将四种类型识别为 NEP 家族。
2. **checkpoint**：配置快照写入当前 `model_type`；`state_dict` 根据类型保存 `common_bias` 或各元素末层 bias。恢复时仍按现有机制恢复权重、优化器状态和训练进度；模型构造依据当前选定类型。类型或参数形状不匹配时按常规模型加载错误处理。
3. **文本力场**：维持现有头形式 `nep4`/`nep5`、可选 `_zbl` 与 `_charge2`，以及原有字段顺序。四种类型分别导出相应头；默认文件名依版本为 `nep4.txt` 或 `nep5.txt`。NEP4 在所有元素网络权重之后写**一个**公共能量 bias，NEP5 写各元素独立能量 bias；带电荷文本中的 `sqrt_epsilon_inf` 等字段保持现有位置和符号约定。导入已有 NEP4 文本时，公共 bias 应重建为单个可训练参数，而不是复制成多个独立 bias。

这里的“头保持不变”指不新增或改名任何 `nep.txt` 头标识；模型类型不同，导出时仍须选用对应的现有 `nep4` 或 `nep5` 头。

### `nep4.txt` 的网络参数顺序

以下顺序直接按 `src/user/nep_param.py` 的 `set_nep_nn_c_param_from_nep_txt()` 反推。设元素数为 `T`、特征数为 `D`、隐藏神经元数为 `H`。头部 `ANN` 行之后，每个数值占一行，并按 `nep.txt` 头中的元素顺序依次写入：

| 次序 | 普通 `NEP4` | `NEP4_charge2` |
| --- | --- | --- |
| 每个元素 `t`，重复 `T` 次 | `W0_t`（`H×D`）、`-b0_t`（`H`）、`W_E,t`（`H`） | `W0_t`（`H×D`）、`-b0_t`（`H`）、`W_E,t`（`H`）、`W_Q,t`（`H`） |
| 所有元素之后 | `-common_bias`（1 个） | `sqrt_epsilon_inf`（1 个）、`-common_bias`（1 个） |
| 网络参数之后 | `c2`、按 `l_max` 决定是否存在的 `c3`、`q_scaler` | 同左 |

`W0_t` 在 checkpoint 中为 `[D,H]`，导出前转置为 `[H,D]` 再展平；解析器用 `reshape(H,D).transpose(1,0)` 还原。`b0_t` 和 `common_bias` 在文本中取负号，因为文本侧的 bias 符号约定与 PyTorch 参数相反。能量头 `W_E,t` 与电荷头 `W_Q,t` 各展平为 `H` 个值；带电荷时，在**每个元素块内**先写该元素全部能量头权重，再写该元素全部电荷头权重，不逐神经元交错。NEP4 的各元素块内都不写输出层 bias，也不写 NEP5 使用的末尾零占位值。

因此普通 NEP4 的网络参数总数为 `T×(D×H + 2H) + 1`；NEP4_charge2 为 `T×(D×H + 3H) + 2`。后续再接描述符参数。导出函数应按此数量检查输出，并用同一个解析函数做往返测试：验证每个元素的 `W0/b0/W_E`、带电荷模型的 `W_Q`、单个公共 bias，以及 `c2/c3/q_scaler` 均按原值恢复。

导出函数现已按四种模型分别计算参数数量，并按表中顺序写出。普通 NEP4 的元素块内没有 `layers.1.bias`，所有元素之后只写一次公共 bias。

### 与 GPUMD 5.2 源码核对

`/data/home/wuxingxing/xcode/GPUMD-5.2/src/main_nep/parameters.cu` 对单隐藏层普通 NEP4 计算的网络参数数目是 `T×(D+2)×H + 1`，带电荷 NEP4 charge2 是 `T×(D+3)×H + 2`。`main_nep/nep.cu` 和 `main_nep/nep_charge.cu` 依次把参数指针分配给每个元素的 `W0`、`b0`、`W1`，最后分配公共 bias；charge2 在公共 bias 前额外分配 `sqrt_epsilon_inf`。这与上表逐项一致。

GPUMD 实际 MD 加载器 `/data/home/wuxingxing/xcode/GPUMD-5.2/src/force/nep.cu`、`force/nep_charge.cu` 使用相同的参数数目与指针偏移。`src/utilities/nep_utilities.cuh` 以 `W0[n×D+d]` 访问输入权重、以 `tanh(W0·q-b0[n])` 计算隐藏层、以 `energy -= b1[0]` 应用公共 bias；charge2 用 `W1[n]` 计算能量、`W1[n+H]` 计算电荷。因此导出时需要转置 `W0`，对两个 bias 取负号，并在每个元素块中连续放置能量头的 `H` 个权重和电荷头的 `H` 个权重。

GPUMD 5.2 的 MD 加载器接受 `nep4`/`nep4_zbl` 与 `nep4_charge2`/`nep4_zbl_charge2`。**它的带电荷加载器不接受 `nep5_charge2`**；因此四种 MatPL 类型中，`NEP_charge2` 的文本不能承诺在此版本 GPUMD 中直接运行。这是 GPUMD 版本兼容范围，与 NEP4 两种布局的核对结果无关。

## 需要改动的代码范围

| 位置 | 主要改动 |
| --- | --- |
| `main.py`、`src/user/input_param.py` 及其他 NEP 分派入口 | 四种类型统一进入既有 NEP 训练/测试流程；集中解析版本和电荷标记，避免各处重复字符串判断。 |
| `src/user/nep_param.py` | JSON 不再解析旧两个字段；`nep.txt` 仍按现有头读取参数；将文件中的公共 bias 交给模型构造时恢复为一个参数。 |
| `src/model/nep_net.py`、`src/model/nep_fitting.py`、`src/model/nep_fused_fitting.py` | 按类型创建能量末层 bias；普通与融合 fitting、输入梯度路径保持一致；仅带电荷类型启用电荷输出。 |
| `src/utils/nep_to_gpumd.py` 及训练导出入口 | 从 `model_type` 选择现有文本头和文件名，按 NEP4/NEP5 分别输出公共或逐元素 bias，核对参数数量。 |
| checkpoint 配置序列化与相关测试 | 保存四种规范类型，不写旧两个字段；验证恢复后的参数结构和推理结果。 |

## 验收标准

- 四种 `model_type` 均能完成模型构建、一次训练更新、checkpoint 保存/恢复和文本力场导出；普通 NEP 的原有行为不变。
- NEP4 仅有一个可训练的能量公共 bias；NEP5 的每元素能量 bias 仍独立。普通与融合 fitting 路径的能量、输入梯度和参数梯度一致。
- `charge2` 两种类型的融合与普通训练路径在能量、受力、维里、电荷输出上保持一致；普通类型不创建电荷输出。
- 四种导出文件沿用现有 `nep4`/`nep5` 头格式，能够由当前 MatPL 文本解析器读取。已在计算节点用 GPUMD 5.2 实际加载 `NEP4` 与 `NEP4_charge2` 的重新导出文件，对 250 原子示例结构做单步 MD，两个模型的受力与原文件逐分量一致（最大绝对差 0）。

实现代码保留在 `nep-dcu/dev` 工作目录，尚未提交或推送 GitHub。CPU 回归测试 42 项通过；3090 计算节点融合训练测试 7 项通过。登录节点安装的 `CalcOps_cpu` 扩展缺少 `calculateNepFeatWithGradContext`，因此该扩展注册检查没有通过，相关其余 CPU 测试通过。
