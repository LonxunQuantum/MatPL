# tNEP (Tensorial NEP) 使用手册

## 1. 简介

tNEP 是 NEP (Neuroevolution Potential) 的张量扩展，能够预测分子的**偶极矩 (dipole moment)** 和**极化率张量 (polarizability tensor)**，用于计算红外 (IR) 和拉曼 (Raman) 光谱。

参考文献: Xu et al., *J. Chem. Theory Comput.* 20, 3273 (2024)

### 与标准 NEP 的关系

| 特性 | 标准 NEP | tNEP |
|------|---------|------|
| 描述符 (descriptor) | Chebyshev + 球谐函数 | **完全相同** |
| 描述符参数 `c_param_2`, `c_param_3` | 训练 | **完全相同** |
| 输出 | 原子能量 (scalar) | 偶极矩 (3分量) 或 极化率 (6分量) |
| 拟合网络 | 每种元素 1 个 | 极化率模式: 每种元素 2 个 (标量头 + 张量头) |
| 损失函数 | 能量 + 力 + 维里 | 仅偶极矩/极化率 (λ_e=0, λ_f=0) |

---

## 2. 训练模式 (train_mode)

| train_mode | 名称 | 输出 | 分量数 | 用途 |
|------------|------|------|--------|------|
| 0 | potential | 原子能量 (标准 NEP) | — | — |
| 1 | dipole | 偶极矩 μ_x, μ_y, μ_z | 3 | IR 光谱 |
| 2 | polarizability | 极化率张量 α_xx, α_yy, α_zz, α_xy, α_yz, α_zx | 6 | Raman 光谱 |

---

## 3. 环境准备

```bash
module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /path/to/matpl-2026.3/bin/activate
export PYTHONPATH=/path/to/MatPL/src:$PYTHONPATH
```

---

## 4. JSON 配置文件

### 4.1 极化率训练 (train_mode=2, nep5 格式)

```json
{
    "model_type": "TNEP",
    "atom_type": [8, 1],
    "train_mode": 2,

    "model": {
        "descriptor": {
            "version": 5,
            "cutoff": [6.0, 4.0],
            "n_max": [4, 4],
            "basis_size": [12, 12],
            "l_max": [4, 2, 0],
            "train_mode": 2
        },
        "fitting_net": {
            "network_size": [40]
        }
    },

    "optimizer": {
        "optimizer": "ADAM",
        "start_lr": 0.001,
        "stop_lr": 1e-8,
        "stop_step": 1000000,
        "decay_step": 5000,
        "epochs": 200,
        "batch_size": 4,
        "train_energy": false,
        "train_force": false,
        "train_polarizability": true,
        "pre_fac_polarizability": 1.0,
        "save_step": 10
    },

    "train_data": ["./train_data"],
    "valid_data": ["./valid_data"]
}
```

### 4.2 偶极矩训练 (train_mode=1, nep5 格式)

```json
{
    "model_type": "TNEP",
    "atom_type": [8, 1],
    "train_mode": 1,

    "model": {
        "descriptor": {
            "version": 5,
            "cutoff": [6.0, 4.0],
            "n_max": [4, 4],
            "basis_size": [12, 12],
            "l_max": [4, 2, 0],
            "train_mode": 1
        },
        "fitting_net": {
            "network_size": [40]
        }
    },

    "optimizer": {
        "optimizer": "ADAM",
        "start_lr": 0.001,
        "stop_lr": 1e-8,
        "stop_step": 1000000,
        "decay_step": 5000,
        "epochs": 200,
        "batch_size": 4,
        "train_energy": false,
        "train_force": false,
        "train_dipole": true,
        "pre_fac_dipole": 1.0,
        "save_step": 10
    },

    "train_data": ["./train_data"],
    "valid_data": ["./valid_data"]
}
```

### 4.3 参数说明

#### model.descriptor

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `version` | int | 5 | NEP 版本: `4` = nep4, `5` = nep5 |
| `train_mode` | int | 0 | 训练模式: `1` = 偶极矩, `2` = 极化率 |
| `cutoff` | [float, float] | [8.0, 4.0] | 径向截断, 角向截断 (Å) |
| `n_max` | [int, int] | [4, 4] | 径向描述符阶数, 角向描述符阶数 |
| `basis_size` | [int, int] | [12, 12] | 径向基函数数量, 角向基函数数量 |
| `l_max` | [int, int, int] | [4, 2, 1] | 3体/4体/5体球谐函数最大角动量 |

#### model.fitting_net

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `network_size` | [int] | [40] | 隐藏层神经元数量 (输出层固定为 1) |

#### optimizer (tNEP 特有)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_energy` | bool | false | tNEP 必须设为 `false` |
| `train_force` | bool | false | tNEP 必须设为 `false` |
| `train_dipole` | bool | false | 偶极矩模式设为 `true` |
| `train_polarizability` | bool | false | 极化率模式设为 `true` |
| `pre_fac_dipole` | float | 1.0 | 偶极矩损失权重 |
| `pre_fac_polarizability` | float | 1.0 | 极化率损失权重 |

> **注意**: `train_energy` 和 `train_force` 对于 tNEP 必须设为 `false`。tNEP 仅拟合偶极矩或极化率。

---

## 5. 训练数据格式

### 5.1 极化率数据 (train_mode=2)

每个结构需要提供 6 个极化率分量 (Voigt 顺序)，单位: Å³：

```
alpha_xx  alpha_yy  alpha_zz  alpha_yz  alpha_xz  alpha_xy
```

这些值存储在标准 PWmat movement 文件的 virial 字段中。

### 5.2 偶极矩数据 (train_mode=1)

每个结构需要提供 3 个偶极矩分量，单位: e·Å：

```
mu_x  mu_y  mu_z
```

存储在 virial 字段的前 3 个分量中。

---

## 6. 运行训练

### 6.1 命令行

```bash
# 极化率训练
python main.py train tnep_polarizability.json

# 偶极矩训练
python main.py train tnep_dipole.json

# 测试/推理
python main.py test tnep_polarizability.json
```

### 6.2 训练输出

训练过程中打印的日志格式：

```
Epoch    1 | Train polarizability_RMSE: 0.052341 | LR: 1.00e-03
Epoch    1 | Valid polarizability_RMSE: 0.048123 | a_xx=0.0321  a_yy=0.0412  a_zz=0.0512  a_yz=0.0123  a_xz=0.0134  a_xy=0.0111
...
tNEP (polarizability) training completed.
```

---

## 7. 模型导出为 GPUMD 格式

训练完成后，将 checkpoint 转换为 GPUMD 兼容的 `nep.txt` 文件：

```bash
# 默认输出为 nep.txt
python main.py totxt MatPL_work/checkpoint/epoch_200.ckpt

# 指定输出文件名
python main.py totxt MatPL_work/checkpoint/epoch_200.ckpt polarizability.txt
```

### 7.1 导出格式

根据 `train_mode` 和 `version`，自动生成正确的文件头：

| 配置 | 导出格式头 |
|------|-----------|
| `train_mode=1, version=4` | `nep4_dipole` |
| `train_mode=1, version=5` | `nep5_dipole` |
| `train_mode=2, version=4` | `nep4_polarizability` |
| `train_mode=2, version=5` | `nep5_polarizability` |

### 7.2 在 GPUMD 中使用

导出的 `nep.txt` 可直接用于 GPUMD 的 MD 模拟：

```
# run.in 中指定
potential nep.txt
```

GPUMD 会在运行时自动识别 `nep5_dipole` / `nep5_polarizability` 头，正确计算偶极矩/极化率。

---

## 8. nep4 与 nep5 格式区别

| 特性 | nep4 | nep5 |
|------|------|------|
| 每个元素的输出层 bias | 无 (共用 common bias) | 有 (每种元素独立 bias) |
| 文件头示例 | `nep4_polarizability 2 O H` | `nep5_polarizability 2 O H` |
| ANN 参数布局 (单头) | w0,b0,w1 × Ntypes → 1个common bias | w0,b0,w1,per-bias × Ntypes → 0.0 |
| ANN 参数布局 (双头) | 张量头(w0,b0,w1) × Ntypes → common_bias_1 → 标量头(w0,b0,w1) × Ntypes → common_bias_2 | 张量头(w0,b0,w1,per-bias) × Ntypes → 标量头(w0,b0,w1,per-bias) × Ntypes → 0.0 |

### 选择建议

- **新项目**: 推荐 `version: 5` (nep5)，每种元素有独立 bias，精度更高
- **兼容旧版 GPUMD**: 使用 `version: 4` (nep4)
- 同一项目中不要混用 nep4 和 nep5 格式

---

## 9. 极化率双头网络架构

对于 `train_mode=2` (极化率)，每种元素有**两个**拟合网络：

```
描述符 (dim=D)
    │
    ├──→ 张量头 (fitting_net)     → t_i (标量) → Fp = ∂t_i/∂q → 各向异性极化率
    │
    └──→ 标量头 (fitting_net_pol) → s_i (标量) → 各向同性对角极化率

总极化率 = 各向同性部分(s) + 各向异性部分(Fp → forces → virial)
```

- **张量头** 的输出通过链式法则 (∂t_i/∂q → ∂q/∂R → force → virial) 贡献各向异性分量
- **标量头** 的输出直接加到极化率对角元 (α_xx, α_yy, α_zz) 上
- 极化率模式的总 ANN 参数量 = 标准 NEP 的 **2 倍**

---

## 10. 已知限制与注意事项

1. **仅支持 virial-only 损失**: tNEP 不拟合能量和力，训练数据无需包含能量/力标签
2. **ZBL 不支持**: tNEP 模式下 ZBL 短程排斥被禁用
3. **CUDA Graph 不支持**: tNEP 暂不支持 CUDA Graph 加速 (未来版本将加入)
4. **多 GPU 训练**: 支持 DDP 多 GPU 训练，与标准 NEP 相同的分布式配置
5. **恢复训练**: 支持从 checkpoint 恢复训练 (`recover_train: true`)

---

## 11. 示例文件

完整示例配置文件位于 `example/tnep/` 目录：

| 文件 | 说明 |
|------|------|
| `tnep_dipole.json` | nep5 偶极矩训练 |
| `tnep_polarizability.json` | nep5 极化率训练 |
| `tnep4_dipole.json` | nep4 偶极矩训练 |
| `tnep4_polarizability.json` | nep4 极化率训练 |
