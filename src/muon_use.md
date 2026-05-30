## 1. Cij 系数适合用什么优化器处理?
但Cij 系数比较特殊 —— 它本质是 4D 张量 (ntypes, ntypes, n_max+1, n_base+1)，不是单纯的"系数"，按你直觉应该用 Adam，但实际上它形状上够 Muon 处理。

## 2. 当前 HybridMuon 的真实路由（GPU 实测）
在 RTX 4090 / Triton 2.2.0 上跑了 muon_route_probe.sh，扫描 NEP 的实际 named_parameters，分流规则是：

参数类别	路由	优化器	weight_decay
c_param_2, c_param_3 (Cij 系数)	muon	Newton-Schulz orth	有
fitting_net.{i}.layers.0.weight (输入层 [35,N])	muon	Newton-Schulz orth	有
fitting_net.{i}.layers.1.weight (隐藏→隐藏 [N,N])	muon	Newton-Schulz orth	有
fitting_net.{i}.layers.{last}.weight (输出层 [N,1])	adam_no_decay	Adam	无（rank<2）
*.bias (所有 bias)	adam_no_decay	Adam	无
*.resnet_dt (残差缩放)	adam_no_decay	Adam	无
具体三个配置的统计：

small_Si (ntypes=1, neuron=[50,1]) — 1800 个参数走 Muon，101 个走 Adam


[muon]          c_param_2 (1,1,5,5)         B=1, R=5, C=5
[muon]          c_param_3 (1,1,5,5)         B=1, R=5, C=5
[muon]          layers.0.weight (35,50)     B=1, R=35, C=50
[adam_no_decay] layers.0.bias (1,50)
[adam_no_decay] layers.1.weight (50,1)      ← rank<2 自动落到 Adam
[adam_no_decay] layers.1.bias (1,1)
water_5k (ntypes=2, neuron=[100,1]) — 7200 个走 Muon，402 个走 Adam

Cij 因为 (2,2,5,5) 在 slice 模式下被展平为 B=4, R=5, C=5，4 个 5×5 矩阵分别做 NS orthogonalization
ntypes=3 + neuron=[100,100,1] — 40950 个走 Muon

多了 layers.1.weight (100,100) 走 Muon
resnet_dt (1,100) 因 effective shape rank=1 自动走 Adam
关于"Cij 系数"的关键点：

按 NEP 物理意义，Cij 是径向/角向描述符的展开系数，不像神经网络权重那样有"特征旋转"的几何结构。但 HybridMuon 的路由是纯 shape-based：只要 effective rank ≥2 且能 view 成矩阵就走 Muon。所以 Cij 当前默认走 Muon。

这是好是坏？

当前 small_Si / water_5k 消融数据 (1.6–3.6× 优势) 是在 Cij 走 Muon 的设定下取得的，说明对 NEP 这个特定模型，Cij 走 Muon 没有破坏收敛。
物理上 Newton-Schulz 把 Cij 的 5×5 矩阵奇异值正交化，在描述符表达上倾向于让径向×角向基函数解相关 —— 实际上可能有益。
但若想验证，可以做一次消融：把 Cij 强制走 Adam，看 force MAE 是否退化。
## 3. 输入文件怎么写
NEP 训练 JSON 的 optimizer 字段：


{
  "model_type": "NEP",
  "...": "...",
  "optimizer": {
    "optimizer": "MUON",
    "epochs": 100,
    "batch_size": 1,
    "learning_rate": 5e-4,
    "stop_lr": 3.51e-8,

    "max_norm": 2.0,
    "norm_type": 2,

    "momentum": 0.95,
    "weight_decay": 1e-3,

    "muon_mode": "slice",
    "muon_lr_adjust": 0.0,
    "muon_lr_adjust_coeff": 0.18,
    "muon_enable_gram": true,
    "muon_flash": true,
    "muon_magma": true,

    "train_energy": true,
    "train_force": true,
    "train_virial": true,
    "start_pre_fac_force": 1000,
    "start_pre_fac_etot": 0.02,
    "start_pre_fac_virial": 50.0,
    "end_pre_fac_force": 1.0,
    "end_pre_fac_etot": 1.0,
    "end_pre_fac_virial": 1.0
  }
}
必填项：

optimizer: "MUON" 选择 HybridMuon
max_norm 或 clip_value 二选一（Phase 3.3 gate 强制要求；缺失会在 InputParam 解析阶段直接报错）
可选 Muon 调参（默认值已是 DPA4/DeepSeek-V4 推荐）：

muon_mode: "2d" / "flat" / "slice"，默认 slice（Cij 4D 张量被分成多个 2D 切片各自正交化）
muon_lr_adjust: 默认 0.0（用 Frobenius scale），>0 时改用 fixed LR scale
muon_lr_adjust_coeff: 默认 0.18（DeepSeek-V4 配方）
muon_enable_gram: 默认 true，启用 Gram-NS（更稳的正交化）
muon_flash: 默认 true，启用 Triton 加速（Triton 不可用时自动回落到 cuBLAS）
muon_magma: 默认 true，启用动量-梯度对齐 damping
Cij/bias 默认行为不需要任何额外配置 —— HybridMuon 在 _build_param_routing 里自动按参数名和形状分桶。

如果你想把 Cij 强制走 Adam做对照实验，目前 hybrid_muon.py 的 get_adam_route 只识别 bias、adam_*、adamw_* 前缀。最干净的 opt-in 方式是改名：把 c_param_2 / c_param_3 重命名为 adam_c_param_2 / adam_c_param_3，会自动落到 Adam 桶。但这是后续消融的事，不是现在要改的。