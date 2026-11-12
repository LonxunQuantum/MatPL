# MatPL NEP 支持 GPUMD `zbl.in`：设计

## 目标

在 NEP 的 `model.descriptor` 中增加与 `zbl` 同级的 `zbl_in` 路径。训练使用文件中逐元素对的柔性 ZBL 参数；导出的 `nep.txt` 自包含这些参数，可直接交给 GPUMD 和 MatPL 的 CPU/GPU test 接口。现有不带 `zbl_in` 的通用 ZBL 行为保持原样。本轮文档仅定义实现，不修改训练代码。

```json
"model": {
  "descriptor": {
    "cutoff": [6.0, 4.0],
    "zbl": 2.0,
    "zbl_in": "zbl.in"
  }
}
```

`zbl_in` 相对 `nep.json` 所在目录解析；绝对路径原样使用。与 GPUMD 的 `nep.in` 一致，指定 `zbl_in` 时仍须指定 `zbl` 以明确启用 ZBL；此时逐元素对的真实内、外截断来自文件，`zbl` 数值不参与柔性势的计算。`zbl_in` 与 `use_typewise_cutoff_zbl` 互斥。

## GPUMD 文件契约

设 `nep.json.atom_type` 顺序为 `0…T−1`。`zbl.in` 必须恰有 `10 × T(T+1)/2` 个有限实数；GPUMD 按上三角顺序 `(0,0),(0,1),…,(0,T−1),(1,1),…,(T−1,T−1)` 读取，每对连续 10 个数：

`[r_inner, r_outer, a1, b1, a2, b2, a3, b3, a4, b4]`

前两项单位为 Å，其余项用于 `φ(x)=Σ(a_k exp(−b_k x))`；`x=2.134563 r (Z_i^0.23+Z_j^0.23)`。势能为 `K_C_SP Z_i Z_j φ(x) f_c(r) / r`；`f_c` 在 `r<r_inner` 为 1，在 `r_inner≤r<r_outer` 为余弦过渡，之后为 0。每原子能量对有向近邻求和后乘 1/2。受力与维里须由同一个势的解析/自动微分导数取得。

示例 `/data/home/wuxingxing/datas/training_test/mgpus/flexiable_zbl/training/zbl.in` 是单元素 W，只有一组 10 个数，内、外截断为 `1/2 Å`，第一对指数系数为 `0.32825/2.54931`；对应 `nep.txt` 的头为 `nep4_zbl 1 W`，下一行是 `zbl 0 0`，文件最后 10 个数就是该组参数。GPUMD 5.2 的 `parameters.cu::read_zbl_in`、`fitness.cu` 与 `src/force/nep.cu` 分别定义读取、导出和推理布局。

## MatPL 实现路径

1. **参数入口**：在 `src/user/nep_param.py` 解析 `zbl_in`，检查文件、参数个数、有限值及每对 `0≤r_inner<r_outer`。检查最大 `r_outer` 不超过当前角向邻居表的截断半径；若希望支持更大值，须先扩展邻居表构建范围。为兼容 GPUMD 5.2 与当前 MatPL CPU/GPU test 端的固定 `para[550]`，柔性 ZBL 限制最多 10 种元素，超出时在解析阶段报错。
2. **配置与 checkpoint**：在 `src/user/input_param.py`、`src/user/nn_feature_type.py` 保留 `zbl_in` 路径，并将已解析的完整数值按原元素顺序写入 checkpoint 配置快照；恢复及 `totxt` 优先使用快照中的数值，不依赖原文件仍在原路径。`zbl.in` 参数固定，不参与优化器更新。仅靠路径保存 checkpoint 不足以保证可移植性。
3. **训练计算**：在 `src/model/nep_net.py` 的现有 ZBL 分支选择“通用”或“柔性”参数。柔性分支按中心/邻居**类型索引**构造上三角元素对索引，从 `[T(T+1)/2,10]` 参数表取得截断和指数系数；不用原子序数决定系数，但 `Z_i Z_j` 和屏蔽长度仍使用真实原子序数。用逐对 `r_outer` 筛选并计算能量、受力、维里；保留现有 force loss 的求导路径。ZBL 不进入拟合网络参数梯度，但其受力必须正确加入总受力标签比较。
4. **文本导出与导入**：在 `src/utils/nep_to_gpumd.py` 输出已有的 `nep4_zbl`/`nep5_zbl`（带电荷版本沿用现有头）及 `zbl 0 0`。在网络参数、描述符系数、`q_scaler` 之后，按原始上三角顺序逐行追加 `10 × T(T+1)/2` 个数。`src/user/nep_param.py::set_nep_nn_c_param_from_nep_txt` 目前对 `zbl 0 0` 抛出“不支持”异常；须解析末尾参数并修正行数/NEP4 网络布局判断，使从 txt 初始化和再导出保持数值不变。MatPL 的 `src/feature` CPU/GPU test 端已有该文本布局读取代码，不需要改变文本格式。

## 验证

- 用 W 单元素示例验证 `zbl.in → checkpoint → nep.txt` 的 10 个数逐项一致，`nep.txt` 由 GPUMD 与 MatPL CPU/GPU test 接口加载。
- 增加两元素测试，特意给三对不同参数，核对 `(0,0),(0,1),(1,1)` 顺序、交换原子类型后的对称性，以及 txt 导入再导出。
- 在计算节点对照 GPUMD 示例的近距结构，比较能量、三维受力和维里；对切换区间及截断边界做有限差分。再进行含受力损失的一步训练，确认参数梯度和 checkpoint 恢复。现有通用 ZBL 回归测试必须继续通过。

## 当前代码边界

实施前，训练端只实现固定系数的通用 ZBL；`src/user/nep_param.py` 明确拒绝 `zbl 0 0`，导出器也只写通用 ZBL 的截断行。CPU/GPU test 端已实现柔性 ZBL，但最多支持 10 种元素。
