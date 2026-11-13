# NEP 模型类型与柔性 ZBL 配置

`model_type: "NEP5"` 与 `"NEP"` 等价。解析后统一保存为 `"NEP"`。

在 `model.descriptor` 中，`zbl` 接受两种值：

- 数值，例如 `"zbl": 2.0`：使用原有通用 ZBL。
- 文件路径，例如 `"zbl": "zbl.in"`：读取逐元素对的柔性 ZBL 参数。相对路径以 `nep.json` 所在目录为基准；不可同时设置 `use_typewise_cutoff_zbl`。

```json
{
  "model_type": "NEP5",
  "model": {
    "descriptor": {
      "cutoff": [6.0, 4.0],
      "zbl": "zbl.in"
    }
  }
}
```

`zbl.in` 按 `atom_type` 的上三角元素对顺序提供参数：`(0,0),(0,1),…,(1,1),…`。每对连续 10 个有限实数：`[r_inner, r_outer, a1, b1, a2, b2, a3, b3, a4, b4]`。要求 `0 ≤ r_inner < r_outer ≤` 角向截断半径，当前最多支持 10 种元素。

配置快照和 checkpoint 保存解析后的数值 `zbl`（所有元素对最大的 `r_outer`）及完整的 `zbl_parameters`，恢复时无需原始文件。导出的 `nep.txt` 写入 `zbl 0 0` 和全部参数，可直接供 GPUMD 与 MatPL test 接口读取。柔性 ZBL 参数固定，不参与优化器更新。
