# NEP checkpoint 恢复规则

训练配置统一以当前 `nep.json` 为准，包括 loss prefactor、学习率与缩放、余弦周期与最低学习率、预热、Adam/AdamW 的 `lambda_2`。

- `optimizer.reset_epoch=false`：恢复模型权重、优化器动量和已完成的更新步数，从 checkpoint 的下一 epoch 继续训练。
- `optimizer.reset_epoch=true`（默认）：继承模型权重，清空优化器历史，从第 1 epoch、更新步数 0 开始。

续训不直接加载旧 scheduler 配置。程序按当前 JSON 创建调度器，并根据已完成步数定位学习率：预热总步数为当前 `warm_epochs × 每轮步数`；余弦调度从扣除预热后的步数定位，指数衰减按累计步数定位。即使新的 JSON 修改了 `learning_rate`、`t_0`、`t_mult`、`stop_lr` 或 `lambda_2`，旧 checkpoint 也不会覆盖它们。

prefactor 使用当前 JSON 的起止值，并按 `已完成步数 / 当前 stop_step` 插值；续训不会自动回到起始权重。修改 `batch_size`、`mix` 或进程数后，累计已完成步数不变，但当前每轮步数会影响以 epoch 为单位配置的预热长度和余弦周期。

旧 checkpoint 缺少 `optimizer_updates` 时，沿用 `已完成 epoch × 当前每轮步数` 的估算；如果训练批量已改变，该估算无法精确还原旧更新次数。缺少 optimizer 状态时使用新优化器，模型权重和可用的训练进度仍可恢复。

checkpoint 中仍保存 scheduler 与预热信息，供检查和旧版本使用；新恢复逻辑不将这些记录作为配置来源。启动日志会打印累计步数、当前预热步数、峰值学习率和下一次更新的学习率。
