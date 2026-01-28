import matplotlib.pyplot as plt
import re
import pandas as pd

# log_path = "/data/home/wuxingxing/datas/debugs/spf/Si-SiO2-HfO2-TiN/norml2_warm/slurm-1624402.out"
# picture_save_path="/data/home/wuxingxing/datas/debugs/spf/Si-SiO2-HfO2-TiN/norml2_warm/lr_new1.png"
log_path = "/data/home/wuxingxing/datas/pwmat_mlff_workdir/hfo2/mgnep/stdallb256/slurm-2554439.out"
picture_save_path= "/data/home/wuxingxing/datas/pwmat_mlff_workdir/hfo2/mgnep/stdallb256/lr.png"
max_epoch = 40

with open(log_path, 'r') as rf:
    data = rf.readlines()

# 初始化存储数据的列表
epochs = []
current_iters = []
learning_rates = []

# 正则表达式模式
pattern = r"Epoch: \[(\d+)\]\[\s*(\d+)/(\d+)\].*?LR (\d+\.\d+e[+-]\d+)"

# 处理每一行数据
for line in data:
    if line.startswith('Epoch:'):
        match = re.search(pattern, line)
        if match:
            epochs.append(int(match.group(1)))
            current_iters.append(int(match.group(2)))
            learning_rates.append(float(match.group(4)))

max_iter = max(current_iters)
epoch_iters = []
x_loc = []
for iter, epoch in zip(current_iters, epochs):
    if epoch > max_epoch:
        continue
    epoch_iters.append(iter + (epoch-1) * max_iter)
    if iter % max_iter == 0:
        x_loc.append(epoch_iters[-1])
# 创建 DataFrame 方便分组计算
df = pd.DataFrame({
    'iter': epoch_iters,
    'lr': learning_rates[:len(epoch_iters)]
})

x_value = list(range(1, len(x_loc)+1))

# 绘制学习率变化曲线（按 epoch 均值）
plt.figure(figsize=(10, 6))
plt.plot(df['iter'], df['lr'], 'b-', marker='o', markersize=6, label='LR')
plt.yscale('log')
plt.xticks(x_loc, x_value)  # (坐标位置, 显示标签)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate vs Epoch (Averaged)')
plt.legend()
plt.grid(True)
plt.savefig(picture_save_path)
