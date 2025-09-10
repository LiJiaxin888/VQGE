import matplotlib.pyplot as plt

import numpy as np

losslist1 = np.loadtxt('loss.txt')
# losslist2 = np.loadtxt('loss_noise.txt')

# 迭代次数根据长度推断
iterlist1 = list(range(1, len(losslist1) + 1))
# iterlist2 = list(range(1, len(losslist2) + 1))


# 创建图形
plt.figure(figsize=(10, 6))

# 绘制第一个 loss 曲线
plt.plot(iterlist1, losslist1, label='Loss', linestyle='-', linewidth=2, marker='o', markersize=3)

# 绘制第二个 loss 曲线
# plt.plot(iterlist2, losslist2, label='Loss with noise', linestyle='--', linewidth=2, marker='s', markersize=3)

# 设置标题和坐标轴
plt.title('Loss Function vs. Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Loss Value')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图例
plt.legend()

# 使用对数 y 轴
plt.yscale('log')

# 显示图形
plt.show()
