import matplotlib.pyplot as plt
import numpy as np
plt.set_cmap('jet')
#       越线条数    线路误差平均值   线路潮流概率精度    越线平均值
# M1:   14         0.64396MW     83.32%           3.40MW
# M2:   7          0.2226MW      95.40%           0.847MW
# M3:   4          0.5662MW      84.75%           1.41MW
# M4:   6          0.1574MW      96.68%           1.00MW
# M5:   3          0.0579MW      99.51%           0.756MW
# 创建数据
x = np.array([0, 1, 2, 3, 4])
y1 = np.array([3.4, 0.847, 1.41, 1, 0.756])
y2 = np.array([14, 7, 4, 6, 3])

# 创建左侧坐标轴并绘制直方图
fig, ax1 = plt.subplots()

ax1.bar(x, y1, width=0.4, label='Average violation', alpha=0.7, color='darksalmon')
# ax1.set_xlabel('X轴标签')
ax1.set_xticklabels(['', 'M1', 'M2', 'M3', 'M4', 'M5'])
ax1.set_ylabel('Average violation(MW)')
ax1.legend(loc='upper center', bbox_to_anchor=(0.708, 0.91))

# 创建右侧坐标轴并绘制折线图
ax2 = ax1.twinx()  # 共享x轴

ax2.plot(x, y2, marker='o', linestyle='-', label='Number of violation lines', color='royalblue')
ax2.set_ylabel('Number of violation lines')
ax2.legend(loc='upper right')


# 显示图形
plt.show()