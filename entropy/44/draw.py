import matplotlib.pyplot as plt
import h5py
import os
import numpy as np

if not os.path.exists("minfo_data.h5"):
    print("minfo_data.h5 not found. Please run the DMRG calculation first.")
    exit()

with h5py.File("minfo_data.h5", "r") as f:
    minfo = f["minfo"][:]
    # ordm1 = f["ordm1"][:]   # 可选
    # ordm2 = f["ordm2"][:]   # 可选

fig, ax = plt.subplots(figsize=(6, 6))  # 图像尺寸可调整

# 创建热力图
cax = ax.imshow(minfo, cmap="ocean_r", vmin=0.0, vmax=0.035)
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Mutual Information", fontsize=10)

# 设置坐标轴刻度为 1~N（从1开始）
n = minfo.shape[0]
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(np.arange(1, n + 1))
ax.set_yticklabels(np.arange(1, n + 1))
ax.tick_params(length=0, labelsize=8)

# 将X轴移动到顶部
ax.xaxis.tick_top()  # 主要修改点：移动x轴刻度到顶部
ax.xaxis.set_label_position("top")  # 确保标签位置正确

# 添加细网格线
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3)
ax.tick_params(which="minor", bottom=False, left=False)

# 保持正方形比例
ax.set_aspect("equal")

# 保存图片
plt.tight_layout()
plt.savefig("minfo_matrix_oceanr.png", dpi=600)
plt.close()
