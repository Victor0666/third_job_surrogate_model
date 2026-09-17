# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# # =========================
# # 0) 全局字体与PDF嵌入
# # =========================
# mpl.rcParams["font.family"] = "Times New Roman"
# mpl.rcParams["mathtext.fontset"] = "stix"
# mpl.rcParams["pdf.fonttype"] = 42
# mpl.rcParams["ps.fonttype"] = 42
#
# # =========================
# # 1) 原始数据（按你图中的表格录入）
# # =========================
# data = {
#     "Scenario": ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"],
#     "FCFS_FIFS": [321114.92, 363175.79, 432193.66, 506870.29, 534289.60, 582367.21, 810911.05, 831290.84, 868121.92],
#     "PD3QN":     [317707.46, 361345.01, 411292.72, 503120.46, 526036.06, 574508.65, 805055.34, 811682.24, 836540.00],
#     "MARL":      [312023.34, 355171.10, 458573.17, 493294.52, 530333.93, 613782.16, 783777.53, 845797.92, 913472.18],
#     "IRWS":      [322630.49, 346111.77, 404966.50, 513292.22, 518350.17, 547586.62, 824116.35, 838096.58, 833653.03],
#     "MAHDRL":    [310665.03, 312465.34, 319776.28, 502905.00, 497653.00, 511576.96, 803683.80, 813018.30, 815547.22],
# }
#
# df = pd.DataFrame(data).set_index("Scenario")
#
# # =========================
# # 2) 计算 RPD
# #    RPD = (当前算法数值 - 该scenario下所有算法最小值) / 当前算法数值
# # =========================
# row_min = df.min(axis=1)
# rpd = (df.sub(row_min, axis=0)).div(df, axis=0)
#
# # =========================
# # 3) 分组柱状图
# # =========================
# scenarios = rpd.index.tolist()
# algos = rpd.columns.tolist()
#
# x = np.arange(len(scenarios))
# n_algos = len(algos)
# bar_w = 0.16
#
# fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=160)
#
# # 颜色映射（你可按需要继续微调）
# color_map = {
#     "FCFS_FIFS": "#c2d0ea",
#     "PD3QN":     "#0070b8",
#     "MARL":      "#00b3b0",
#     "IRWS":      "#d40d8c",
#     "MAHDRL":    "#6A5ACD",
# }
# # color_map = {
# #     "MAHDRL_NTA": "#c2d0ea",
# #     "MAHDRL_NHA": "#0070b8",
# #     "MAHDRL_NVA": "#00b3b0",
# #     "MAHDRL_TL":  "#d40d8c",
# #     "MAHDRL":     "#6A5ACD",
# # }
# # 为 0 值设置最小可见高度，仅用于显示
# ymax = float(rpd.to_numpy().max())
# eps = 0.015 * ymax if ymax > 0 else 0.001
#
# for i, algo in enumerate(algos):
#     offset = (i - (n_algos - 1) / 2) * bar_w
#
#     vals = rpd[algo].values.copy()
#     zero_mask = np.isclose(vals, 0.0)
#
#     vals_plot = vals.copy()
#     vals_plot[zero_mask] = eps
#
#     bars = ax.bar(
#         x + offset,
#         vals_plot,
#         width=bar_w,
#         label=algo,
#         color=color_map.get(algo, "gray"),
#         edgecolor="black",
#         linewidth=0.5,
#     )
#
#     # 真实为0的柱子用纹理标识
#     for b, is_zero in zip(bars, zero_mask):
#         if is_zero:
#             b.set_hatch("//")
#             b.set_alpha(0.7)
#
# # 坐标轴
# ax.set_xticks(x)
# ax.set_xticklabels(scenarios, fontsize=10)
# ax.set_ylabel("RPD", fontsize=11)
# ax.set_xlabel("Scenario", fontsize=11)
#
# # 科学计数法
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# ax.yaxis.get_offset_text().set_size(10)
#
# # y轴留一点底部空隙
# gap = 0.06 * ymax if ymax > 0 else 0.001
# ax.set_ylim(-gap, ymax * 1.15 if ymax > 0 else 0.01)
#
# # 图例
# ax.legend(
#     ncol=3,
#     fontsize=8,
#     frameon=True,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.18)
# )
#
# # 边框与刻度
# for spine in ax.spines.values():
#     spine.set_linewidth(0.8)
# ax.tick_params(axis="both", labelsize=10, width=0.8)
#
# plt.tight_layout()
#
# # =========================
# # 4) 导出 PDF
# # =========================
# fig.savefig(
#     "algorithm_rpd.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     transparent=True,
# )
#
# plt.show()
#
# # =========================
# # 5) 如需查看RPD数值，可打印
# # =========================
# print("RPD values:")
# print(rpd)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# 0) 全局字体与PDF嵌入
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# =========================
# 1) 原始数据（按你图中的表格录入）
# =========================
data = {
    "Scenario": ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"],
    "FCFS_FIFS": [321114.92, 363175.79, 432193.66, 506870.29, 534289.60, 582367.21, 810911.05, 831290.84, 868121.92],
    "PD3QN":     [317707.46, 361345.01, 411292.72, 503120.46, 526036.06, 574508.65, 805055.34, 811682.24, 836540.00],
    "MARL":      [312023.34, 355171.10, 458573.17, 493294.52, 530333.93, 613782.16, 783777.53, 845797.92, 913472.18],
    "IRWS":      [322630.49, 346111.77, 404966.50, 513292.22, 518350.17, 547586.62, 824116.35, 838096.58, 833653.03],
    "MAHDRL":    [310665.03, 312465.34, 319776.28, 502905.00, 497653.00, 511576.96, 803683.80, 813018.30, 815547.22],
}

df = pd.DataFrame(data).set_index("Scenario")

# =========================
# 2) 计算 RPD
#    RPD = (当前算法数值 - 该scenario下所有算法最小值) / 当前算法数值
# =========================
row_min = df.min(axis=1)
rpd = (df.sub(row_min, axis=0)).div(df, axis=0)

# =========================
# 3) 准备箱线图数据
#    每个算法 across 9 scenarios 的 RPD 构成一个箱子
# =========================
algos = rpd.columns.tolist()
box_data = [rpd[col].values for col in algos]

# 颜色映射
# color_map = {
#     "FCFS_FIFS": "#c2d0ea",
#     "PD3QN":     "#7b95c6",
#     "MARL":      "#67a583",
#     "IRWS":      "#f0c987",
#     "MAHDRL":    "#e49b9b",
# }
color_map = {
    "FCFS_FIFS": "#c2d0ea",
    "PD3QN":     "#0070b8",
    "MARL":      "#00b3b0",
    "IRWS":      "#d40d8c",
    "MAHDRL":    "#6A5ACD",
}
# =========================
# 4) 绘制箱线图
# =========================
fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=160)

bp = ax.boxplot(
    box_data,
    patch_artist=True,
    widths=0.55,
    showmeans=True,
    meanline=False,
    showfliers=False,   # 不显示异常值
    boxprops=dict(linewidth=0.9, color="black"),
    whiskerprops=dict(linewidth=0.9, color="black"),
    capprops=dict(linewidth=0.9, color="black"),
    medianprops=dict(linewidth=1.2, color="black"),
    meanprops=dict(
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=4
    )
)

# 给每个箱子填充颜色
for patch, algo in zip(bp["boxes"], algos):
    patch.set_facecolor(color_map.get(algo, "lightgray"))
    patch.set_alpha(0.9)

# 可选：叠加散点，显示每个 scenario 的真实 RPD
# for i, vals in enumerate(box_data, start=1):
#     x_jitter = np.random.normal(loc=i, scale=0.04, size=len(vals))
#     ax.scatter(
#         x_jitter,
#         vals,
#         s=18,
#         facecolors="white",
#         edgecolors="black",
#         linewidths=0.5,
#         zorder=3
#     )

# =========================
# 5) 坐标轴与样式
# =========================
ax.set_xticks(np.arange(1, len(algos) + 1))
ax.set_xticklabels(algos, fontsize=10)
ax.set_ylabel("RPD", fontsize=11)
ax.set_xlabel("Algorithm", fontsize=11)

# 科学计数法
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.yaxis.get_offset_text().set_size(10)

# 网格线
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

# 边框与刻度
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
ax.tick_params(axis="both", labelsize=10, width=0.8)

plt.tight_layout()

# =========================
# 6) 导出 PDF
# =========================
fig.savefig(
    "algorithm_boxplot_rpd.pdf",
    format="pdf",
    bbox_inches="tight",
    transparent=True,
)

plt.show()

# =========================
# 7) 打印 RPD 数值
# =========================
print("RPD values:")
print(rpd)

print("\nDescriptive statistics of RPD:")
print(rpd.describe())