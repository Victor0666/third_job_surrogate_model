# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# 0) 全局字体与 PDF 嵌入
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# =========================
# 1) Excel 路径
# =========================
excel_path = r"ablation_30.xlsx"   # 改成你的 Excel 路径

# 9 个 sheet 名
scenarios = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]

# Excel 中的算法列名
algos = ["MAHDRL_NTA", "MAHDRL_NHA", "MAHDRL_NVA", "MAHDRL_TL", "MAHDRL"]

# 横坐标显示名
display_names = {
    "MAHDRL_NTA": "MAHDRL-NTA",
    "MAHDRL_NHA": "MAHDRL-NHA",
    "MAHDRL_NVA": "MAHDRL-NVA",
    "MAHDRL_TL":  "MAHDRL-TL",
    "MAHDRL":     "MAHDRL",
}

# 配色
color_map = {
    "MAHDRL_NTA": "#c2d0ea",
    "MAHDRL_NHA": "#0070b8",
    "MAHDRL_NVA": "#00b3b0",
    "MAHDRL_TL":  "#d40d8c",
    "MAHDRL":     "#6A5ACD",
}

# =========================
# 2) 从 Excel 读取所有 sheet 的原始数据
#    每个 sheet：
#    第1行为表头
#    第2~31行为 30 个种子结果
# =========================
all_algo_values = {algo: [] for algo in algos}

for scenario in scenarios:
    df_sheet = pd.read_excel(excel_path, sheet_name=scenario)

    # 只保留目标列并转数值
    df_sheet = df_sheet[algos].apply(pd.to_numeric, errors="coerce")

    # 取前 30 行原始 seed 数据
    raw_df = df_sheet.iloc[0:30].copy()

    # 累加到总集合
    for algo in algos:
        vals = raw_df[algo].dropna().values.astype(float)
        all_algo_values[algo].extend(vals.tolist())

# 转成作图数据
violin_data = [np.array(all_algo_values[algo], dtype=float) for algo in algos]

# =========================
# 3) 绘制单张小提琴图
# =========================
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)

vp = ax.violinplot(
    violin_data,
    positions=np.arange(1, len(algos) + 1),
    widths=0.8,
    showmeans=False,
    showmedians=True,
    showextrema=True
)

# 小提琴着色
for i, body in enumerate(vp["bodies"]):
    algo = algos[i]
    body.set_facecolor(color_map.get(algo, "gray"))
    body.set_edgecolor("black")
    body.set_linewidth(0.5)
    body.set_alpha(0.9)

# 中位数、极值线样式
for partname in ["cbars", "cmins", "cmaxes", "cmedians"]:
    if partname in vp:
        vp[partname].set_edgecolor("black")
        vp[partname].set_linewidth(0.8)

# 叠加均值点
for i, algo in enumerate(algos, start=1):
    y = violin_data[i - 1]
    mean_val = np.mean(y)
    ax.scatter(i, mean_val, marker="o", s=22, color="black", zorder=3)

# =========================
# 4) 坐标轴与样式
# =========================
all_vals = np.concatenate(violin_data)
ymin = np.min(all_vals)
ymax = np.max(all_vals)

if ymax > ymin:
    gap = 0.05 * (ymax - ymin)
else:
    gap = max(abs(ymax) * 0.05, 1.0)

ax.set_xticks(np.arange(1, len(algos) + 1))
ax.set_xticklabels([display_names[a] for a in algos], fontsize=18)
ax.set_ylabel("Energy Consumption", fontsize=20)
ax.set_xlabel("Algorithm", fontsize=20)

ax.set_ylim(ymin - gap, ymax + gap)

# 科学计数法
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.yaxis.get_offset_text().set_size(18)

# 网格
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

# 边框与刻度
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
ax.tick_params(axis="both", labelsize=18, width=0.8)

plt.tight_layout()

# =========================
# 5) 导出
# =========================
fig.savefig(
    "all_algorithms_violin_from_excel.pdf",
    format="pdf",
    bbox_inches="tight",
    transparent=True
)

fig.savefig(
    "all_algorithms_violin_from_excel.png",
    format="png",
    bbox_inches="tight",
    transparent=True,
    dpi=300
)

plt.show()

# =========================
# 6) 打印样本量信息
# =========================
print("Sample size of each algorithm:")
for algo in algos:
    print(f"{algo}: {len(all_algo_values[algo])}")