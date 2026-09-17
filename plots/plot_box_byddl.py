# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

# =========================
# 0) 全局字体与 PDF 嵌入
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ===== 全局字号 =====
mpl.rcParams["font.size"] = 28
mpl.rcParams["axes.labelsize"] = 28
mpl.rcParams["axes.titlesize"] = 28
mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25
mpl.rcParams["legend.fontsize"] = 25
mpl.rcParams["legend.title_fontsize"] = 25

# =========================
# 1) 三种 deadline 条件的 Excel 路径
# =========================
excel_paths = {
    "Loose":  r"excel/mix_L.xlsx",
    "Medium": r"excel/mix_M.xlsx",
    "Tight":  r"excel/mix_T.xlsx",
}

# 9 个 scenario / sheet 名
scenarios = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]

# Excel 中的算法列名（保持不变，必须与表头一致）
algos = ["ERTF_FIFS", "PD3QN", "MARL", "IRWS", "MAHDRL"]

# 显示名称：这里只改显示，不改 Excel 列名
display_names = {
    "ERTF_FIFS": "ERTF-H-FIFS",
    "PD3QN": "PD3QN",
    "MARL": "MARL",
    "IRWS": "IRWS",
    "MAHDRL": "MAHDRL",
}

# 算法颜色
color_map = {
    "ERTF_FIFS": "#c2d0ea",
    "PD3QN":     "#0070b8",
    "MARL":      "#00b3b0",
    "IRWS":      "#d40d8c",
    "MAHDRL":    "#6A5ACD",
}

condition_order = ["Loose", "Medium", "Tight"]

# 输出目录
out_dir = "fig_box_byddl"
os.makedirs(out_dir, exist_ok=True)

# =========================
# 2) 读取三种条件下的 9 个 sheet 数据
#    假设：
#    - 第1行为表头
#    - 第2~31行为30个seed原始数据
# =========================
raw_data_dict = {cond: {} for cond in condition_order}

for cond in condition_order:
    excel_path = excel_paths[cond]

    for scenario in scenarios:
        df_sheet = pd.read_excel(excel_path, sheet_name=scenario)

        # 只保留目标算法列，并强制转为数值
        df_sheet = df_sheet[algos].apply(pd.to_numeric, errors="coerce")

        # 前30行作为30个seed原始数据
        raw_df = df_sheet.iloc[0:30].copy()

        raw_data_dict[cond][scenario] = raw_df

# =========================
# 3) 逐个 scenario 画箱线图
#    每张图：一个 scenario
#    横轴：Loose / Medium / Tight
#    每个松紧度下：5个算法并排箱线图
# =========================
for scenario in scenarios:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)

    # 三个 ddl 条件组中心（缩小组间距）
    condition_centers = np.arange(len(condition_order)) * 1.45 + 1.0

    # 每组内 5 个算法的偏移
    offsets = [-0.44, -0.22, 0.0, 0.22, 0.44]
    box_width = 0.15

    box_data = []
    positions = []
    meta_info = []
    all_vals = []

    # 组织数据：按“ddl条件组” -> “5个算法”
    for i, cond in enumerate(condition_order):
        for j, algo in enumerate(algos):
            vals = raw_data_dict[cond][scenario][algo].dropna().values.astype(float)
            if len(vals) == 0:
                continue

            box_data.append(vals)
            positions.append(condition_centers[i] + offsets[j])
            meta_info.append((cond, algo))
            all_vals.extend(vals.tolist())

    # 画箱线图
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        boxprops=dict(linewidth=0.9, color="black"),
        whiskerprops=dict(linewidth=0.9, color="black"),
        capprops=dict(linewidth=0.9, color="black"),
        medianprops=dict(linewidth=1.2, color="black")
    )

    # 手动缩小左右留白
    left_pad = 0.25
    right_pad = 0.25
    ax.set_xlim(min(positions) - left_pad, max(positions) + right_pad)

    # 箱体着色：按算法着色
    for patch, (_, algo) in zip(bp["boxes"], meta_info):
        patch.set_facecolor(color_map.get(algo, "lightgray"))
        patch.set_alpha(0.9)

    # =========================
    # y轴自适应：使用该 scenario 在三种条件下、所有算法的所有值
    # =========================
    all_vals = np.array(all_vals, dtype=float)
    all_vals = all_vals[~np.isnan(all_vals)]

    y_min = np.min(all_vals)
    y_max = np.max(all_vals)

    if y_max > y_min:
        y_gap = 0.08 * (y_max - y_min)
    else:
        y_gap = max(abs(y_max) * 0.05, 1.0)

    ax.set_ylim(y_min - y_gap, y_max + y_gap)

    # =========================
    # 坐标轴与标签
    # =========================
    ax.set_xticks(condition_centers)
    ax.set_xticklabels(condition_order, fontsize=28)
    ax.set_ylabel("Energy Consumption", fontsize=22)
    ax.set_xlabel("Deadline Tightness", fontsize=22)

    # 科学计数法
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_size(22)

    # 网格
    # ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # 组间淡分隔线
    for i in range(len(condition_centers) - 1):
        x_sep = (condition_centers[i] + condition_centers[i + 1]) / 2.0
        ax.axvline(x=x_sep, color="gray", linewidth=0.4, alpha=0.18)

    # 边框与刻度
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=22, width=0.8)

    # 主图中不显示图例，只生成图例句柄
    legend_handles = [
        Patch(facecolor=color_map[a], edgecolor="black", label=display_names[a])
        for a in algos
    ]

    plt.tight_layout()

    out_path_pdf = os.path.join(out_dir, f"{scenario}_LMT.pdf")
    out_path_png = os.path.join(out_dir, f"{scenario}_LMT.png")

    fig.savefig(out_path_pdf, format="pdf", bbox_inches="tight", transparent=True)
    fig.savefig(out_path_png, format="png", bbox_inches="tight", transparent=True, dpi=300)

    plt.close(fig)

# =========================
# 4) 单独生成图例图片（算法图例横排）
# =========================
fig_leg, ax_leg = plt.subplots(figsize=(9.0, 0.75), dpi=300)
ax_leg.axis("off")
ax_leg.set_position([0.0, 0.0, 1.0, 1.0])

legend_handles = [
    Patch(facecolor=color_map[a], edgecolor="black", label=display_names[a])
    for a in algos
]

ax_leg.legend(
    handles=legend_handles,
    loc="center",
    bbox_to_anchor=(0.5, 0.5),
    ncol=5,
    frameon=False,
    fontsize=25,
    handlelength=1.6,
    handleheight=0.9,
    columnspacing=0.9,
    handletextpad=0.35,
    borderaxespad=0.0,
    labelspacing=0.2
)

legend_pdf = os.path.join(out_dir, "legend_LMT.pdf")
legend_png = os.path.join(out_dir, "legend_LMT.png")

fig_leg.savefig(
    legend_pdf,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0,
    transparent=True
)
fig_leg.savefig(
    legend_png,
    format="png",
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
    dpi=300
)

plt.close(fig_leg)

print("All scenario boxplots and algorithm legend have been saved.")