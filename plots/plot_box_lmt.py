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
mpl.rcParams["font.size"] = 20
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 18
mpl.rcParams["legend.title_fontsize"] = 18

# =========================
# 1) 三种 deadline 条件的 Excel 路径
#    这里改成你的实际路径
# =========================
excel_paths = {
    "Loose":  r"excel/mix_L.xlsx",
    "Medium": r"excel/mix_M.xlsx",
    "Tight":  r"excel/mix_T.xlsx",
}

# 9 个 scenario / sheet 名
scenarios = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]

# 算法列名（必须与 Excel 表头一致）
algos = ["ERTF_FIFS", "PD3QN", "MARL", "IRWS", "MAHDRL"]

# 横坐标显示名称
display_names = {
    "ERTF_FIFS": "ERTF-FIFS",
    "PD3QN": "PD3QN",
    "MARL": "MARL",
    "IRWS": "IRWS",
    "MAHDRL": "MAHDRL",
}

# 算法颜色：保留你原来的配色
color_map = {
    "ERTF_FIFS": "#c2d0ea",
    "PD3QN":     "#0070b8",
    "MARL":      "#00b3b0",
    "IRWS":      "#d40d8c",
    "MAHDRL":    "#6A5ACD",
}

# 用 hatch 区分三种 deadline 条件
# 这样可以保留“算法颜色”不变，同时又能看出 Loose / Medium / Tight
hatch_map = {
    "Loose": "",
    "Medium": "//",
    "Tight": "xx",
}

condition_order = ["Loose", "Medium", "Tight"]

# 输出目录
out_dir = "fig_box_lmt"
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

        # 取前30行作为30个seed原始数据
        raw_df = df_sheet.iloc[0:30].copy()

        raw_data_dict[cond][scenario] = raw_df

# =========================
# 3) 逐个 scenario 画“合并后的”箱线图
#    每张图：一个 scenario
#    横轴：算法
#    每个算法下：Loose / Medium / Tight 三个箱线图
# =========================
for scenario in scenarios:
    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=180)

    # 每个算法一组，组中心位置
    # group_centers = np.arange(len(algos)) * 2.2 + 1.5
    group_centers = np.arange(len(algos)) * 1.3 + 1.0

    # 每组内三个箱线图的偏移
    offsets = [-0.42, 0.0, 0.42]
    box_width = 0.30

    box_data = []
    positions = []
    meta_info = []
    all_vals = []

    # 组织数据：按“算法组” -> “三种deadline条件”
    for i, algo in enumerate(algos):
        for j, cond in enumerate(condition_order):
            vals = raw_data_dict[cond][scenario][algo].dropna().values.astype(float)
            if len(vals) == 0:
                continue

            box_data.append(vals)
            positions.append(group_centers[i] + offsets[j])
            meta_info.append((algo, cond))
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
    left_pad = 0.28
    right_pad = 0.28
    ax.set_xlim(min(positions) - left_pad, max(positions) + right_pad)

    # 箱体着色 + hatch
    for patch, (algo, cond) in zip(bp["boxes"], meta_info):
        patch.set_facecolor(color_map.get(algo, "lightgray"))
        patch.set_alpha(0.9)
        patch.set_hatch(hatch_map[cond])

    # =========================
    # y轴自适应：使用该 scenario 在三种条件下的所有值
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
    ax.set_xticks(group_centers)
    ax.set_xticklabels([display_names[a] for a in algos], fontsize=22)
    ax.set_ylabel("Energy Consumption", fontsize=22)
    ax.set_xlabel("Algorithm", fontsize=22)
    # ax.set_title(f"({scenario})", fontsize=18, pad=8)

    # 科学计数法
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_size(18)

    # 网格
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # 组间淡分隔线
    for i in range(len(group_centers) - 1):
        x_sep = (group_centers[i] + group_centers[i + 1]) / 2.0
        ax.axvline(x=x_sep, color="gray", linewidth=0.4, alpha=0.18)

    # 边框与刻度
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=16, width=0.8)

    # =========================
    # 图例：只标识 deadline 条件
    # 算法本身由横轴分组体现
    # =========================
    # legend_handles = [
    #     Patch(facecolor="white", edgecolor="black", hatch=hatch_map[c], label=c)
    #     for c in condition_order
    # ]
    # ax.legend(
    #     handles=legend_handles,
    #     title="Deadline",
    #     fontsize=18,
    #     title_fontsize=20,
    #     frameon=True,
    #     loc="upper right"
    # )
    # =========================
    # 不在主图中显示图例
    # =========================
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch_map[c], label=c)
        for c in condition_order
    ]

    plt.tight_layout()

    out_path_pdf = os.path.join(out_dir, f"{scenario}_LMT.pdf")
    out_path_png = os.path.join(out_dir, f"{scenario}_LMT.png")

    fig.savefig(out_path_pdf, format="pdf", bbox_inches="tight", transparent=True)
    fig.savefig(out_path_png, format="png", bbox_inches="tight", transparent=True, dpi=300)

    plt.close(fig)

# =========================
# 4) 单独生成图例图片（3个图例横排）
# =========================
fig_leg, ax_leg = plt.subplots(figsize=(5.5, 0.65), dpi=300)
ax_leg.axis("off")

# 让坐标轴区域尽量铺满整张图
ax_leg.set_position([0.0, 0.0, 1.0, 1.0])

legend_handles = [
    Patch(facecolor="white", edgecolor="black", hatch=hatch_map[c], label=c)
    for c in condition_order
]

leg = ax_leg.legend(
    handles=legend_handles,
    loc="center",
    bbox_to_anchor=(0.5, 0.5),
    ncol=3,
    frameon=False,
    fontsize=20,
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

print("All merged scenario boxplots and legend have been saved.")