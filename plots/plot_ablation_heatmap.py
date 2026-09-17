# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm

# =========================
# 0) 全局字体与 PDF 嵌入
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ===== 全局字号 =====
mpl.rcParams["font.size"] = 22
mpl.rcParams["axes.labelsize"] = 24
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["legend.fontsize"] = 20
mpl.rcParams["legend.title_fontsize"] = 20

# =========================
# 1) 三种 deadline 条件的 Excel 路径
#    改成你的实际路径
# =========================
excel_paths = {
    "Loose":  r"excel/ab_L.xlsx",
    "Medium": r"excel/ab_M.xlsx",
    "Tight":  r"excel/ab_T.xlsx",
}

condition_order = ["Loose", "Medium", "Tight"]

# 9 个 scenario / sheet 名
scenarios = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]

# 算法列名（必须与 Excel 表头一致）
algos = ["MAHDRL_NTA", "MAHDRL_NHA", "MAHDRL_NVA", "MAHDRL_TL", "MAHDRL"]
full_algo = "MAHDRL"
ablation_algos = ["MAHDRL_NTA", "MAHDRL_NHA", "MAHDRL_NVA", "MAHDRL_TL"]

# 算法显示名称
display_names = {
    "MAHDRL_NTA": "MAHDRL_NTA",
    "MAHDRL_NHA": "MAHDRL_NHA",
    "MAHDRL_NVA": "MAHDRL_NVA",
    "MAHDRL_TL":  "MAHDRL_TL",
    "MAHDRL":     "MAHDRL",
}

# 算法颜色
color_map = {
    "MAHDRL_NTA": "#c2d0ea",
    "MAHDRL_NHA": "#0070b8",
    "MAHDRL_NVA": "#00b3b0",
    "MAHDRL_TL":  "#d40d8c",
    "MAHDRL":     "#6A5ACD",
}

# 输出目录
out_dir = "fig_ablation_new"
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
# 3) 工具函数
# =========================
task_levels = ["S", "M", "L"]
resource_levels = ["S", "M", "L"]

def scenario_to_matrix_index(scenario):
    """
    scenario: 'SS','SM',...,'LL'
    行：任务规模 S/M/L
    列：资源规模 S/M/L
    """
    task_char = scenario[0]
    res_char = scenario[1]
    i = task_levels.index(task_char)
    j = resource_levels.index(res_char)
    return i, j

def build_degradation_matrix(cond, ab_algo):
    """
    返回 3x3 矩阵，每个元素是相对完整模型MAHDRL的能耗退化百分比
    Degradation(%) = (mean(ablation) - mean(full)) / mean(full) * 100
    """
    mat = np.full((3, 3), np.nan, dtype=float)

    for scenario in scenarios:
        df = raw_data_dict[cond][scenario]

        ab_vals = df[ab_algo].dropna().values.astype(float)
        full_vals = df[full_algo].dropna().values.astype(float)

        if len(ab_vals) == 0 or len(full_vals) == 0:
            continue

        ab_mean = np.mean(ab_vals)
        full_mean = np.mean(full_vals)

        if np.isclose(full_mean, 0):
            degr = np.nan
        else:
            degr = (ab_mean - full_mean) / full_mean * 100.0

        r, c = scenario_to_matrix_index(scenario)
        mat[r, c] = degr

    return mat

# =========================
# 4) 生成正文主图：
#    27场景相对能耗退化热力图
#    布局：4行(消融) × 3列(ddl)
#    每个子图内部：3×3场景矩阵
# =========================
heatmap_data = {ab: {} for ab in ablation_algos}
all_heat_values = []

for ab in ablation_algos:
    for cond in condition_order:
        mat = build_degradation_matrix(cond, ab)
        heatmap_data[ab][cond] = mat
        valid_vals = mat[~np.isnan(mat)]
        if valid_vals.size > 0:
            all_heat_values.extend(valid_vals.tolist())

# all_heat_values = np.array(all_heat_values, dtype=float)
#
# if all_heat_values.size == 0:
#     raise ValueError("No valid values found for heatmap plotting.")
#
# # 以0为中心
# vmin = np.min(all_heat_values)
# vmax = np.max(all_heat_values)
#
# vmin = min(vmin, 0.0)
# vmax = max(vmax, 0.0)
#
# if np.isclose(vmin, vmax):
#     vmin -= 1.0
#     vmax += 1.0
#
# norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

all_heat_values = np.array(all_heat_values, dtype=float)

if all_heat_values.size == 0:
    raise ValueError("No valid values found for heatmap plotting.")

# =========================
# 关键修改：使用关于 0 对称的范围
# =========================
# max_abs = np.nanmax(np.abs(all_heat_values))
#
# if np.isclose(max_abs, 0.0):
#     max_abs = 1.0
#
# vmin = -max_abs
# vmax =  max_abs
#
# norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

# =========================
# 使用关于 0 对称、且刻度规整的 colorbar 范围
# =========================
max_abs = np.nanmax(np.abs(all_heat_values))

if np.isclose(max_abs, 0.0):
    max_abs = 1.0

# 向上取整到 5 的倍数
tick_base = 5.0
max_abs_round = np.ceil(max_abs / tick_base) * tick_base

vmin = -max_abs_round
vmax =  max_abs_round

norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

fig, axes = plt.subplots(
    nrows=len(ablation_algos),
    ncols=len(condition_order),
    figsize=(12.2, 12.8),
    dpi=300
)

cmap = plt.get_cmap("RdYlBu_r")

for row, ab in enumerate(ablation_algos):
    for col, cond in enumerate(condition_order):
        ax = axes[row, col]
        mat = heatmap_data[ab][cond]

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")

        # 列标题
        if row == 0:
            ax.set_title(cond, fontsize=20, pad=10)

        # 行标签
        if col == 0:
            ax.set_ylabel(display_names[ab], fontsize=20, labelpad=14)

        # 坐标轴
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(resource_levels, fontsize=18)
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(task_levels, fontsize=18)

        if row == len(ablation_algos) - 1:
            ax.set_xlabel("Resource Scale", fontsize=20)

        # 网格线
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)

        # 格子数值
        for i in range(3):
            for j in range(3):
                val = mat[i, j]
                if np.isnan(val):
                    text = "NA"
                else:
                    text = f"{val:+.2f}"
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    fontsize=16,
                    color="black"
                )

# ========= 关键修改：右侧单独留空间放 colorbar =========
fig.subplots_adjust(
    left=0.08,
    right=0.88,
    top=0.96,
    bottom=0.07,
    wspace=0.01,  # 列与列之间的水平间距，越小越紧凑
    hspace=0.15  # 行与行之间的垂直间距，越小越紧凑
)

# 单独创建右侧颜色条区域
# [left, bottom, width, height]
cax = fig.add_axes([0.90, 0.26, 0.018, 0.48])

# cbar = fig.colorbar(im, cax=cax)
# cbar.set_label("Relative Energy Degradation (%)", fontsize=20)
# cbar.ax.tick_params(labelsize=20)

# =========================
# 关键修改：手动指定 colorbar 刻度
# =========================
# ticks = np.linspace(vmin, vmax, 11)   # 例如 7 个刻度：负-零-正
#
# cbar = fig.colorbar(im, cax=cax, ticks=ticks)
# cbar.set_label("Relative Energy Degradation (%)", fontsize=20)
# cbar.ax.tick_params(labelsize=20)
# 可选：强制保留两位小数
# cbar.ax.set_yticklabels([f"{t:.0f}" for t in ticks])


ticks = np.arange(vmin, vmax + tick_base, tick_base)

cbar = fig.colorbar(im, cax=cax, ticks=ticks)
cbar.set_label("Relative Energy Degradation (%)", fontsize=20)
cbar.ax.tick_params(labelsize=20)
cbar.ax.set_yticklabels([f"{t:.0f}" for t in ticks])





heatmap_pdf = os.path.join(out_dir, "main_relative_degradation_heatmap.pdf")
heatmap_png = os.path.join(out_dir, "main_relative_degradation_heatmap.png")

fig.savefig(heatmap_pdf, format="pdf", bbox_inches="tight", transparent=True)
fig.savefig(heatmap_png, format="png", bbox_inches="tight", transparent=True, dpi=400)
plt.close(fig)

# =========================
# 5) 生成补充图：
#    每个ddl一张箱线图
#    横轴：9个scenario
#    每个scenario下：5个算法的30次重复箱线图
# =========================
for cond in condition_order:
    fig, ax = plt.subplots(figsize=(18, 6.2), dpi=250)

    n_scen = len(scenarios)
    n_alg = len(algos)

    group_centers = np.arange(n_scen) * 1.8 + 1.0
    offsets = np.linspace(-0.48, 0.48, n_alg)
    box_width = 0.18

    box_data = []
    positions = []
    color_list = []
    all_vals = []

    for s_idx, scenario in enumerate(scenarios):
        df = raw_data_dict[cond][scenario]

        for a_idx, algo in enumerate(algos):
            vals = df[algo].dropna().values.astype(float)
            if len(vals) == 0:
                continue

            box_data.append(vals)
            positions.append(group_centers[s_idx] + offsets[a_idx])
            color_list.append(color_map[algo])
            all_vals.extend(vals.tolist())

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

    for patch, fc in zip(bp["boxes"], color_list):
        patch.set_facecolor(fc)
        patch.set_alpha(0.90)

    # x轴
    ax.set_xticks(group_centers)
    ax.set_xticklabels(scenarios, fontsize=20)
    ax.set_xlabel("Scenario", fontsize=20)
    ax.set_ylabel("Energy Consumption", fontsize=20)

    # y轴科学计数法
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_size(13)

    # y轴自适应
    all_vals = np.array(all_vals, dtype=float)
    all_vals = all_vals[~np.isnan(all_vals)]
    y_min = np.min(all_vals)
    y_max = np.max(all_vals)

    if y_max > y_min:
        y_gap = 0.08 * (y_max - y_min)
    else:
        y_gap = max(abs(y_max) * 0.05, 1.0)

    ax.set_ylim(y_min - y_gap, y_max + y_gap)

    # 网格
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

    # 场景分隔线
    for i in range(len(group_centers) - 1):
        x_sep = (group_centers[i] + group_centers[i + 1]) / 2.0
        ax.axvline(x=x_sep, color="gray", linewidth=0.45, alpha=0.18)

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.tick_params(axis="both", labelsize=13, width=0.8)

    # 图例
    legend_handles = [
        Patch(facecolor=color_map[a], edgecolor="black", label=display_names[a])
        for a in algos
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=5,
        frameon=False,
        fontsize=20,
        handlelength=1.4,
        columnspacing=0.9,
        handletextpad=0.4
    )

    plt.tight_layout()

    out_pdf = os.path.join(out_dir, f"supp_boxplot_{cond}.pdf")
    out_png = os.path.join(out_dir, f"supp_boxplot_{cond}.png")

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", transparent=True)
    fig.savefig(out_png, format="png", bbox_inches="tight", transparent=True, dpi=400)
    plt.close(fig)

print("Done.")
print(f"Outputs saved to: {out_dir}")