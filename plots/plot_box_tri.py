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
excel_path = r"excel/mix_T.xlsx"   # <<< 改成你的 Excel 文件路径

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

# 配色
color_map = {
    "ERTF_FIFS": "#c2d0ea",
    "PD3QN":     "#0070b8",
    "MARL":      "#00b3b0",
    "IRWS":      "#d40d8c",
    "MAHDRL":    "#6A5ACD",
}

# 输出目录
out_dir = "fig_box_tri"
os.makedirs(out_dir, exist_ok=True)

# =========================
# 2) 读取 9 个 sheet 数据
#    假设：
#    - 第1行为表头
#    - 第2~31行为30个seed原始数据
# =========================
raw_data_dict = {}

for scenario in scenarios:
    df_sheet = pd.read_excel(excel_path, sheet_name=scenario)

    # 只保留目标算法列，并强制转为数值
    df_sheet = df_sheet[algos].apply(pd.to_numeric, errors="coerce")

    # 第2~31行：30个种子原始数据
    raw_df = df_sheet.iloc[0:30].copy()

    raw_data_dict[scenario] = raw_df

# =========================
# 3) 逐个 scenario 画原始数据箱线图（9张）
#    每张图自适应 y 轴
# =========================
for scenario in scenarios:
    df_raw = raw_data_dict[scenario]

    # 每个算法对应一组30个原始值
    box_data = [df_raw[algo].dropna().values for algo in algos]

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)

    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        widths=0.55,
        showmeans=False,     # 不显示均值圆点
        showfliers=False,    # 不显示异常值圆点
        boxprops=dict(linewidth=0.9, color="black"),
        whiskerprops=dict(linewidth=0.9, color="black"),
        capprops=dict(linewidth=0.9, color="black"),
        medianprops=dict(linewidth=1.2, color="black")
    )

    # 箱体着色
    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(color_map.get(algo, "lightgray"))
        patch.set_alpha(0.9)

    # ===== 每张图单独计算 y 轴范围 =====
    scenario_vals = df_raw[algos].to_numpy().flatten().astype(float)
    scenario_vals = scenario_vals[~np.isnan(scenario_vals)]

    scenario_ymin = np.min(scenario_vals)
    scenario_ymax = np.max(scenario_vals)

    if scenario_ymax > scenario_ymin:
        scenario_gap = 0.05 * (scenario_ymax - scenario_ymin)
        text_offset = 0.02 * (scenario_ymax - scenario_ymin)
    else:
        scenario_gap = max(abs(scenario_ymax) * 0.05, 1.0)
        text_offset = scenario_gap * 0.4

    # 标均值数字（只保留数值文字，不加圆点）
    # for i, algo in enumerate(algos, start=1):
    #     y = df_raw[algo].dropna().values
    #     mean_val = np.mean(y)
    #     ax.text(
    #         i,
    #         mean_val + text_offset,
    #         f"{mean_val:.2f}",
    #         ha="center",
    #         va="bottom",
    #         fontsize=15
    #     )

    ax.set_xticks(np.arange(1, len(algos) + 1))
    ax.set_xticklabels([display_names[a] for a in algos], fontsize=20)
    ax.set_ylabel("Energy Consumption", fontsize=20)
    ax.set_xlabel("Algorithm", fontsize=20)
    # ax.set_title(f"Scenario: {scenario}", fontsize=15)

    # 自适应 y 轴
    ax.set_ylim(scenario_ymin - scenario_gap, scenario_ymax + scenario_gap)

    # 科学计数法
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_size(20)

    # 网格
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # 边框与刻度
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=18, width=0.8)

    plt.tight_layout()

    # model = "L"
    # model = "M"
    model = "T"

    out_path_pdf = os.path.join(out_dir, f"{scenario}_{model}.pdf")
    out_path_png = os.path.join(out_dir, f"{scenario}_{model}.png")

    fig.savefig(out_path_pdf, format="pdf", bbox_inches="tight", transparent=True)
    fig.savefig(out_path_png, format="png", bbox_inches="tight", transparent=True, dpi=300)

    plt.close(fig)

print("All scenario boxplots have been saved.")