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
# 1) 原始数据（来自你图里的表）
# =========================
data = {
    "Scenario": ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"],
    "MAHDRL_NTA": [311212.70, 316388.92, 319264.57, 503256.22, 501359.02, 514882.00, 804098.75, 817504.11, 821587.99],
    "MAHDRL_NHA": [316986.50, 350653.78, 409328.02, 504090.91, 522260.71, 561380.64, 807702.93, 828174.68, 873615.97],
    "MAHDRL_NVA": [312615.20, 319597.03, 328552.24, 505952.40, 503927.59, 518864.80, 822712.06, 818594.86, 814082.97],
    "MAHDRL_TL":  [310801.14, 329310.14, 385296.73, 504000.02, 506778.73, 530096.17, 811767.03, 826019.29, 851225.32],
    "MAHDRL":     [310665.03, 312465.34, 319776.28, 502685.12, 498824.00, 512155.00, 803683.80, 813018.30, 815547.22],
}
df = pd.DataFrame(data).set_index("Scenario")

# =========================
# 2) 计算 RPD
#    RPD = (当前算法数值 - 该scenario下所有算法最小值) / 当前算法数值
# =========================
row_min = df.min(axis=1)
print(row_min)
rpd = (df.sub(row_min, axis=0)).div(df, axis=0)

# =========================
# 3) 分组柱状图（每个 scenario 5 个柱子）
# =========================
scenarios = rpd.index.tolist()
algos = rpd.columns.tolist()

x = np.arange(len(scenarios))
n_algos = len(algos)
bar_w = 0.16

fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=160)

# 颜色映射
color_map = {
    "MAHDRL_NTA": "#c2d0ea",
    "MAHDRL_NHA": "#0070b8",
    "MAHDRL_NVA": "#00b3b0",
    "MAHDRL_TL":  "#d40d8c",
    "MAHDRL":     "#6A5ACD",
}

# ✅ 关键：给 0 值一个“最小可见高度”（仅用于显示，不改变真实数值含义）
ymax = float(rpd.to_numpy().max())
eps = 0.001 * ymax   # 0柱子显示高度（可调：0.01~0.03）
# eps = 0

for i, algo in enumerate(algos):
    offset = (i - (n_algos - 1) / 2) * bar_w

    vals = rpd[algo].values
    zero_mask = (vals == 0)

    vals_plot = vals.copy()
    vals_plot[zero_mask] = eps  # 仅显示时替换 0

    bars = ax.bar(
        x + offset,
        vals_plot,
        width=bar_w,
        label=algo,
        color=color_map.get(algo, "gray"),
        edgecolor="black",
        linewidth=0.4,
    )

    # 对原本为 0 的柱子：加纹理+半透明，表示这是“视觉占位”
    for b, is_zero in zip(bars, zero_mask):
        if is_zero:
            b.set_hatch("//")
            b.set_alpha(0.7)
            # 可选：在占位柱上标一个“0”
            # ax.text(b.get_x() + b.get_width()/2, eps, "0",
            #         ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=20)
ax.set_ylabel("RPD", fontsize=20)
ax.set_xlabel("Scenario", fontsize=20)
ax.yaxis.get_offset_text().set_size(20)

# 科学计数法显示
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# ✅ 柱子底部不贴 x 轴：y轴下界留一点空隙
gap = 0.06 * ymax
ax.set_ylim(-gap, ymax * 1.15)

ax.legend(ncol=2, fontsize=12, frameon=True)
ax.tick_params(axis="both", labelsize=20)
plt.tight_layout()

# 导出 PDF
fig.savefig(
    "ablation_boxPlot.pdf",
    format="pdf",
    bbox_inches="tight",
    transparent=True,
)

plt.show()