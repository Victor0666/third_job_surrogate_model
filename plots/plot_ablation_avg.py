import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# 0) 全局字体（Times New Roman）与PDF嵌入
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# =========================
# 1) 你的RPD数据（单行）
# =========================
rpd = {
    "MAHDRL_NTA": 0.004867532,
    "MAHDRL_NHA": 0.064276167,
    "MAHDRL_NVA": 0.013253624,
    "MAHDRL_TL":  0.038601799,
    "MAHDRL":     0.000377291,
}

# 颜色映射（沿用你之前的配色）
# color_map = {
    # "MAHDRL_NTA": "#49c2d9",
    # "MAHDRL_NHA": "#a1d8e8",
    # "MAHDRL_NVA": "#67a583",
    # "MAHDRL_TL":  "#a2c986",
    # "MAHDRL":     "#6A5ACD",

    # "MAHDRL_NTA": "#A8D19D",
    # "MAHDRL_NHA": "#88CBAE",
    # "MAHDRL_NVA": "#6FBBA4",
    # "MAHDRL_TL":  "#55A79B",
    # "MAHDRL":     "#6A5ACD",

#     "MAHDRL_NTA": "#E2CFC2",
#     "MAHDRL_NHA": "#A9D6E5",
#     "MAHDRL_NVA": "#F7E9A2",
#     "MAHDRL_TL": "#FFB5A1",
#     "MAHDRL": "#6A5ACD",
# }

color_map = {
    "MAHDRL_NTA": "#c2d0ea",
    "MAHDRL_NHA": "#0070b8",
    "MAHDRL_NVA": "#00b3b0",
    "MAHDRL_TL":  "#d40d8c",
    "MAHDRL":     "#6A5ACD",
}

algos = list(rpd.keys())
vals = np.array([rpd[a] for a in algos], dtype=float)

x = np.arange(len(algos))

fig, ax = plt.subplots(figsize=(6, 2.6), dpi=160)

ax.bar(
    x, vals,
    width=0.6,
    color=[color_map.get(a, "gray") for a in algos],
    edgecolor="black",
    linewidth=0.5,
)

ax.set_xticks(x)
ax.set_xticklabels(algos, rotation=0)
# ax.set_xticklabels(algos, rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylabel("RPD")
ax.set_xlabel("Algorithm")

# 科学计数法（看起来更像论文图）
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# 让柱子底部不贴x轴
ymax = float(vals.max())
gap = 0.08 * ymax if ymax > 0 else 0.01
ax.set_ylim(-gap, ymax * 1.15)

plt.tight_layout()

# 导出PDF（可改路径）
fig.savefig("ablation_boxPlot_ave.pdf", format="pdf", bbox_inches="tight", transparent=True)

plt.show()