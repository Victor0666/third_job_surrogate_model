import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.font_manager import FontProperties

symbol_font = FontProperties(family='DejaVu Sans')

# =========================
# 1. 数据
# =========================
scenarios = ['SS', 'SM', 'SL', 'MS', 'MM', 'ML', 'LS', 'LM', 'LL']
algorithms = ['MAHDRL', 'MARL', 'PD3QN', 'IRWS', 'ERTF_FIFS']

# Loose
data_loose = np.array([
    [0.996666667, 0.997333, 1.000000, 0.964000, 0.996667],  # SS
    [0.990000000, 0.992667, 1.000000, 0.999333, 0.991333],  # SM
    [0.990667000, 0.993333, 1.000000, 1.000000, 0.994667],  # SL
    [0.997333000, 1.000000, 1.000000, 0.900000, 1.000000],  # MS
    [0.995333000, 0.991333, 1.000000, 0.998000, 0.991333],  # MM
    [0.993333000, 0.994000, 1.000000, 1.000000, 0.994000],  # ML
    [0.998667000, 1.000000, 1.000000, 0.944000, 1.000000],  # LS
    [0.998000000, 0.998000, 1.000000, 0.936000, 0.999333],  # LM
    [0.993333000, 0.996000, 0.999333, 0.996000, 0.992667],  # LL
])

# Medium
data_medium = np.array([
    [0.994000, 0.996000, 1.000000, 0.960667, 0.994000],  # SS
    [0.975333, 0.981333, 1.000000, 0.999333, 0.982667],  # SM
    [0.982000, 0.985333, 1.000000, 1.000000, 0.984667],  # SL
    [0.998667, 1.000000, 0.999333, 0.875333, 0.999333],  # MS
    [0.987333, 0.988667, 1.000000, 0.992667, 0.982667],  # MM
    [0.974000, 0.986667, 1.000000, 0.999333, 0.977333],  # ML
    [0.996000, 1.000000, 1.000000, 0.932000, 1.000000],  # LS
    [0.990667, 0.996667, 1.000000, 0.918000, 0.996667],  # LM
    [0.984000, 0.985333333, 1.000000, 0.993333, 0.986667],  # LL
])

# Tight
data_tight = np.array([
    [0.986667, 0.992000, 1.000000, 0.937333, 0.988000],  # SS
    [0.973333, 0.969333, 0.999333, 0.997333, 0.974667],  # SM
    [0.972000, 0.974667, 1.000000, 1.000000, 0.973333],  # SL
    [0.988667, 0.999333, 0.999333, 0.854000, 0.998000],  # MS
    [0.970000, 0.982667, 1.000000, 0.992667, 0.976667],  # MM
    [0.966667, 0.966000, 1.000000, 0.998667, 0.965333],  # ML
    [0.980667, 0.999333, 1.000000, 0.922667, 1.000000],  # LS
    [0.991333, 0.998667, 0.998667, 0.892000, 0.993333],  # LM
    [0.970667, 0.972667, 0.998000, 0.990667, 0.980667],  # LL
])

# Loose
# data_loose = np.array([
#     [1.00, 1.00, 1.00, 0.96, 1.00],  # SS
#     [0.99, 0.99, 1.00, 1.00, 0.99],  # SM
#     [0.99, 0.99, 1.00, 1.00, 0.99],  # SL
#     [1.00, 1.00, 1.00, 0.90, 1.00],  # MS
#     [1.00, 0.99, 1.00, 1.00, 0.99],  # MM
#     [0.99, 0.99, 1.00, 1.00, 0.99],  # ML
#     [1.00, 1.00, 1.00, 0.94, 1.00],  # LS
#     [1.00, 1.00, 1.00, 0.94, 1.00],  # LM
#     [0.99, 1.00, 1.00, 1.00, 0.99],  # LL
# ])
#
# # Medium
# data_medium = np.array([
#     [0.99, 1.00, 1.00, 0.96, 0.99],  # SS
#     [0.98, 0.98, 1.00, 1.00, 0.98],  # SM
#     [0.98, 0.99, 1.00, 1.00, 0.98],  # SL
#     [1.00, 1.00, 1.00, 0.88, 1.00],  # MS
#     [0.99, 0.99, 1.00, 0.99, 0.98],  # MM
#     [0.97, 0.99, 1.00, 1.00, 0.98],  # ML
#     [1.00, 1.00, 1.00, 0.93, 1.00],  # LS
#     [0.99, 1.00, 1.00, 0.92, 1.00],  # LM
#     [0.98, 0.99, 1.00, 0.99, 0.99],  # LL
# ])
#
# # Tight
# data_tight = np.array([
#     [0.99, 0.99, 1.00, 0.94, 0.99],  # SS
#     [0.97, 0.97, 1.00, 1.00, 0.97],  # SM
#     [0.97, 0.97, 1.00, 1.00, 0.97],  # SL
#     [0.99, 1.00, 1.00, 0.85, 1.00],  # MS
#     [0.97, 0.98, 1.00, 0.99, 0.98],  # MM
#     [0.97, 0.97, 1.00, 1.00, 0.97],  # ML
#     [0.98, 1.00, 1.00, 0.92, 1.00],  # LS
#     [0.99, 1.00, 1.00, 0.89, 0.99],  # LM
#     [0.97, 0.97, 1.00, 0.99, 0.98],  # LL
# ])


# 左到右顺序：Loose / Medium / Tight
datasets = [data_loose, data_medium, data_tight]
dataset_short_labels = ['L', 'M', 'T']

# =========================
# 2. 阈值与状态编码
# 0 -> <95%
# 1 -> [95%, 97%)
# 2 -> >=97%
# =========================
thr95 = 0.95
thr97 = 0.97

def encode_state(data, thr95=0.95, thr97=0.97):
    state = np.zeros_like(data, dtype=int)
    state[(data >= thr95) & (data < thr97)] = 1
    state[data >= thr97] = 2
    return state

states = [encode_state(d, thr95, thr97) for d in datasets]

# =========================
# 3. 画图风格
# =========================
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

bg_color = '#FFFFFF'
color_fail = '#D8A2A2'      # <95%
color_95 = '#E7C98B'        # 95%~97%
color_97 = '#5B8C85'        # >=97%
shadow_color = '#D8D2C8'
text_dark = '#2F2F2F'
subtle_text = '#777777'

n_rows = len(scenarios)
n_cols = len(algorithms)

fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=300, facecolor=bg_color)
ax.set_facecolor(bg_color)

# =========================
# 4. 参数：每个大格子拆成 3 个小块
# =========================
group_w = 0.92
group_h = 0.68
rounding = 0.10

inner_pad_x = 0.04
inner_gap = 0.04
sub_w = (group_w - 2 * inner_pad_x - 2 * inner_gap) / 3
sub_h = 0.60

# 三个小块中心相对大格子中心的偏移
x_offsets = np.array([
    -group_w / 2 + inner_pad_x + sub_w / 2,
    -group_w / 2 + inner_pad_x + sub_w / 2 + (sub_w + inner_gap),
    -group_w / 2 + inner_pad_x + sub_w / 2 + 2 * (sub_w + inner_gap),
])

# =========================
# 5. 绘制三分格矩阵
# =========================
for i in range(n_rows):
    for j in range(n_cols):
        x = j
        y = i

        # 整组阴影
        shadow = FancyBboxPatch(
            (x - group_w / 2 + 0.03, y - group_h / 2 + 0.04),
            group_w, group_h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            linewidth=0,
            facecolor=shadow_color,
            alpha=0.28,
            zorder=1
        )
        ax.add_patch(shadow)

        # 三个子块：L / M / T
        for k in range(3):
            s = states[k][i, j]

            if s == 0:
                color = color_fail
                mark = '×'
            elif s == 1:
                color = color_95
                mark = '•'
            else:
                color = color_97
                mark = '✓'

            sub_x_center = x + x_offsets[k]

            box = FancyBboxPatch(
                (sub_x_center - sub_w / 2, y - sub_h / 2),
                sub_w, sub_h,
                boxstyle="round,pad=0.02,rounding_size=0.07",
                linewidth=0,
                facecolor=color,
                zorder=2
            )
            ax.add_patch(box)

            ax.text(
                sub_x_center, y, mark,
                ha='center', va='center',
                fontsize=10.5, fontweight='bold',
                color='white', zorder=3,
                fontproperties=symbol_font
            )


# =========================
# 6. 坐标文字
# =========================
# 顶部算法名
for j, alg in enumerate(algorithms):
    ax.text(
        j, -0.72, alg,
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color=text_dark
    )

# 每列顶部增加 L / M / T 提示
for j in range(n_cols):
    for k, lab in enumerate(dataset_short_labels):
        ax.text(
            j + x_offsets[k], -0.44, lab,
            ha='center', va='center',
            fontsize=10, color=text_dark
        )

# 左侧场景名
for i, sc in enumerate(scenarios):
    ax.text(
        -0.60, i,
        sc,
        ha='center', va='center',
        fontsize=15, fontweight='bold',
        color=text_dark
    )

# =========================
# 7. 图例
# =========================
legend_handles = [
    Patch(facecolor=color_fail, edgecolor='none', label='< 95%'),
    Patch(facecolor=color_95, edgecolor='none', label='95% ≤ SR < 97%'),
    Patch(facecolor=color_97, edgecolor='none', label='≥ 97%'),
]

ax.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    frameon=False,
    fontsize=15
)

# =========================
# 8. 范围与清理
# =========================
ax.set_xlim(-0.50, n_cols - 1 + 0.46)
ax.set_ylim(n_rows - 1 + 0.52, -0.88)

ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(rect=[0.005, 0.01, 0.997, 0.995])

# =========================
# 9. 保存为 PDF
# =========================
plt.savefig(
    'success_rate_LMT.pdf',
    format='pdf',
    bbox_inches='tight',
    pad_inches=0.0
)
plt.show()

# # =========================
# # 6. 坐标文字
# # =========================
# # 顶部算法名
# for j, alg in enumerate(algorithms):
#     ax.text(
#         j, -0.82, alg,
#         ha='center', va='center',
#         fontsize=12, fontweight='bold',
#         color=text_dark
#     )
#
# # 每列顶部增加 L / M / T 提示
# for j in range(n_cols):
#     for k, lab in enumerate(dataset_short_labels):
#         ax.text(
#             j + x_offsets[k], -0.43, lab,
#             ha='center', va='center',
#             fontsize=10, color=text_dark
#         )
#
# # 左侧场景名
# for i, sc in enumerate(scenarios):
#     ax.text(
#         -0.60, i,
#         sc,
#         ha='center', va='center',
#         fontsize=15, fontweight='bold',
#         color=text_dark
#     )
#
# # =========================
# # 7. 图例
# # =========================
# legend_handles = [
#     Patch(facecolor=color_fail, edgecolor='none', label='< 95%'),
#     Patch(facecolor=color_95, edgecolor='none', label='95% ≤ SR < 97%'),
#     Patch(facecolor=color_97, edgecolor='none', label='≥ 97%'),
# ]
#
# ax.legend(
#     handles=legend_handles,
#     loc='upper center',
#     bbox_to_anchor=(0.5, 0.03),
#     ncol=3,
#     frameon=False,
#     fontsize=15
# )
#
# # =========================
# # 8. 范围与清理
# # =========================
# ax.set_xlim(-0.88, n_cols - 1 + 0.58)
# ax.set_ylim(n_rows - 1 + 0.72, -1.02)
#
# ax.set_xticks([])
# ax.set_yticks([])
#
# for spine in ax.spines.values():
#     spine.set_visible(False)
#
# plt.tight_layout(rect=[0.01, 0.03, 0.995, 0.985])
#
# # =========================
# # 9. 保存为 PDF
# # =========================
# plt.savefig(
#     'success_rate_LMT.pdf',
#     format='pdf',
#     bbox_inches='tight',
#     pad_inches=0.002
# )
# plt.show()