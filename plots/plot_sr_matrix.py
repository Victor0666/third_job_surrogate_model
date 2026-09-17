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

#Loose
data = np.array([
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


#Medium
# data = np.array([
#     [0.994000, 0.996000, 1.000000, 0.960667, 0.994000],  # SS
#     [0.975333, 0.981333, 1.000000, 0.999333, 0.982667],  # SM
#     [0.982000, 0.985333, 1.000000, 1.000000, 0.984667],  # SL
#     [0.998667, 1.000000, 0.999333, 0.875333, 0.999333],  # MS
#     [0.987333, 0.988667, 1.000000, 0.992667, 0.982667],  # MM
#     [0.974000, 0.986667, 1.000000, 0.999333, 0.977333],  # ML
#     [0.996000, 1.000000, 1.000000, 0.932000, 1.000000],  # LS
#     [0.990667, 0.996667, 1.000000, 0.918000, 0.996667],  # LM
#     [0.984000, 0.985333333, 1.000000, 0.993333, 0.986667],  # LL
# ])

#Tight
# data = np.array([
#     [0.986667, 0.992000, 1.000000, 0.937333, 0.988000],  # SS
#     [0.973333, 0.969333, 0.999333, 0.997333, 0.974667],  # SM
#     [0.972000, 0.974667, 1.000000, 1.000000, 0.973333],  # SL
#     [0.988667, 0.999333, 0.999333, 0.854000, 0.998000],  # MS
#     [0.970000, 0.982667, 1.000000, 0.992667, 0.976667],  # MM
#     [0.966667, 0.966000, 1.000000, 0.998667, 0.965333],  # ML
#     [0.980667, 0.999333, 1.000000, 0.922667, 1.000000],  # LS
#     [0.991333, 0.998667, 0.998667, 0.892000, 0.993333],  # LM
#     [0.970667, 0.972667, 0.998000, 0.990667, 0.980667],  # LL
# ])

# =========================
# 2. 阈值与状态编码
# 0 -> <95%
# 1 -> [95%, 97%)
# 2 -> >=97%
# =========================
thr95 = 0.95
thr97 = 0.97

state = np.zeros_like(data, dtype=int)
state[(data >= thr95) & (data < thr97)] = 1
state[data >= thr97] = 2

# 底部统计：每个算法在多少个场景里 >=95 / >=97
count_ge95_by_alg = (data >= thr95).sum(axis=0)
count_ge97_by_alg = (data >= thr97).sum(axis=0)

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

n_rows = len(scenarios)
n_cols = len(algorithms)

fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=300, facecolor=bg_color)
ax.set_facecolor(bg_color)

# =========================
# 4. 绘制圆角矩阵
# =========================
cell_w = 0.82
cell_h = 0.72
rounding = 0.12

for i in range(n_rows):
    for j in range(n_cols):
        x = j
        y = i

        s = state[i, j]
        if s == 0:
            color = color_fail
            mark = '×'
            txt_color = 'white'
        elif s == 1:
            color = color_95
            mark = '•'
            txt_color = 'white'
        else:
            color = color_97
            mark = '✓'
            txt_color = 'white'

        # 阴影
        shadow = FancyBboxPatch(
            (x - cell_w / 2 + 0.03, y - cell_h / 2 + 0.04),
            cell_w, cell_h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            linewidth=0,
            facecolor=shadow_color,
            alpha=0.35,
            zorder=1
        )
        ax.add_patch(shadow)

        # 主色块
        box = FancyBboxPatch(
            (x - cell_w / 2, y - cell_h / 2),
            cell_w, cell_h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            linewidth=0,
            facecolor=color,
            zorder=2
        )
        ax.add_patch(box)

        # 符号
        ax.text(
            x, y, mark,
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color=txt_color, zorder=3,
            fontproperties=symbol_font
        )

# =========================
# 5. 坐标文字
# =========================
# 顶部算法名
for j, alg in enumerate(algorithms):
    ax.text(
        j, -0.95, alg,
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color=text_dark
    )

# 左侧场景名
for i, sc in enumerate(scenarios):
    ax.text(
        -1.05, i, sc,
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color=text_dark
    )

# 底部两行统计：每个算法在多少个场景里 >=95 / >=97
ax.text(-1.05, n_rows - 0.10, '≥95%', ha='center', va='center',
        fontsize=11.5, fontweight='bold', color=text_dark)
ax.text(-1.05, n_rows + 0.55, '≥97%', ha='center', va='center',
        fontsize=11.5, fontweight='bold', color=text_dark)

for j in range(n_cols):
    ax.text(j, n_rows - 0.10, f'{count_ge95_by_alg[j]}/{n_rows}',
            ha='center', va='center',
            fontsize=11.5, fontweight='bold', color=text_dark)
    ax.text(j, n_rows + 0.55, f'{count_ge97_by_alg[j]}/{n_rows}',
            ha='center', va='center',
            fontsize=11.5, fontweight='bold', color=text_dark)

# =========================
# 6. 图例
# =========================
legend_handles = [
    Patch(facecolor=color_fail, edgecolor='none', label='< 95%'),
    Patch(facecolor=color_95, edgecolor='none', label='95% ≤ SR < 97%'),
    Patch(facecolor=color_97, edgecolor='none', label='≥ 97%')
]
ax.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.015),
    ncol=3,
    frameon=False,
    fontsize=11
)

# =========================
# 7. 范围与清理
# =========================
# 去掉右侧统计后，把右边界收紧；上下左右也整体收紧
ax.set_xlim(-1.30, n_cols - 1 + 0.75)
ax.set_ylim(n_rows + 0.72, -1.20)

ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(rect=[0.015, 0.03, 0.99, 0.99])

# =========================
# 8. 保存为 PDF
# =========================
plt.savefig('success_rate_L.pdf',
            format='pdf', bbox_inches='tight', pad_inches=0.01)
plt.show()