import numpy as np
from scipy.stats import wilcoxon

# ---------------------------
# 1) 原始数据（能耗，越小越好）
# ---------------------------
pd3qn = np.array([
    317541.1078,
    348649.456,
    402942.8914,
    505978.8135,
    521456.1405,
    550468.5663,
    808457.3975,
    831122.1266,
    837417.754
], dtype=float)

marl_evl1 = np.array([
    316054.814,
    325973.558,
    338048.158,
    510176.775,
    512528.513,
    548973.144,
    808075.02,
    811744.101,
    833389.869
], dtype=float)

hrl_tri = np.array([
    311807.664,
    312867.3375,
    317575.503,
    508706.558,
    505690.5865,
    527030.309,
    811538.5362,
    815752.815,
    825833.364
], dtype=float)

# ---------------------------
# 2) Wilcoxon 单侧检验：hrl_tri 是否更小（更优）
# ---------------------------
def wilcoxon_one_sided_less(x, y, name_x="x", name_y="y"):
    # 统计差值，便于检查有没有全0、ties等情况
    d = x - y
    print(f"\n=== {name_x} < {name_y} ? (Wilcoxon, paired, one-sided) ===")
    print(f"n={len(x)}, mean(x)={x.mean():.3f}, mean(y)={y.mean():.3f}, mean(x-y)={d.mean():.3f}")

    # method='exact' 更适合小样本；若有大量 ties/zeros 可能回退到近似
    res = wilcoxon(x, y, alternative="less", zero_method="wilcox", method="exact")
    print(f"Wilcoxon statistic W = {res.statistic}, p(one-sided) = {res.pvalue:.6g}")
    return res.pvalue

p1 = wilcoxon_one_sided_less(hrl_tri, pd3qn, name_x="hrl_tri", name_y="pd3qn_FCFS")
p2 = wilcoxon_one_sided_less(hrl_tri, marl_evl1, name_x="hrl_tri", name_y="marl_evl1")

# ---------------------------
# 3)（可选）两次比较的多重检验校正：Holm / Bonferroni
# ---------------------------
pvals = np.array([p1, p2], dtype=float)

# Bonferroni
p_bonf = np.minimum(pvals * 2, 1.0)

# Holm（手写一个简单版）
order = np.argsort(pvals)
p_holm = np.empty_like(pvals)
m = len(pvals)
for k, idx in enumerate(order):
    p_holm[idx] = min((m - k) * pvals[idx], 1.0)

print("\n--- Multiple comparison correction (2 tests) ---")
print(f"raw p-values        : {pvals}")
print(f"Bonferroni adjusted : {p_bonf}")
print(f"Holm adjusted       : {p_holm}")

# 判定示例（alpha=0.05）
alpha = 0.05
print("\n--- Decisions at alpha=0.05 ---")
print(f"hrl_tri < pd3qn_FCFS ?     {'YES' if p1 < alpha else 'NO'} (raw),  {'YES' if p_holm[0] < alpha else 'NO'} (Holm)")
print(f"hrl_tri < marl_evl1 ? {'YES' if p2 < alpha else 'NO'} (raw),  {'YES' if p_holm[1] < alpha else 'NO'} (Holm)")
