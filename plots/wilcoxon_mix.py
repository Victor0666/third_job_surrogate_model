# -*- coding: utf-8 -*-
"""
对 MAHDRL 与其他算法做 Wilcoxon signed-rank test
使用 Loose / Medium / Tight 三种情况下共 27 条数据联合检验

检验目标：
H1: MAHDRL 的 energy 显著小于对比算法（即 MAHDRL 更优）

输出：
1. 原始 p 值
2. Bonferroni 校正 p 值
3. FDR(BH) 校正 p 值
4. 可直接用于论文表格填写的数据
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR 校正，返回 q 值"""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    order = np.argsort(pvals)
    ranked_p = pvals[order]

    ranked_q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = ranked_p[i] * n / rank
        q = min(q, prev, 1.0)
        ranked_q[i] = q
        prev = q

    qvals = np.empty(n, dtype=float)
    qvals[order] = ranked_q
    return qvals


def sig_label(p, alpha=0.05):
    """把显著性结果写成 SS / NS"""
    # print(p)
    return "SS" if p < alpha else "NS"


def sci_fmt(p):
    """论文里常用科学计数法格式"""
    return f"{p:.4E}"


# =========================
# 1. 三种情况下的数据
# =========================
data_Loose = {
    "Scenario": ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"],
    "ERT_FIFS": [322257.64, 369822.06, 433136.97, 506804.15, 539210.24, 586336.04, 810415.14, 828322.49, 866153.32],
    "PD3QN":    [317920.38, 361446.92, 411540.49, 503354.29, 526557.08, 573988.19, 805313.13, 810911.68, 836230.86],
    "MARL":     [312339.69, 357209.25, 458530.66, 492347.68, 534306.22, 605988.10, 782687.98, 830788.47, 889778.28],
    "IRWS":     [320236.34, 333211.63, 396666.04, 517066.82, 505730.67, 538874.54, 833147.41, 823829.82, 825074.99],
    "MAHDRL":   [310750.03, 314137.81, 322505.11, 501546.19, 496615.86, 520598.96, 804567.93, 808656.14, 815465.02],
}

data_Medium = {
    "Scenario": ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"],
    "ERT_FIFS": [322257.64, 369822.06, 433136.97, 506804.15, 539210.24, 586336.04, 810415.14, 828322.49, 866153.32],
    "PD3QN":    [317707.46, 360714.53, 411471.01, 503120.46, 525897.37, 574247.20, 805055.34, 811990.02, 834786.93],
    "MARL":     [315079.66, 358202.15, 436595.24, 493838.40, 526300.36, 593159.68, 786180.59, 834510.29, 898590.72],
    "IRWS":     [318408.02, 330397.45, 395562.62, 512153.34, 503900.15, 536000.74, 825359.91, 837669.61, 827277.13],
    "MAHDRL":   [310158.20, 325177.49, 330382.94, 500942.63, 502952.96, 527234.71, 804642.27, 810019.56, 816521.51],
}

data_Tight = {
    "Scenario": ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"],
    "ERT_FIFS": [322257.64, 369822.06, 433136.97, 506804.15, 539210.24, 586336.04, 810415.14, 828322.49, 866153.32],
    "PD3QN":    [317965.13, 360512.60, 411964.79, 503222.75, 525918.01, 573968.30, 805544.54, 812203.03, 834348.89],
    "MARL":     [312896.34, 357295.06, 443040.54, 491838.91, 531664.58, 601923.90, 781779.31, 833051.35, 898612.03],
    "IRWS":     [315483.64, 334727.38, 400034.27, 507573.32, 509760.32, 541658.82, 816115.00, 831075.06, 831547.52],
    "MAHDRL":   [309986.84, 321647.93, 324051.56, 500851.52, 501806.34, 522417.45, 804402.33, 810063.11, 818459.50],
}


# =========================
# 2. 合并为 27 条数据
# =========================
df_loose = pd.DataFrame(data_Loose)
df_loose["Deadline"] = "Loose"

df_medium = pd.DataFrame(data_Medium)
df_medium["Deadline"] = "Medium"

df_tight = pd.DataFrame(data_Tight)
df_tight["Deadline"] = "Tight"

df_all = pd.concat([df_loose, df_medium, df_tight], ignore_index=True)

# 可选：查看合并后的27条数据
print("=== 合并后的 27 条数据 ===")
print(df_all[["Deadline", "Scenario", "ERT_FIFS", "PD3QN", "MARL", "IRWS", "MAHDRL"]])
print()


# =========================
# 3. 与 MAHDRL 做单侧 Wilcoxon 检验
#    H1: MAHDRL < competitor
# =========================
baseline = "MAHDRL"
comparators = ["ERT_FIFS", "PD3QN", "MARL", "IRWS"]

raw_pvals = []
stats = []

for alg in comparators:
    x = df_all[baseline].values   # 27条
    y = df_all[alg].values        # 27条

    stat, p = wilcoxon(
        x, y,
        alternative="less",   # MAHDRL 更小 => 更优
        zero_method="wilcox",
        method="auto"         # 27条数据用 auto 更稳妥
    )

    stats.append(stat)
    raw_pvals.append(p)

raw_pvals = np.array(raw_pvals, dtype=float)


# =========================
# 4. Bonferroni 与 FDR 校正
# =========================
m = len(raw_pvals)

bonf_pvals = np.minimum(raw_pvals * m, 1.0)
fdr_pvals = benjamini_hochberg(raw_pvals)


# =========================
# 5. 输出完整结果
# =========================
result = pd.DataFrame({
    "Algorithm": comparators,
    "Wilcoxon_Statistic": stats,
    "P_Value": raw_pvals,
    "Bonferroni_p": bonf_pvals,
    "FDR_p": fdr_pvals,
    "Bonferroni": [sig_label(p) for p in bonf_pvals],
    "FDR": [sig_label(p) for p in fdr_pvals],
})

print("=== 完整统计结果（基于27条数据） ===")
print(result)
print()


# =========================
# 6. 生成适合论文表格填写的版本
# =========================
paper_table = pd.DataFrame({
    "Algorithm": result["Algorithm"],
    "P-Value": [sci_fmt(p) for p in result["P_Value"]],
    "Bonferroni": result["Bonferroni"],
    "FDR": result["FDR"],
})

print("=== 可直接填表版本 ===")
print(paper_table)
print()


# =========================
# 7. 保存结果
# =========================
result.to_csv("wilcoxon_full_results_27cases.csv", index=False, encoding="utf-8-sig")
paper_table.to_csv("wilcoxon_paper_table_27cases.csv", index=False, encoding="utf-8-sig")

print("已保存：")
print("1) wilcoxon_full_results_27cases.csv")
print("2) wilcoxon_paper_table_27cases.csv")


# =========================
# 8. 额外打印更明确的数值
# =========================
print("\n=== 详细数值（便于核对） ===")
for i, alg in enumerate(comparators):
    print(
        f"{alg:9s} | "
        f"raw p = {raw_pvals[i]:.8f} | "
        f"Bonferroni p = {bonf_pvals[i]:.8f} | "
        f"FDR p = {fdr_pvals[i]:.8f} | "
        f"Bonf = {sig_label(bonf_pvals[i])} | "
        f"FDR = {sig_label(fdr_pvals[i])}"
    )