# -*- coding: utf-8 -*-
# Wilcoxon signed-rank tests to verify whether MAHDRL is significantly better than others
# Output table: ONLY "P-Value" and "FDR(BH)" (scientific notation), optionally also "SS/NS" tag.

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import inspect

LOWER_IS_BETTER = True   # lower metric is better (e.g., energy/cost). If higher is better -> False
alpha = 0.05

# -----------------------------
# 1) Data (from your table)
# -----------------------------
data = {
    "Scenario": ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"],
    "FCFS_FIFS": [321114.92, 363175.79, 432193.66, 506870.29, 534289.60, 582367.21, 810911.05, 831290.84, 868121.92],
    "PD3QN":     [317707.46, 361345.01, 411292.72, 503120.46, 526036.06, 574508.65, 805055.34, 811682.24, 836540.00],
    "MARL":      [312023.34, 355171.10, 458573.17, 493294.52, 530333.93, 613782.16, 783777.53, 845797.92, 913472.18],
    "IRWS":      [322630.49, 346111.77, 404966.50, 513292.22, 518350.17, 547586.62, 824116.35, 838096.58, 833653.03],
    "MAHDRL":    [310665.03, 312465.34, 319776.28, 502905.00, 497653.00, 511576.96, 803683.80, 813018.30, 815547.22],
}
df = pd.DataFrame(data).set_index("Scenario")

# -----------------------------
# 2) FDR (Benjamini–Hochberg) adjustment
# -----------------------------
def fdr_bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR adjusted p-values (q-values).
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = pvals[order]

    q_sorted = np.empty(m, dtype=float)
    for i in range(m):
        rank = i + 1
        q_sorted[i] = p_sorted[i] * m / rank

    # enforce monotonicity from the end
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])

    q_sorted = np.minimum(q_sorted, 1.0)

    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q

# -----------------------------
# 3) Scientific notation formatter (e.g., 3.91E-03)
# -----------------------------
def fmt_sci(x: float, sig: int = 2) -> str:
    s = f"{float(x):.{sig}E}"      # e.g., 3.91E-03
    mant, exp = s.split("E")
    return f"{mant}E{int(exp):+03d}"  # normalize exponent width/sign: E-03 / E+00 / E+01

# -----------------------------
# 4) Paired Wilcoxon tests: H1 MAHDRL < baseline (if LOWER_IS_BETTER)
# -----------------------------
target = "MAHDRL"
baselines = [c for c in df.columns if c != target]
alternative = "less" if LOWER_IS_BETTER else "greater"

# SciPy compatibility (some versions differ)
wilcoxon_sig = inspect.signature(wilcoxon)
supports_alternative = "alternative" in wilcoxon_sig.parameters
supports_method = "method" in wilcoxon_sig.parameters

rows, pvals = [], []

for b in baselines:
    x = df[target].to_numpy(dtype=float)
    y = df[b].to_numpy(dtype=float)

    kwargs = dict(zero_method="wilcox", correction=False)
    if supports_alternative:
        kwargs["alternative"] = alternative
    if supports_method:
        kwargs["method"] = "auto"

    stat, p = wilcoxon(x, y, **kwargs)
    pvals.append(p)

    rows.append({
        "baseline": b,
        "wilcoxon_stat": stat,
        "p_value": p,
    })

res = pd.DataFrame(rows)
res["p_fdr"] = fdr_bh_adjust(res["p_value"].to_numpy(dtype=float))

# (optional) sort by FDR then raw p
res = res.sort_values(["p_fdr", "p_value"]).reset_index(drop=True)

# -----------------------------
# 5) Final tables (ONLY P-Value and FDR) in scientific notation
# -----------------------------
table_num = res[["baseline", "p_value", "p_fdr"]].copy()
table_num = table_num.rename(columns={
    "baseline": "Algorithm",
    "p_value": "P-Value",
    "p_fdr": "FDR"
})
table_num["P-Value"] = table_num["P-Value"].apply(lambda v: fmt_sci(v, sig=2))
table_num["FDR"]     = table_num["FDR"].apply(lambda v: fmt_sci(v, sig=2))

print("=== P-Value and FDR(BH) (scientific notation) ===")
print(table_num.to_string(index=False))

# If you prefer the screenshot style with "SS/NS" in the FDR column, use this:
table_tag = pd.DataFrame({
    "Algorithm": res["baseline"],
    "P-Value": res["p_value"].apply(lambda v: fmt_sci(v, sig=2)),
    "FDR": np.where(res["p_fdr"] < alpha, "SS", "NS"),
})
print("\n=== Screenshot-style (P-Value + FDR SS/NS) ===")
print(table_tag.to_string(index=False))