# -*- coding: utf-8 -*-
"""
wilcoxon_from_long_csv.py
从“长表”CSV执行 Wilcoxon 配对符号秩检验（按 seed 对齐）：
必需列：seed, algo, energy_j

功能：
- 对同一 seed 下 algoA 与 algoB 的 energy_j 做 Wilcoxon（双侧/单侧可选）
- 计算 Rank-Biserial 效应量与 Hodges–Lehmann（配对差分中位数）估计
- 打印描述性统计
- 生成两张图（箱线图、成对连线图）

用法示例：
python wilcoxon_from_long_csv.py --csv data.csv --algoA HRL_MLL --algoB HRL_MLL_NM
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="输入CSV路径（包含 seed, algo, energy_j 列）")
    p.add_argument("--algoA", default="", help="算法A名称（留空则自动按字母序取前两个）")
    p.add_argument("--algoB", default="", help="算法B名称")
    p.add_argument("--seed-col", default="seed", help="seed 列名（默认 seed）")
    p.add_argument("--algo-col", default="algo", help="算法列名（默认 algo）")
    p.add_argument("--value-col", default="energy_j", help="数值列名（默认 energy_j）")
    p.add_argument("--alternative", choices=["two-sided","less","greater"], default="two-sided",
                   help="Wilcoxon 备择假设（默认 two-sided）")
    p.add_argument("--outdir", default="", help="输出图保存目录（默认与CSV同目录）")
    p.add_argument("--no-plots", action="store_true", help="不绘图")
    return p.parse_args()

def describe(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    x = x[mask]
    if x.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, q25=np.nan,
                    median=np.nan, q75=np.nan, max=np.nan)
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
        min=float(np.min(x)),
        q25=float(np.percentile(x, 25)),
        median=float(np.median(x)),
        q75=float(np.percentile(x, 75)),
        max=float(np.max(x)),
    )

def wilcoxon_effect_size(a: np.ndarray, b: np.ndarray):
    """返回 (rank-biserial r, HL估计, 非零差分对数)"""
    diff = a - b
    diff_nz = diff[diff != 0]
    if diff_nz.size == 0:
        return 0.0, 0.0, 0
    ranks = rankdata(np.abs(diff_nz), method="average")
    W_pos = float(np.sum(ranks[diff_nz > 0]))
    W_neg = float(np.sum(ranks[diff_nz < 0]))
    S = diff_nz.size * (diff_nz.size + 1) / 2.0
    r_rb = (W_pos - W_neg) / S
    HL = float(np.median(diff))  # Hodges–Lehmann 的常用近似（配对差分中位数）
    return r_rb, HL, int(diff_nz.size)

def main():
    args = parse_args()

    # 读取并“大小写不敏感”对齐列名
    df_raw = pd.read_csv(args.csv)
    cols_lower = {c.lower(): c for c in df_raw.columns}
    def pick(name: str) -> str:
        key = name.lower()
        if key not in cols_lower:
            raise ValueError(f"CSV 中缺少必须列：{name} ；现有列：{list(df_raw.columns)}")
        return cols_lower[key]
    col_seed = pick(args.seed_col)
    col_algo = pick(args.algo_col)
    col_val  = pick(args.value_col)

    # 去重聚合：同一 (seed, algo) 多行时取均值（也可改成 .last()/.median()）
    df = (df_raw
          .groupby([col_seed, col_algo], as_index=False)[col_val]
          .mean())

    # 决定比较的两种算法
    algos = sorted(df[col_algo].unique().tolist())
    if len(algos) < 2:
        raise ValueError(f"algo 种类少于2：{algos}")
    if args.algoA and args.algoB:
        if args.algoA not in algos or args.algoB not in algos:
            raise ValueError(f"指定算法不在数据中：{args.algoA}, {args.algoB}；现有：{algos}")
        algA, algB = args.algoA, args.algoB
    else:
        algA, algB = algos[0], algos[1]

    # 透视为宽表并按 seed 对齐，只保留两列都存在的 seed
    wide = df.pivot(index=col_seed, columns=col_algo, values=col_val)
    if algA not in wide.columns or algB not in wide.columns:
        raise ValueError(f"透视后未找到要比较的两列：{algA}, {algB}；现有列：{wide.columns.tolist()}")
    paired = wide[[algA, algB]].dropna()
    if paired.shape[0] == 0:
        raise ValueError("没有包含两种算法的配对 seed（透视后全为空）。")

    a = paired[algA].to_numpy()
    b = paired[algB].to_numpy()

    # Wilcoxon（默认剔除零差，zero_method='wilcox'）
    res = wilcoxon(a, b, alternative=args.alternative, zero_method="wilcox",
                   correction=False, mode="auto")
    W = float(res.statistic)
    p = float(res.pvalue)

    # 效应量与 HL
    r_rb, HL, n_nz = wilcoxon_effect_size(a, b)

    # 打印结果
    print("===== 数据 / 对齐信息 =====")
    print(f"文件: {args.csv}")
    print(f"算法: {algA} (A)  vs  {algB} (B)")
    print(f"总 seed 数: {wide.shape[0]}，有效配对 seed 数: {paired.shape[0]}")

    print("\n===== Wilcoxon 符号秩检验 =====")
    print(f"备择假设: {args.alternative}")
    print(f"W 统计量: {W}")
    print(f"P 值: {p:.6g}")
    print(f"非零差分对数: {n_nz}")
    print("结论:", "显著差异（拒绝 H0）" if p < 0.05 else "无显著差异（不拒绝 H0）")

    print("\n===== 效应量与稳健估计 =====")
    print(f"Rank-Biserial 效应量 r_rb: {r_rb:.4f}")
    print(f"Hodges–Lehmann 估计（A-B 的差分中位数）: {HL:.6g}")

    da, db, dd = describe(a), describe(b), describe(a - b)
    print("\n===== 描述性统计 =====")
    print(f"{algA}: n={da['n']}, mean={da['mean']:.6g}, std={da['std']:.6g}, "
          f"median={da['median']:.6g}, q25={da['q25']:.6g}, q75={da['q75']:.6g}")
    print(f"{algB}: n={db['n']}, mean={db['mean']:.6g}, std={db['std']:.6g}, "
          f"median={db['median']:.6g}, q25={db['q25']:.6g}, q75={db['q75']:.6g}")
    print(f"(A-B): n={dd['n']}, mean={dd['mean']:.6g}, std={dd['std']:.6g}, "
          f"median={dd['median']:.6g}, q25={dd['q25']:.6g}, q75={dd['q75']:.6g}")

    # 绘图输出
    if not args.no_plots:
        outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv)) or "."
        os.makedirs(outdir, exist_ok=True)

        # 1) 箱线图
        plt.figure(figsize=(8, 5))
        plt.boxplot([a, b], labels=[algA, algB], showmeans=True)
        plt.title(f"{algA} vs {algB} (Wilcoxon)")
        plt.ylabel(args.value_col)
        plt.grid(True, linestyle="--", alpha=0.5)
        path_box = os.path.join(outdir, "wilcoxon_boxplot.png")
        plt.savefig(path_box, bbox_inches="tight", dpi=150)
        plt.close()

        # 2) 成对连线图（每个 seed 一条线）
        plt.figure(figsize=(9, 6))
        for i in range(len(a)):
            plt.plot([0, 1], [a[i], b[i]], marker="o")  # 不指定颜色
        plt.xticks([0, 1], [algA, algB])
        plt.title(f"Paired Lines by seed (n={len(a)})")
        plt.ylabel(args.value_col)
        plt.grid(True, linestyle="--", alpha=0.5)
        path_lines = os.path.join(outdir, "wilcoxon_paired_lines.png")
        plt.savefig(path_lines, bbox_inches="tight", dpi=150)
        plt.close()

        print(f"\n图像已保存：\n- {path_box}\n- {path_lines}")

if __name__ == "__main__":
    main()
