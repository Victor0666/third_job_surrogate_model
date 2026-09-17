# -*- coding: utf-8 -*-
"""
plot_train_eval_pd3qn_cost_metrics.py

PD3QN 训练日志绘图（适配你“修改后的 PD3QN”训练脚本输出的 CSVLogger 字段）。

特点：
1) PD3QN 日志通常每个 episode 只写一行（type="episode"），同一行里既有 train 指标，也有 eval_* 指标。
2) 输出两张图：
   - evolution_train_pd3qn_metrics.png : energy / task_avg_tardiness / makespan
   - evolution_eval_pd3qn_metrics.png  : eval_energy / eval_task_avg_tardiness / eval_makespan
3) 列名别名映射：支持你未来改名（episode_energy / total_energy 等）。

横轴：episode
纵轴：metric（as-is）
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 平滑/中心线（沿用你 HRL 脚本风格）
# -----------------------------
def moving_avg(x, k):
    if k is None or k <= 1:
        return np.asarray(x, dtype=float)
    return pd.Series(x, dtype="float64").rolling(window=int(k), min_periods=1).mean().values


def center_line(y, k=101, method="mean"):
    y = np.asarray(y, dtype=float)
    if k is None or k <= 1:
        return y
    ser = pd.Series(y, dtype="float64")
    if str(method).lower().strip() == "median":
        return ser.rolling(window=int(k), min_periods=1, center=True).median().values
    return ser.rolling(window=int(k), min_periods=1, center=True).mean().values


def _plot_series(ax, x, y, title, ylabel, smooth_k=11, center_k=101, center_method="mean"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    y_s = moving_avg(y, smooth_k)
    ax.plot(x, y_s, linewidth=1.8, linestyle="-", label=f"smoothed k={smooth_k}")

    y_c = center_line(y, k=center_k, method=center_method)
    ax.plot(x, y_c, linewidth=2.3, linestyle="--", label=f"center {center_method}, k={center_k}")

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")


def save_triplet(ep, y1, y2, y3, out_path, fig_title,
                 titles=("Metric 1", "Metric 2", "Metric 3"),
                 ylabels=("Value", "Value", "Value"),
                 smooth_k=11, center_k=101, center_method="mean"):
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 9.2), sharex=True)

    _plot_series(
        axes[0], ep, y1, title=titles[0], ylabel=ylabels[0],
        smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )
    _plot_series(
        axes[1], ep, y2, title=titles[1], ylabel=ylabels[1],
        smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )
    _plot_series(
        axes[2], ep, y3, title=titles[2], ylabel=ylabels[2],
        smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )

    fig.suptitle(fig_title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


# -----------------------------
# CSV 列名归一（PD3QN 兼容）
# -----------------------------
_CANONICAL_ALIASES = {
    # 基础
    "episode": ["episode", "ep", "Episode"],
    "type": ["type", "log_type", "kind", "stage"],

    # train metrics
    "episode_energy": ["episode_energy", "total_energy", "train_energy", "energy", "episode_total_energy"],
    "task_avg_tardiness": ["task_avg_tardiness", "task_avg_tardy", "avg_task_tardiness", "task_tardy_mean"],
    "makespan": ["makespan", "episode_makespan", "ep_makespan"],

    # eval metrics
    "eval_energy": ["eval_energy", "eval_total_energy", "eval_episode_energy"],
    "eval_task_avg_tardiness": ["eval_task_avg_tardiness", "eval_task_avg_tardy", "eval_task_tardy_mean"],
    "eval_makespan": ["eval_makespan", "eval_episode_makespan"],

    # 可选：如果你后来把回报也写进 CSV，这里也能自动识别
    "episode_return": ["episode_return", "train_return", "return", "ep_return", "episode_reward"],
    "eval_return": ["eval_return", "eval_episode_return", "eval_reward"],
}


def _pick_first_existing(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = list(df.columns)

    rename_map = {}
    for canonical, cands in _CANONICAL_ALIASES.items():
        hit = _pick_first_existing(cols, cands)
        if hit is not None and hit != canonical:
            rename_map[hit] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)

    # 保证 canonical 列存在
    for canonical in _CANONICAL_ALIASES.keys():
        if canonical not in df.columns:
            df[canonical] = np.nan

    # 类型与数值转换
    df["type"] = df["type"].astype(str)

    num_cols = [
        "episode",
        "episode_energy", "task_avg_tardiness", "makespan",
        "eval_energy", "eval_task_avg_tardiness", "eval_makespan",
        "episode_return", "eval_return",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def safe_load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV 不存在：{csv_path}")
        return pd.DataFrame()

    # 常见编码兜底
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")

    df = canonicalize_columns(df)
    return df


def _is_episode_row(s: str) -> bool:
    s = str(s).lower().strip()
    # 你 train_pd3qn 里写的是 type="episode"
    return s in {"episode", "train_episode", "train", "ep"}


# -----------------------------
# 从日志构造 train/eval 曲线数据
# -----------------------------
def build_pd3qn_episode_df(df: pd.DataFrame) -> pd.DataFrame:
    # 只取有 episode 的行
    d = df.dropna(subset=["episode"]).copy()

    # 如果 type 列乱了，也不强行过滤（只要 metrics 存在就行）
    # 但优先用 episode 行
    ep = d[d["type"].apply(_is_episode_row)].copy()
    if ep.empty:
        ep = d.copy()

    ep = ep.sort_values("episode").reset_index(drop=True)
    return ep


def main(csv_path, out_dir,
         smooth_k=11, center_k=101, center_method="mean"):
    os.makedirs(out_dir, exist_ok=True)

    df = safe_load_csv(csv_path)
    if df.empty:
        print("[WARN] 日志为空或读取失败。")
        return

    epdf = build_pd3qn_episode_df(df)
    if epdf.empty:
        print("[WARN] 没有可用的 episode 行。")
        return

    ep = epdf["episode"].values

    # -------------------------
    # Train 图：energy / task_avg_tardiness / makespan
    # -------------------------
    out_train = os.path.join(out_dir, "evolution_train_pd3qn_metrics.png")
    save_triplet(
        ep,
        epdf["episode_energy"].values,
        epdf["task_avg_tardiness"].values,
        epdf["makespan"].values,
        out_path=out_train,
        fig_title="PD3QN Training Evolution  [episode x metrics]",
        titles=("Train Energy (J)", "Train Task Avg Tardiness", "Train Makespan"),
        ylabels=("Energy (J)", "Avg Tardiness", "Makespan"),
        smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method
    )
    print(f"[OK] 保存：{os.path.abspath(out_train)}")

    # -------------------------
    # Eval 图：eval_energy / eval_task_avg_tardiness / eval_makespan
    # 若 eval 列全空则跳过
    # -------------------------
    has_eval = (
        (~epdf["eval_energy"].isna()).any()
        or (~epdf["eval_task_avg_tardiness"].isna()).any()
        or (~epdf["eval_makespan"].isna()).any()
    )
    if has_eval:
        out_eval = os.path.join(out_dir, "evolution_eval_pd3qn_metrics.png")
        save_triplet(
            ep,
            epdf["eval_energy"].values,
            epdf["eval_task_avg_tardiness"].values,
            epdf["eval_makespan"].values,
            out_path=out_eval,
            fig_title="PD3QN Evaluation Evolution  [episode x metrics]",
            titles=("Eval Energy (J)", "Eval Task Avg Tardiness", "Eval Makespan"),
            ylabels=("Energy (J)", "Avg Tardiness", "Makespan"),
            smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method
        )
        print(f"[OK] 保存：{os.path.abspath(out_eval)}")
    else:
        print("[WARN] 未检测到 eval_* 指标列（或全为空），跳过 eval 图。")

    # -------------------------
    # 可选：如果你后来把 return 写进日志，就顺便画 return
    # -------------------------
    has_ret = (~epdf["episode_return"].isna()).any() or (~epdf["eval_return"].isna()).any()
    if has_ret:
        fig = plt.figure(figsize=(10.0, 4.6))
        ax = fig.add_subplot(1, 1, 1)

        # 训练 return
        if (~epdf["episode_return"].isna()).any():
            y = epdf["episode_return"].values
            ax.plot(ep, moving_avg(y, smooth_k), linewidth=1.8, linestyle="-", label=f"train smoothed k={smooth_k}")
            ax.plot(ep, center_line(y, center_k, center_method), linewidth=2.3, linestyle="--",
                    label=f"train center {center_method}, k={center_k}")

        # 评测 return
        if (~epdf["eval_return"].isna()).any():
            y = epdf["eval_return"].values
            ax.plot(ep, moving_avg(y, smooth_k), linewidth=1.8, linestyle="-.", label=f"eval smoothed k={smooth_k}")
            ax.plot(ep, center_line(y, center_k, center_method), linewidth=2.3, linestyle=":",
                    label=f"eval center {center_method}, k={center_k}")

        ax.set_title("PD3QN Return Evolution  [episode x return]")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (as-is)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

        out_ret = os.path.join(out_dir, "evolution_pd3qn_return.png")
        plt.tight_layout()
        plt.savefig(out_ret, dpi=160)
        plt.close(fig)
        print(f"[OK] 保存：{os.path.abspath(out_ret)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=str,
        default="../out/logs/logs_pd3qn_cost_deadlineCACHE_FCFS_alpha15_smallTask_largeRes_per_banditWarmup_saveTime_ddlAplha1/train_metrics_pd3qn_cost.csv"
    )
    ap.add_argument(
        "--out", type=str,
        default="../out/plots/plots_pd3qn_cost_deadlineCACHE_FCFS_alpha15_smallTask_smallRes_per_banditWarmup_saveTime_ddlAlpha1_sl"
    )
    ap.add_argument("--smooth", type=int, default=11)
    ap.add_argument("--center_k", type=int, default=101)
    ap.add_argument("--center_method", type=str, default="mean", choices=["mean", "median"])
    args = ap.parse_args()

    main(
        csv_path=args.csv,
        out_dir=args.out,
        smooth_k=args.smooth,
        center_k=args.center_k,
        center_method=args.center_method,
    )
