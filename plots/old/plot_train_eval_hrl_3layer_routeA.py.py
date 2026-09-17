# -*- coding: utf-8 -*-
"""
plot_train_eval_hrl_3layer_triplet.py

三层进化曲线：三行三个坐标系（VM / Host / Manager），但在一张图中。
横轴：episode
纵轴：reward（as-is，不做单位换算）

输入 CSV：train_metrics_3layer_routeA.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 平滑/中心线
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


def _plot_series(ax, x, y, title, ylabel="Reward", smooth_k=11, center_k=101, center_method="mean"):
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


def save_triplet(ep, y_vm, y_host, y_mgr, out_path, fig_title,
                 smooth_k=11, center_k=101, center_method="mean",
                 ylabel="Reward (as-is)"):
    """
    三行子图：VM / Host / Manager，每行一个坐标系，整体保存为一张图。
    """
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 9.2), sharex=True)

    _plot_series(
        axes[0], ep, y_vm,
        title="VM Layer Reward Evolution",
        ylabel=ylabel, smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )
    _plot_series(
        axes[1], ep, y_host,
        title="Host Layer Reward Evolution",
        ylabel=ylabel, smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )
    _plot_series(
        axes[2], ep, y_mgr,
        title="Manager Layer Reward Evolution",
        ylabel=ylabel, smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )

    fig.suptitle(fig_title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


# -----------------------------
# CSV 读取与聚合
# -----------------------------
def safe_load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV 不存在：{csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, encoding="utf-8")

    need_cols = [
        "episode", "type",
        "assign_cnt", "phase_size_sum_mi",
        "r_vm_phase_mean", "r_host_phase_mean", "r_manager_raw",
        "eval_vm", "eval_host", "eval_mgr",
    ]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    for c in ["episode", "assign_cnt", "phase_size_sum_mi",
              "r_vm_phase_mean", "r_host_phase_mean", "r_manager_raw",
              "eval_vm", "eval_host", "eval_mgr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["type"] = df["type"].astype(str)
    return df


def _weighted_avg(values: np.ndarray, weights: np.ndarray):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = (~np.isnan(values)) & (~np.isnan(weights)) & (weights > 0)
    if mask.sum() == 0:
        return float(np.nanmean(values)) if not np.isnan(values).all() else np.nan
    v = values[mask]
    w = weights[mask]
    return float(np.sum(v * w) / np.sum(w))


def build_train_episode_rewards(df: pd.DataFrame, mgr_weight_mode="phase_size_sum_mi") -> pd.DataFrame:
    """
    train：从 type=phase 构造 per-episode 三层 reward
    - VM/Host：用 assign_cnt 加权（更接近任务数意义）
    - Manager：默认用 phase_size_sum_mi 加权；若全为0，则回退 assign_cnt；再不行等权
    """
    ph = df[df["type"].isin(["phase"])].copy()
    ph = ph.dropna(subset=["episode"])
    if ph.empty:
        return pd.DataFrame(columns=["episode", "train_vm", "train_host", "train_mgr"])

    ph["w_task"] = ph["assign_cnt"].fillna(0.0)

    if mgr_weight_mode == "phase_size_sum_mi":
        w_mgr = ph["phase_size_sum_mi"].fillna(0.0)
        if float(w_mgr.sum()) <= 0:
            w_mgr = ph["w_task"].copy()
        if float(w_mgr.sum()) <= 0:
            w_mgr = pd.Series(1.0, index=ph.index)
        ph["w_mgr"] = w_mgr
    elif mgr_weight_mode == "assign_cnt":
        w_mgr = ph["w_task"].copy()
        if float(w_mgr.sum()) <= 0:
            w_mgr = pd.Series(1.0, index=ph.index)
        ph["w_mgr"] = w_mgr
    else:
        ph["w_mgr"] = 1.0

    rows = []
    for ep, g in ph.groupby("episode", sort=True):
        rows.append({
            "episode": float(ep),
            "train_vm": _weighted_avg(g["r_vm_phase_mean"].values, g["w_task"].values),
            "train_host": _weighted_avg(g["r_host_phase_mean"].values, g["w_task"].values),
            "train_mgr": _weighted_avg(g["r_manager_raw"].values, g["w_mgr"].values),
        })

    return pd.DataFrame(rows).sort_values("episode")


def build_eval_episode_rewards(df: pd.DataFrame) -> pd.DataFrame:
    ep = df[df["type"].isin(["episode", "episode_reward"])].copy()
    ep = ep.dropna(subset=["episode"]).sort_values("episode")
    if ep.empty:
        return pd.DataFrame(columns=["episode", "eval_vm", "eval_host", "eval_mgr"])
    return ep[["episode", "eval_vm", "eval_host", "eval_mgr"]].copy()


def main(csv_path, out_dir,
         smooth_k=11, center_k=101, center_method="mean",
         mgr_weight_mode="phase_size_sum_mi"):
    os.makedirs(out_dir, exist_ok=True)
    df = safe_load_csv(csv_path)
    if df.empty:
        print("[WARN] 日志为空或读取失败。")
        return

    # -------- Train（三层，episode×reward）--------
    train_df = build_train_episode_rewards(df, mgr_weight_mode=mgr_weight_mode)
    if not train_df.empty:
        ep = train_df["episode"].values
        out = os.path.join(out_dir, "evolution_train_3layer_triplet.png")
        save_triplet(
            ep,
            train_df["train_vm"].values,
            train_df["train_host"].values,
            train_df["train_mgr"].values,
            out_path=out,
            fig_title="Training Evolution (3-layer)  [episode x reward]",
            smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method,
            ylabel="Reward (as-is)"
        )
        print(f"[OK] 保存：{os.path.abspath(out)}")
    else:
        print("[WARN] 未找到 type=phase 数据，跳过 train 三层图。")

    # -------- Eval（三层，episode×reward）--------
    eval_df = build_eval_episode_rewards(df)
    if not eval_df.empty:
        ep = eval_df["episode"].values
        out = os.path.join(out_dir, "evolution_eval_3layer_triplet.png")
        save_triplet(
            ep,
            eval_df["eval_vm"].values,
            eval_df["eval_host"].values,
            eval_df["eval_mgr"].values,
            out_path=out,
            fig_title="Evaluation Evolution (3-layer)  [episode x reward]",
            smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method,
            ylabel="Reward (as-is)"
        )
        print(f"[OK] 保存：{os.path.abspath(out)}")
    else:
        print("[WARN] 未找到 type=episode 数据，跳过 eval 三层图。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str,
                    default="../out/logs/logs_hrl_3layer_option_critic_smallTask_smallRes/train_metrics_3layer_option_critic.csv")
    ap.add_argument("--out", type=str,
                    default="../out/plots/plots_hrl_3layer_option_critic_smallTask_smallRes")
    ap.add_argument("--smooth", type=int, default=11)
    ap.add_argument("--center_k", type=int, default=101)
    ap.add_argument("--center_method", type=str, default="mean", choices=["mean", "median"])
    ap.add_argument("--mgr_weight_mode", type=str, default="phase_size_sum_mi",
                    choices=["phase_size_sum_mi", "assign_cnt", "equal"],
                    help="train 聚合时 manager 的加权方式（默认按 phase_size_sum_mi）")
    args = ap.parse_args()

    main(
        csv_path=args.csv,
        out_dir=args.out,
        smooth_k=args.smooth,
        center_k=args.center_k,
        center_method=args.center_method,
        mgr_weight_mode=args.mgr_weight_mode
    )
