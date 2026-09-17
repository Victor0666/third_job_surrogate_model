# -*- coding: utf-8 -*-
"""
plot_train_eval_hrl_3layer_triplet_mgc.py

适配 hrl_mgc 训练日志的三层曲线绘图脚本（VM / Host / Manager 三行子图）。

关键兼容点（hrl_mgc）：
1) 训练日志字段名可能存在差异（例如 r_mgr / eval_manager / phase_mi_sum 等）：
   -> 本脚本会做“列名别名映射（alias -> canonical）”，统一到固定列名后再聚合。
2) type 字段可能出现不同写法（phase / train_phase / episode / eval 等）：
   -> 本脚本更鲁棒地识别 phase 行与 eval 行。
3) Manager 口径：默认忽略 micro-phase（is_micro==1）对齐 “Q(s,o) 只在真实推进边界更新”。
   -> 若 is_micro 列不存在，则默认全视为 0（不过滤）。

横轴：episode
纵轴：reward（as-is）
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
# CSV 读取与“hrl_mgc 兼容”列名归一
# -----------------------------
_CANONICAL_ALIASES = {
    # 基础
    "episode": ["episode", "ეპ", "ep", "Episode"],
    "type": ["type", "log_type", "stage", "kind"],

    # phase 统计
    "assign_cnt": ["assign_cnt", "phase_assign_cnt", "assigned", "task_cnt", "n_assign"],
    "phase_size_sum_mi": ["phase_size_sum_mi", "sum_mi_phase", "phase_mi_sum", "phase_sum_mi"],

    # train reward（phase）
    "r_vm_phase_mean": ["r_vm_phase_mean", "r_vm_mean", "vm_reward_mean", "r_vm_mean_phase", "r_vm"],
    "r_host_phase_mean": ["r_host_phase_mean", "r_host_mean", "host_reward_mean", "r_host_mean_phase", "r_host"],
    "r_manager_raw": ["r_manager_raw", "r_mgr_raw", "r_manager", "r_mgr", "r_manager_phase", "r_mgr_phase"],

    # eval reward（episode）
    "eval_vm": ["eval_vm", "eval_r_vm", "eval_vm_reward", "eval_vm_return"],
    "eval_host": ["eval_host", "eval_r_host", "eval_host_reward", "eval_host_return"],
    "eval_mgr": ["eval_mgr", "eval_manager", "eval_r_mgr", "eval_mgr_reward", "eval_mgr_return"],

    # micro-phase 标记
    "is_micro": ["is_micro", "micro", "is_micro_phase", "mgr_is_micro"],
}


def _pick_first_existing(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ★ MOD(hrl_mgc): 将不同训练脚本可能产生的“列名差异”统一映射到 canonical 名称。
    """
    df = df.copy()
    cols = list(df.columns)

    rename_map = {}
    for canonical, cands in _CANONICAL_ALIASES.items():
        hit = _pick_first_existing(cols, cands)
        if hit is not None and hit != canonical:
            rename_map[hit] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)

    # 确保 canonical 列都存在（不存在则补 NaN）
    for canonical in _CANONICAL_ALIASES.keys():
        if canonical not in df.columns:
            df[canonical] = np.nan

    # 数值列转数值
    num_cols = [
        "episode", "assign_cnt", "phase_size_sum_mi",
        "r_vm_phase_mean", "r_host_phase_mean", "r_manager_raw",
        "eval_vm", "eval_host", "eval_mgr",
        "is_micro",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # type 转字符串（缺失也能处理）
    df["type"] = df["type"].astype(str)
    return df


def safe_load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV 不存在：{csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, encoding="utf-8")
    df = canonicalize_columns(df)  # ★ MOD(hrl_mgc)
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


def _is_phase_type(s: str) -> bool:
    s = str(s).lower().strip()
    return s in {"phase", "train_phase", "phase_train", "micro_phase"}  # ★ MOD(hrl_mgc): 更鲁棒


def _is_eval_type(s: str) -> bool:
    s = str(s).lower().strip()
    return s in {"episode", "episode_reward", "episode_eval", "eval", "evaluation", "eval_episode"}  # ★ MOD


def build_train_episode_rewards(
    df: pd.DataFrame,
    mgr_weight_mode="phase_size_sum_mi",
    mgr_ignore_micro: bool = True,
) -> pd.DataFrame:
    """
    train：从 phase 行构造 per-episode 三层 reward

    - VM/Host：默认用 assign_cnt 加权（保留 micro 行）
    - Manager：默认用 phase_size_sum_mi 加权；若 mgr_ignore_micro=True 则过滤 is_micro==1
    """
    ph = df[df["type"].apply(_is_phase_type)].copy()  # ★ MOD(hrl_mgc)
    ph = ph.dropna(subset=["episode"])
    if ph.empty:
        return pd.DataFrame(columns=["episode", "train_vm", "train_host", "train_mgr"])

    ph["w_task"] = ph["assign_cnt"].fillna(0.0)

    # manager 权重
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
        # VM/Host：含 micro 行，用任务数加权
        train_vm = _weighted_avg(g["r_vm_phase_mean"].values, g["w_task"].values)
        train_host = _weighted_avg(g["r_host_phase_mean"].values, g["w_task"].values)

        # Manager：默认忽略 micro 行（is_micro==1）
        g_mgr = g
        if mgr_ignore_micro:
            # ★ MOD(hrl_mgc): is_micro 缺失时默认 0，不会把数据全过滤掉
            is_micro_int = g["is_micro"].fillna(0).astype(int)
            g_mgr_filtered = g[is_micro_int == 0]
            if not g_mgr_filtered.empty:
                g_mgr = g_mgr_filtered

        train_mgr = _weighted_avg(g_mgr["r_manager_raw"].values, g_mgr["w_mgr"].values)

        rows.append({
            "episode": float(ep),
            "train_vm": train_vm,
            "train_host": train_host,
            "train_mgr": train_mgr,
        })

    return pd.DataFrame(rows).sort_values("episode")


def build_eval_episode_rewards(df: pd.DataFrame) -> pd.DataFrame:
    """
    eval：从 episode/eval 行读取每 episode 三层 eval reward
    兼容：有些脚本可能在同一 episode 写多次 eval 行 -> 取“最后一次非空值”。
    """
    ep = df[df["type"].apply(_is_eval_type)].copy()  # ★ MOD(hrl_mgc)
    ep = ep.dropna(subset=["episode"]).sort_values("episode")
    if ep.empty:
        # 兜底：如果 type 没写 eval，但 eval_* 列有值，则也抓出来
        cand = df.dropna(subset=["episode"]).copy()
        has_eval = (~cand["eval_vm"].isna()) | (~cand["eval_host"].isna()) | (~cand["eval_mgr"].isna())
        cand = cand[has_eval].sort_values("episode")
        ep = cand

    if ep.empty:
        return pd.DataFrame(columns=["episode", "eval_vm", "eval_host", "eval_mgr"])

    # 每个 episode 取最后一行（更贴近“每回合评测一次”的写入习惯）
    ep2 = ep.groupby("episode", sort=True).tail(1)
    return ep2[["episode", "eval_vm", "eval_host", "eval_mgr"]].copy()


def main(csv_path, out_dir,
         smooth_k=11, center_k=101, center_method="mean",
         mgr_weight_mode="phase_size_sum_mi",
         mgr_ignore_micro=True):
    os.makedirs(out_dir, exist_ok=True)
    df = safe_load_csv(csv_path)
    if df.empty:
        print("[WARN] 日志为空或读取失败。")
        return

    # -------- Train（三层，episode×reward）--------
    train_df = build_train_episode_rewards(
        df,
        mgr_weight_mode=mgr_weight_mode,
        mgr_ignore_micro=bool(mgr_ignore_micro),
    )
    if not train_df.empty:
        ep = train_df["episode"].values
        out = os.path.join(out_dir, "evolution_train_3layer_triplet.png")
        save_triplet(
            ep,
            train_df["train_vm"].values,
            train_df["train_host"].values,
            train_df["train_mgr"].values,
            out_path=out,
            fig_title="Training Evolution (3-layer, hrl_mgc compatible)  [episode x reward]",
            smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method,
            ylabel="Reward (as-is)"
        )
        print(f"[OK] 保存：{os.path.abspath(out)}")
    else:
        print("[WARN] 未找到 phase 数据，跳过 train 三层图。")

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
            fig_title="Evaluation Evolution (3-layer, hrl_mgc compatible)  [episode x reward]",
            smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method,
            ylabel="Reward (as-is)"
        )
        print(f"[OK] 保存：{os.path.abspath(out)}")
    else:
        print("[WARN] 未找到 eval/episode 数据，跳过 eval 三层图。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=str,
        # ★ MOD(hrl_mgc): 默认路径你按自己的 logs 目录改即可；这里只给一个更中性的默认
        # default="../out/logs/logs_hrl_3layer_routeA_mgc_ave_smallTask_smallRes_seed1_warm003_alpha_Exhid_Dehid_MaxreadyAuto_UniformPar_deadlineCACHE_FCFS_alpha075_dalpha15/train_metrics_3layer_routeA.csv"
        default="../out/logs/logs_pd3qn_cost_deadlineCACHE_FCFS_alpha15_smallTask_largeRes_per_banditWarmup/train_metrics_pd3qn_cost.csv"
        # default="../out/logs/logs_hrl_3layer_routeA_mgc_ave_smallTask_largeRes/train_metrics_3layer_routeA.csv"
        # default = "../out/logs/logs_hrl_3layer_routeA_mgc_ave_smallTask_large_largeRes/train_metrics_3layer_routeA.csv"

    )
    ap.add_argument(
        "--out", type=str,
        # default="../out/plots/plots_hrl_mgc_ave_seed1_warm003_alpha_Exhid_Dehid_MaxreadyAuto_UniformPar_deadlineCACHE_FCFS_alpha075_dalpha15_ss"
        default="../out/plots/logs_pd3qn_cost_deadlineCACHE_FCFS_alpha15_smallTask_largeRes_per_banditWarmup_sl"
    )
    ap.add_argument("--smooth", type=int, default=11)
    ap.add_argument("--center_k", type=int, default=101)
    ap.add_argument("--center_method", type=str, default="mean", choices=["mean", "median"])
    ap.add_argument(
        "--mgr_weight_mode", type=str, default="phase_size_sum_mi",
        choices=["phase_size_sum_mi", "assign_cnt", "equal"],
        help="train 聚合时 manager 的加权方式（默认按 phase_size_sum_mi）"
    )
    ap.add_argument(
        "--mgr_ignore_micro", type=int, default=1, choices=[0, 1],
        help="1=manager 聚合忽略 is_micro==1 的 phase 行（默认开启，推荐与训练口径对齐）"
    )
    args = ap.parse_args()

    main(
        csv_path=args.csv,
        out_dir=args.out,
        smooth_k=args.smooth,
        center_k=args.center_k,
        center_method=args.center_method,
        mgr_weight_mode=args.mgr_weight_mode,
        mgr_ignore_micro=bool(args.mgr_ignore_micro),
    )
