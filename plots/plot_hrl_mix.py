# -*- coding: utf-8 -*-
"""
plot_train_eval_hrl_new_triplet.py

适配“hrl新训练代码”的三层进化曲线绘图脚本。

对应训练脚本：
train_hrl_tri_003_alpha15_smallTask_smallRes_seed5_ddlFCFS_alpha075_dalphaMix_Loose.py

日志特点（固定）：
1) phase 行：type == "phase"
   字段包括：
   - assign_cnt
   - phase_size_sum_mi
   - r_vm_phase_mean
   - r_host_phase_mean
   - r_manager_raw
   - phase_energy
   - phase_delay
   - phase_energy_cost
   - phase_delay_cost
   - manager_reward_old
   - manager_reward_new

2) episode 行：type == "episode"
   字段包括：
   - episode_energy
   - ep_total_lateness
   - ep_avg_lateness
   - wf_avg_lateness
   - eval_vm
   - eval_host
   - eval_mgr
   - eval_energy

说明：
- 你的“hrl新训练代码”里 logger 写入的 r_manager_raw，
  实际上是 finish_phase_and_advance() 返回的“新 manager reward”。
- 因此默认 manager 曲线使用 r_manager_raw。
- 如需对比 old/new manager reward，可通过参数切换。
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 基础工具
# =========================================================
def moving_avg(x, k):
    x = np.asarray(x, dtype=float)
    if k is None or int(k) <= 1:
        return x
    return pd.Series(x, dtype="float64").rolling(window=int(k), min_periods=1).mean().values


def center_line(y, k=101, method="mean"):
    y = np.asarray(y, dtype=float)
    if k is None or int(k) <= 1:
        return y
    ser = pd.Series(y, dtype="float64")
    method = str(method).lower().strip()
    if method == "median":
        return ser.rolling(window=int(k), min_periods=1, center=True).median().values
    return ser.rolling(window=int(k), min_periods=1, center=True).mean().values


def weighted_avg(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    mask = (~np.isnan(v)) & (~np.isnan(w)) & (w > 0)
    if mask.sum() == 0:
        if np.all(np.isnan(v)):
            return np.nan
        return float(np.nanmean(v))

    v = v[mask]
    w = w[mask]
    return float(np.sum(v * w) / np.sum(w))


def ensure_numeric(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_str(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str)
    return df


# =========================================================
# 读取日志
# =========================================================
def safe_load_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV 不存在：{csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, encoding="utf-8")

    num_cols = [
        "step", "episode", "ep_length", "env_time",
        "episode_energy", "wf_completed", "wf_target",
        "eps_vm", "eps_host", "eps_mgr",
        "assign_cnt", "phase_size_sum_mi",
        "r_vm_phase_mean", "r_host_phase_mean", "r_manager_raw",
        "phase_energy", "phase_delay",
        "phase_energy_cost", "phase_delay_cost",
        "manager_reward_old", "manager_reward_new",
        "ep_total_lateness", "ep_avg_lateness", "wf_avg_lateness",
        "ep_wf_lateness_sum",
        "eval_vm", "eval_host", "eval_mgr", "eval_energy",
    ]
    str_cols = ["type"]

    df = ensure_numeric(df, num_cols)
    df = ensure_str(df, str_cols)

    return df


# =========================================================
# phase / episode 数据构造
# =========================================================
def build_train_episode_rewards(df, manager_reward_col="r_manager_raw", manager_weight_mode="phase_size_sum_mi"):
    """
    从 type=phase 的日志构造每个 episode 的训练曲线数据。

    VM/Host：
        默认按 assign_cnt 加权求该 episode 内各 phase 的平均 reward。

    Manager：
        默认按 phase_size_sum_mi 加权；
        也可切换为 assign_cnt 或 equal。
    """
    ph = df[df["type"].str.lower().str.strip() == "phase"].copy()
    ph = ph.dropna(subset=["episode"])

    if ph.empty:
        return pd.DataFrame(columns=["episode", "train_vm", "train_host", "train_mgr"])

    if manager_reward_col not in ph.columns:
        raise ValueError(f"manager_reward_col={manager_reward_col} 不存在于 CSV 列中")

    ph["w_task"] = ph["assign_cnt"].fillna(0.0)

    if manager_weight_mode == "phase_size_sum_mi":
        ph["w_mgr"] = ph["phase_size_sum_mi"].fillna(0.0)
        if float(ph["w_mgr"].sum()) <= 0:
            ph["w_mgr"] = ph["w_task"].copy()
        if float(ph["w_mgr"].sum()) <= 0:
            ph["w_mgr"] = 1.0
    elif manager_weight_mode == "assign_cnt":
        ph["w_mgr"] = ph["w_task"].copy()
        if float(ph["w_mgr"].sum()) <= 0:
            ph["w_mgr"] = 1.0
    else:
        ph["w_mgr"] = 1.0

    rows = []
    for ep, g in ph.groupby("episode", sort=True):
        train_vm = weighted_avg(g["r_vm_phase_mean"].values, g["w_task"].values)
        train_host = weighted_avg(g["r_host_phase_mean"].values, g["w_task"].values)
        train_mgr = weighted_avg(g[manager_reward_col].values, g["w_mgr"].values)

        rows.append({
            "episode": float(ep),
            "train_vm": train_vm,
            "train_host": train_host,
            "train_mgr": train_mgr,
        })

    return pd.DataFrame(rows).sort_values("episode")


def build_eval_episode_rewards(df):
    """
    从 type=episode 的日志读取每个 episode 的评测曲线数据。
    """
    ep = df[df["type"].str.lower().str.strip() == "episode"].copy()
    ep = ep.dropna(subset=["episode"]).sort_values("episode")

    if ep.empty:
        return pd.DataFrame(columns=["episode", "eval_vm", "eval_host", "eval_mgr", "eval_energy"])

    # 同一 episode 若重复写入，取最后一条
    ep = ep.groupby("episode", sort=True).tail(1)

    return ep[["episode", "eval_vm", "eval_host", "eval_mgr", "eval_energy"]].copy()


def build_episode_scalar_metrics(df):
    """
    从 type=episode 的日志提取 episode 级指标：
    - episode_energy
    - ep_total_lateness
    - ep_avg_lateness
    - wf_avg_lateness
    - eval_energy
    """
    ep = df[df["type"].str.lower().str.strip() == "episode"].copy()
    ep = ep.dropna(subset=["episode"]).sort_values("episode")

    if ep.empty:
        return pd.DataFrame(columns=[
            "episode",
            "episode_energy",
            "ep_total_lateness",
            "ep_avg_lateness",
            "wf_avg_lateness",
            "eval_energy",
        ])

    ep = ep.groupby("episode", sort=True).tail(1)

    return ep[[
        "episode",
        "episode_energy",
        "ep_total_lateness",
        "ep_avg_lateness",
        "wf_avg_lateness",
        "eval_energy",
    ]].copy()


# =========================================================
# 绘图函数
# =========================================================
def plot_series(ax, x, y, title, ylabel, smooth_k=11, center_k=101, center_method="mean"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    y_s = moving_avg(y, smooth_k)
    y_c = center_line(y, center_k, center_method)

    ax.plot(x, y_s, linewidth=1.8, linestyle="-", label=f"smoothed k={smooth_k}")
    ax.plot(x, y_c, linewidth=2.2, linestyle="--", label=f"center {center_method}, k={center_k}")

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")


def save_triplet(ep, y_vm, y_host, y_mgr, out_path, fig_title,
                 smooth_k=11, center_k=101, center_method="mean",
                 ylabel="Reward"):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.3), sharex=True)

    plot_series(
        axes[0], ep, y_vm,
        title="VM Layer Reward Evolution",
        ylabel=ylabel,
        smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )
    plot_series(
        axes[1], ep, y_host,
        title="Host Layer Reward Evolution",
        ylabel=ylabel,
        smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )
    plot_series(
        axes[2], ep, y_mgr,
        title="Manager Layer Reward Evolution",
        ylabel=ylabel,
        smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )

    fig.suptitle(fig_title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_single_curve(ep, y, out_path, fig_title, ylabel,
                      smooth_k=11, center_k=101, center_method="mean"):
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    plot_series(
        ax, ep, y,
        title=fig_title,
        ylabel=ylabel,
        smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_double_curve(ep, y1, y2, out_path, fig_title,
                      label1, label2, ylabel,
                      smooth_k=11):
    fig, ax = plt.subplots(figsize=(10.2, 4.8))

    ep = np.asarray(ep, dtype=float)
    y1 = moving_avg(np.asarray(y1, dtype=float), smooth_k)
    y2 = moving_avg(np.asarray(y2, dtype=float), smooth_k)

    ax.plot(ep, y1, linewidth=2.0, linestyle="-", label=label1)
    ax.plot(ep, y2, linewidth=2.0, linestyle="--", label=label2)

    ax.set_title(fig_title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


# =========================================================
# 主流程
# =========================================================
def main(csv_path, out_dir,
         smooth_k=11, center_k=101, center_method="mean",
         manager_reward_col="r_manager_raw",
         manager_weight_mode="phase_size_sum_mi"):
    os.makedirs(out_dir, exist_ok=True)

    df = safe_load_csv(csv_path)
    if df.empty:
        print("[WARN] 日志为空或读取失败。")
        return

    # ---------------- 1) Train 三层 reward ----------------
    train_df = build_train_episode_rewards(
        df,
        manager_reward_col=manager_reward_col,
        manager_weight_mode=manager_weight_mode,
    )
    if not train_df.empty:
        out = os.path.join(out_dir, "evolution_train_3layer_triplet.png")
        save_triplet(
            train_df["episode"].values,
            train_df["train_vm"].values,
            train_df["train_host"].values,
            train_df["train_mgr"].values,
            out_path=out,
            fig_title="Training Evolution (HRL New, 3-layer) [episode x reward]",
            smooth_k=smooth_k,
            center_k=center_k,
            center_method=center_method,
            ylabel="Reward",
        )
        print(f"[OK] 保存：{os.path.abspath(out)}")
    else:
        print("[WARN] 未找到 type=phase 的数据，跳过训练三层图。")

    # ---------------- 2) Eval 三层 reward ----------------
    eval_df = build_eval_episode_rewards(df)
    if not eval_df.empty:
        out = os.path.join(out_dir, "evolution_eval_3layer_triplet.png")
        save_triplet(
            eval_df["episode"].values,
            eval_df["eval_vm"].values,
            eval_df["eval_host"].values,
            eval_df["eval_mgr"].values,
            out_path=out,
            fig_title="Evaluation Evolution (HRL New, 3-layer) [episode x reward]",
            smooth_k=smooth_k,
            center_k=center_k,
            center_method=center_method,
            ylabel="Reward",
        )
        print(f"[OK] 保存：{os.path.abspath(out)}")
    else:
        print("[WARN] 未找到 type=episode 的 eval 数据，跳过评测三层图。")

    # ---------------- 3) episode 能耗曲线 ----------------
    scalar_df = build_episode_scalar_metrics(df)
    if not scalar_df.empty:
        if not scalar_df["eval_energy"].isna().all():
            out = os.path.join(out_dir, "episode_energy_eval_curve.png")
            save_single_curve(
                scalar_df["episode"].values,
                scalar_df["eval_energy"].values,
                out_path=out,
                fig_title="Evaluation Energy Evolution",
                ylabel="Energy (J)",
                smooth_k=smooth_k,
                center_k=center_k,
                center_method=center_method,
            )
            print(f"[OK] 保存：{os.path.abspath(out)}")

        if (not scalar_df["ep_avg_lateness"].isna().all()) or (not scalar_df["wf_avg_lateness"].isna().all()):
            out = os.path.join(out_dir, "episode_lateness_curve.png")
            save_double_curve(
                scalar_df["episode"].values,
                scalar_df["ep_avg_lateness"].values,
                scalar_df["wf_avg_lateness"].values,
                out_path=out,
                fig_title="Lateness Evolution",
                label1="Task Avg Lateness",
                label2="Workflow Avg Lateness",
                ylabel="Lateness (s)",
                smooth_k=smooth_k,
            )
            print(f"[OK] 保存：{os.path.abspath(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # resModel = "smallRes"
    # resModel = "medRes"
    resModel = "largeRes"

    # model = "Loose"
    model = "Medium"
    # model = "Tight"
    ap.add_argument(
        "--csv",
        type=str,
        default="../out/logs/logs_hrl_3layer_routeA_mgc_ave_smallTask_"+ resModel +"_seed1_mgrDelayEnergy_mix_rAlpha075_dalphaMix_HVrn_" + model + "/train_metrics_3layer_routeA.csv",
        help="hrl新训练代码生成的训练日志 CSV 路径"
    )
    ap.add_argument(
        "--out",
        type=str,
        default="../out/plots/plots_hrl_new_smallTask_"+ resModel +"_mgrDelayEnergy_mix_rAlpha075_dalphaMix_HVrn_" + model,
        help="输出图片目录"
    )

    ap.add_argument("--smooth", type=int, default=101)
    ap.add_argument("--center_k", type=int, default=101)
    ap.add_argument("--center_method", type=str, default="mean", choices=["mean", "median"])

    ap.add_argument(
        "--manager_reward_col",
        type=str,
        default="r_manager_raw",
        choices=["r_manager_raw", "manager_reward_new", "manager_reward_old"],
        help=(
            "manager 曲线使用哪一列。"
            "默认 r_manager_raw；在 hrl新训练代码中，它实际就是环境返回的新的 manager reward。"
        )
    )

    ap.add_argument(
        "--manager_weight_mode",
        type=str,
        default="phase_size_sum_mi",
        choices=["phase_size_sum_mi", "assign_cnt", "equal"],
        help="训练阶段聚合 manager reward 时的权重方式"
    )

    args = ap.parse_args()

    main(
        csv_path=args.csv,
        out_dir=args.out,
        smooth_k=int(args.smooth),
        center_k=int(args.center_k),
        center_method=args.center_method,
        manager_reward_col=args.manager_reward_col,
        manager_weight_mode=args.manager_weight_mode,
    )