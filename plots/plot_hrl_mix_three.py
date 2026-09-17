# -*- coding: utf-8 -*-
"""
plot_train_eval_hrl_new_triplet_LMT.py

把 hrl 新训练代码在 Loose / Medium / Tight 三种 model 下的结果，
合并绘制到同一张大图中。

输出：
1) evolution_train_3layer_triplet_LMT.png   -> 3行(层) x 3列(model)
2) evolution_eval_3layer_triplet_LMT.png    -> 3行(层) x 3列(model)
3) episode_energy_eval_curve_LMT.png        -> 1行 x 3列(model)
4) episode_lateness_curve_LMT.png           -> 1行 x 3列(model)

说明：
- 每一列对应一个 model：Loose / Medium / Tight
- 对于三层图：
    第1行：VM
    第2行：Host
    第3行：Manager
- 默认仍沿用你原来的日志目录命名规则
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
    VM/Host 默认按 assign_cnt 加权。
    Manager 默认按 phase_size_sum_mi 加权。
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
        rows.append({
            "episode": float(ep),
            "train_vm": weighted_avg(g["r_vm_phase_mean"].values, g["w_task"].values),
            "train_host": weighted_avg(g["r_host_phase_mean"].values, g["w_task"].values),
            "train_mgr": weighted_avg(g[manager_reward_col].values, g["w_mgr"].values),
        })

    return pd.DataFrame(rows).sort_values("episode")


def build_eval_episode_rewards(df):
    ep = df[df["type"].str.lower().str.strip() == "episode"].copy()
    ep = ep.dropna(subset=["episode"]).sort_values("episode")

    if ep.empty:
        return pd.DataFrame(columns=["episode", "eval_vm", "eval_host", "eval_mgr", "eval_energy"])

    ep = ep.groupby("episode", sort=True).tail(1)
    return ep[["episode", "eval_vm", "eval_host", "eval_mgr", "eval_energy"]].copy()


def build_episode_scalar_metrics(df):
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
# 路径构造
# =========================================================
def build_csv_path(root_logs, res_model, model):
    """
    根据你原来的命名规则生成 CSV 路径
    """
    folder = (
        f"logs_hrl_3layer_routeA_mgc_ave_smallTask_{res_model}"
        f"_seed1_mgrDelayEnergy_mix_rAlpha075_dalphaMix_HVrn_{model}"
    )
    return os.path.join(root_logs, folder, "train_metrics_3layer_routeA.csv")


def load_all_models(root_logs, res_model, models):
    data = {}
    for m in models:
        csv_path = build_csv_path(root_logs, res_model, m)
        print(f"[INFO] 读取 {m}: {csv_path}")
        df = safe_load_csv(csv_path)
        data[m] = {
            "csv_path": csv_path,
            "raw": df,
            "ok": (not df.empty)
        }
    return data


# =========================================================
# 单子图绘制工具
# =========================================================
def plot_one_series(ax, x, y, title, ylabel,
                    smooth_k=11, center_k=101, center_method="mean",
                    color_smooth="#1f77b4", color_center="#d62728"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0 or len(y) == 0:
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.grid(alpha=0.25)
        return

    y_s = moving_avg(y, smooth_k)
    y_c = center_line(y, center_k, center_method)

    ax.plot(x, y_s, linewidth=1.8, linestyle="-", color=color_smooth, label=f"smoothed k={smooth_k}")
    ax.plot(x, y_c, linewidth=2.0, linestyle="--", color=color_center, label=f"center {center_method}, k={center_k}")

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def plot_two_series(ax, x, y1, y2, title, ylabel,
                    label1="Task Avg Lateness", label2="Workflow Avg Lateness",
                    smooth_k=11):
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.grid(alpha=0.25)
        return

    y1 = moving_avg(np.asarray(y1, dtype=float), smooth_k)
    y2 = moving_avg(np.asarray(y2, dtype=float), smooth_k)

    ax.plot(x, y1, linewidth=2.0, linestyle="-", label=label1)
    ax.plot(x, y2, linewidth=2.0, linestyle="--", label=label2)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


# =========================================================
# 合并绘图
# =========================================================
def save_triplet_merged(train_or_eval_dict, out_path, fig_title,
                        value_keys=("vm", "host", "mgr"),
                        model_order=("Loose", "Medium", "Tight"),
                        smooth_k=11, center_k=101, center_method="mean",
                        ylabel="Reward"):
    row_titles = ["VM Layer", "Host Layer", "Manager Layer"]

    fig, axes = plt.subplots(
        3, len(model_order),
        figsize=(5.3 * len(model_order), 10.0),
        sharex=False
    )

    if len(model_order) == 1:
        axes = np.array(axes).reshape(3, 1)

    for col, model in enumerate(model_order):
        df = train_or_eval_dict.get(model, pd.DataFrame())

        if df is None or df.empty:
            for row in range(3):
                ax = axes[row, col]
                ax.set_title(f"{model} - {row_titles[row]}")
                ax.set_xlabel("Episode")
                ax.set_ylabel(ylabel)
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.grid(alpha=0.25)
            continue

        x = df["episode"].values

        key_map = {
            "vm": [c for c in df.columns if c.endswith("_vm") or c == "eval_vm" or c == "train_vm"],
            "host": [c for c in df.columns if c.endswith("_host") or c == "eval_host" or c == "train_host"],
            "mgr": [c for c in df.columns if c.endswith("_mgr") or c == "eval_mgr" or c == "train_mgr"],
        }

        y_vm = df[key_map["vm"][0]].values
        y_host = df[key_map["host"][0]].values
        y_mgr = df[key_map["mgr"][0]].values
        ys = [y_vm, y_host, y_mgr]

        for row in range(3):
            ax = axes[row, col]
            plot_one_series(
                ax,
                x, ys[row],
                title=f"{model} - {row_titles[row]}",
                ylabel=ylabel,
                smooth_k=smooth_k,
                center_k=center_k,
                center_method=center_method
            )

    fig.suptitle(fig_title, y=0.995, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_energy_merged(scalar_dict, out_path, fig_title,
                       model_order=("Loose", "Medium", "Tight"),
                       smooth_k=11, center_k=101, center_method="mean"):
    fig, axes = plt.subplots(1, len(model_order), figsize=(5.4 * len(model_order), 4.8), sharey=False)
    if len(model_order) == 1:
        axes = [axes]

    for i, model in enumerate(model_order):
        ax = axes[i]
        df = scalar_dict.get(model, pd.DataFrame())

        if df is None or df.empty or df["eval_energy"].isna().all():
            ax.set_title(f"{model} - Evaluation Energy")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Energy (J)")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(alpha=0.25)
            continue

        plot_one_series(
            ax,
            df["episode"].values,
            df["eval_energy"].values,
            title=f"{model} - Evaluation Energy",
            ylabel="Energy (J)",
            smooth_k=smooth_k,
            center_k=center_k,
            center_method=center_method
        )

    fig.suptitle(fig_title, y=0.995, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_lateness_merged(scalar_dict, out_path, fig_title,
                         model_order=("Loose", "Medium", "Tight"),
                         smooth_k=11):
    fig, axes = plt.subplots(1, len(model_order), figsize=(5.4 * len(model_order), 4.8), sharey=False)
    if len(model_order) == 1:
        axes = [axes]

    for i, model in enumerate(model_order):
        ax = axes[i]
        df = scalar_dict.get(model, pd.DataFrame())

        no_data = (
            df is None or df.empty or
            (df["ep_avg_lateness"].isna().all() and df["wf_avg_lateness"].isna().all())
        )

        if no_data:
            ax.set_title(f"{model} - Lateness")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Lateness (s)")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(alpha=0.25)
            continue

        plot_two_series(
            ax,
            df["episode"].values,
            df["ep_avg_lateness"].values,
            df["wf_avg_lateness"].values,
            title=f"{model} - Lateness",
            ylabel="Lateness (s)",
            label1="Task Avg Lateness",
            label2="Workflow Avg Lateness",
            smooth_k=smooth_k
        )

    fig.suptitle(fig_title, y=0.995, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


# =========================================================
# 主流程
# =========================================================
def main(root_logs, out_dir, res_model,
         models=("Loose", "Medium", "Tight"),
         smooth_k=101, center_k=101, center_method="mean",
         manager_reward_col="r_manager_raw",
         manager_weight_mode="phase_size_sum_mi"):
    os.makedirs(out_dir, exist_ok=True)

    data = load_all_models(root_logs=root_logs, res_model=res_model, models=models)

    train_dict = {}
    eval_dict = {}
    scalar_dict = {}

    for model in models:
        df = data[model]["raw"]

        if df is None or df.empty:
            train_dict[model] = pd.DataFrame()
            eval_dict[model] = pd.DataFrame()
            scalar_dict[model] = pd.DataFrame()
            continue

        train_dict[model] = build_train_episode_rewards(
            df,
            manager_reward_col=manager_reward_col,
            manager_weight_mode=manager_weight_mode,
        )
        eval_dict[model] = build_eval_episode_rewards(df)
        scalar_dict[model] = build_episode_scalar_metrics(df)

    # 1) train 三层合并图
    out_train = os.path.join(out_dir, "evolution_train_3layer_triplet_LMT.png")
    save_triplet_merged(
        train_or_eval_dict=train_dict,
        out_path=out_train,
        fig_title=f"Training Evolution (HRL New, 3-layer) - {res_model}",
        model_order=models,
        smooth_k=smooth_k,
        center_k=center_k,
        center_method=center_method,
        ylabel="Reward",
    )
    print(f"[OK] 保存：{os.path.abspath(out_train)}")

    # 2) eval 三层合并图
    out_eval = os.path.join(out_dir, "evolution_eval_3layer_triplet_LMT.png")
    save_triplet_merged(
        train_or_eval_dict=eval_dict,
        out_path=out_eval,
        fig_title=f"Evaluation Evolution (HRL New, 3-layer) - {res_model}",
        model_order=models,
        smooth_k=smooth_k,
        center_k=center_k,
        center_method=center_method,
        ylabel="Reward",
    )
    print(f"[OK] 保存：{os.path.abspath(out_eval)}")

    # 3) eval_energy 合并图
    out_energy = os.path.join(out_dir, "episode_energy_eval_curve_LMT.png")
    save_energy_merged(
        scalar_dict=scalar_dict,
        out_path=out_energy,
        fig_title=f"Evaluation Energy Evolution - {res_model}",
        model_order=models,
        smooth_k=smooth_k,
        center_k=center_k,
        center_method=center_method,
    )
    print(f"[OK] 保存：{os.path.abspath(out_energy)}")

    # 4) lateness 合并图
    out_late = os.path.join(out_dir, "episode_lateness_curve_LMT.png")
    save_lateness_merged(
        scalar_dict=scalar_dict,
        out_path=out_late,
        fig_title=f"Lateness Evolution - {res_model}",
        model_order=models,
        smooth_k=smooth_k,
    )
    print(f"[OK] 保存：{os.path.abspath(out_late)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # resModel 可改为 smallRes / medRes / largeRes
    resModel = "largeRes"

    ap.add_argument(
        "--root_logs",
        type=str,
        default="../out/logs",
        help="日志总目录，例如 ../out/logs"
    )
    ap.add_argument(
        "--out",
        type=str,
        default="../out/plots/plots_hrl_new_smallTask_" + resModel + "_mgrDelayEnergy_mix_rAlpha075_dalphaMix_HVrn_LMT",
        help="输出图片目录"
    )
    ap.add_argument(
        "--res_model",
        type=str,
        default=resModel,
        choices=["smallRes", "medRes", "largeRes"],
        help="资源规模"
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["Loose", "Medium", "Tight"],
        help="要合并的 model 列表"
    )

    ap.add_argument("--smooth", type=int, default=101)
    ap.add_argument("--center_k", type=int, default=101)
    ap.add_argument("--center_method", type=str, default="mean", choices=["mean", "median"])

    ap.add_argument(
        "--manager_reward_col",
        type=str,
        default="r_manager_raw",
        choices=["r_manager_raw", "manager_reward_new", "manager_reward_old"],
        help="manager 曲线使用哪一列"
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
        root_logs=args.root_logs,
        out_dir=args.out,
        res_model=args.res_model,
        models=tuple(args.models),
        smooth_k=int(args.smooth),
        center_k=int(args.center_k),
        center_method=args.center_method,
        manager_reward_col=args.manager_reward_col,
        manager_weight_mode=args.manager_weight_mode,
    )