# -*- coding: utf-8 -*-
"""
plot_train_eval_hrl_3layer_overlay.py

把 4 个训练脚本（例如 batch64/128/256/512）输出的 CSV 叠加画到同一坐标系：
- 三层三个子图（VM / Host / Manager）
- 每个子图 4 条不同颜色曲线（对应 4 个 batch 的日志）

输出：
- evolution_train_3layer_overlay.png
- evolution_eval_3layer_overlay.png

说明：
- 训练日志 type=phase 里包含 is_micro（micro-phase 边界）
- Manager 的 train 聚合默认仅用 is_micro==0 的 phase 行（mgr_ignore_micro=1）
"""

import os
import re
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
        "is_micro",
    ]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    for c in [
        "episode", "assign_cnt", "phase_size_sum_mi",
        "r_vm_phase_mean", "r_host_phase_mean", "r_manager_raw",
        "eval_vm", "eval_host", "eval_mgr",
        "is_micro",
    ]:
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


def build_train_episode_rewards(
    df: pd.DataFrame,
    mgr_weight_mode="phase_size_sum_mi",
    mgr_ignore_micro: bool = True,
) -> pd.DataFrame:
    """
    train：从 type=phase 构造 per-episode 三层 reward

    - VM/Host：用 assign_cnt 加权（含 micro 行）
    - Manager：默认用 phase_size_sum_mi 加权；并且可忽略 micro 行（is_micro==1）
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
        train_vm = _weighted_avg(g["r_vm_phase_mean"].values, g["w_task"].values)
        train_host = _weighted_avg(g["r_host_phase_mean"].values, g["w_task"].values)

        g_mgr = g
        if mgr_ignore_micro and ("is_micro" in g.columns):
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
    ep = df[df["type"].str.lower().isin(["episode", "episode_reward"])].copy()
    ep = ep.dropna(subset=["episode"]).sort_values("episode")
    if ep.empty:
        return pd.DataFrame(columns=["episode", "eval_vm", "eval_host", "eval_mgr"])
    return ep[["episode", "eval_vm", "eval_host", "eval_mgr"]].copy()


# -----------------------------
# 叠加绘图
# -----------------------------


def _infer_label_from_path(p: str) -> str:
    """
    图例名称：Batch Size = 64/128/256/512
    优先从路径中识别 batch64/batch128/batch256/batch512；
    识别不到再从文件名里找；
    还识别不到就回退父目录名/文件名。
    """
    s = p.replace("\\", "/")
    m = re.search(r"batch[_-]?(64|128|256|512)", s, flags=re.IGNORECASE)
    if m:
        return f"Batch Size = {m.group(1)}"

    m2 = re.search(r"(64|128|256|512)", os.path.basename(s))
    if m2:
        return f"Batch Size = {m2.group(1)}"

    parent = os.path.basename(os.path.dirname(s))
    base = os.path.splitext(os.path.basename(s))[0]
    return parent if parent else base


def _plot_multi(ax, series_list, title, ylabel,
               smooth_k=11, center_k=101, center_method="mean",
               plot_mode="smoothed"):
    """
    series_list: list of dict {label, x, y}
    强制把 label 清洗成：Batch Size = 64/128/256/512
    """
    for s in series_list:
        raw_label = str(s.get("label", ""))

        # ✅ 关键：无论 raw_label 是不是路径，都再走一遍 _infer_label_from_path
        # 如果 raw_label 是长路径 -> 提取 batch64 -> "Batch Size = 64"
        # 如果 raw_label 已经是 "Batch Size = 64" -> 仍然会稳定返回 "Batch Size = 64"
        label = _infer_label_from_path(raw_label)

        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)

        mask = (~np.isnan(x)) & (~np.isnan(y))
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue

        if plot_mode == "raw":
            y_plot = y
        elif plot_mode == "center":
            y_plot = center_line(y, k=center_k, method=center_method)
        else:
            y_plot = moving_avg(y, smooth_k)

        ax.plot(x, y_plot, linewidth=2.0, linestyle="-", label=label)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")


def save_overlay_triplet(series_vm, series_host, series_mgr,
                         out_path, fig_title,
                         smooth_k=11, center_k=101, center_method="mean",
                         ylabel="Reward (as-is)",
                         plot_mode="smoothed"):
    """
    三行子图：VM / Host / Manager。每个子图多条曲线叠加（4条不同颜色）。
    """
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 9.4), sharex=True)

    _plot_multi(
        axes[0], series_vm,
        title="VM Layer Reward Evolution (overlay)",
        ylabel=ylabel,
        smooth_k=smooth_k, center_k=center_k, center_method=center_method,
        plot_mode=plot_mode
    )
    _plot_multi(
        axes[1], series_host,
        title="Host Layer Reward Evolution (overlay)",
        ylabel=ylabel,
        smooth_k=smooth_k, center_k=center_k, center_method=center_method,
        plot_mode=plot_mode
    )
    _plot_multi(
        axes[2], series_mgr,
        title="Manager Layer Reward Evolution (overlay)",
        ylabel=ylabel,
        smooth_k=smooth_k, center_k=center_k, center_method=center_method,
        plot_mode=plot_mode
    )

    fig.suptitle(fig_title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


# -----------------------------
# 主流程：读 4 个 CSV -> 分别构造 train/eval -> 叠加画图
# -----------------------------
def main(csv_paths, out_dir,
         smooth_k=11, center_k=101, center_method="mean",
         mgr_weight_mode="phase_size_sum_mi",
         mgr_ignore_micro=True,
         plot_mode="smoothed"):

    os.makedirs(out_dir, exist_ok=True)

    # 收集：每个 CSV 对应一条曲线
    train_vm_series, train_host_series, train_mgr_series = [], [], []
    eval_vm_series, eval_host_series, eval_mgr_series = [], [], []

    any_train = False
    any_eval = False

    for p in csv_paths:
        label = _infer_label_from_path(p)
        df = safe_load_csv(p)
        if df.empty:
            print(f"[WARN] 跳过（空/读取失败）：{p}")
            continue

        # ---- train ----
        train_df = build_train_episode_rewards(
            df,
            mgr_weight_mode=mgr_weight_mode,
            mgr_ignore_micro=bool(mgr_ignore_micro),
        )
        if not train_df.empty:
            any_train = True
            ep = train_df["episode"].values
            train_vm_series.append({"label": label, "x": ep, "y": train_df["train_vm"].values})
            train_host_series.append({"label": label, "x": ep, "y": train_df["train_host"].values})
            train_mgr_series.append({"label": label, "x": ep, "y": train_df["train_mgr"].values})
        else:
            print(f"[WARN] {label} 未找到 type=phase（train）数据：{p}")

        # ---- eval ----
        eval_df = build_eval_episode_rewards(df)
        if not eval_df.empty:
            any_eval = True
            ep2 = eval_df["episode"].values
            eval_vm_series.append({"label": label, "x": ep2, "y": eval_df["eval_vm"].values})
            eval_host_series.append({"label": label, "x": ep2, "y": eval_df["eval_host"].values})
            eval_mgr_series.append({"label": label, "x": ep2, "y": eval_df["eval_mgr"].values})
        else:
            print(f"[WARN] {label} 未找到 type=episode（eval）数据：{p}")

    # 画 train overlay
    if any_train:
        out = os.path.join(out_dir, "evolution_train_3layer_overlay.png")
        save_overlay_triplet(
            train_vm_series, train_host_series, train_mgr_series,
            out_path=out,
            fig_title="Training Evolution (3-layer overlay)  [episode x reward]",
            smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method,
            ylabel="Reward (as-is)",
            plot_mode=plot_mode
        )
        print(f"[OK] 保存：{os.path.abspath(out)}")
    else:
        print("[WARN] 所有 CSV 都没有 train(phase) 数据，跳过 train overlay。")

    # 画 eval overlay
    if any_eval:
        out = os.path.join(out_dir, "evolution_eval_3layer_overlay.png")
        save_overlay_triplet(
            eval_vm_series, eval_host_series, eval_mgr_series,
            out_path=out,
            fig_title="Evaluation Evolution (3-layer overlay)  [episode x reward]",
            smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method,
            ylabel="Reward (as-is)",
            plot_mode=plot_mode
        )
        print(f"[OK] 保存：{os.path.abspath(out)}")
    else:
        print("[WARN] 所有 CSV 都没有 eval(episode) 数据，跳过 eval overlay。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # 关键：支持一次性传入 4 个 CSV
    ap.add_argument(
        "--csvs", type=str, nargs="+", required=True,
        help="多个CSV路径（例如 batch64/batch128/batch256/batch512 各一个）"
    )
    ap.add_argument(
        "--out", type=str,
        default="../out/plots/plots_hrl_tri_overlay",
        help="输出目录"
    )

    # 平滑/中心线参数
    ap.add_argument("--smooth", type=int, default=51)
    ap.add_argument("--center_k", type=int, default=101)
    ap.add_argument("--center_method", type=str, default="mean", choices=["mean", "median"])

    # train 聚合口径（保持你原来的）
    ap.add_argument(
        "--mgr_weight_mode", type=str, default="phase_size_sum_mi",
        choices=["phase_size_sum_mi", "assign_cnt", "equal"],
        help="train 聚合时 manager 的加权方式（默认按 phase_size_sum_mi）"
    )
    ap.add_argument(
        "--mgr_ignore_micro", type=int, default=1, choices=[0, 1],
        help="1=manager 聚合忽略 is_micro==1 的 phase 行（默认开启，推荐与训练口径对齐）"
    )

    # 多曲线情况下默认只画 smoothed，更清爽
    ap.add_argument(
        "--plot_mode", type=str, default="smoothed",
        choices=["smoothed", "center", "raw"],
        help="叠加绘图模式：smoothed(默认)/center/raw"
    )

    args = ap.parse_args()

    main(
        csv_paths=args.csvs,
        out_dir=args.out,
        smooth_k=args.smooth,
        center_k=args.center_k,
        center_method=args.center_method,
        mgr_weight_mode=args.mgr_weight_mode,
        mgr_ignore_micro=bool(args.mgr_ignore_micro),
        plot_mode=args.plot_mode,
    )


r"""
python plot_train_eval_hrl_3layer_overlay.py `
  --csvs `
  ..\out\logs\logs_hrl_3layer_routeA_smallTask_medRes_b64\train_metrics_3layer_routeA.csv `
  ..\out\logs\logs_hrl_3layer_routeA_smallTask_medRes_b128\train_metrics_3layer_routeA.csv `
  ..\out\logs\logs_hrl_3layer_routeA_smallTask_medRes\train_metrics_3layer_routeA.csv `
  ..\out\logs\logs_hrl_3layer_routeA_smallTask_medRes_b512\train_metrics_3layer_routeA.csv `
  --out ..\out\plots\plots_hrl_tri_medRes_overlay

"""
