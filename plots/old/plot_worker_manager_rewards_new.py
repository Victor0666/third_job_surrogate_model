# -*- coding: utf-8 -*-
"""
plot_train_eval_hrl_mll.py  (human-friendly units)

在“奖励拆分环境”HRL训练日志上同时绘制 **原始口径** 与 **人眼友好口径**（单位换算后）的
训练与评测进化曲线。

新增能力：
- Worker(eval/train 可选)：把奖励换算为 (deadline - finish) 的 **秒数**（早完为正，迟到为负）
- Manager(eval/train 可选)：把奖励换算为 **mJ/MI**（正值=能耗强度，便于直观比较）

注意：
- 评测(eval)的 reward 一般是“原始 raw”，建议开启换算（默认开启）。
- 训练(train)的 ep_reward_* 在很多脚本里是“归一化后的值”（≈[-1,1]），默认不进行单位换算，
  以免误导；如你确定日志里就是 raw，可用 --assume_train_raw 开启换算。

用法示例（项目根目录）：
python -m compare.plot_train_eval_hrl_mll \
  --csv logs_hrl_new_design/train_metrics_energy_reward.csv \
  --out logs_hrl_new_design/plots_train_eval \
  --beta 0.05 --norm 300 --energy_scale 1e-3 \
  --smooth 5 --center_k 101 --center_method mean
"""

import os
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXPECTED_ALL = [
    "episode","type",
    "ep_reward_worker","ep_reward_manager",
    "eval_reward_worker","eval_reward_manager",
]

def safe_load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV 不存在：{csv_path}")
        return pd.DataFrame(columns=EXPECTED_ALL)

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=EXPECTED_ALL)
        name_to_idx = {name: i for i, name in enumerate(header)}
        for r in reader:
            row = {}
            for name in EXPECTED_ALL:
                row[name] = r[name_to_idx[name]] if name in name_to_idx and name_to_idx[name] < len(r) else ""
            rows.append(row)

    df = pd.DataFrame(rows, columns=EXPECTED_ALL)
    # num_cols = ["episode","ep_reward_worker","ep_reward_manager","eval_worker_weighted_mean","eval_reward_manager"]
    num_cols = ["episode","ep_reward_worker","ep_reward_manager","eval_reward_worker","eval_reward_manager"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def moving_avg(x, k):
    if k is None or k <= 1:
        return np.asarray(x, dtype=float)
    return pd.Series(x, dtype="float64").rolling(window=int(k), min_periods=1).mean().values

def center_line(y, k=101, method="mean"):
    y = np.asarray(y, dtype=float)
    if k is None or k <= 1:
        return y
    ser = pd.Series(y, dtype="float64")
    if method.lower().strip() == "median":
        return ser.rolling(window=int(k), min_periods=1, center=True).median().values
    return ser.rolling(window=int(k), min_periods=1, center=True).mean().values

def _plot_series(ax, x, y, title, ylabel="Reward", smooth_k=5, center_k=101, center_method="mean"):
    # ax.plot(x, y, linestyle="None", marker="o", markersize=2.5, alpha=0.45, label="points")
    y_s = moving_avg(y, smooth_k)
    ax.plot(x, y_s, linewidth=1.8, linestyle="-", label=f"smoothed k={smooth_k}")
    y_c = center_line(y, k=center_k, method=center_method)
    ax.plot(x, y_c, linewidth=2.4, linestyle="--", label=f"center {center_method}, k={center_k}")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

def save_dual(ep, y_top, y_bottom, out_path, top_title, bottom_title,
              top_ylabel="Reward", bottom_ylabel="Reward",
              smooth_k=5, center_k=101, center_method="mean"):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    _plot_series(
        axes[0], ep, y_top, top_title,
        ylabel=top_ylabel, smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )
    _plot_series(
        axes[1], ep, y_bottom, bottom_title,
        ylabel=bottom_ylabel, smooth_k=smooth_k, center_k=center_k, center_method=center_method
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def save_single(ep, y, out_path, title, ylabel="Reward", smooth_k=5, center_k=101, center_method="mean"):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    _plot_series(ax, ep, y, title, ylabel=ylabel,
                 smooth_k=smooth_k, center_k=center_k, center_method=center_method)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# === 单位换算 ===
def convert_worker_to_seconds(arr, beta, norm):
    """deadline-finish 的秒数（早完为正，迟到为负）"""
    if arr is None:
        return None
    if beta is None or norm is None or beta == 0:
        return None
    return arr * (float(norm) / float(beta))

def convert_manager_to_mJ_per_MI(arr, energy_scale):
    """mJ/MI（正值=能耗强度），把 reward 里的负号与 energy_scale 还原"""
    if arr is None:
        return None
    if energy_scale is None or energy_scale == 0:
        return None
    return (-arr / float(energy_scale)) * 1e3  # J→mJ

def main(csv_path, out_dir, smooth_k=5, center_k=101, center_method="mean",
         beta=0.05, norm=300.0, energy_scale=1e-3,
         convert_eval=True, assume_train_raw=False, also_show=False):
    os.makedirs(out_dir, exist_ok=True)
    df = safe_load_csv(csv_path)
    if df.empty:
        print("[WARN] 日志为空或缺少必要列。")
        return

    # 只保留 episode 级记录（兼容 'episode' / 'episode_reward'）
    if "type" in df.columns and df["type"].notna().any():
        ep_df = df[df["type"].isin(["episode", "episode_reward"])].copy()
        if ep_df.empty:
            ep_df = df.dropna(subset=["episode"]).copy()
    else:
        ep_df = df.dropna(subset=["episode"]).copy()

    ep_df = ep_df.dropna(subset=["episode"]).sort_values("episode")
    ep = ep_df["episode"].values.astype(float)

    # ====== 评测（eval）曲线：原始 + 人眼单位 ======
    wr_eval = ep_df["eval_reward_worker"].values.astype(float) if "eval_reward_worker" in ep_df else None
    mr_eval = ep_df["eval_reward_manager"].values.astype(float) if "eval_reward_manager" in ep_df else None

    if wr_eval is not None and mr_eval is not None:
        dual_eval = os.path.join(out_dir, "evolution_eval_dual.png")
        save_dual(ep, wr_eval, mr_eval, dual_eval,
                  top_title="Worker Evaluation Reward (raw)",
                  bottom_title="Manager Evaluation Reward (raw)",
                  top_ylabel="Reward", bottom_ylabel="Reward",
                  smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
        print(f"[OK] 双子图（eval, raw）保存：{os.path.abspath(dual_eval)}")

    if wr_eval is not None and not np.isnan(wr_eval).all():
        w_eval = os.path.join(out_dir, "evolution_eval_worker_weighted_mean.png")
        save_single(ep, wr_eval, w_eval, "Worker Evaluation Reward (raw)",
                    ylabel="Reward",
                    smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
        print(f"[OK] Worker eval（raw）保存：{os.path.abspath(w_eval)}")

    if mr_eval is not None and not np.isnan(mr_eval).all():
        m_eval = os.path.join(out_dir, "evolution_eval_reward_manager.png")
        save_single(ep, mr_eval, m_eval, "Manager Evaluation Reward (raw)",
                    ylabel="Reward",
                    smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
        print(f"[OK] Manager eval（raw）保存：{os.path.abspath(m_eval)}")

    # —— 评测的人眼单位版本（推荐看这个） ——
    if convert_eval:
        wr_eval_sec   = convert_worker_to_seconds(wr_eval, beta, norm)
        mr_eval_mJMI  = convert_manager_to_mJ_per_MI(mr_eval, energy_scale)

        if wr_eval_sec is not None and not np.isnan(wr_eval_sec).all():
            out = os.path.join(out_dir, "evolution_eval_worker_seconds.png")
            save_single(ep, wr_eval_sec, out,
                        title="Worker Evaluation (lateness lead, seconds)",
                        ylabel="Seconds (+early / −late)",
                        smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
            print(f"[OK] Worker eval（秒）保存：{os.path.abspath(out)}")

        if mr_eval_mJMI is not None and not np.isnan(mr_eval_mJMI).all():
            out = os.path.join(out_dir, "evolution_eval_manager_mJ_per_MI.png")
            save_single(ep, mr_eval_mJMI, out,
                        title="Manager Evaluation (energy intensity, mJ/MI)",
                        ylabel="mJ per MI",
                        smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
            print(f"[OK] Manager eval（mJ/MI）保存：{os.path.abspath(out)}")

        if (wr_eval_sec is not None) and (mr_eval_mJMI is not None):
            out = os.path.join(out_dir, "evolution_eval_dual_human.png")
            save_dual(ep, wr_eval_sec, mr_eval_mJMI, out,
                      top_title="Worker Evaluation (seconds)",
                      bottom_title="Manager Evaluation (mJ/MI)",
                      top_ylabel="Seconds (+early / −late)",
                      bottom_ylabel="mJ per MI",
                      smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
            print(f"[OK] 双子图（eval, human units）保存：{os.path.abspath(out)}")

    # ====== 训练（train）曲线：原始 +（可选）人眼单位 ======
    wr_train = ep_df["ep_reward_worker"].values.astype(float) if "ep_reward_worker" in ep_df else None
    mr_train = ep_df["ep_reward_manager"].values.astype(float) if "ep_reward_manager" in ep_df else None

    if wr_train is not None and mr_train is not None:
        dual_train = os.path.join(out_dir, "evolution_train_dual.png")
        save_dual(ep, wr_train, mr_train, dual_train,
                  top_title="Worker Training Reward (per-episode, as-is)",
                  bottom_title="Manager Training Reward (per-episode, as-is)",
                  top_ylabel="Reward (as-is)", bottom_ylabel="Reward (as-is)",
                  smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
        print(f"[OK] 双子图（train, as-is）保存：{os.path.abspath(dual_train)}")

    if wr_train is not None and not np.isnan(wr_train).all():
        out = os.path.join(out_dir, "evolution_train_reward_worker.png")
        save_single(ep, wr_train, out,
                    title="Worker Training Reward (as-is)",
                    ylabel="Reward (as-is)",
                    smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
        print(f"[OK] Worker train（as-is）保存：{os.path.abspath(out)}")

    if mr_train is not None and not np.isnan(mr_train).all():
        out = os.path.join(out_dir, "evolution_train_reward_manager.png")
        save_single(ep, mr_train, out,
                    title="Manager Training Reward (as-is)",
                    ylabel="Reward (as-is)",
                    smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
        print(f"[OK] Manager train（as-is）保存：{os.path.abspath(out)}")

    # —— 训练的人眼单位版本（仅当你确认 ep_reward_* 是 raw 才开启） ——
    if assume_train_raw:
        wr_train_sec  = convert_worker_to_seconds(wr_train, beta, norm)
        mr_train_mJMI = convert_manager_to_mJ_per_MI(mr_train, energy_scale)

        if wr_train_sec is not None and not np.isnan(wr_train_sec).all():
            out = os.path.join(out_dir, "evolution_train_worker_seconds.png")
            save_single(ep, wr_train_sec, out,
                        title="Worker Training (seconds, assuming raw)",
                        ylabel="Seconds (+early / −late)",
                        smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
            print(f"[OK] Worker train（秒, raw 假设）保存：{os.path.abspath(out)}")

        if mr_train_mJMI is not None and not np.isnan(mr_train_mJMI).all():
            out = os.path.join(out_dir, "evolution_train_manager_mJ_per_MI.png")
            save_single(ep, mr_train_mJMI, out,
                        title="Manager Training (mJ/MI, assuming raw)",
                        ylabel="mJ per MI",
                        smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
            print(f"[OK] Manager train（mJ/MI, raw 假设）保存：{os.path.abspath(out)}")

        if (wr_train_sec is not None) and (mr_train_mJMI is not None):
            out = os.path.join(out_dir, "evolution_train_dual_human.png")
            save_dual(ep, wr_train_sec, mr_train_mJMI, out,
                      top_title="Worker Training (seconds, assuming raw)",
                      bottom_title="Manager Training (mJ/MI, assuming raw)",
                      top_ylabel="Seconds (+early / −late)",
                      bottom_ylabel="mJ per MI",
                      smooth_k=int(smooth_k), center_k=int(center_k), center_method=center_method)
            print(f"[OK] 双子图（train, human units, raw 假设）保存：{os.path.abspath(out)}")

    if also_show:
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="../out/logs/logs_hrl_3layer_routeA_smallTask_smallRes/train_metrics_3layer_routeA.csv")
    ap.add_argument("--out", type=str, default="../out/plots/plots_train_eval_hrl_3layer_routeA_smallTask_smallRes")
    ap.add_argument("--smooth", type=int, default=11, help="滚动平均窗口（基本平滑）")
    ap.add_argument("--center_k", type=int, default=101, help="中心线窗口（建议奇数）")
    ap.add_argument("--center_method", type=str, default="mean", choices=["mean","median"], help="中心线统计方法")
    # 单位换算所需参数（与训练脚本保持一致）
    ap.add_argument("--beta", type=float, default=1, help="task_baseline_beta")
    ap.add_argument("--norm", type=float, default=60.0, help="task_baseline_norm")
    ap.add_argument("--energy_scale", type=float, default=1, help="energy_reward_scale")
    ap.add_argument("--convert_eval", action="store_true", default=True,
                    help="对 eval_* 进行单位换算（默认 True）")
    ap.add_argument("--no_convert_eval", action="store_false", dest="convert_eval")
    ap.add_argument("--assume_train_raw", action="store_true",
                    help="若 ep_reward_* 为 raw，则对其做单位换算（默认 False）")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    main(csv_path=args.csv,
         out_dir=args.out,
         smooth_k=args.smooth,
         center_k=args.center_k,
         center_method=args.center_method,
         beta=args.beta, norm=args.norm, energy_scale=args.energy_scale,
         convert_eval=args.convert_eval,
         assume_train_raw=args.assume_train_raw,
         also_show=args.show)
