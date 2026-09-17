# -*- coding: utf-8 -*-
"""
plot_train_eval_hrl_new_triplet_overlay_LMT.py

功能：
1) 同时读取 Loose / Medium / Tight 三种情况下的日志
2) 对于上/中/下三层，相同层画在同一坐标系内
3) evolution_eval_3layer_overlay.png:
   - 删除实线（center_line）
   - 只保留更明显的虚线（moving_avg）
   - 图例放在右下角
   - 纵坐标使用科学计数法
4) 所有标题已删除
5) 同时保存 PNG 和 PDF
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# =========================
# 全局字体设置
# =========================
FONT_FAMILY = "Times New Roman"   # 可改成 "Arial"、"SimSun" 等
FONT_SIZE = 20                    # 全局默认字号
AXIS_LABEL_SIZE = 20              # 坐标轴标题
TICK_LABEL_SIZE = 20              # 坐标轴刻度
LEGEND_SIZE = 20                  # 图例正文
LEGEND_TITLE_SIZE = 20            # 图例标题
SCI_OFFSET_SIZE = 20              # 科学计数法 1eX 的字号

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.labelsize"] = AXIS_LABEL_SIZE
plt.rcParams["xtick.labelsize"] = TICK_LABEL_SIZE
plt.rcParams["ytick.labelsize"] = TICK_LABEL_SIZE
plt.rcParams["legend.fontsize"] = LEGEND_SIZE

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
        print(f"[WARN] 文件不存在，跳过：{csv_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        print(f"[WARN] 读取失败：{csv_path}\n{e}")
        return pd.DataFrame()

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
# 从单个日志中构造 train / eval / scalar
# =========================================================
def build_train_episode_rewards(df, manager_reward_col="r_manager_raw", manager_weight_mode="phase_size_sum_mi"):
    ph = df[df["type"].str.lower().str.strip() == "phase"].copy()
    ph = ph.dropna(subset=["episode"])

    if ph.empty:
        return pd.DataFrame(columns=["episode", "train_vm", "train_host", "train_mgr"])

    if manager_reward_col not in ph.columns:
        raise ValueError(f"manager_reward_col={manager_reward_col} 不存在于 CSV 中")

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
        return pd.DataFrame(columns=["episode", "eval_vm", "eval_host", "eval_mgr"])

    ep = ep.groupby("episode", sort=True).tail(1)
    return ep[["episode", "eval_vm", "eval_host", "eval_mgr"]].copy()


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
    return os.path.join(
        root_logs,
        f"logs_hrl_3layer_routeA_mgc_ave_smallTask_{res_model}"
        f"_seed1_mgrDelayEnergy_mix_rAlpha075_dalphaMix_HVrn_{model}",
        "train_metrics_3layer_routeA.csv"
    )


def load_all_models(root_logs, res_model, models):
    result = {}
    for model in models:
        csv_path = build_csv_path(root_logs, res_model, model)
        print(f"[INFO] 读取 {model}: {csv_path}")
        df = safe_load_csv(csv_path)

        if df.empty:
            result[model] = {
                "raw": pd.DataFrame(),
                "train": pd.DataFrame(),
                "eval": pd.DataFrame(),
                "scalar": pd.DataFrame(),
            }
            continue

        result[model] = {
            "raw": df,
            "train": None,
            "eval": None,
            "scalar": None,
        }

    return result


# =========================================================
# 通用坐标轴设置
# =========================================================
def set_yaxis_scientific(ax):
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(SCI_OFFSET_SIZE)


def save_png_and_pdf(fig, out_path, dpi=180):
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")


# =========================================================
# 相同层叠加绘图：reward / energy
# mode:
#   "trend_only"  -> 只画实线(center_line)
#   "smooth_only" -> 只画虚线(moving_avg)
# =========================================================
def plot_overlay_series(ax, data_dict, key, ylabel,
                        smooth_k=11, center_k=101, center_method="mean",
                        color_map=None,
                        mode="trend_only",
                        legend_loc="upper right",
                        sci_y=False):
    if color_map is None:
        color_map = {
            "Loose": "#1f77b4",
            "Medium": "#ff7f0e",
            "Tight": "#2ca02c",
        }

    found = False
    model_handles = []

    for model, df in data_dict.items():
        if df is None or df.empty or key not in df.columns or "episode" not in df.columns:
            continue

        x = np.asarray(df["episode"].values, dtype=float)
        y = np.asarray(df[key].values, dtype=float)
        if len(x) == 0:
            continue

        y_s = moving_avg(y, smooth_k)
        y_c = center_line(y, center_k, center_method)
        color = color_map.get(model, None)

        if mode == "smooth_only":
            ax.plot(
                x, y_s,
                linewidth=3.0,
                alpha=1.0,
                color=color,
                linestyle=(0, (12, 4)),
                zorder=3
            )
            model_handles.append(
                Line2D([0], [0], color=color, lw=3.0, linestyle=(0, (12, 4)), label=model)
            )
        else:
            ax.plot(
                x, y_c,
                linewidth=2.6,
                alpha=0.98,
                color=color,
                linestyle="-",
                zorder=3
            )
            model_handles.append(
                Line2D([0], [0], color=color, lw=2.6, linestyle="-", label=model)
            )

        found = True

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25)

    if sci_y:
        set_yaxis_scientific(ax)

    if found:
        ax.legend(
            handles=model_handles,
            title="Scenario",
            loc=legend_loc,
            fontsize=LEGEND_SIZE,
            title_fontsize=LEGEND_TITLE_SIZE,
            frameon=True
        )
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


# =========================================================
# lateness 叠加绘图
# =========================================================
def save_lateness_overlay(scalar_dict, out_path, smooth_k=11):
    color_map = {
        "Loose": "#1f77b4",
        "Medium": "#ff7f0e",
        "Tight": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    found = False

    for model, df in scalar_dict.items():
        if df is None or df.empty or "ep_avg_lateness" not in df.columns or "episode" not in df.columns:
            continue

        x = np.asarray(df["episode"].values, dtype=float)
        y = np.asarray(df["ep_avg_lateness"].values, dtype=float)
        if len(x) == 0:
            continue

        y_s = moving_avg(y, smooth_k)
        ax.plot(
            x, y_s,
            linewidth=2.3,
            linestyle="-",
            color=color_map.get(model, None)
        )
        found = True

    for model, df in scalar_dict.items():
        if df is None or df.empty or "wf_avg_lateness" not in df.columns or "episode" not in df.columns:
            continue

        x = np.asarray(df["episode"].values, dtype=float)
        y = np.asarray(df["wf_avg_lateness"].values, dtype=float)
        if len(x) == 0:
            continue

        y_s = moving_avg(y, smooth_k)
        ax.plot(
            x, y_s,
            linewidth=2.8,
            linestyle=(0, (8, 4)),
            color=color_map.get(model, None),
            alpha=1.0
        )
        found = True

    ax.set_xlabel("Episode")
    ax.set_ylabel("Lateness (s)")
    ax.grid(alpha=0.25)

    if found:
        scenario_handles = [
            Line2D([0], [0], color=color_map["Loose"], lw=2.4, linestyle="-", label="Loose"),
            Line2D([0], [0], color=color_map["Medium"], lw=2.4, linestyle="-", label="Medium"),
            Line2D([0], [0], color=color_map["Tight"], lw=2.4, linestyle="-", label="Tight"),
        ]
        legend1 = ax.legend(
            handles=scenario_handles,
            title="Scenario",
            loc="upper right",
            fontsize=LEGEND_SIZE,
            title_fontsize=LEGEND_TITLE_SIZE,
            frameon=True
        )
        ax.add_artist(legend1)

        metric_handles = [
            Line2D([0], [0], color="black", lw=2.3, linestyle="-", label="Task avg lateness"),
            Line2D([0], [0], color="black", lw=2.8, linestyle=(0, (8, 4)), label="Workflow avg lateness"),
        ]
        ax.legend(
            handles=metric_handles,
            title="Line meaning",
            loc="lower right",
            fontsize=LEGEND_SIZE,
            title_fontsize=LEGEND_TITLE_SIZE,
            frameon=True
        )
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    save_png_and_pdf(fig, out_path, dpi=180)
    plt.close(fig)


# =========================================================
# 保存图
# =========================================================
def save_triplet_overlay(data_dict, keys, out_path,
                         smooth_k=11, center_k=101, center_method="mean",
                         ylabel="Reward",
                         mode="trend_only",
                         legend_loc="upper right",
                         sci_y=False):
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 12.2), sharex=False)

    for ax, key in zip(axes, keys):
        plot_overlay_series(
            ax=ax,
            data_dict=data_dict,
            key=key,
            ylabel=ylabel,
            smooth_k=smooth_k,
            center_k=center_k,
            center_method=center_method,
            mode=mode,
            legend_loc=legend_loc,
            sci_y=sci_y
        )

    plt.tight_layout()
    save_png_and_pdf(fig, out_path, dpi=180)
    plt.close(fig)


def save_energy_overlay(scalar_dict, out_path,
                        smooth_k=11, center_k=101, center_method="mean"):
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    plot_overlay_series(
        ax=ax,
        data_dict=scalar_dict,
        key="eval_energy",
        ylabel="Energy (J)",
        smooth_k=smooth_k,
        center_k=center_k,
        center_method=center_method,
        mode="trend_only",
        legend_loc="upper right",
        sci_y=False
    )
    plt.tight_layout()
    save_png_and_pdf(fig, out_path, dpi=180)
    plt.close(fig)


# =========================================================
# 主流程
# =========================================================
def main(root_logs, out_dir,
         # res_model="smallRes",
         # res_model="medRes",
         res_model="largeRes",
         models=("Loose", "Medium", "Tight"),
         smooth_k=101, center_k=101, center_method="mean",
         manager_reward_col="r_manager_raw",
         manager_weight_mode="phase_size_sum_mi"):
    os.makedirs(out_dir, exist_ok=True)

    all_data = load_all_models(root_logs=root_logs, res_model=res_model, models=models)

    train_dict = {}
    eval_dict = {}
    scalar_dict = {}

    for model in models:
        df = all_data[model]["raw"]

        if df is None or df.empty:
            train_dict[model] = pd.DataFrame()
            eval_dict[model] = pd.DataFrame()
            scalar_dict[model] = pd.DataFrame()
            continue

        train_dict[model] = build_train_episode_rewards(
            df,
            manager_reward_col=manager_reward_col,
            manager_weight_mode=manager_weight_mode
        )
        eval_dict[model] = build_eval_episode_rewards(df)
        scalar_dict[model] = build_episode_scalar_metrics(df)

    out1 = os.path.join(out_dir, "evolution_train_3layer_overlay.png")
    save_triplet_overlay(
        data_dict=train_dict,
        keys=["train_vm", "train_host", "train_mgr"],
        out_path=out1,
        smooth_k=smooth_k,
        center_k=center_k,
        center_method=center_method,
        ylabel="Reward",
        mode="trend_only",
        legend_loc="upper right",
        sci_y=False
    )
    print(f"[OK] 保存：{os.path.abspath(out1)}")

    out2 = os.path.join(out_dir, "evolution_eval_3layer_overlay.png")
    save_triplet_overlay(
        data_dict=eval_dict,
        keys=["eval_vm", "eval_host", "eval_mgr"],
        out_path=out2,
        smooth_k=smooth_k,
        center_k=center_k,
        center_method=center_method,
        ylabel="Reward",
        mode="smooth_only",
        legend_loc="lower right",
        sci_y=True
    )
    print(f"[OK] 保存：{os.path.abspath(out2)}")

    out3 = os.path.join(out_dir, "episode_energy_eval_overlay.png")
    save_energy_overlay(
        scalar_dict=scalar_dict,
        out_path=out3,
        smooth_k=smooth_k,
        center_k=center_k,
        center_method=center_method
    )
    print(f"[OK] 保存：{os.path.abspath(out3)}")

    out4 = os.path.join(out_dir, "episode_lateness_overlay.png")
    save_lateness_overlay(
        scalar_dict=scalar_dict,
        out_path=out4,
        smooth_k=smooth_k
    )
    print(f"[OK] 保存：{os.path.abspath(out4)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    resModel = "smallRes"
    # resModel = "medRes"
    # resModel = "largeRes"

    ap.add_argument(
        "--root_logs",
        type=str,
        default="../out/logs",
        help="日志根目录"
    )
    ap.add_argument(
        "--out",
        type=str,
        default="../out/plots/plots_hrl_new_smallTask_" + resModel + "_mgrDelayEnergy_mix_rAlpha075_dalphaMix_HVrn_overlay_LMT",
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
        help="对比的 model 列表"
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
        manager_weight_mode=args.manager_weight_mode
    )