from pathlib import Path
import argparse
import csv
import re

import matplotlib.pyplot as plt


BEST_PATTERN = re.compile(
    r"Best obj:\s*([+\-\d.eE]+),\s*"
    r"Constraint key:\s*\(([^)]+)\)"
)
FE_PATTERN = re.compile(r"Function Evals:\s*(\d+)")


def load_run_context(log_path):
    """从日志目录的有效配置中识别 modal/fuzzy 指标口径。"""
    context = {
        "fuzzy_enabled": None,
        "energy_uncertainty_weight": None,
        "deadline_eta": None,
    }
    config_paths = sorted(log_path.parent.glob("effective_*_config.yaml"))
    if not config_paths:
        return context

    try:
        import yaml
    except ImportError:
        return context

    try:
        with open(config_paths[0], "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        fuzzy = config.get("fuzzy", {}) if isinstance(config, dict) else {}
        context["fuzzy_enabled"] = bool(fuzzy.get("enabled", False))
        context["energy_uncertainty_weight"] = float(
            fuzzy.get("energy_uncertainty_weight", 1.0)
        )
        context["deadline_eta"] = float(fuzzy.get("deadline_eta", 0.95))
    except (OSError, TypeError, ValueError):
        # 配置只用于生成准确标签；日志解析与绘图不应因此失败。
        return {
            "fuzzy_enabled": None,
            "energy_uncertainty_weight": None,
            "deadline_eta": None,
        }
    return context


def parse_log(log_path):
    """提取每个完整进化代结束时的全局最优结果。"""
    last_record_by_fe = {}
    pending = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            best_match = BEST_PATTERN.search(line)
            if best_match:
                values = [
                    float(value.strip())
                    for value in best_match.group(2).split(",")
                ]

                pending = {
                    # fuzzy 模式下该值是风险调整模糊能耗分数，而非 modal 总能耗。
                    "objective": float(best_match.group(1)),
                    # comparison key = (infeasible, primary, secondary, objective)
                    "infeasible": int(values[0] >= 0.5),
                    "constraint_violation": values[1],
                    "total_lateness": values[2],
                }
                continue

            fe_match = FE_PATTERN.search(line)
            if fe_match and pending is not None:
                # 同一个 Function Evals 会记录交叉、自进化和变异阶段。
                # 保留最后一条，即该完整进化代结束后的全局最优结果。
                function_eval = int(fe_match.group(1))
                last_record_by_fe[function_eval] = pending
                pending = None

    records = []
    for function_eval in sorted(last_record_by_fe):
        record = dict(last_record_by_fe[function_eval])
        record["generation"] = function_eval + 1
        records.append(record)

    if not records:
        raise RuntimeError("没有从日志中找到有效的收敛记录。")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="SeEvo main.log 文件")
    parser.add_argument(
        "--output",
        default=None,
        help="输出图片路径，默认保存在日志目录",
    )
    args = parser.parse_args()

    log_path = Path(args.log).resolve()
    records = parse_log(log_path)
    context = load_run_context(log_path)

    generations = [r["generation"] for r in records]
    objectives = [r["objective"] for r in records]
    violations = [r["constraint_violation"] for r in records]
    lateness = [r["total_lateness"] for r in records]
    infeasible = [r["infeasible"] for r in records]

    output_path = (
        Path(args.output).resolve()
        if args.output
        else log_path.with_name("seevo_convergence.png")
    )
    csv_path = output_path.with_suffix(".csv")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 9),
        sharex=True,
        constrained_layout=True,
    )

    # SeEvo 的比较键按“可行性、违反程度、总延期、能耗目标”的顺序排序。
    # 分面展示可避免把不同单位强行加权，也避免双纵轴造成视觉误读。
    axes[0].plot(
        generations,
        violations,
        marker="s",
        color="tab:orange",
        label="Constraint violation",
    )
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_ylabel("DDL violation")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        generations,
        lateness,
        marker="^",
        color="tab:red",
        label="Total lateness",
    )
    axes[1].set_ylabel("Total lateness (s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fuzzy_enabled = context["fuzzy_enabled"]
    if fuzzy_enabled is True:
        weight = context["energy_uncertainty_weight"]
        objective_label = "Fuzzy energy score (J)"
        objective_legend = (
            "Lexicographic incumbent"
            if weight is None
            else f"Incumbent: mean + {weight:g} std"
        )
        eta = context["deadline_eta"]
        constraint_context = (
            "risk-adjusted DDL"
            if eta is None
            else f"risk-adjusted DDL, eta={eta:g}"
        )
    elif fuzzy_enabled is False:
        objective_label = "Modal total energy (J)"
        objective_legend = "Lexicographic incumbent"
        constraint_context = "modal DDL"
    else:
        objective_label = "Optimization objective (J)"
        objective_legend = "Lexicographic incumbent"
        constraint_context = "DDL"

    axes[2].plot(
        generations,
        objectives,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label=objective_legend,
    )
    axes[2].set_xlabel("Evolution generation")
    axes[2].set_ylabel(objective_label)
    axes[2].grid(alpha=0.3)
    axes[2].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[2].legend()

    feasible_generations = [
        generation
        for generation, is_infeasible in zip(generations, infeasible)
        if not is_infeasible
    ]
    if feasible_generations:
        first_feasible = feasible_generations[0]
        for axis in axes:
            axis.axvline(
                first_feasible,
                color="tab:green",
                linewidth=1.2,
                linestyle=":",
            )
        title_suffix = f"first feasible generation = {first_feasible}"
    else:
        title_suffix = "no feasible incumbent"
    axes[0].set_title(
        "SeEvo lexicographic convergence\n"
        f"{constraint_context}; {title_suffix}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "generation",
                "objective",
                "infeasible",
                "constraint_violation",
                "total_lateness",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"图片已保存：{output_path}")
    print(f"数据已保存：{csv_path}")


if __name__ == "__main__":
    main()
