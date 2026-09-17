from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "rename_manifest.csv"


DIR_MOVES = [
    ("a_set", "common"),
    ("a_dax", "data/dax"),
    ("d_output", "out"),
    ("out/deadlines", "data/deadlines/fcfs"),
    ("out/deadlines_HEFT", "data/deadlines/heft"),
    ("e_draw_plot", "plots"),
    ("a_generate_episode_manifest", "tools/manifest"),
    ("b_FCFS_FIFS", "baseline_fcfs"),
    ("b_hrl_tri_mgc_ave_ddlFCFS_mgrn_Mix_ddlAlpha_HVrn", "hrl_mix"),
    ("c_run", "run"),
    ("run/hrl_tri_ddlFCFS_alpha075_FCFS_MIX_Alpha_HVrn", "run/hrl_mix"),
    ("plots/20260326_Mix_ddlAlpha_result", "plots/excel"),
    ("plots/d_output", "plots/out"),
    ("plots/scenario_boxplots_byDDL_groupedAlgo", "plots/fig_box_byddl"),
    ("plots/scenario_boxplots_merged_LMT", "plots/fig_box_lmt"),
    ("plots/scenario_boxplots_merged_LMT_ab", "plots/fig_box_lmt_ab"),
    ("plots/scenario_boxplots_from_excel_tri", "plots/fig_box_tri"),
    ("plots/scenario_boxplots_from_excel_1", "plots/fig_box_raw"),
    ("plots/scenario_boxplots_from_excel_adaptive_y", "plots/fig_box_adapt"),
    ("plots/scenario_boxplots_from_excel_no_o", "plots/fig_box_nooutlier"),
    ("plots/scenario_violinplots_from_excel", "plots/fig_violin"),
    ("plots/ablation_main_heatmap_and_supp_boxplots", "plots/fig_ablation"),
    ("plots/ablation_main_heatmap_and_supp_boxplots_new", "plots/fig_ablation_new"),
]


TEXT_REPLACEMENTS = [
    (b"from a_set", b"from common"),
    (b"import a_set", b"import common"),
    (b'"a_set"', b'"common"'),
    (b"from b_FCFS_FIFS", b"from baseline_fcfs"),
    (b"import b_FCFS_FIFS", b"import baseline_fcfs"),
    (
        b"baseline_fcfs.env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_FSFS",
        b"baseline_fcfs.env_fcfs",
    ),
    (b"from base.d3qn_agent_EXDehid import D3QNAgent", b"from base.d3qn_agent import D3QNAgent"),
    (b"from base.d3qn_agent_PER import D3QNAgent", b"from base.d3qn_agent import D3QNAgent"),
    (b"base.d3qn_agent_EXDehid.D3QNAgent", b"base.d3qn_agent.D3QNAgent"),
    (b"base.d3qn_agent_PER.D3QNAgent", b"base.d3qn_agent.D3QNAgent"),
    (
        b"from b_hrl_tri_mgc_ave_ddlFCFS_mgrn_Mix_ddlAlpha_HVrn",
        b"from hrl_mix",
    ),
    (
        b"import b_hrl_tri_mgc_ave_ddlFCFS_mgrn_Mix_ddlAlpha_HVrn",
        b"import hrl_mix",
    ),
    (b"from hrl_mix.env_fcfs", b"from base.hrl_env"),
    (b"from hrl_mix.env_heft", b"from base.hrl_env"),
    (b"import hrl_mix.env_fcfs", b"import base.hrl_env"),
    (b"import hrl_mix.env_heft", b"import base.hrl_env"),
    (
        b"b_hrl_tri_mgc_ave_ddlFCFS_mgrn_Mix_ddlAlpha_HVrn"
        b".env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLCACHE_FCFS",
        b"base.hrl_env",
    ),
    (
        b"b_hrl_tri_mgc_ave_ddlFCFS_mgrn_Mix_ddlAlpha_HVrn"
        b".env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLHEFT",
        b"base.hrl_env",
    ),
    (b"env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLCACHE_FCFS", b"hrl_env"),
    (b"env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLHEFT", b"hrl_env"),
    (b'os.path.join(ROOT_DIR, "a_dax")', b'os.path.join(ROOT_DIR, "data", "dax")'),
    (b'os.path.join(PROJECT_DIR, "a_dax")', b'os.path.join(PROJECT_DIR, "data", "dax")'),
    (b'"out", "deadlines"', b'"data", "deadlines", "fcfs"'),
    (b'"out", "deadlines_HEFT"', b'"data", "deadlines", "heft"'),
    (b"out/deadlines_HEFT", b"data/deadlines/heft"),
    (b"out/deadlines", b"data/deadlines/fcfs"),
    (b"out\\deadlines_HEFT", b"data\\deadlines\\heft"),
    (b"out\\deadlines", b"data\\deadlines\\fcfs"),
    (b"d_output", b"out"),
    (b"e_draw_plot", b"plots"),
    (b"20260326_Mix_ddlAlpha_result", b"excel"),
    (b"scenario_boxplots_byDDL_groupedAlgo", b"fig_box_byddl"),
    (b"scenario_boxplots_merged_LMT_ab", b"fig_box_lmt_ab"),
    (b"scenario_boxplots_merged_LMT", b"fig_box_lmt"),
    (b"scenario_boxplots_from_excel_tri", b"fig_box_tri"),
    (b"scenario_boxplots_from_excel_1", b"fig_box_raw"),
    (b"scenario_boxplots_from_excel_adaptive_y", b"fig_box_adapt"),
    (b"scenario_boxplots_from_excel_no_o", b"fig_box_nooutlier"),
    (b"scenario_violinplots_from_excel", b"fig_violin"),
    (b"ablation_main_heatmap_and_supp_boxplots_new", b"fig_ablation_new"),
    (b"ablation_main_heatmap_and_supp_boxplots", b"fig_ablation"),
    (b"Mix_ddlAlpha30_Tight_ab.xlsx", b"ab_T.xlsx"),
    (b"Mix_ddlAlpha30_Medium_ab.xlsx", b"ab_M.xlsx"),
    (b"Mix_ddlAlpha30_Loose_ab.xlsx", b"ab_L.xlsx"),
    (b"Mix_ddlAlpha30_Tight.xlsx", b"mix_T.xlsx"),
    (b"Mix_ddlAlpha30_Medium.xlsx", b"mix_M.xlsx"),
    (b"Mix_ddlAlpha30_Loose.xlsx", b"mix_L.xlsx"),
    (b"deadline_cache_fcfs_vmfirst_", b"fcfs_"),
    (b"deadline_cache_heft_", b"heft_"),
    (
        b"episode_manifest_smallTask_smallRes_seed0-1000.json",
        b"manifest_smallTask_smallRes_seed0-1000.json",
    ),
]


PLOT_FILE_MOVES = {
    "algorithm_boxPlot.py": "plot_box_tri.py",
    "algorithm_boxPlot_ave.py": "plot_box_avg.py",
    "algorithm_boxPlot_Merge.py": "plot_box_lmt.py",
    "algorithm_boxPlot_Merge_ab.py": "plot_ablation_heatmap.py",
    "algorithm_boxPlot_Merge_byDDL_groupedAlgo.py": "plot_box_byddl.py",
    "algorithm_boxplot_rpd.pdf": "rpd_box.pdf",
    "algorithm_rpd.pdf": "rpd.pdf",
    "all_algorithms_violin_from_excel.pdf": "violin_all.pdf",
    "all_algorithms_violin_from_excel.png": "violin_all.png",
    "ablation_boxPlot.py": "plot_ablation_box.py",
    "ablation_boxPlot.pdf": "ablation_box.pdf",
    "ablation_boxPlot_ave.py": "plot_ablation_avg.py",
    "ablation_boxPlot_ave.pdf": "ablation_avg.pdf",
    "ablation_violinplot.py": "plot_ablation_violin.py",
    "ablation_violinPlot.pdf": "ablation_violin.pdf",
    "plot_success_rate_matrix_95_97.py": "plot_sr_matrix.py",
    "plot_success_rate_matrix_95_97_Merge.py": "plot_sr_matrix_merge.py",
    "plot_train_eval_hrl_new_triplet_Mix_ddlAlpha.py": "plot_hrl_mix.py",
    "plot_train_eval_hrl_new_triplet_Mix_ddlAlpha_merge_One.py": "plot_hrl_mix_one.py",
    "plot_train_eval_hrl_new_triplet_Mix_ddlAlpha_merge_One_short.py": "plot_hrl_mix_one_short.py",
    "plot_train_eval_hrl_new_triplet_Mix_ddlAlpha_merge_Three.py": "plot_hrl_mix_three.py",
    "Wilcoxon signed-rank.py": "wilcoxon_signed_rank.py",
    "Wilcoxon_signed_rank_Mix_ddlAlpha_OK.py": "wilcoxon_mix.py",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def move_path(src_rel: str, dst_rel: str, rows: list[tuple[str, str, str]], apply: bool) -> None:
    src = ROOT / src_rel
    dst = ROOT / dst_rel
    if not src.exists() or dst.exists():
        return
    rows.append(("move", src_rel, dst_rel))
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def rename_file(src: Path, dst: Path, rows: list[tuple[str, str, str]], apply: bool) -> None:
    if not src.exists() or dst.exists():
        return
    rows.append(("rename", rel(src), rel(dst)))
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)


def scenario_code(task: str, res: str) -> str:
    return {"small": "S", "med": "M", "large": "L"}[task] + {"small": "S", "med": "M", "large": "L"}[res]


def ddl_code(ddl: str) -> str:
    return {"Tight": "T", "Medium": "M", "Loose": "L"}[ddl]


def rename_training_scripts(rows: list[tuple[str, str, str]], apply: bool) -> None:
    root = ROOT / "hrl_mix"
    pat = re.compile(
        r"train_hrl_tri_003_alpha15_(small|med|large)Task_"
        r"(small|med|large)Res_seed5_ddlFCFS_alpha075_dalphaMix_"
        r"(Tight|Medium|Loose)\.py"
    )
    for path in root.glob("*.py"):
        m = pat.fullmatch(path.name)
        if not m:
            continue
        dst = path.with_name(f"train_{scenario_code(m.group(1), m.group(2))}_{ddl_code(m.group(3))}.py")
        rename_file(path, dst, rows, apply)


def rename_eval_scripts(rows: list[tuple[str, str, str]], apply: bool) -> None:
    root = ROOT / "run" / "hrl_mix"
    pat = re.compile(
        r"run_hrl_routeA_(small|med|large)Task_(small|med|large)Res_"
        r"deadlineCACHE_FCFS_alpha075_dalphaMix_HVrn_(Tight|Medium|Loose)\.py"
    )
    for path in root.glob("*.py"):
        m = pat.fullmatch(path.name)
        if not m:
            continue
        dst = path.with_name(f"eval_{scenario_code(m.group(1), m.group(2))}_{ddl_code(m.group(3))}.py")
        rename_file(path, dst, rows, apply)


def rename_fixed_files(rows: list[tuple[str, str, str]], apply: bool) -> None:
    fixed = {
        "baseline_fcfs/env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_FSFS.py": "baseline_fcfs/env_fcfs.py",
        "baseline_fcfs/train_d3qn_single_controller_reward_MIX_Tsize_FCFS.py": "baseline_fcfs/train_fcfs.py",
        "base/env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15.py": "base/env.py",
        "base/d3qn_agent_EXDehid.py": "base/d3qn_agent.py",
        "base/d3qn_agent_PER.py": "base/d3qn_agent.py",
        "hrl_mix/env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLCACHE_FCFS.py": "base/hrl_env.py",
        "hrl_mix/env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLHEFT.py": "base/hrl_env.py",
        "tools/manifest/generate_episode_manifest_smallTask_smallRes.py": "tools/manifest/generate_manifest_SS.py",
    }
    for src, dst in fixed.items():
        rename_file(ROOT / src, ROOT / dst, rows, apply)
    plot_root = ROOT / "plots"
    for src, dst in PLOT_FILE_MOVES.items():
        rename_file(plot_root / src, plot_root / dst, rows, apply)


def rename_result_files(rows: list[tuple[str, str, str]], apply: bool) -> None:
    for folder in [ROOT / "data" / "deadlines" / "fcfs", ROOT / "data" / "deadlines" / "heft"]:
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            new = path.name
            new = new.replace("deadline_cache_fcfs_vmfirst_", "fcfs_")
            new = new.replace("deadline_cache_heft_", "heft_")
            new = new.replace(" - ", "_")
            if new != path.name:
                rename_file(path, path.with_name(new), rows, apply)

    manifest_dir = ROOT / "out" / "episode_manifests"
    if manifest_dir.exists():
        for path in manifest_dir.glob("episode_manifest_*.json"):
            rename_file(path, path.with_name(path.name.replace("episode_manifest_", "manifest_")), rows, apply)

    excel_dir = ROOT / "plots" / "excel"
    excel_map = {
        "Mix_ddlAlpha30_Tight.xlsx": "mix_T.xlsx",
        "Mix_ddlAlpha30_Medium.xlsx": "mix_M.xlsx",
        "Mix_ddlAlpha30_Loose.xlsx": "mix_L.xlsx",
        "Mix_ddlAlpha30_Tight_ab.xlsx": "ab_T.xlsx",
        "Mix_ddlAlpha30_Medium_ab.xlsx": "ab_M.xlsx",
        "Mix_ddlAlpha30_Loose_ab.xlsx": "ab_L.xlsx",
    }
    for src, dst in excel_map.items():
        rename_file(excel_dir / src, excel_dir / dst, rows, apply)


def rewrite_text(rows: list[tuple[str, str, str]], apply: bool) -> None:
    skip_parts = {".venv", ".idea", "__pycache__"}
    suffixes = {".py", ".md", ".txt", ".bat", ".ps1"}
    for path in ROOT.rglob("*"):
        if path == Path(__file__).resolve():
            continue
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        data = path.read_bytes()
        new_data = data
        for old, new in TEXT_REPLACEMENTS:
            new_data = new_data.replace(old, new)
        if new_data != data:
            rows.append(("rewrite", rel(path), rel(path)))
            if apply:
                path.write_bytes(new_data)


def write_manifest(rows: list[tuple[str, str, str]], apply: bool) -> None:
    if not apply:
        return
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["action", "old_path", "new_path"])
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Shorten project paths and record a rename manifest.")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Without this flag, only prints the plan.")
    args = parser.parse_args()

    rows: list[tuple[str, str, str]] = []

    for src, dst in DIR_MOVES:
        move_path(src, dst, rows, args.apply)

    rewrite_text(rows, args.apply)
    rename_fixed_files(rows, args.apply)
    rename_training_scripts(rows, args.apply)
    rename_eval_scripts(rows, args.apply)
    rename_result_files(rows, args.apply)
    write_manifest(rows, args.apply)

    for action, old, new in rows:
        print(f"{action}: {old} -> {new}")
    print(f"{'applied' if args.apply else 'planned'} {len(rows)} changes")
    if args.apply:
        print(f"manifest: {rel(MANIFEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
