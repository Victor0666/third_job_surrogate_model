# -*- coding: utf-8 -*-
import os
import json
import numpy as np

from common.read_xml_opt_Tsize import poisson_arrival_times

def build_episode_manifest(
    seed: int,
    dax_list,
    workflows_per_episode: int,
    arrival_lambda: float,
    horizon: float,
):
    # 1) arrival 随机流
    arrival_times = poisson_arrival_times(
        arrival_lambda,
        horizon,
        seed=seed,
    )

    # 如果你的 poisson_arrival_times 不能直接保证恰好生成 50 个，
    # 这里要按你环境里的“生成 episode arrival 序列”的真实逻辑改
    arrival_times = list(arrival_times[:workflows_per_episode])

    if len(arrival_times) < workflows_per_episode:
        raise RuntimeError(
            f"seed={seed}: arrival_times only has {len(arrival_times)} < {workflows_per_episode}"
        )

    # 2) dax 随机流（独立于 arrival）
    dax_rng = np.random.RandomState(seed + 1000000)

    workflows = []
    for i in range(workflows_per_episode):
        dax_path = str(dax_rng.choice(dax_list))
        payload_seed = int(seed * 100000 + i)

        workflows.append(
            {
                "wf_id": int(i),
                "dax_path": os.path.basename(dax_path),
                "arrival_time": float(arrival_times[i]),
                "payload_seed": int(payload_seed),
            }
        )

    return {
        "seed": int(seed),
        "workflows": workflows,
    }


def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dax_dir = os.path.join(ROOT_DIR, "data", "dax")
    dax_list = [
        os.path.join(dax_dir, "CyberShake_30.xml"),
        os.path.join(dax_dir, "Epigenomics_24.xml"),
        os.path.join(dax_dir, "LIGO_30.xml"),
        os.path.join(dax_dir, "Montage_25.xml"),
        os.path.join(dax_dir, "Sipht_29.xml"),
    ]

    workflows_per_episode = 50
    arrival_lambda = 0.03
    horizon = 1e9

    manifest = {
        "meta": {
            "version": "episode_manifest_v1",
            "workflows_per_episode": int(workflows_per_episode),
            "arrival_lambda": float(arrival_lambda),
            "horizon": float(horizon),
            "randomize_payloads": True,
            "length_mi_range": [2000, 3000],
            "io_file_count_range": [1, 3],
            "file_size_mb_range": [512, 1024],
            "dax_paths": [os.path.basename(p) for p in dax_list],
        },
        "episodes": [],
    }

    for seed in range(0, 1001):
        ep = build_episode_manifest(
            seed=seed,
            dax_list=dax_list,
            workflows_per_episode=workflows_per_episode,
            arrival_lambda=arrival_lambda,
            horizon=horizon,
        )
        manifest["episodes"].append(ep)

    out_dir = os.path.join(ROOT_DIR, "out", "episode_manifests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        "manifest_smallTask_smallRes_seed0-1000.json"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved manifest to: {out_path}")


if __name__ == "__main__":
    main()