"""把已完成的 CEWS 评价报告注册为版本化安全启发式记录。

该工具不会导入或执行候选 Python。候选必须已经由 CEWS evaluator 显式评价，
且源码和评价报告分别位于 manifest 声明的可信目录中。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.heuristic_admission import (  # noqa: E402
    append_admission_record,
    build_admission_record,
)


def _load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required; install "
            "algorithms/llm_safe_hrl/LLM/requirements.txt"
        ) from exc
    with Path(path).resolve().open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must be a YAML mapping")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--heuristic-id", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--seevo-iteration", required=True, type=int)
    parser.add_argument("--seevo-individual", required=True, type=int)
    parser.add_argument(
        "--experiment-protocol",
        default=None,
        help=(
            "YAML/JSON file carrying the experiment protocol identity, "
            "either nested under 'experiment_protocol' or as the "
            "mapping itself. The identity is stamped into the record "
            "AND checked against the manifest, which is what binds a "
            "registered rule to one protocol. If omitted, the protocol "
            "is taken from --config when that file declares one."
        ),
    )
    parser.add_argument(
        "--allow-missing-protocol",
        action="store_true",
        help=(
            "Register a rule with no protocol identity. Only for "
            "legacy protocol-less manifests; the resulting manifest "
            "lineage cannot be bound to a protocol afterwards."
        ),
    )
    return parser.parse_args(argv)


def _load_protocol_identity(path) -> dict:
    """从独立文件读取实验协议身份，接受嵌套或裸映射两种写法。"""
    payload = _load_yaml(path)
    nested = payload.get("experiment_protocol")
    if isinstance(nested, dict):
        return nested
    return payload


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    expected_protocol = (
        _load_protocol_identity(args.experiment_protocol)
        if args.experiment_protocol
        else None
    )
    record = build_admission_record(
        heuristic_id=args.heuristic_id,
        display_name=args.display_name,
        source_path=args.candidate,
        evaluation_report_path=args.evaluation_report,
        evaluation_config=_load_yaml(args.config),
        manifest_path=manifest_path,
        seevo_iteration=args.seevo_iteration,
        seevo_individual=args.seevo_individual,
        experiment_protocol=expected_protocol,
    )
    # 没有显式协议、评价配置里也没有时，记录会不带任何协议身份，并进一步建出
    # 同样不带协议的 manifest 谱系。这种谱系事后无法再绑定协议，因此必须显式
    # 声明才允许，不能静默发生。
    if (
        record.get("experiment_protocol") is None
        and not args.allow_missing_protocol
    ):
        raise SystemExit(
            "refusing to register a heuristic without an experiment "
            "protocol identity: pass --experiment-protocol, or declare "
            "it in --config, or pass --allow-missing-protocol"
        )
    append_admission_record(
        manifest_path,
        record,
        expected_protocol_identity=expected_protocol,
    )
    print(
        f"heuristic_id={record['heuristic_id']} "
        f"version={record['version']} "
        f"status={record['admission_status']}"
    )
    if record["rejection_reasons"]:
        print(
            "rejection_reasons="
            + ";".join(record["rejection_reasons"])
        )
    return 0 if record["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
