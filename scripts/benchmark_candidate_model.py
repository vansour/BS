#!/usr/bin/env python3
"""
Benchmark one candidate fog-model checkpoint against the configured baselines.

This script builds a temporary model list that contains:
1. The existing baseline models from `configs/benchmark_models.json`.
2. One candidate model checkpoint provided on the command line.

It then calls `scripts/run_multi_model_benchmark.py` and produces a
candidate-focused summary with rankings and per-baseline deltas.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_CONFIG = ROOT / "configs" / "benchmark_videos.json"
DEFAULT_BASE_MODEL_CONFIG = ROOT / "configs" / "benchmark_models.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "Candidate_Benchmark_v1"
DEFAULT_YOLO_WEIGHTS = ROOT / "yolo11n.pt"
DEFAULT_PASS_RULES = {
    "allowed_unified_mean_drop": 0.0,
    "allowed_unified_ratio_drop": 0.0,
    "allowed_fog_switch_increase": 0.02,
    "allowed_beta_abs_delta_increase": 0.005,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a candidate fog-model checkpoint against the baseline model set."
    )
    parser.add_argument(
        "--candidate-weights",
        required=True,
        help="候选雾模型权重路径。",
    )
    parser.add_argument(
        "--candidate-label",
        required=True,
        help="候选模型展示名称。",
    )
    parser.add_argument(
        "--candidate-slug",
        default=None,
        help="候选模型 slug；默认由 label 派生。",
    )
    parser.add_argument(
        "--candidate-tags",
        nargs="*",
        default=None,
        help="候选模型标签列表。",
    )
    parser.add_argument(
        "--candidate-config",
        default=None,
        help="候选模型专属配置文件路径（.json/.yaml/.yml）。",
    )
    parser.add_argument(
        "--candidate-notes",
        default="",
        help="候选模型备注。",
    )
    parser.add_argument(
        "--benchmark-config",
        default=str(DEFAULT_BENCHMARK_CONFIG),
        help="Benchmark 视频配置文件。",
    )
    parser.add_argument(
        "--base-model-config",
        default=str(DEFAULT_BASE_MODEL_CONFIG),
        help="基线模型配置文件。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="候选模型对比输出目录。",
    )
    parser.add_argument(
        "--yolo-weights",
        default=str(DEFAULT_YOLO_WEIGHTS),
        help="混合方案车辆检测权重。",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help="覆盖所有模型运行时的 sample stride。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="覆盖所有模型运行时的 max frames。",
    )
    parser.add_argument(
        "--unified-conf",
        type=float,
        default=None,
        help="覆盖所有模型运行时的 unified confidence threshold。",
    )
    parser.add_argument(
        "--hybrid-conf",
        type=float,
        default=None,
        help="覆盖所有模型运行时的 hybrid confidence threshold。",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="覆盖所有模型运行时的 hybrid YOLO imgsz。",
    )
    parser.add_argument(
        "--topk-preview",
        type=int,
        default=None,
        help="覆盖所有模型运行时的 preview 数量。",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="若多模型输出目录已有结果，则直接复用。",
    )
    parser.add_argument(
        "--allowed-unified-mean-drop",
        type=float,
        default=DEFAULT_PASS_RULES["allowed_unified_mean_drop"],
        help="允许候选模型相对 baseline 的 unified mean detections/frame 最大下降量。",
    )
    parser.add_argument(
        "--allowed-unified-ratio-drop",
        type=float,
        default=DEFAULT_PASS_RULES["allowed_unified_ratio_drop"],
        help="允许候选模型相对 baseline 的 unified frames_with_detections_ratio 最大下降量。",
    )
    parser.add_argument(
        "--allowed-fog-switch-increase",
        type=float,
        default=DEFAULT_PASS_RULES["allowed_fog_switch_increase"],
        help="允许候选模型相对 baseline 的 fog switch rate 最大上升量。",
    )
    parser.add_argument(
        "--allowed-beta-abs-delta-increase",
        type=float,
        default=DEFAULT_PASS_RULES["allowed_beta_abs_delta_increase"],
        help="允许候选模型相对 baseline 的 beta abs delta mean 最大上升量。",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def sanitize_slug(value: str) -> str:
    import re

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._") or "candidate"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Base model config not found: {path}")
    payload = load_json(path)
    if isinstance(payload, dict):
        models = payload.get("models", [])
        description = str(payload.get("description", "") or "").strip()
    elif isinstance(payload, list):
        models = payload
        description = ""
        payload = {
            "description": description,
            "models": models,
        }
    else:
        raise TypeError("Base model config must be either an object or a list.")
    if not isinstance(models, list):
        raise TypeError("Base model config field `models` must be a list.")
    payload["description"] = description
    payload["models"] = models
    return payload


def build_candidate_entry(args: argparse.Namespace, candidate_weights: Path) -> dict[str, Any]:
    candidate_slug = args.candidate_slug or sanitize_slug(args.candidate_label)
    candidate_tags = list(args.candidate_tags or ["candidate"])
    candidate_config = resolve_path(args.candidate_config) if args.candidate_config else None
    if candidate_config is not None and not candidate_config.exists():
        raise FileNotFoundError(f"Candidate config not found: {candidate_config}")
    return {
        "label": args.candidate_label,
        "slug": candidate_slug,
        "fog_weights": str(candidate_weights),
        "config": str(candidate_config) if candidate_config is not None else "",
        "tags": candidate_tags,
        "notes": args.candidate_notes,
    }


def merge_candidate_model(
    base_payload: dict[str, Any],
    candidate_entry: dict[str, Any],
) -> dict[str, Any]:
    candidate_slug = str(candidate_entry["slug"])
    candidate_label = str(candidate_entry["label"])

    filtered_models = []
    replaced_labels = []
    for item in base_payload["models"]:
        if not isinstance(item, dict):
            continue
        same_slug = str(item.get("slug") or sanitize_slug(str(item.get("label") or ""))) == candidate_slug
        same_label = str(item.get("label") or "").strip() == candidate_label
        if same_slug or same_label:
            replaced_labels.append(str(item.get("label") or item.get("slug") or "unknown"))
            continue
        filtered_models.append(item)

    return {
        "description": base_payload["description"],
        "models": filtered_models + [candidate_entry],
        "candidate_merge": {
            "candidate_label": candidate_label,
            "candidate_slug": candidate_slug,
            "replaced_models": replaced_labels,
            "baseline_model_count_before": len(base_payload["models"]),
            "model_count_after": len(filtered_models) + 1,
        },
    }


def build_multi_model_command(
    args: argparse.Namespace,
    benchmark_config: Path,
    model_config: Path,
    output_dir: Path,
    yolo_weights: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_multi_model_benchmark.py"),
        "--benchmark-config",
        str(benchmark_config),
        "--model-config",
        str(model_config),
        "--output-dir",
        str(output_dir),
        "--yolo-weights",
        str(yolo_weights),
    ]

    overrides = {
        "--sample-stride": args.sample_stride,
        "--max-frames": args.max_frames,
        "--unified-conf": args.unified_conf,
        "--hybrid-conf": args.hybrid_conf,
        "--imgsz": args.imgsz,
        "--topk-preview": args.topk_preview,
    }
    for flag, value in overrides.items():
        if value is not None:
            command.extend([flag, str(value)])
    if args.reuse_existing:
        command.append("--reuse-existing")
    return command


def rank_models(
    model_summaries: list[dict[str, Any]],
    key: str,
    *,
    reverse: bool,
) -> dict[str, int]:
    ordered = sorted(
        model_summaries,
        key=lambda item: float(item["aggregate"][key]),
        reverse=reverse,
    )
    ranks: dict[str, int] = {}
    current_rank = 0
    previous_value: float | None = None
    for index, item in enumerate(ordered, start=1):
        current_value = round(float(item["aggregate"][key]), 12)
        if previous_value is None or current_value != previous_value:
            current_rank = index
            previous_value = current_value
        ranks[str(item["slug"])] = current_rank
    return ranks


def build_pass_rules(args: argparse.Namespace) -> dict[str, float]:
    """从命令行参数构造候选模型通过规则。"""
    return {
        "allowed_unified_mean_drop": float(args.allowed_unified_mean_drop),
        "allowed_unified_ratio_drop": float(args.allowed_unified_ratio_drop),
        "allowed_fog_switch_increase": float(args.allowed_fog_switch_increase),
        "allowed_beta_abs_delta_increase": float(args.allowed_beta_abs_delta_increase),
    }


def evaluate_candidate_vs_baseline(
    delta: dict[str, float],
    pass_rules: dict[str, float],
) -> dict[str, Any]:
    """根据阈值规则判断候选模型是否通过单个 baseline。"""
    checks = {
        "unified_mean_count_per_frame": (
            float(delta["delta_unified_mean_count_per_frame"])
            >= -float(pass_rules["allowed_unified_mean_drop"])
        ),
        "unified_frames_with_detections_ratio": (
            float(delta["delta_unified_frames_with_detections_ratio"])
            >= -float(pass_rules["allowed_unified_ratio_drop"])
        ),
        "fog_switch_rate": (
            float(delta["delta_fog_switch_rate"])
            <= float(pass_rules["allowed_fog_switch_increase"])
        ),
        "beta_abs_delta_mean": (
            float(delta["delta_beta_abs_delta_mean"])
            <= float(pass_rules["allowed_beta_abs_delta_increase"])
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
    }


def build_candidate_summary(
    multi_model_summary: dict[str, Any],
    candidate_slug: str,
    pass_rules: dict[str, float],
) -> dict[str, Any]:
    model_summaries = multi_model_summary["models"]
    candidate = next(
        (item for item in model_summaries if str(item["slug"]) == candidate_slug),
        None,
    )
    if candidate is None:
        raise RuntimeError(f"Candidate slug {candidate_slug!r} was not found in multi-model summary.")

    rank_unified_mean = rank_models(
        model_summaries,
        "weighted_unified_mean_count_per_frame",
        reverse=True,
    )
    rank_unified_ratio = rank_models(
        model_summaries,
        "weighted_unified_frames_with_detections_ratio",
        reverse=True,
    )
    rank_switch_rate = rank_models(
        model_summaries,
        "weighted_dominant_switch_rate",
        reverse=False,
    )
    rank_beta_delta = rank_models(
        model_summaries,
        "weighted_beta_abs_delta_mean",
        reverse=False,
    )

    baseline_deltas = []
    candidate_aggregate = candidate["aggregate"]
    for item in model_summaries:
        if str(item["slug"]) == candidate_slug:
            continue
        aggregate = item["aggregate"]
        delta = {
            "label": item["label"],
            "slug": item["slug"],
            "role": item.get("role", ""),
            "status": item.get("status", ""),
            "delta_unified_mean_count_per_frame": (
                float(candidate_aggregate["weighted_unified_mean_count_per_frame"])
                - float(aggregate["weighted_unified_mean_count_per_frame"])
            ),
            "delta_unified_frames_with_detections_ratio": (
                float(candidate_aggregate["weighted_unified_frames_with_detections_ratio"])
                - float(aggregate["weighted_unified_frames_with_detections_ratio"])
            ),
            "delta_fog_switch_rate": (
                float(candidate_aggregate["weighted_dominant_switch_rate"])
                - float(aggregate["weighted_dominant_switch_rate"])
            ),
            "delta_beta_abs_delta_mean": (
                float(candidate_aggregate["weighted_beta_abs_delta_mean"])
                - float(aggregate["weighted_beta_abs_delta_mean"])
            ),
        }
        delta["gate"] = evaluate_candidate_vs_baseline(delta, pass_rules)
        baseline_deltas.append(delta)

    overall_gate = {
        "pass": all(item["gate"]["pass"] for item in baseline_deltas),
        "passed_baseline_count": sum(1 for item in baseline_deltas if item["gate"]["pass"]),
        "total_baseline_count": len(baseline_deltas),
    }

    return {
        "candidate": {
            "label": candidate["label"],
            "slug": candidate["slug"],
            "fog_weights": candidate["fog_weights"],
            "tags": candidate["tags"],
            "notes": candidate["notes"],
            "aggregate": candidate_aggregate,
            "rankings": {
                "weighted_unified_mean_count_per_frame": rank_unified_mean[candidate_slug],
                "weighted_unified_frames_with_detections_ratio": rank_unified_ratio[candidate_slug],
                "weighted_dominant_switch_rate": rank_switch_rate[candidate_slug],
                "weighted_beta_abs_delta_mean": rank_beta_delta[candidate_slug],
            },
        },
        "pass_rules": pass_rules,
        "overall_gate": overall_gate,
        "baseline_deltas": baseline_deltas,
        "leaders": multi_model_summary["leaders"],
        "multi_model_artifacts": multi_model_summary["artifacts"],
    }


def write_markdown_report(
    payload: dict[str, Any],
    output_path: Path,
):
    candidate = payload["candidate"]
    aggregate = candidate["aggregate"]
    rankings = candidate["rankings"]
    overall_gate = payload["overall_gate"]
    pass_rules = payload["pass_rules"]

    lines = [
        "# Candidate Benchmark Summary",
        "",
        "## Candidate",
        "",
        f"- Label: `{candidate['label']}`",
        f"- Slug: `{candidate['slug']}`",
        f"- Fog weights: `{candidate['fog_weights']}`",
        f"- Tags: `{candidate['tags']}`",
        f"- Notes: {candidate['notes'] or 'N/A'}",
        "",
        "## Candidate Aggregate",
        "",
        f"- Weighted unified mean detections/frame: `{aggregate['weighted_unified_mean_count_per_frame']:.3f}`",
        f"- Weighted unified nonzero-detection ratio: `{aggregate['weighted_unified_frames_with_detections_ratio']:.3f}`",
        f"- Weighted fog switch rate: `{aggregate['weighted_dominant_switch_rate']:.3f}`",
        f"- Weighted beta abs delta mean: `{aggregate['weighted_beta_abs_delta_mean']:.5f}`",
        "",
        "## Gate Result",
        "",
        f"- Overall pass: `{overall_gate['pass']}`",
        f"- Passed baselines: `{overall_gate['passed_baseline_count']}/{overall_gate['total_baseline_count']}`",
        "",
        "## Rankings",
        "",
        f"- Unified mean detections/frame rank: `{rankings['weighted_unified_mean_count_per_frame']}`",
        f"- Unified nonzero-detection ratio rank: `{rankings['weighted_unified_frames_with_detections_ratio']}`",
        f"- Fog switch rate rank: `{rankings['weighted_dominant_switch_rate']}`",
        f"- Beta abs delta mean rank: `{rankings['weighted_beta_abs_delta_mean']}`",
        "",
        "## Pass Rules",
        "",
        f"- Allowed unified mean drop: `{pass_rules['allowed_unified_mean_drop']}`",
        f"- Allowed unified ratio drop: `{pass_rules['allowed_unified_ratio_drop']}`",
        f"- Allowed fog switch increase: `{pass_rules['allowed_fog_switch_increase']}`",
        f"- Allowed beta abs delta increase: `{pass_rules['allowed_beta_abs_delta_increase']}`",
        "",
        "## Deltas vs Baselines",
        "",
    ]

    for item in payload["baseline_deltas"]:
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- Baseline role: `{item['role'] or 'N/A'}`",
                f"- Baseline status: `{item['status'] or 'N/A'}`",
                f"- Pass this baseline gate: `{item['gate']['pass']}`",
                f"- Delta unified mean detections/frame: `{item['delta_unified_mean_count_per_frame']:+.3f}`",
                f"- Delta unified nonzero-detection ratio: `{item['delta_unified_frames_with_detections_ratio']:+.3f}`",
                f"- Delta fog switch rate: `{item['delta_fog_switch_rate']:+.3f}`",
                f"- Delta beta abs delta mean: `{item['delta_beta_abs_delta_mean']:+.5f}`",
                f"- Unified mean check: `{item['gate']['checks']['unified_mean_count_per_frame']}`",
                f"- Unified ratio check: `{item['gate']['checks']['unified_frames_with_detections_ratio']}`",
                f"- Fog switch check: `{item['gate']['checks']['fog_switch_rate']}`",
                f"- Beta abs delta check: `{item['gate']['checks']['beta_abs_delta_mean']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Multi-Model Artifacts",
            "",
            f"- Model overview CSV: `{payload['multi_model_artifacts']['model_overview_csv']}`",
            f"- Model-video matrix CSV: `{payload['multi_model_artifacts']['model_video_matrix_csv']}`",
            f"- Combined multi-model summary JSON: `{payload['multi_model_artifacts']['summary_json']}`",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    candidate_weights = resolve_path(args.candidate_weights)
    if not candidate_weights.exists():
        raise FileNotFoundError(f"Candidate weights not found: {candidate_weights}")

    benchmark_config = resolve_path(args.benchmark_config)
    base_model_config = resolve_path(args.base_model_config)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo_weights = resolve_path(args.yolo_weights)
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    base_payload = load_model_config(base_model_config)
    candidate_entry = build_candidate_entry(args, candidate_weights)
    pass_rules = build_pass_rules(args)
    merged_payload = merge_candidate_model(base_payload, candidate_entry)

    generated_model_config = output_dir / "candidate_model_config.json"
    generated_model_config.write_text(
        json.dumps(merged_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    multi_model_output_dir = output_dir / "multi_model"
    command = build_multi_model_command(
        args,
        benchmark_config,
        generated_model_config,
        multi_model_output_dir,
        yolo_weights,
    )

    print(f"Candidate weights: {candidate_weights}")
    print(f"Candidate label: {args.candidate_label}")
    print(f"Generated model config: {generated_model_config}")
    print("Running multi-model benchmark:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)

    multi_model_summary_json = multi_model_output_dir / "multi_model_benchmark_summary.json"
    multi_model_summary = load_json(multi_model_summary_json)
    candidate_summary = build_candidate_summary(
        multi_model_summary,
        str(candidate_entry["slug"]),
        pass_rules,
    )

    summary_json = output_dir / "candidate_benchmark_summary.json"
    summary_md = output_dir / "candidate_benchmark_summary.md"
    summary_json.write_text(
        json.dumps(candidate_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(candidate_summary, summary_md)

    print(f"Candidate summary JSON: {summary_json}")
    print(f"Candidate summary Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
