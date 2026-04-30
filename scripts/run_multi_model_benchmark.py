#!/usr/bin/env python3
"""
Run the same route-evaluation benchmark across multiple fog-model baselines.

This script keeps the existing `scripts/evaluate_inference_routes.py` as the
single-model evaluator and adds one orchestration layer on top:
1. Read a shared benchmark video list.
2. Read a model list with multiple fog-weight checkpoints.
3. Run the evaluator once per model.
4. Aggregate the results into cross-model CSV/Markdown/JSON summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_CONFIG = ROOT / "configs" / "benchmark_videos.json"
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "benchmark_models.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "Benchmark_Model_Compare_v1"
DEFAULT_YOLO_WEIGHTS = ROOT / "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark video set across multiple fog-model baselines."
    )
    parser.add_argument(
        "--benchmark-config",
        default=str(DEFAULT_BENCHMARK_CONFIG),
        help="Benchmark 视频配置文件。",
    )
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_MODEL_CONFIG),
        help="多模型基线配置文件。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="多模型 benchmark 输出目录。",
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
        help="如果某模型输出目录中已有 route_eval_summary.json，则直接复用。",
    )
    return parser.parse_args()


def sanitize_slug(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._") or "model"


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_entries(config_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    payload = load_json(config_path)
    if isinstance(payload, dict):
        raw_models = payload.get("models", [])
        benchmark_id = str(payload.get("benchmark_id", "") or "").strip()
        description = str(payload.get("description", "") or "").strip()
        selection_rules = payload.get("selection_rules", [])
    elif isinstance(payload, list):
        raw_models = payload
        benchmark_id = ""
        description = ""
        selection_rules = []
    else:
        raise TypeError("Model config must be either an object or a list.")

    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("Model config does not contain any model entries.")
    if selection_rules is None:
        selection_rules = []
    if not isinstance(selection_rules, list):
        raise TypeError("Model config field `selection_rules` must be a list.")

    entries: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for index, item in enumerate(raw_models, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Model entry must be an object, got {type(item)!r}")

        label = str(item.get("label") or "").strip()
        fog_weights_raw = str(item.get("fog_weights") or "").strip()
        if not label:
            raise ValueError("Each model entry must provide `label`.")
        if not fog_weights_raw:
            raise ValueError(f"Model entry {label!r} must provide `fog_weights`.")

        fog_weights = resolve_path(fog_weights_raw)
        if not fog_weights.exists():
            raise FileNotFoundError(f"Fog weights not found: {fog_weights}")
        config_raw = str(item.get("config") or item.get("config_path") or "").strip()
        config_path = resolve_path(config_raw) if config_raw else None
        if config_path is not None and not config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")

        slug = str(item.get("slug") or sanitize_slug(label))
        if slug in seen_slugs:
            raise ValueError(f"Duplicate model slug detected: {slug}")
        seen_slugs.add(slug)

        tags = item.get("tags", [])
        if tags is None:
            tags = []
        if not isinstance(tags, list):
            raise TypeError(f"Model entry {label!r} field `tags` must be a list.")

        entry = {
            "entry_id": int(index),
            "label": label,
            "slug": slug,
            "fog_weights": fog_weights,
            "fog_weights_raw": fog_weights_raw,
            "config_path": str(config_path) if config_path is not None else "",
            "role": str(item.get("role", "") or "").strip(),
            "status": str(item.get("status", "active") or "active").strip().lower(),
            "notes": str(item.get("notes", "") or "").strip(),
            "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
        }
        entries.append(entry)

    return entries, {
        "benchmark_id": benchmark_id,
        "model_config": str(config_path),
        "model_config_description": description,
        "model_selection_rules": selection_rules,
    }


def build_eval_command(
    args: argparse.Namespace,
    benchmark_config: Path,
    model_output_dir: Path,
    model_entry: dict[str, Any],
    fog_weights: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_inference_routes.py"),
        "--benchmark-config",
        str(benchmark_config),
        "--output-dir",
        str(model_output_dir),
        "--fog-weights",
        str(fog_weights),
        "--yolo-weights",
        str(resolve_path(args.yolo_weights)),
    ]
    if model_entry.get("config_path"):
        command.extend(["--config", str(model_entry["config_path"])])

    optional_overrides = {
        "--sample-stride": args.sample_stride,
        "--max-frames": args.max_frames,
        "--unified-conf": args.unified_conf,
        "--hybrid-conf": args.hybrid_conf,
        "--imgsz": args.imgsz,
        "--topk-preview": args.topk_preview,
    }
    for flag, value in optional_overrides.items():
        if value is not None:
            command.extend([flag, str(value)])
    return command


def run_single_model(
    args: argparse.Namespace,
    benchmark_config: Path,
    output_root: Path,
    model_entry: dict[str, Any],
) -> dict[str, Any]:
    model_output_dir = output_root / model_entry["slug"]
    model_output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = model_output_dir / "route_eval_summary.json"

    if args.reuse_existing and summary_json.exists():
        print(f"Reusing existing summary for model {model_entry['label']}: {summary_json}")
        route_summary = load_json(summary_json)
    else:
        command = build_eval_command(
            args,
            benchmark_config,
            model_output_dir,
            model_entry,
            model_entry["fog_weights"],
        )
        print("Running benchmark for model:")
        print(f"  label={model_entry['label']}")
        print(f"  fog_weights={model_entry['fog_weights']}")
        print("  command=" + " ".join(command))
        subprocess.run(command, cwd=str(ROOT), check=True)
        route_summary = load_json(summary_json)

    return {
        "entry_id": int(model_entry["entry_id"]),
        "label": str(model_entry["label"]),
        "slug": str(model_entry["slug"]),
        "fog_weights": str(model_entry["fog_weights"]),
        "config_path": str(model_entry.get("config_path", "")),
        "role": str(model_entry["role"]),
        "status": str(model_entry["status"]),
        "notes": str(model_entry["notes"]),
        "tags": list(model_entry["tags"]),
        "artifacts": {
            "model_output_dir": str(model_output_dir),
            "route_eval_summary_json": str(summary_json),
            "route_eval_summary_md": str(model_output_dir / "route_eval_summary.md"),
            "benchmark_overview_csv": str(model_output_dir / "benchmark_overview.csv"),
        },
        "aggregate": route_summary["aggregate"],
        "videos": route_summary["videos"],
    }


def pick_best_model(
    model_summaries: list[dict[str, Any]],
    key: str,
    *,
    reverse: bool,
) -> dict[str, Any]:
    return sorted(
        model_summaries,
        key=lambda item: float(item["aggregate"][key]),
        reverse=reverse,
    )[0]


def build_leaderboard(model_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    best_unified_count = pick_best_model(
        model_summaries,
        "weighted_unified_mean_count_per_frame",
        reverse=True,
    )
    best_unified_ratio = pick_best_model(
        model_summaries,
        "weighted_unified_frames_with_detections_ratio",
        reverse=True,
    )
    lowest_switch_rate = pick_best_model(
        model_summaries,
        "weighted_dominant_switch_rate",
        reverse=False,
    )
    lowest_beta_delta = pick_best_model(
        model_summaries,
        "weighted_beta_abs_delta_mean",
        reverse=False,
    )

    return {
        "best_unified_mean_count_per_frame": {
            "label": best_unified_count["label"],
            "value": best_unified_count["aggregate"][
                "weighted_unified_mean_count_per_frame"
            ],
        },
        "best_unified_frames_with_detections_ratio": {
            "label": best_unified_ratio["label"],
            "value": best_unified_ratio["aggregate"][
                "weighted_unified_frames_with_detections_ratio"
            ],
        },
        "lowest_fog_switch_rate": {
            "label": lowest_switch_rate["label"],
            "value": lowest_switch_rate["aggregate"][
                "weighted_dominant_switch_rate"
            ],
        },
        "lowest_beta_abs_delta_mean": {
            "label": lowest_beta_delta["label"],
            "value": lowest_beta_delta["aggregate"][
                "weighted_beta_abs_delta_mean"
            ],
        },
    }


def write_model_overview_csv(
    model_summaries: list[dict[str, Any]],
    output_path: Path,
):
    fieldnames = [
        "entry_id",
        "label",
        "slug",
        "role",
        "status",
        "tags",
        "video_count",
        "total_sampled_frames",
        "weighted_unified_mean_count_per_frame",
        "weighted_unified_frames_with_detections_ratio",
        "weighted_hybrid_mean_count_per_frame",
        "weighted_hybrid_frames_with_detections_ratio",
        "weighted_mean_count_gap_hybrid_minus_unified",
        "weighted_mean_abs_count_gap",
        "weighted_beta_mean",
        "weighted_dominant_switch_rate",
        "weighted_beta_abs_delta_mean",
        "weighted_unified_heuristic_persistent_static_fp_per_min",
        "weighted_hybrid_heuristic_persistent_static_fp_per_min",
        "weighted_unified_confirmation_latency_frames",
        "weighted_hybrid_confirmation_latency_frames",
        "recommendation_hist",
        "route_eval_summary_json",
        "notes",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in model_summaries:
            aggregate = item["aggregate"]
            writer.writerow(
                {
                    "entry_id": int(item["entry_id"]),
                    "label": item["label"],
                    "slug": item["slug"],
                    "role": item["role"],
                    "status": item["status"],
                    "tags": "|".join(item["tags"]),
                    "video_count": int(aggregate["video_count"]),
                    "total_sampled_frames": int(aggregate["total_sampled_frames"]),
                    "weighted_unified_mean_count_per_frame": round(
                        float(aggregate["weighted_unified_mean_count_per_frame"]),
                        6,
                    ),
                    "weighted_unified_frames_with_detections_ratio": round(
                        float(
                            aggregate["weighted_unified_frames_with_detections_ratio"]
                        ),
                        6,
                    ),
                    "weighted_hybrid_mean_count_per_frame": round(
                        float(aggregate["weighted_hybrid_mean_count_per_frame"]),
                        6,
                    ),
                    "weighted_hybrid_frames_with_detections_ratio": round(
                        float(aggregate["weighted_hybrid_frames_with_detections_ratio"]),
                        6,
                    ),
                    "weighted_mean_count_gap_hybrid_minus_unified": round(
                        float(
                            aggregate[
                                "weighted_mean_count_gap_hybrid_minus_unified"
                            ]
                        ),
                        6,
                    ),
                    "weighted_mean_abs_count_gap": round(
                        float(aggregate["weighted_mean_abs_count_gap"]),
                        6,
                    ),
                    "weighted_beta_mean": round(
                        float(aggregate["weighted_beta_mean"]),
                        6,
                    ),
                    "weighted_dominant_switch_rate": round(
                        float(aggregate["weighted_dominant_switch_rate"]),
                        6,
                    ),
                    "weighted_beta_abs_delta_mean": round(
                        float(aggregate["weighted_beta_abs_delta_mean"]),
                        6,
                    ),
                    "weighted_unified_heuristic_persistent_static_fp_per_min": round(
                        float(
                            aggregate.get(
                                "weighted_unified_heuristic_persistent_static_fp_per_min",
                                0.0,
                            )
                        ),
                        6,
                    ),
                    "weighted_hybrid_heuristic_persistent_static_fp_per_min": round(
                        float(
                            aggregate.get(
                                "weighted_hybrid_heuristic_persistent_static_fp_per_min",
                                0.0,
                            )
                        ),
                        6,
                    ),
                    "weighted_unified_confirmation_latency_frames": round(
                        float(
                            aggregate.get(
                                "weighted_unified_confirmation_latency_frames",
                                0.0,
                            )
                        ),
                        6,
                    ),
                    "weighted_hybrid_confirmation_latency_frames": round(
                        float(
                            aggregate.get(
                                "weighted_hybrid_confirmation_latency_frames",
                                0.0,
                            )
                        ),
                        6,
                    ),
                    "recommendation_hist": json.dumps(
                        aggregate["recommendation_hist"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "route_eval_summary_json": item["artifacts"][
                        "route_eval_summary_json"
                    ],
                    "notes": item["notes"],
                }
            )


def write_model_video_matrix_csv(
    model_summaries: list[dict[str, Any]],
    output_path: Path,
):
    fieldnames = [
        "model_label",
        "model_slug",
        "model_role",
        "model_status",
        "video_label",
        "video_name",
        "tags",
        "sampled_frames",
        "recommendation",
        "unified_mean_count_per_frame",
        "unified_frames_with_detections_ratio",
        "hybrid_mean_count_per_frame",
        "hybrid_frames_with_detections_ratio",
        "mean_count_gap_hybrid_minus_unified",
        "beta_mean",
        "beta_std",
        "majority_fog_label",
        "dominant_switch_rate",
        "beta_abs_delta_mean",
        "frame_metrics_csv",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in model_summaries:
            for video in item["videos"]:
                writer.writerow(
                    {
                        "model_label": item["label"],
                        "model_slug": item["slug"],
                        "model_role": item["role"],
                        "model_status": item["status"],
                        "video_label": video["benchmark"]["label"],
                        "video_name": video["video_name"],
                        "tags": "|".join(video["benchmark"]["tags"]),
                        "sampled_frames": int(video["sampled_frames"]),
                        "recommendation": video["heuristic_recommendation"]["route"],
                        "unified_mean_count_per_frame": round(
                            float(video["unified"]["mean_count_per_frame"]),
                            6,
                        ),
                        "unified_frames_with_detections_ratio": round(
                            float(video["unified"]["frames_with_detections_ratio"]),
                            6,
                        ),
                        "hybrid_mean_count_per_frame": round(
                            float(video["hybrid"]["mean_count_per_frame"]),
                            6,
                        ),
                        "hybrid_frames_with_detections_ratio": round(
                            float(video["hybrid"]["frames_with_detections_ratio"]),
                            6,
                        ),
                        "mean_count_gap_hybrid_minus_unified": round(
                            float(
                                video["comparison"][
                                    "mean_count_gap_hybrid_minus_unified"
                                ]
                            ),
                            6,
                        ),
                        "beta_mean": round(float(video["fog"]["beta"]["mean"]), 6),
                        "beta_std": round(float(video["fog"]["beta"]["std"]), 6),
                        "majority_fog_label": video["fog"]["stability"][
                            "majority_fog_label"
                        ],
                        "dominant_switch_rate": round(
                            float(video["fog"]["stability"]["dominant_switch_rate"]),
                            6,
                        ),
                        "beta_abs_delta_mean": round(
                            float(video["fog"]["stability"]["beta_abs_delta_mean"]),
                            6,
                        ),
                        "frame_metrics_csv": video["artifacts"]["frame_metrics_csv"],
                    }
                )


def write_markdown_report(
    payload: dict[str, Any],
    output_path: Path,
):
    lines = [
        "# Multi-Model Benchmark Report",
        "",
        "## Scope",
        "",
        f"- Benchmark ID: `{payload['meta'].get('benchmark_id', '') or 'N/A'}`",
        f"- Benchmark config: `{payload['meta']['benchmark_config']}`",
        f"- Model config: `{payload['meta']['model_config']}`",
        f"- Yolo weights: `{payload['meta']['yolo_weights']}`",
        f"- Output dir: `{payload['meta']['output_dir']}`",
        f"- Reuse existing: `{payload['meta']['reuse_existing']}`",
        "",
    ]

    selection_rules = payload["meta"].get("model_selection_rules", [])
    if selection_rules:
        lines.extend(
            [
                "## Model Selection Rules",
                "",
            ]
        )
        for rule in selection_rules:
            lines.append(f"- {rule}")
        lines.append("")

    lines.extend(
        [
        "## Leaders",
        "",
        (
            "- Best unified mean detections/frame: "
            f"`{payload['leaders']['best_unified_mean_count_per_frame']['label']}` "
            f"({payload['leaders']['best_unified_mean_count_per_frame']['value']:.3f})"
        ),
        (
            "- Best unified nonzero-detection ratio: "
            f"`{payload['leaders']['best_unified_frames_with_detections_ratio']['label']}` "
            f"({payload['leaders']['best_unified_frames_with_detections_ratio']['value']:.3f})"
        ),
        (
            "- Lowest fog switch rate: "
            f"`{payload['leaders']['lowest_fog_switch_rate']['label']}` "
            f"({payload['leaders']['lowest_fog_switch_rate']['value']:.3f})"
        ),
        (
            "- Lowest beta abs delta mean: "
            f"`{payload['leaders']['lowest_beta_abs_delta_mean']['label']}` "
            f"({payload['leaders']['lowest_beta_abs_delta_mean']['value']:.5f})"
        ),
        "",
        "## Model Summary",
        "",
        ]
    )

    for item in payload["models"]:
        aggregate = item["aggregate"]
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- Fog weights: `{item['fog_weights']}`",
                f"- Role: `{item['role'] or 'N/A'}`",
                f"- Status: `{item['status']}`",
                f"- Tags: `{item['tags']}`",
                f"- Notes: {item['notes'] or 'N/A'}",
                f"- Weighted unified mean detections/frame: `{aggregate['weighted_unified_mean_count_per_frame']:.3f}`",
                f"- Weighted unified nonzero-detection ratio: `{aggregate['weighted_unified_frames_with_detections_ratio']:.3f}`",
                f"- Weighted hybrid mean detections/frame: `{aggregate['weighted_hybrid_mean_count_per_frame']:.3f}`",
                f"- Weighted beta mean: `{aggregate['weighted_beta_mean']:.5f}`",
                f"- Weighted fog switch rate: `{aggregate['weighted_dominant_switch_rate']:.3f}`",
                f"- Weighted beta abs delta mean: `{aggregate['weighted_beta_abs_delta_mean']:.5f}`",
                f"- Weighted unified heuristic static-FP/min: `{aggregate.get('weighted_unified_heuristic_persistent_static_fp_per_min', 0.0):.5f}`",
                f"- Weighted hybrid heuristic static-FP/min: `{aggregate.get('weighted_hybrid_heuristic_persistent_static_fp_per_min', 0.0):.5f}`",
                f"- Weighted unified confirmation latency (frames): `{aggregate.get('weighted_unified_confirmation_latency_frames', 0.0):.3f}`",
                f"- Weighted hybrid confirmation latency (frames): `{aggregate.get('weighted_hybrid_confirmation_latency_frames', 0.0):.3f}`",
                f"- Recommendation histogram: `{aggregate['recommendation_hist']}`",
                f"- Route eval JSON: `{item['artifacts']['route_eval_summary_json']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Artifact Paths",
            "",
            f"- Model overview CSV: `{payload['artifacts']['model_overview_csv']}`",
            f"- Model-video matrix CSV: `{payload['artifacts']['model_video_matrix_csv']}`",
            f"- Combined JSON: `{payload['artifacts']['summary_json']}`",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    benchmark_config = resolve_path(args.benchmark_config)
    model_config = resolve_path(args.model_config)
    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if not benchmark_config.exists():
        raise FileNotFoundError(f"Benchmark config not found: {benchmark_config}")

    yolo_weights = resolve_path(args.yolo_weights)
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    model_entries, model_meta = load_model_entries(model_config)

    print(f"Benchmark config: {benchmark_config}")
    print(f"Model config: {model_config}")
    print(f"Output dir: {output_root}")
    print(f"Models to evaluate: {len(model_entries)}")

    model_summaries = [
        run_single_model(args, benchmark_config, output_root, model_entry)
        for model_entry in model_entries
    ]

    payload = {
        "meta": {
            "benchmark_id": model_meta["benchmark_id"],
            "benchmark_config": str(benchmark_config),
            "model_config": str(model_config),
            "model_config_description": model_meta["model_config_description"],
            "model_selection_rules": model_meta["model_selection_rules"],
            "yolo_weights": str(yolo_weights),
            "output_dir": str(output_root),
            "reuse_existing": bool(args.reuse_existing),
            "overrides": {
                "sample_stride": args.sample_stride,
                "max_frames": args.max_frames,
                "unified_conf": args.unified_conf,
                "hybrid_conf": args.hybrid_conf,
                "imgsz": args.imgsz,
                "topk_preview": args.topk_preview,
            },
        },
        "leaders": build_leaderboard(model_summaries),
        "artifacts": {},
        "models": model_summaries,
    }

    overview_csv = output_root / "model_overview.csv"
    matrix_csv = output_root / "model_video_matrix.csv"
    summary_json = output_root / "multi_model_benchmark_summary.json"
    summary_md = output_root / "multi_model_benchmark_summary.md"

    write_model_overview_csv(model_summaries, overview_csv)
    write_model_video_matrix_csv(model_summaries, matrix_csv)
    payload["artifacts"] = {
        "model_overview_csv": str(overview_csv.relative_to(output_root)),
        "model_video_matrix_csv": str(matrix_csv.relative_to(output_root)),
        "summary_json": str(summary_json.relative_to(output_root)),
        "summary_md": str(summary_md.relative_to(output_root)),
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(payload, summary_md)

    print(f"Model overview CSV: {overview_csv}")
    print(f"Model-video matrix CSV: {matrix_csv}")
    print(f"Combined JSON: {summary_json}")
    print(f"Combined Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
