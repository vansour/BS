#!/usr/bin/env python3
"""
Postprocess a completed formal fog-focused training run.

This script:
1. Locates the latest completed training run under the given output directory.
2. Runs offline route evaluation with the new best model.
3. Generates a compact markdown/json summary for experiment logging and thesis use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess a completed fog-focused formal training run."
    )
    parser.add_argument(
        "--train-output-dir",
        default=str(ROOT / "outputs" / "Fog_Detection_Project_fogfocus_full"),
        help="Formal fog-focused training output directory.",
    )
    parser.add_argument(
        "--video",
        default=str(ROOT / "gettyimages-1353950094-640_adpp.mp4"),
        help="Video used for route evaluation.",
    )
    parser.add_argument(
        "--route-eval-output",
        default=str(ROOT / "outputs" / "Route_Eval_fogfocus_full_formal"),
        help="Route evaluation output directory.",
    )
    parser.add_argument(
        "--candidate-benchmark-config",
        default=str(ROOT / "configs" / "benchmark_videos.json"),
        help="Benchmark 视频配置文件，用于候选模型横向对比。",
    )
    parser.add_argument(
        "--baseline-model-config",
        default=str(ROOT / "configs" / "benchmark_models.json"),
        help="Baseline 模型配置文件，用于候选模型横向对比。",
    )
    parser.add_argument(
        "--candidate-benchmark-output",
        default=str(ROOT / "outputs" / "Candidate_Benchmark_fogfocus_full_formal"),
        help="候选模型 benchmark 输出目录。",
    )
    parser.add_argument(
        "--candidate-benchmark-label",
        default="fogfocus_full_formal_candidate",
        help="候选模型在 benchmark 对比中的展示名称。",
    )
    parser.add_argument(
        "--skip-candidate-benchmark",
        action="store_true",
        help="跳过候选模型与 baseline 集合的横向对比。",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=5,
        help="Sampling stride for route evaluation.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum sampled frames for route evaluation.",
    )
    return parser.parse_args()


def latest_completed_run(train_output_dir: Path) -> Path:
    candidates = sorted(train_output_dir.glob("runs/*/summary.json"))
    if not candidates:
        raise RuntimeError(
            f"No completed training run summary was found under: {train_output_dir / 'runs'}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_route_eval(
    fog_weights: Path,
    video: Path,
    route_eval_output: Path,
    sample_stride: int,
    max_frames: int,
):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_inference_routes.py"),
        "--video",
        str(video),
        "--fog-weights",
        str(fog_weights),
        "--sample-stride",
        str(sample_stride),
        "--max-frames",
        str(max_frames),
        "--topk-preview",
        "8",
        "--output-dir",
        str(route_eval_output),
    ]
    print("Running route evaluation:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_candidate_benchmark(
    fog_weights: Path,
    candidate_label: str,
    benchmark_config: Path,
    baseline_model_config: Path,
    output_dir: Path,
    sample_stride: int,
    max_frames: int,
):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_candidate_model.py"),
        "--candidate-weights",
        str(fog_weights),
        "--candidate-label",
        str(candidate_label),
        "--benchmark-config",
        str(benchmark_config),
        "--base-model-config",
        str(baseline_model_config),
        "--output-dir",
        str(output_dir),
        "--sample-stride",
        str(sample_stride),
        "--max-frames",
        str(max_frames),
        "--reuse-existing",
    ]
    print("Running candidate benchmark:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def maybe_load_previous_baseline() -> dict | None:
    baseline = ROOT / "outputs" / "Route_Eval_fogfocus_final" / "route_eval_summary.json"
    if not baseline.exists():
        return None
    return load_json(baseline)


def build_summary(
    train_output_dir: Path,
    run_dir: Path,
    route_eval_output: Path,
    training_summary: dict,
    route_eval_summary: dict,
    candidate_benchmark_summary: dict | None,
    candidate_benchmark_output: Path | None,
    baseline_summary: dict | None,
) -> dict:
    latest_video = route_eval_summary["videos"][0]
    best_epoch_record = None
    for item in training_summary.get("phase_summaries", []):
        if item.get("phase") == "fp32":
            best_epoch_record = item
    summary = {
        "train_output_dir": str(train_output_dir),
        "run_dir": str(run_dir),
        "artifacts": {
            "best_model": str(train_output_dir / "unified_model_best.pt"),
            "final_model": str(train_output_dir / "unified_model.pt"),
            "training_summary_json": str(run_dir / "summary.json"),
            "route_eval_summary_json": str(route_eval_output / "route_eval_summary.json"),
            "route_eval_summary_md": str(route_eval_output / "route_eval_summary.md"),
        },
        "training": {
            "status": training_summary.get("status"),
            "best_loss": training_summary.get("best_loss"),
            "best_qat_loss": training_summary.get("best_qat_loss"),
            "last_epoch_record": best_epoch_record,
        },
        "route_eval": {
            "video_name": latest_video["video_name"],
            "sampled_frames": latest_video["sampled_frames"],
            "unified": latest_video["unified"],
            "hybrid": latest_video["hybrid"],
            "comparison": latest_video["comparison"],
            "fog": latest_video["fog"],
            "recommendation": latest_video["heuristic_recommendation"],
        },
    }

    if candidate_benchmark_summary and candidate_benchmark_output is not None:
        summary["artifacts"]["candidate_benchmark_summary_json"] = str(
            candidate_benchmark_output / "candidate_benchmark_summary.json"
        )
        summary["artifacts"]["candidate_benchmark_summary_md"] = str(
            candidate_benchmark_output / "candidate_benchmark_summary.md"
        )
        summary["candidate_benchmark"] = candidate_benchmark_summary

    if baseline_summary:
        baseline_video = baseline_summary["videos"][0]
        summary["baseline_comparison"] = {
            "baseline_route_eval": str(
                ROOT / "outputs" / "Route_Eval_fogfocus_final" / "route_eval_summary.json"
            ),
            "unified_mean_detection_before": baseline_video["unified"]["mean_count_per_frame"],
            "unified_mean_detection_after": latest_video["unified"]["mean_count_per_frame"],
            "hybrid_mean_detection_before": baseline_video["hybrid"]["mean_count_per_frame"],
            "hybrid_mean_detection_after": latest_video["hybrid"]["mean_count_per_frame"],
            "beta_mean_before": baseline_video["fog"]["beta"]["mean"],
            "beta_mean_after": latest_video["fog"]["beta"]["mean"],
            "dominant_hist_before": baseline_video["fog"]["probs"]["dominant_hist"],
            "dominant_hist_after": latest_video["fog"]["probs"]["dominant_hist"],
        }
    return summary


def write_markdown(path: Path, summary: dict):
    route = summary["route_eval"]
    training = summary["training"]
    lines = [
        "# Fog-Focused Formal Experiment Summary",
        "",
        "## Training",
        "",
        f"- Train output dir: `{summary['train_output_dir']}`",
        f"- Run dir: `{summary['run_dir']}`",
        f"- Status: `{training['status']}`",
        f"- Best loss: `{training['best_loss']}`",
        f"- Best model: `{summary['artifacts']['best_model']}`",
        "",
        "## Route Evaluation",
        "",
        f"- Video: `{route['video_name']}`",
        f"- Sampled frames: `{route['sampled_frames']}`",
        f"- Unified mean detections/frame: `{route['unified']['mean_count_per_frame']:.3f}`",
        f"- Unified nonzero-detection ratio: `{route['unified']['frames_with_detections_ratio']:.3f}`",
        f"- Hybrid mean detections/frame: `{route['hybrid']['mean_count_per_frame']:.3f}`",
        f"- Hybrid nonzero-detection ratio: `{route['hybrid']['frames_with_detections_ratio']:.3f}`",
        f"- Mean beta: `{route['fog']['beta']['mean']:.5f}`",
        f"- Dominant fog histogram: `{route['fog']['probs']['dominant_hist']}`",
        f"- Recommendation: `{route['recommendation']['route']}`",
        f"- Recommendation reason: {route['recommendation']['reason']}",
        "",
    ]

    baseline = summary.get("baseline_comparison")
    if baseline:
        lines.extend(
            [
                "## Baseline Comparison",
                "",
                f"- Unified mean detections/frame: `{baseline['unified_mean_detection_before']:.3f}` -> `{baseline['unified_mean_detection_after']:.3f}`",
                f"- Hybrid mean detections/frame: `{baseline['hybrid_mean_detection_before']:.3f}` -> `{baseline['hybrid_mean_detection_after']:.3f}`",
                f"- Mean beta: `{baseline['beta_mean_before']:.5f}` -> `{baseline['beta_mean_after']:.5f}`",
                f"- Dominant fog histogram before: `{baseline['dominant_hist_before']}`",
                f"- Dominant fog histogram after: `{baseline['dominant_hist_after']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Artifact Paths",
            "",
            f"- Training summary: `{summary['artifacts']['training_summary_json']}`",
            f"- Route eval JSON: `{summary['artifacts']['route_eval_summary_json']}`",
            f"- Route eval Markdown: `{summary['artifacts']['route_eval_summary_md']}`",
            "",
        ]
    )

    candidate = summary.get("candidate_benchmark")
    if candidate:
        lines.extend(
            [
                "## Candidate Benchmark",
                "",
                f"- Candidate label: `{candidate['candidate']['label']}`",
                f"- Unified mean detections/frame rank: `{candidate['candidate']['rankings']['weighted_unified_mean_count_per_frame']}`",
                f"- Unified nonzero-detection ratio rank: `{candidate['candidate']['rankings']['weighted_unified_frames_with_detections_ratio']}`",
                f"- Fog switch rate rank: `{candidate['candidate']['rankings']['weighted_dominant_switch_rate']}`",
                f"- Beta abs delta mean rank: `{candidate['candidate']['rankings']['weighted_beta_abs_delta_mean']}`",
                f"- Candidate benchmark JSON: `{summary['artifacts']['candidate_benchmark_summary_json']}`",
                f"- Candidate benchmark Markdown: `{summary['artifacts']['candidate_benchmark_summary_md']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    train_output_dir = Path(args.train_output_dir).resolve()
    video = Path(args.video).resolve()
    route_eval_output = Path(args.route_eval_output).resolve()
    route_eval_output.mkdir(parents=True, exist_ok=True)
    candidate_benchmark_output = Path(args.candidate_benchmark_output).resolve()
    benchmark_config = Path(args.candidate_benchmark_config).resolve()
    baseline_model_config = Path(args.baseline_model_config).resolve()

    run_dir = latest_completed_run(train_output_dir)
    training_summary = load_json(run_dir / "summary.json")
    best_model = train_output_dir / "unified_model_best.pt"
    if not best_model.exists():
        raise FileNotFoundError(f"Best model not found: {best_model}")

    run_route_eval(
        best_model,
        video,
        route_eval_output,
        args.sample_stride,
        args.max_frames,
    )

    route_eval_summary = load_json(route_eval_output / "route_eval_summary.json")
    candidate_benchmark_summary = None
    if not args.skip_candidate_benchmark:
        candidate_benchmark_output.mkdir(parents=True, exist_ok=True)
        run_candidate_benchmark(
            best_model,
            args.candidate_benchmark_label,
            benchmark_config,
            baseline_model_config,
            candidate_benchmark_output,
            args.sample_stride,
            args.max_frames,
        )
        candidate_benchmark_summary = load_json(
            candidate_benchmark_output / "candidate_benchmark_summary.json"
        )
    baseline_summary = maybe_load_previous_baseline()

    summary = build_summary(
        train_output_dir,
        run_dir,
        route_eval_output,
        training_summary,
        route_eval_summary,
        candidate_benchmark_summary,
        candidate_benchmark_output if candidate_benchmark_summary else None,
        baseline_summary,
    )

    summary_json = train_output_dir / "formal_experiment_summary.json"
    summary_md = train_output_dir / "formal_experiment_summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary_md, summary)

    print(f"Formal experiment summary JSON: {summary_json}")
    print(f"Formal experiment summary Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
