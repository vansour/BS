#!/usr/bin/env python3
"""
离线比较统一模型推理与混合推理路线。

该脚本面向“先做路线判断，再决定是否继续优化统一推理”这个阶段，
把两条当前可用方案放到同一套离线评估框架里：

1. 统一方案：
   使用 `HighwayFogSystem` 同时输出雾分类、beta 和车辆检测。
2. 混合方案：
   使用 `HighwayFogSystem` 输出雾分类、beta，
   再用独立 `yolo11n.pt` 做车辆检测。

脚本不会伪造 AP/mAP 这类没有标注就无法成立的指标，而是输出当前阶段
真正有价值的离线观测量：
- 每帧 fog 概率和 beta；
- 两条路线的检测数量、平均置信度、最高置信度；
- “混合方案比统一方案多检出 / 少检出”的帧数；
- 若干差异最大的代表帧截图。

输出内容包括：
- `route_eval_summary.json`
- `route_eval_summary.md`
- 每个视频一个 `frame_metrics.csv`
- 每个视频若干差异帧 `preview_*.jpg`
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.inference import HighwayFogSystem
from src.temporal_vehicle_filter import TemporalVehicleFilter
from src.utils import resolve_model_weights


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
ENTRY_OVERRIDE_KEYS = (
    "sample_stride",
    "max_frames",
    "unified_conf",
    "hybrid_conf",
    "imgsz",
    "topk_preview",
)
ACTIVE_VIDEO_STATUSES = {"active", "enabled", "runnable"}
CLI_OVERRIDE_FLAGS = {
    "sample_stride": "--sample-stride",
    "max_frames": "--max-frames",
    "unified_conf": "--unified-conf",
    "hybrid_conf": "--hybrid-conf",
    "imgsz": "--imgsz",
    "topk_preview": "--topk-preview",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline comparison between unified inference and hybrid inference."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="可选配置文件路径（.json/.yaml/.yml）。",
    )
    parser.add_argument(
        "--benchmark-config",
        default=None,
        help="Benchmark 配置 JSON。若提供，则优先于 --video 和 --video-dir。",
    )
    parser.add_argument(
        "--video",
        default=str(PROJECT_ROOT / "gettyimages-1353950094-640_adpp.mp4"),
        help="单个输入视频路径。若同时提供 --video-dir，则忽略该参数。",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="批量评估目录。会扫描常见视频扩展名。",
    )
    parser.add_argument(
        "--fog-weights",
        default=None,
        help="雾模型权重。默认自动解析最佳可用权重。",
    )
    parser.add_argument(
        "--yolo-weights",
        default=str(PROJECT_ROOT / "yolo11n.pt"),
        help="混合方案车辆检测权重。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "Route_Eval"),
        help="评估报告输出目录。",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=5,
        help="每隔多少帧采样 1 帧。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="每个视频最多评估多少个采样帧。0 表示不限制。",
    )
    parser.add_argument(
        "--unified-conf",
        type=float,
        default=0.25,
        help="统一模型检测结果统计阈值。",
    )
    parser.add_argument(
        "--hybrid-conf",
        type=float,
        default=0.10,
        help="混合方案 YOLO 检测统计阈值。",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="混合方案 YOLO 推理尺寸。",
    )
    parser.add_argument(
        "--topk-preview",
        type=int,
        default=6,
        help="每个视频最多保存多少张差异代表帧。",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_cli_override_flags(argv: list[str]) -> set[str]:
    flags = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        flags.add(token.split("=", 1)[0])
    return flags


def sanitize_output_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._") or "video"


def resolve_video_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def normalize_video_entry(
    raw_entry: dict[str, Any],
    *,
    index: int,
    source: str,
) -> dict[str, Any]:
    if not isinstance(raw_entry, dict):
        raise TypeError(f"Video entry must be an object, got {type(raw_entry)!r}")

    raw_path = raw_entry.get("path")
    if not raw_path or not isinstance(raw_path, str):
        raise ValueError("Each video entry must provide a non-empty string field `path`.")

    video_path = resolve_video_path(raw_path)
    status = str(raw_entry.get("status", "active") or "active").strip().lower()
    enabled = bool(raw_entry.get("enabled", status in ACTIVE_VIDEO_STATUSES))
    if enabled and not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    label = str(raw_entry.get("label") or video_path.stem).strip() or video_path.stem
    tags = raw_entry.get("tags", [])
    if tags is None:
        tags = []
    if not isinstance(tags, list):
        raise TypeError("Video entry field `tags` must be a list of strings.")
    normalized_tags = [str(item).strip() for item in tags if str(item).strip()]
    notes = str(raw_entry.get("notes", "") or "").strip()

    entry = {
        "entry_id": int(index),
        "source": source,
        "raw_path": raw_path,
        "video_path": video_path,
        "label": label,
        "status": status,
        "enabled": enabled,
        "tags": normalized_tags,
        "notes": notes,
        "output_name": f"{index:02d}_{sanitize_output_name(label)}",
    }
    for key in ENTRY_OVERRIDE_KEYS:
        if key in raw_entry and raw_entry[key] is not None:
            entry[key] = raw_entry[key]
    return entry


def load_benchmark_entries(
    config_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_videos = payload.get("videos", [])
        benchmark_id = str(payload.get("benchmark_id", "") or "").strip()
        description = str(payload.get("description", "") or "").strip()
        default_runtime = payload.get("default_runtime", {})
    elif isinstance(payload, list):
        raw_videos = payload
        benchmark_id = ""
        description = ""
        default_runtime = {}
    else:
        raise TypeError("Benchmark config must be either an object or a list.")

    if not isinstance(raw_videos, list) or not raw_videos:
        raise ValueError("Benchmark config does not contain any video entries.")
    if default_runtime is None:
        default_runtime = {}
    if not isinstance(default_runtime, dict):
        raise TypeError("Benchmark config field `default_runtime` must be an object.")

    all_entries = [
        normalize_video_entry(
            {**default_runtime, **raw_entry},
            index=index,
            source="benchmark_config",
        )
        for index, raw_entry in enumerate(raw_videos, start=1)
    ]
    entries = [entry for entry in all_entries if entry["enabled"]]
    if not entries:
        raise ValueError("Benchmark config does not contain any active/enabled video entries.")

    inactive_entries = [entry for entry in all_entries if not entry["enabled"]]
    return entries, {
        "video_source_mode": "benchmark_config",
        "benchmark_config": str(config_path),
        "benchmark_id": benchmark_id,
        "benchmark_description": description,
        "benchmark_default_runtime": default_runtime,
        "benchmark_total_entries": len(all_entries),
        "benchmark_active_entries": len(entries),
        "benchmark_inactive_entries": len(inactive_entries),
        "benchmark_inactive_labels": [entry["label"] for entry in inactive_entries],
    }


def collect_video_entries(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.benchmark_config:
        return load_benchmark_entries(resolve_video_path(args.benchmark_config))

    if args.video_dir:
        root = Path(args.video_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Video directory not found: {root}")
        videos = sorted(
            [
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
            ]
        )
        if not videos:
            raise RuntimeError(f"No supported videos were found in: {root}")
        entries = [
            normalize_video_entry(
                {"path": str(path), "label": path.stem},
                index=index,
                source="video_dir",
            )
            for index, path in enumerate(videos, start=1)
        ]
        return entries, {
            "video_source_mode": "video_dir",
            "video_dir": str(root),
        }

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return [
        normalize_video_entry(
            {"path": str(video_path), "label": video_path.stem},
            index=1,
            source="single_video",
        )
    ], {
        "video_source_mode": "single_video",
        "video": str(video_path),
    }


def filter_unified_detections(
    detections: torch.Tensor,
    conf_thres: float,
) -> torch.Tensor:
    if detections.numel() == 0:
        return detections
    return detections[detections[:, 4] >= float(conf_thres)]


def extract_hybrid_detections(result, conf_thres: float) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    hybrid_detections: list[dict[str, Any]] = []
    xyxy_list = boxes.xyxy.cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()
    cls_list = boxes.cls.int().cpu().tolist()
    for xyxy, score, cls_id in zip(xyxy_list, conf_list, cls_list):
        if float(score) < float(conf_thres) or int(cls_id) not in VEHICLE_CLASS_IDS:
            continue
        hybrid_detections.append(
            {
                "xyxy": [float(v) for v in xyxy],
                "conf": float(score),
                "cls_id": int(cls_id),
                "name": VEHICLE_CLASS_NAMES.get(int(cls_id), str(cls_id)),
            }
        )
    return hybrid_detections


def update_route_accumulator(accumulator: dict[str, float], count: int, confs: list[float]):
    accumulator["frames"] += 1
    accumulator["total_count"] += int(count)
    accumulator["max_count"] = max(accumulator["max_count"], int(count))
    if count > 0:
        accumulator["frames_with_det"] += 1
        accumulator["max_conf"] = max(
            accumulator["max_conf"], max(float(score) for score in confs)
        )
        accumulator["conf_sum"] += sum(float(score) for score in confs)
        accumulator["conf_count"] += int(count)


def summarize_route(accumulator: dict[str, float]) -> dict[str, float]:
    frames = max(1, int(accumulator["frames"]))
    conf_count = int(accumulator["conf_count"])
    return {
        "frames": int(accumulator["frames"]),
        "frames_with_detections": int(accumulator["frames_with_det"]),
        "frames_with_detections_ratio": accumulator["frames_with_det"] / frames,
        "mean_count_per_frame": accumulator["total_count"] / frames,
        "max_count_per_frame": int(accumulator["max_count"]),
        "mean_conf": accumulator["conf_sum"] / conf_count if conf_count > 0 else 0.0,
        "max_conf": float(accumulator["max_conf"]),
        "total_detections": int(accumulator["total_count"]),
    }


def summarize_beta(beta_values: list[float]) -> dict[str, float]:
    if not beta_values:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    beta_array = np.asarray(beta_values, dtype=np.float32)
    return {
        "mean": float(beta_array.mean()),
        "min": float(beta_array.min()),
        "max": float(beta_array.max()),
        "std": float(beta_array.std()),
    }


def summarize_fog_probs(prob_values: list[np.ndarray]) -> dict[str, Any]:
    if not prob_values:
        return {
            "mean_probs": [0.0, 0.0, 0.0],
            "dominant_hist": {
                label: 0 for label in HighwayFogSystem.FOG_CLASS_NAMES
            },
        }

    prob_array = np.stack(prob_values, axis=0).astype(np.float32)
    dominant_indices = prob_array.argmax(axis=1)
    hist = {
        HighwayFogSystem.FOG_CLASS_NAMES[index]: int((dominant_indices == index).sum())
        for index in range(prob_array.shape[1])
    }
    return {
        "mean_probs": [float(v) for v in prob_array.mean(axis=0)],
        "dominant_hist": hist,
    }


def summarize_fog_stability(
    prob_values: list[np.ndarray],
    beta_values: list[float],
) -> dict[str, Any]:
    if not prob_values:
        return {
            "majority_fog_label": "N/A",
            "dominant_switch_count": 0,
            "dominant_switch_rate": 0.0,
            "mean_top1_prob": 0.0,
            "mean_margin": 0.0,
            "beta_abs_delta_mean": 0.0,
            "beta_abs_delta_max": 0.0,
            "beta_delta_std": 0.0,
        }

    prob_array = np.stack(prob_values, axis=0).astype(np.float32)
    dominant_indices = prob_array.argmax(axis=1)
    dominant_hist = np.bincount(dominant_indices, minlength=prob_array.shape[1])
    majority_index = int(dominant_hist.argmax()) if dominant_hist.size > 0 else 0

    top1 = prob_array.max(axis=1)
    if prob_array.shape[1] > 1:
        partitioned = np.partition(prob_array, kth=prob_array.shape[1] - 2, axis=1)
        top2 = partitioned[:, -2]
    else:
        top2 = np.zeros_like(top1)

    transitions = max(prob_array.shape[0] - 1, 0)
    switch_count = (
        int(np.count_nonzero(dominant_indices[1:] != dominant_indices[:-1]))
        if transitions > 0
        else 0
    )

    beta_array = np.asarray(beta_values, dtype=np.float32)
    if beta_array.size > 1:
        beta_deltas = np.diff(beta_array)
        beta_abs_deltas = np.abs(beta_deltas)
        beta_abs_delta_mean = float(beta_abs_deltas.mean())
        beta_abs_delta_max = float(beta_abs_deltas.max())
        beta_delta_std = float(beta_deltas.std())
    else:
        beta_abs_delta_mean = 0.0
        beta_abs_delta_max = 0.0
        beta_delta_std = 0.0

    return {
        "majority_fog_label": HighwayFogSystem.FOG_CLASS_NAMES[majority_index],
        "dominant_switch_count": switch_count,
        "dominant_switch_rate": (
            float(switch_count / transitions) if transitions > 0 else 0.0
        ),
        "mean_top1_prob": float(top1.mean()),
        "mean_margin": float((top1 - top2).mean()),
        "beta_abs_delta_mean": beta_abs_delta_mean,
        "beta_abs_delta_max": beta_abs_delta_max,
        "beta_delta_std": beta_delta_std,
    }


def resolve_runtime_setting(
    video_entry: dict[str, Any],
    args: argparse.Namespace,
    key: str,
):
    provided_flags = getattr(args, "_provided_flags", set())
    flag_name = CLI_OVERRIDE_FLAGS[key]
    if flag_name in provided_flags:
        return getattr(args, key)
    return video_entry.get(key, getattr(args, key))


def recommend_route(video_summary: dict[str, Any]) -> dict[str, str]:
    unified = video_summary["unified"]
    hybrid = video_summary["hybrid"]
    comparison = video_summary["comparison"]

    if (
        hybrid["mean_count_per_frame"] >= unified["mean_count_per_frame"] * 1.25
        and comparison["hybrid_more_detection_frames"]
        > comparison["unified_more_detection_frames"]
    ):
        return {
            "route": "hybrid",
            "reason": (
                "Hybrid route keeps a noticeably higher mean detection count on sampled "
                "frames and more often out-detects the unified route."
            ),
        }

    if (
        unified["mean_count_per_frame"] >= hybrid["mean_count_per_frame"] * 0.90
        and unified["frames_with_detections_ratio"]
        >= hybrid["frames_with_detections_ratio"] * 0.90
    ):
        return {
            "route": "unified",
            "reason": (
                "Unified route stays close enough to the hybrid detector on sampled "
                "frames, so keeping one model is still defensible."
            ),
        }

    return {
        "route": "manual_review",
        "reason": (
            "Neither route dominates strongly enough from detection-count heuristics "
            "alone; review the saved preview frames before fixing the final demo path."
        ),
    }


def maybe_add_preview_candidate(
    candidates: list[dict[str, Any]],
    candidate: dict[str, Any],
    topk: int,
):
    if topk <= 0:
        return

    candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            item["score_gap"],
            item["score_det"],
            item["score_conf"],
            -item["frame_index"],
        ),
        reverse=True,
    )
    del candidates[topk:]


def draw_unified_panel(
    frame: np.ndarray,
    detections: torch.Tensor,
    fog_probs: np.ndarray,
    beta: float,
    conf_thres: float,
) -> np.ndarray:
    canvas = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det[:4].round().int().tolist()
        conf = float(det[4].item())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 230, 60), 2)
        cv2.putText(
            canvas,
            f"vehicle {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 230, 60),
            2,
        )

    fog_idx = int(np.argmax(fog_probs))
    fog_name = HighwayFogSystem.FOG_CLASS_NAMES[fog_idx]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 88), (30, 30, 30), -1)
    cv2.putText(
        canvas,
        "Unified Route",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"fog={fog_name} beta={beta:.4f} conf>={conf_thres:.2f}",
        (18, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        HighwayFogSystem.FOG_COLORS[fog_idx],
        1,
    )
    cv2.putText(
        canvas,
        (
            f"clear={fog_probs[0]:.2f} uniform={fog_probs[1]:.2f} "
            f"patchy={fog_probs[2]:.2f} det={len(detections)}"
        ),
        (18, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (225, 225, 225),
        1,
    )
    return canvas


def draw_hybrid_panel(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    fog_probs: np.ndarray,
    beta: float,
    conf_thres: float,
) -> np.ndarray:
    canvas = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in det["xyxy"]]
        conf = float(det["conf"])
        name = str(det["name"])
        color = (0, 255, 255) if name in {"bus", "truck"} else (80, 255, 80)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{name} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    fog_idx = int(np.argmax(fog_probs))
    fog_name = HighwayFogSystem.FOG_CLASS_NAMES[fog_idx]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 88), (30, 30, 30), -1)
    cv2.putText(
        canvas,
        "Hybrid Route",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"fog={fog_name} beta={beta:.4f} conf>={conf_thres:.2f}",
        (18, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        HighwayFogSystem.FOG_COLORS[fog_idx],
        1,
    )
    cv2.putText(
        canvas,
        (
            f"clear={fog_probs[0]:.2f} uniform={fog_probs[1]:.2f} "
            f"patchy={fog_probs[2]:.2f} det={len(detections)}"
        ),
        (18, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (225, 225, 225),
        1,
    )
    return canvas


def render_preview(
    candidate: dict[str, Any],
    output_path: Path,
    unified_conf: float,
    hybrid_conf: float,
):
    frame = candidate["frame"]
    fog_probs = candidate["fog_probs"]
    beta = float(candidate["beta"])

    left = draw_unified_panel(
        frame,
        candidate["unified_detections"],
        fog_probs,
        beta,
        unified_conf,
    )
    right = draw_hybrid_panel(
        frame,
        candidate["hybrid_detections"],
        fog_probs,
        beta,
        hybrid_conf,
    )
    comparison_bar = np.full((64, frame.shape[1] * 2, 3), 24, dtype=np.uint8)
    cv2.putText(
        comparison_bar,
        (
            f"frame={candidate['frame_index']} time={candidate['timestamp_sec']:.2f}s "
            f"unified={candidate['unified_count']} hybrid={candidate['hybrid_count']} "
            f"gap={candidate['count_gap']:+d}"
        ),
        (18, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        comparison_bar,
        candidate["gap_reason"],
        (18, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )
    combined = cv2.vconcat([comparison_bar, cv2.hconcat([left, right])])
    cv2.imwrite(str(output_path), combined)


def evaluate_video(
    video_entry: dict[str, Any],
    fog_system: HighwayFogSystem,
    yolo_model: YOLO,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    video_path = Path(video_entry["video_path"])
    sample_stride = max(1, int(resolve_runtime_setting(video_entry, args, "sample_stride")))
    max_frames = max(0, int(resolve_runtime_setting(video_entry, args, "max_frames")))
    unified_conf = float(resolve_runtime_setting(video_entry, args, "unified_conf"))
    hybrid_conf = float(resolve_runtime_setting(video_entry, args, "hybrid_conf"))
    imgsz = max(32, int(resolve_runtime_setting(video_entry, args, "imgsz")))
    topk_preview = max(0, int(resolve_runtime_setting(video_entry, args, "topk_preview")))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    safe_name = str(video_entry["output_name"])
    video_output_dir = ensure_dir(output_root / safe_name)
    frame_csv_path = video_output_dir / "frame_metrics.csv"
    unified_track_log_path = video_output_dir / "unified_track_log.jsonl"
    hybrid_track_log_path = video_output_dir / "hybrid_track_log.jsonl"
    unified_frame_filter_log_path = video_output_dir / "unified_filter_frames.jsonl"
    hybrid_frame_filter_log_path = video_output_dir / "hybrid_filter_frames.jsonl"
    fog_system.reset_temporal_state()
    hybrid_filter = TemporalVehicleFilter.from_config(
        fog_system.cfg,
        route_name="hybrid",
    )

    unified_acc = {
        "frames": 0,
        "frames_with_det": 0,
        "total_count": 0,
        "max_count": 0,
        "conf_sum": 0.0,
        "conf_count": 0,
        "max_conf": 0.0,
    }
    hybrid_acc = {
        "frames": 0,
        "frames_with_det": 0,
        "total_count": 0,
        "max_count": 0,
        "conf_sum": 0.0,
        "conf_count": 0,
        "max_conf": 0.0,
    }
    comparison_acc = {
        "hybrid_more_detection_frames": 0,
        "unified_more_detection_frames": 0,
        "same_detection_frames": 0,
        "hybrid_nonzero_unified_zero_frames": 0,
        "unified_nonzero_hybrid_zero_frames": 0,
        "total_count_gap": 0.0,
        "total_abs_count_gap": 0.0,
        "max_positive_gap": 0,
        "max_negative_gap": 0,
    }

    beta_values: list[float] = []
    fog_prob_values: list[np.ndarray] = []
    preview_candidates: list[dict[str, Any]] = []

    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "beta",
        "fog_clear",
        "fog_uniform",
        "fog_patchy",
        "dominant_fog",
        "dominant_fog_conf",
        "fog_margin",
        "dominant_fog_changed",
        "beta_delta_from_prev",
        "unified_count",
        "unified_raw_count_before_temporal",
        "unified_suppressed_count",
        "unified_persistent_static_candidate_track_count",
        "unified_mean_conf",
        "unified_max_conf",
        "hybrid_count",
        "hybrid_raw_count_before_temporal",
        "hybrid_suppressed_count",
        "hybrid_persistent_static_candidate_track_count",
        "hybrid_mean_conf",
        "hybrid_max_conf",
        "count_gap_hybrid_minus_unified",
    ]

    processed = 0
    frame_index = -1
    prev_beta: float | None = None
    prev_fog_idx: int | None = None
    with frame_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if frame_index % sample_stride != 0:
                continue

            if max_frames > 0 and processed >= max_frames:
                break
            processed += 1

            timestamp_sec = (
                float(frame_index / fps) if fps and fps > 0 else float(processed - 1)
            )

            fog_probs, beta, unified_detections = fog_system.predict(
                frame,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )
            fog_probs = np.asarray(fog_probs, dtype=np.float32)
            unified_temporal_report = fog_system.get_last_temporal_report()
            unified_filtered = filter_unified_detections(
                unified_detections, unified_conf
            )

            hybrid_result = yolo_model.predict(
                source=frame,
                verbose=False,
                conf=hybrid_conf,
                iou=0.45,
                imgsz=imgsz,
                classes=VEHICLE_CLASS_IDS,
                device=fog_system.cfg.DEVICE,
            )[0]
            hybrid_detections = extract_hybrid_detections(
                hybrid_result, hybrid_conf
            )
            hybrid_detections, hybrid_temporal_report = hybrid_filter.filter_detection_dicts(
                frame,
                hybrid_detections,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )

            unified_confs = (
                unified_filtered[:, 4].cpu().tolist()
                if unified_filtered.numel() > 0
                else []
            )
            hybrid_confs = [float(det["conf"]) for det in hybrid_detections]
            unified_count = int(len(unified_confs))
            hybrid_count = int(len(hybrid_confs))
            count_gap = int(hybrid_count - unified_count)
            fog_idx = int(np.argmax(fog_probs))
            dominant_fog = HighwayFogSystem.FOG_CLASS_NAMES[fog_idx]
            dominant_fog_conf = float(fog_probs[fog_idx])
            if fog_probs.shape[0] > 1:
                fog_margin = float(dominant_fog_conf - np.partition(fog_probs, -2)[-2])
            else:
                fog_margin = dominant_fog_conf
            dominant_fog_changed = int(
                prev_fog_idx is not None and fog_idx != prev_fog_idx
            )
            beta_delta_from_prev = (
                float(beta - prev_beta) if prev_beta is not None else 0.0
            )
            prev_fog_idx = fog_idx
            prev_beta = float(beta)

            update_route_accumulator(unified_acc, unified_count, unified_confs)
            update_route_accumulator(hybrid_acc, hybrid_count, hybrid_confs)
            comparison_acc["total_count_gap"] += float(count_gap)
            comparison_acc["total_abs_count_gap"] += float(abs(count_gap))
            comparison_acc["max_positive_gap"] = max(
                comparison_acc["max_positive_gap"], int(count_gap)
            )
            comparison_acc["max_negative_gap"] = min(
                comparison_acc["max_negative_gap"], int(count_gap)
            )
            if count_gap > 0:
                comparison_acc["hybrid_more_detection_frames"] += 1
            elif count_gap < 0:
                comparison_acc["unified_more_detection_frames"] += 1
            else:
                comparison_acc["same_detection_frames"] += 1

            if hybrid_count > 0 and unified_count == 0:
                comparison_acc["hybrid_nonzero_unified_zero_frames"] += 1
            if unified_count > 0 and hybrid_count == 0:
                comparison_acc["unified_nonzero_hybrid_zero_frames"] += 1

            beta_values.append(float(beta))
            fog_prob_values.append(fog_probs.copy())

            unified_mean_conf = (
                float(np.mean(unified_confs)) if unified_confs else 0.0
            )
            hybrid_mean_conf = (
                float(np.mean(hybrid_confs)) if hybrid_confs else 0.0
            )
            unified_max_conf = max(unified_confs) if unified_confs else 0.0
            hybrid_max_conf = max(hybrid_confs) if hybrid_confs else 0.0

            writer.writerow(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 4),
                    "beta": round(float(beta), 6),
                    "fog_clear": round(float(fog_probs[0]), 6),
                    "fog_uniform": round(float(fog_probs[1]), 6),
                    "fog_patchy": round(float(fog_probs[2]), 6),
                    "dominant_fog": dominant_fog,
                    "dominant_fog_conf": round(dominant_fog_conf, 6),
                    "fog_margin": round(fog_margin, 6),
                    "dominant_fog_changed": dominant_fog_changed,
                    "beta_delta_from_prev": round(beta_delta_from_prev, 6),
                    "unified_count": unified_count,
                    "unified_raw_count_before_temporal": int(
                        unified_temporal_report.get("input_count", unified_count)
                    ),
                    "unified_suppressed_count": int(
                        unified_temporal_report.get("suppressed_count", 0)
                    ),
                    "unified_persistent_static_candidate_track_count": int(
                        unified_temporal_report.get(
                            "persistent_static_candidate_track_count",
                            0,
                        )
                    ),
                    "unified_mean_conf": round(unified_mean_conf, 6),
                    "unified_max_conf": round(float(unified_max_conf), 6),
                    "hybrid_count": hybrid_count,
                    "hybrid_raw_count_before_temporal": int(
                        hybrid_temporal_report.get("input_count", hybrid_count)
                    ),
                    "hybrid_suppressed_count": int(
                        hybrid_temporal_report.get("suppressed_count", 0)
                    ),
                    "hybrid_persistent_static_candidate_track_count": int(
                        hybrid_temporal_report.get(
                            "persistent_static_candidate_track_count",
                            0,
                        )
                    ),
                    "hybrid_mean_conf": round(hybrid_mean_conf, 6),
                    "hybrid_max_conf": round(float(hybrid_max_conf), 6),
                    "count_gap_hybrid_minus_unified": count_gap,
                }
            )

            if count_gap > 0:
                gap_reason = (
                    "Hybrid route detects more vehicles on this sampled frame."
                )
            elif count_gap < 0:
                gap_reason = (
                    "Unified route detects more vehicles on this sampled frame."
                )
            else:
                gap_reason = (
                    "Detection counts are tied; this frame is kept for qualitative review."
                )

            maybe_add_preview_candidate(
                preview_candidates,
                {
                    "frame_index": frame_index,
                    "timestamp_sec": float(timestamp_sec),
                    "frame": frame.copy(),
                    "fog_probs": fog_probs.copy(),
                    "beta": float(beta),
                    "unified_detections": unified_filtered.detach().cpu().clone(),
                    "hybrid_detections": list(hybrid_detections),
                    "unified_count": unified_count,
                    "hybrid_count": hybrid_count,
                    "count_gap": count_gap,
                    "gap_reason": gap_reason,
                    "score_gap": abs(count_gap),
                    "score_det": max(unified_count, hybrid_count),
                    "score_conf": max(
                        float(unified_max_conf),
                        float(hybrid_max_conf),
                    ),
                },
                topk_preview,
            )

            if processed % 25 == 0:
                print(
                    f"[{video_path.name}] processed sampled frames: {processed}"
                )

    cap.release()
    fog_system.flush_temporal_state()
    hybrid_filter.flush()

    write_jsonl(unified_track_log_path, fog_system.export_temporal_event_log())
    write_jsonl(hybrid_track_log_path, hybrid_filter.export_event_log())
    write_jsonl(unified_frame_filter_log_path, fog_system.export_temporal_frame_reports())
    write_jsonl(hybrid_frame_filter_log_path, hybrid_filter.export_frame_reports())

    sampled_frames = int(unified_acc["frames"])
    unified_summary = summarize_route(unified_acc)
    hybrid_summary = summarize_route(hybrid_acc)
    unified_summary["temporal_filter"] = fog_system.get_temporal_summary(fps=fps)
    hybrid_summary["temporal_filter"] = hybrid_filter.build_summary(fps=fps)
    comparison_summary = {
        "hybrid_more_detection_frames": int(
            comparison_acc["hybrid_more_detection_frames"]
        ),
        "unified_more_detection_frames": int(
            comparison_acc["unified_more_detection_frames"]
        ),
        "same_detection_frames": int(comparison_acc["same_detection_frames"]),
        "hybrid_nonzero_unified_zero_frames": int(
            comparison_acc["hybrid_nonzero_unified_zero_frames"]
        ),
        "unified_nonzero_hybrid_zero_frames": int(
            comparison_acc["unified_nonzero_hybrid_zero_frames"]
        ),
        "mean_count_gap_hybrid_minus_unified": (
            comparison_acc["total_count_gap"] / max(sampled_frames, 1)
        ),
        "mean_abs_count_gap": (
            comparison_acc["total_abs_count_gap"] / max(sampled_frames, 1)
        ),
        "max_positive_gap_hybrid_minus_unified": int(
            comparison_acc["max_positive_gap"]
        ),
        "max_negative_gap_hybrid_minus_unified": int(
            comparison_acc["max_negative_gap"]
        ),
    }

    preview_paths: list[str] = []
    for index, candidate in enumerate(preview_candidates, start=1):
        preview_path = video_output_dir / (
            f"preview_{index:02d}_frame_{candidate['frame_index']:06d}.jpg"
        )
        render_preview(
            candidate,
            preview_path,
            unified_conf,
            hybrid_conf,
        )
        preview_paths.append(str(preview_path.relative_to(output_root)))

    runtime_settings = {
        "sample_stride": sample_stride,
        "max_frames": max_frames,
        "unified_conf": unified_conf,
        "hybrid_conf": hybrid_conf,
        "imgsz": imgsz,
        "topk_preview": topk_preview,
    }
    video_summary = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "sampled_frames": sampled_frames,
        "sample_stride": sample_stride,
        "settings": runtime_settings,
        "fps": fps,
        "frame_size": [width, height],
        "total_frames": total_frames,
        "benchmark": {
            "entry_id": int(video_entry["entry_id"]),
            "label": str(video_entry["label"]),
            "tags": list(video_entry["tags"]),
            "notes": str(video_entry["notes"]),
            "raw_path": str(video_entry["raw_path"]),
        },
        "fog": {
            "beta": summarize_beta(beta_values),
            "probs": summarize_fog_probs(fog_prob_values),
            "stability": summarize_fog_stability(fog_prob_values, beta_values),
        },
        "unified": unified_summary,
        "hybrid": hybrid_summary,
        "comparison": comparison_summary,
        "heuristic_recommendation": {},
        "artifacts": {
            "frame_metrics_csv": str(frame_csv_path.relative_to(output_root)),
            "unified_track_log_jsonl": str(unified_track_log_path.relative_to(output_root)),
            "hybrid_track_log_jsonl": str(hybrid_track_log_path.relative_to(output_root)),
            "unified_filter_frames_jsonl": str(
                unified_frame_filter_log_path.relative_to(output_root)
            ),
            "hybrid_filter_frames_jsonl": str(
                hybrid_frame_filter_log_path.relative_to(output_root)
            ),
            "preview_frames": preview_paths,
        },
    }
    video_summary["heuristic_recommendation"] = recommend_route(video_summary)
    return video_summary


def build_aggregate_summary(video_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    recommendation_hist: dict[str, int] = {}
    dominant_fog_hist = {
        label: 0 for label in HighwayFogSystem.FOG_CLASS_NAMES
    }
    total_sampled_frames = sum(int(video["sampled_frames"]) for video in video_summaries)
    total_transitions = sum(
        max(int(video["sampled_frames"]) - 1, 0) for video in video_summaries
    )

    unified_total_detections = sum(
        int(video["unified"]["total_detections"]) for video in video_summaries
    )
    hybrid_total_detections = sum(
        int(video["hybrid"]["total_detections"]) for video in video_summaries
    )
    unified_frames_with_det = sum(
        int(video["unified"]["frames_with_detections"]) for video in video_summaries
    )
    hybrid_frames_with_det = sum(
        int(video["hybrid"]["frames_with_detections"]) for video in video_summaries
    )
    total_abs_gap = sum(
        float(video["comparison"]["mean_abs_count_gap"]) * int(video["sampled_frames"])
        for video in video_summaries
    )
    weighted_beta_mean = sum(
        float(video["fog"]["beta"]["mean"]) * int(video["sampled_frames"])
        for video in video_summaries
    )
    total_switches = sum(
        int(video["fog"]["stability"]["dominant_switch_count"])
        for video in video_summaries
    )
    weighted_beta_abs_delta = sum(
        float(video["fog"]["stability"]["beta_abs_delta_mean"])
        * max(int(video["sampled_frames"]) - 1, 0)
        for video in video_summaries
    )
    unified_static_fp_count = sum(
        int(video["unified"]["temporal_filter"]["heuristic_persistent_static_fp_count"])
        for video in video_summaries
    )
    hybrid_static_fp_count = sum(
        int(video["hybrid"]["temporal_filter"]["heuristic_persistent_static_fp_count"])
        for video in video_summaries
    )
    unified_suppressed_static_fp_count = sum(
        int(
            video["unified"]["temporal_filter"][
                "suppressed_heuristic_persistent_static_fp_count"
            ]
        )
        for video in video_summaries
    )
    hybrid_suppressed_static_fp_count = sum(
        int(
            video["hybrid"]["temporal_filter"][
                "suppressed_heuristic_persistent_static_fp_count"
            ]
        )
        for video in video_summaries
    )
    total_eval_minutes = sum(
        float(video["unified"]["temporal_filter"]["evaluated_minutes"])
        for video in video_summaries
    )
    weighted_unified_confirmation_latency_frames = sum(
        float(video["unified"]["temporal_filter"]["mean_confirmation_latency_frames"])
        * int(video["sampled_frames"])
        for video in video_summaries
    )
    weighted_hybrid_confirmation_latency_frames = sum(
        float(video["hybrid"]["temporal_filter"]["mean_confirmation_latency_frames"])
        * int(video["sampled_frames"])
        for video in video_summaries
    )

    for video in video_summaries:
        route = str(video["heuristic_recommendation"]["route"])
        recommendation_hist[route] = recommendation_hist.get(route, 0) + 1
        for label, count in video["fog"]["probs"]["dominant_hist"].items():
            dominant_fog_hist[label] = dominant_fog_hist.get(label, 0) + int(count)

    return {
        "video_count": len(video_summaries),
        "total_sampled_frames": total_sampled_frames,
        "recommendation_hist": recommendation_hist,
        "dominant_fog_hist": dominant_fog_hist,
        "weighted_unified_mean_count_per_frame": (
            unified_total_detections / total_sampled_frames
            if total_sampled_frames > 0
            else 0.0
        ),
        "weighted_hybrid_mean_count_per_frame": (
            hybrid_total_detections / total_sampled_frames
            if total_sampled_frames > 0
            else 0.0
        ),
        "weighted_unified_frames_with_detections_ratio": (
            unified_frames_with_det / total_sampled_frames
            if total_sampled_frames > 0
            else 0.0
        ),
        "weighted_hybrid_frames_with_detections_ratio": (
            hybrid_frames_with_det / total_sampled_frames
            if total_sampled_frames > 0
            else 0.0
        ),
        "weighted_mean_count_gap_hybrid_minus_unified": (
            (hybrid_total_detections - unified_total_detections) / total_sampled_frames
            if total_sampled_frames > 0
            else 0.0
        ),
        "weighted_mean_abs_count_gap": (
            total_abs_gap / total_sampled_frames if total_sampled_frames > 0 else 0.0
        ),
        "weighted_beta_mean": (
            weighted_beta_mean / total_sampled_frames if total_sampled_frames > 0 else 0.0
        ),
        "weighted_dominant_switch_rate": (
            total_switches / total_transitions if total_transitions > 0 else 0.0
        ),
        "weighted_beta_abs_delta_mean": (
            weighted_beta_abs_delta / total_transitions if total_transitions > 0 else 0.0
        ),
        "weighted_unified_heuristic_persistent_static_fp_per_min": (
            unified_static_fp_count / total_eval_minutes if total_eval_minutes > 0 else 0.0
        ),
        "weighted_hybrid_heuristic_persistent_static_fp_per_min": (
            hybrid_static_fp_count / total_eval_minutes if total_eval_minutes > 0 else 0.0
        ),
        "weighted_unified_suppressed_static_fp_per_min": (
            unified_suppressed_static_fp_count / total_eval_minutes
            if total_eval_minutes > 0
            else 0.0
        ),
        "weighted_hybrid_suppressed_static_fp_per_min": (
            hybrid_suppressed_static_fp_count / total_eval_minutes
            if total_eval_minutes > 0
            else 0.0
        ),
        "weighted_unified_confirmation_latency_frames": (
            weighted_unified_confirmation_latency_frames / total_sampled_frames
            if total_sampled_frames > 0
            else 0.0
        ),
        "weighted_hybrid_confirmation_latency_frames": (
            weighted_hybrid_confirmation_latency_frames / total_sampled_frames
            if total_sampled_frames > 0
            else 0.0
        ),
    }


def write_benchmark_overview_csv(
    video_summaries: list[dict[str, Any]],
    output_path: Path,
):
    fieldnames = [
        "entry_id",
        "label",
        "video_name",
        "tags",
        "sampled_frames",
        "recommendation",
        "unified_mean_count_per_frame",
        "unified_frames_with_detections_ratio",
        "hybrid_mean_count_per_frame",
        "hybrid_frames_with_detections_ratio",
        "mean_count_gap_hybrid_minus_unified",
        "mean_abs_count_gap",
        "beta_mean",
        "beta_std",
        "majority_fog_label",
        "dominant_switch_rate",
        "beta_abs_delta_mean",
        "unified_heuristic_static_fp_per_min",
        "hybrid_heuristic_static_fp_per_min",
        "unified_confirmation_latency_frames",
        "hybrid_confirmation_latency_frames",
        "preview_count",
        "frame_metrics_csv",
        "unified_track_log_jsonl",
        "hybrid_track_log_jsonl",
        "notes",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for video in video_summaries:
            writer.writerow(
                {
                    "entry_id": int(video["benchmark"]["entry_id"]),
                    "label": video["benchmark"]["label"],
                    "video_name": video["video_name"],
                    "tags": "|".join(video["benchmark"]["tags"]),
                    "sampled_frames": int(video["sampled_frames"]),
                    "recommendation": video["heuristic_recommendation"]["route"],
                    "unified_mean_count_per_frame": round(
                        float(video["unified"]["mean_count_per_frame"]), 6
                    ),
                    "unified_frames_with_detections_ratio": round(
                        float(video["unified"]["frames_with_detections_ratio"]), 6
                    ),
                    "hybrid_mean_count_per_frame": round(
                        float(video["hybrid"]["mean_count_per_frame"]), 6
                    ),
                    "hybrid_frames_with_detections_ratio": round(
                        float(video["hybrid"]["frames_with_detections_ratio"]), 6
                    ),
                    "mean_count_gap_hybrid_minus_unified": round(
                        float(
                            video["comparison"][
                                "mean_count_gap_hybrid_minus_unified"
                            ]
                        ),
                        6,
                    ),
                    "mean_abs_count_gap": round(
                        float(video["comparison"]["mean_abs_count_gap"]),
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
                    "unified_heuristic_static_fp_per_min": round(
                        float(
                            video["unified"]["temporal_filter"][
                                "heuristic_persistent_static_fp_per_min"
                            ]
                        ),
                        6,
                    ),
                    "hybrid_heuristic_static_fp_per_min": round(
                        float(
                            video["hybrid"]["temporal_filter"][
                                "heuristic_persistent_static_fp_per_min"
                            ]
                        ),
                        6,
                    ),
                    "unified_confirmation_latency_frames": round(
                        float(
                            video["unified"]["temporal_filter"][
                                "mean_confirmation_latency_frames"
                            ]
                        ),
                        6,
                    ),
                    "hybrid_confirmation_latency_frames": round(
                        float(
                            video["hybrid"]["temporal_filter"][
                                "mean_confirmation_latency_frames"
                            ]
                        ),
                        6,
                    ),
                    "preview_count": len(video["artifacts"]["preview_frames"]),
                    "frame_metrics_csv": video["artifacts"]["frame_metrics_csv"],
                    "unified_track_log_jsonl": video["artifacts"]["unified_track_log_jsonl"],
                    "hybrid_track_log_jsonl": video["artifacts"]["hybrid_track_log_jsonl"],
                    "notes": video["benchmark"]["notes"],
                }
            )


def write_markdown_report(summary: dict[str, Any], output_path: Path):
    lines = [
        "# Route Evaluation Report",
        "",
        "## Purpose",
        "",
        "This report compares the current unified inference route against the hybrid route on sampled video frames.",
        "",
        f"- Fog weights: `{summary['meta']['fog_weights']}`",
        f"- Hybrid detector weights: `{summary['meta']['yolo_weights']}`",
        f"- Sample stride: `{summary['meta']['sample_stride']}`",
        f"- Max frames per video: `{summary['meta']['max_frames']}`",
        f"- Unified confidence threshold: `{summary['meta']['unified_conf']}`",
        f"- Hybrid confidence threshold: `{summary['meta']['hybrid_conf']}`",
        f"- Video source mode: `{summary['meta']['video_source_mode']}`",
        "",
    ]

    if summary["meta"].get("benchmark_config"):
        lines.extend(
            [
                "## Benchmark Config",
                "",
                f"- Benchmark ID: `{summary['meta'].get('benchmark_id', '') or 'N/A'}`",
                f"- Config: `{summary['meta']['benchmark_config']}`",
                f"- Description: {summary['meta'].get('benchmark_description', '') or 'N/A'}",
                f"- Total entries: `{summary['meta'].get('benchmark_total_entries', 0)}`",
                f"- Active entries: `{summary['meta'].get('benchmark_active_entries', 0)}`",
                f"- Inactive/planned entries: `{summary['meta'].get('benchmark_inactive_entries', 0)}`",
                f"- Inactive/planned labels: `{summary['meta'].get('benchmark_inactive_labels', [])}`",
                "",
            ]
        )

    aggregate = summary["aggregate"]
    lines.extend(
        [
            "## Aggregate Summary",
            "",
            f"- Video count: `{aggregate['video_count']}`",
            f"- Total sampled frames: `{aggregate['total_sampled_frames']}`",
            f"- Weighted unified mean detections/frame: `{aggregate['weighted_unified_mean_count_per_frame']:.3f}`",
            f"- Weighted hybrid mean detections/frame: `{aggregate['weighted_hybrid_mean_count_per_frame']:.3f}`",
            f"- Weighted mean count gap (hybrid - unified): `{aggregate['weighted_mean_count_gap_hybrid_minus_unified']:.3f}`",
            f"- Weighted beta mean: `{aggregate['weighted_beta_mean']:.5f}`",
            f"- Weighted fog switch rate: `{aggregate['weighted_dominant_switch_rate']:.3f}`",
            f"- Weighted beta abs delta mean: `{aggregate['weighted_beta_abs_delta_mean']:.5f}`",
            f"- Weighted unified heuristic static-FP/min: `{aggregate['weighted_unified_heuristic_persistent_static_fp_per_min']:.3f}`",
            f"- Weighted hybrid heuristic static-FP/min: `{aggregate['weighted_hybrid_heuristic_persistent_static_fp_per_min']:.3f}`",
            f"- Weighted unified confirmation latency (frames): `{aggregate['weighted_unified_confirmation_latency_frames']:.3f}`",
            f"- Weighted hybrid confirmation latency (frames): `{aggregate['weighted_hybrid_confirmation_latency_frames']:.3f}`",
            f"- Recommendation histogram: `{aggregate['recommendation_hist']}`",
            f"- Dominant fog histogram: `{aggregate['dominant_fog_hist']}`",
            f"- Overview CSV: `{summary['artifacts']['benchmark_overview_csv']}`",
            "",
            "## Video Summary",
            "",
        ]
    )

    for video in summary["videos"]:
        recommendation = video["heuristic_recommendation"]
        lines.extend(
            [
                f"### {video['benchmark']['label']}",
                "",
                f"- Video file: `{video['video_name']}`",
                f"- Tags: `{video['benchmark']['tags']}`",
                f"- Notes: {video['benchmark']['notes'] or 'N/A'}",
                f"- Sampled frames: `{video['sampled_frames']}`",
                f"- Unified mean detections/frame: `{video['unified']['mean_count_per_frame']:.3f}`",
                f"- Hybrid mean detections/frame: `{video['hybrid']['mean_count_per_frame']:.3f}`",
                f"- Hybrid more-detection frames: `{video['comparison']['hybrid_more_detection_frames']}`",
                f"- Unified more-detection frames: `{video['comparison']['unified_more_detection_frames']}`",
                f"- Mean count gap (hybrid - unified): `{video['comparison']['mean_count_gap_hybrid_minus_unified']:.3f}`",
                f"- Beta mean/std: `{video['fog']['beta']['mean']:.5f} / {video['fog']['beta']['std']:.5f}`",
                f"- Majority fog label: `{video['fog']['stability']['majority_fog_label']}`",
                f"- Fog switch rate: `{video['fog']['stability']['dominant_switch_rate']:.3f}`",
                f"- Beta abs delta mean: `{video['fog']['stability']['beta_abs_delta_mean']:.5f}`",
                f"- Unified heuristic static-FP/min: `{video['unified']['temporal_filter']['heuristic_persistent_static_fp_per_min']:.3f}`",
                f"- Hybrid heuristic static-FP/min: `{video['hybrid']['temporal_filter']['heuristic_persistent_static_fp_per_min']:.3f}`",
                f"- Unified confirmation latency (frames): `{video['unified']['temporal_filter']['mean_confirmation_latency_frames']:.3f}`",
                f"- Hybrid confirmation latency (frames): `{video['hybrid']['temporal_filter']['mean_confirmation_latency_frames']:.3f}`",
                f"- Recommendation: `{recommendation['route']}`",
                f"- Recommendation reason: {recommendation['reason']}",
                f"- Frame metrics CSV: `{video['artifacts']['frame_metrics_csv']}`",
                f"- Unified track log: `{video['artifacts']['unified_track_log_jsonl']}`",
                f"- Hybrid track log: `{video['artifacts']['hybrid_track_log_jsonl']}`",
            ]
        )
        if video["artifacts"]["preview_frames"]:
            lines.append("- Preview frames:")
            for preview in video["artifacts"]["preview_frames"]:
                lines.append(f"  - `{preview}`")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args._provided_flags = collect_cli_override_flags(sys.argv[1:])
    cfg = Config(config_path=args.config)

    video_entries, source_info = collect_video_entries(args)
    output_root = ensure_dir(Path(args.output_dir).resolve())

    fog_weights = args.fog_weights or resolve_model_weights(
        cfg.OUTPUT_DIR,
        cfg.CHECKPOINT_DIR,
        preferred_files=["unified_model_best.pt", "unified_model.pt"],
    )
    if not fog_weights:
        raise RuntimeError("No usable fog-model weights were found.")

    yolo_weights = Path(args.yolo_weights).resolve()
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    print(f"Device: {cfg.DEVICE}")
    print(f"Fog weights: {fog_weights}")
    print(f"YOLO weights: {yolo_weights}")
    print(f"Videos to evaluate: {len(video_entries)}")
    if source_info.get("benchmark_config"):
        print(f"Benchmark config: {source_info['benchmark_config']}")

    fog_system = HighwayFogSystem(
        str(fog_weights),
        video_source=0,
        cfg=cfg,
    )
    yolo_model = YOLO(str(yolo_weights))

    video_summaries = []
    for video_entry in video_entries:
        print(
            "Evaluating video: "
            f"{video_entry['video_path']} "
            f"(label={video_entry['label']}, tags={video_entry['tags']})"
        )
        video_summary = evaluate_video(
            video_entry,
            fog_system,
            yolo_model,
            args,
            output_root,
        )
        recommendation = video_summary["heuristic_recommendation"]
        print(
            f"Finished {video_summary['video_name']}: recommendation={recommendation['route']}, "
            f"mean_gap={video_summary['comparison']['mean_count_gap_hybrid_minus_unified']:.3f}"
        )
        video_summaries.append(video_summary)

    summary = {
        "meta": {
            "device": cfg.DEVICE,
            "config_file": cfg.CONFIG_FILE or "",
            "fog_weights": str(fog_weights),
            "yolo_weights": str(yolo_weights),
            "sample_stride": max(1, args.sample_stride),
            "max_frames": max(0, args.max_frames),
            "unified_conf": float(args.unified_conf),
            "hybrid_conf": float(args.hybrid_conf),
            "imgsz": int(args.imgsz),
            "output_dir": str(output_root),
            "temporal_filter_enabled": bool(cfg.TEMPORAL_FILTER_ENABLED),
            "temporal_min_hits": int(cfg.TEMPORAL_MIN_HITS),
            "temporal_max_missing": int(cfg.TEMPORAL_MAX_MISSING),
            "temporal_iou_match_thres": float(cfg.TEMPORAL_IOU_MATCH_THRES),
            "temporal_static_center_shift_thres": float(
                cfg.TEMPORAL_STATIC_CENTER_SHIFT_THRES
            ),
            "temporal_static_area_change_thres": float(
                cfg.TEMPORAL_STATIC_AREA_CHANGE_THRES
            ),
            "temporal_static_motion_thres": float(cfg.TEMPORAL_STATIC_MOTION_THRES),
            "temporal_static_frame_limit": int(cfg.TEMPORAL_STATIC_FRAME_LIMIT),
            "temporal_low_conf_static_suppress": float(
                cfg.TEMPORAL_LOW_CONF_STATIC_SUPPRESS
            ),
            "temporal_enable_second_stage_classifier": bool(
                cfg.TEMPORAL_ENABLE_SECOND_STAGE_CLASSIFIER
            ),
            **source_info,
        },
        "aggregate": build_aggregate_summary(video_summaries),
        "artifacts": {},
        "videos": video_summaries,
    }

    summary_json = output_root / "route_eval_summary.json"
    summary_md = output_root / "route_eval_summary.md"
    overview_csv = output_root / "benchmark_overview.csv"
    write_benchmark_overview_csv(video_summaries, overview_csv)
    summary["artifacts"] = {
        "route_eval_summary_json": str(summary_json),
        "route_eval_summary_md": str(summary_md),
        "benchmark_overview_csv": str(overview_csv.relative_to(output_root)),
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(summary, summary_md)

    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")
    print(f"Benchmark overview CSV: {overview_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
