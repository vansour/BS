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
import os
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
from src.utils import resolve_model_weights


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline comparison between unified inference and hybrid inference."
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


def collect_videos(args: argparse.Namespace) -> list[Path]:
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
        return videos

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return [video_path]


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
    video_path: Path,
    fog_system: HighwayFogSystem,
    yolo_model: YOLO,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    safe_name = video_path.stem
    video_output_dir = ensure_dir(output_root / safe_name)
    frame_csv_path = video_output_dir / "frame_metrics.csv"

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
        "unified_count",
        "unified_mean_conf",
        "unified_max_conf",
        "hybrid_count",
        "hybrid_mean_conf",
        "hybrid_max_conf",
        "count_gap_hybrid_minus_unified",
    ]

    processed = 0
    frame_index = -1
    with frame_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if frame_index % max(1, args.sample_stride) != 0:
                continue

            if args.max_frames > 0 and processed >= args.max_frames:
                break
            processed += 1

            fog_probs, beta, unified_detections = fog_system.predict(frame)
            fog_probs = np.asarray(fog_probs, dtype=np.float32)
            unified_filtered = filter_unified_detections(
                unified_detections, args.unified_conf
            )

            hybrid_result = yolo_model.predict(
                source=frame,
                verbose=False,
                conf=args.hybrid_conf,
                iou=0.45,
                imgsz=args.imgsz,
                classes=VEHICLE_CLASS_IDS,
                device=fog_system.cfg.DEVICE,
            )[0]
            hybrid_detections = extract_hybrid_detections(
                hybrid_result, args.hybrid_conf
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
            timestamp_sec = (
                float(frame_index / fps) if fps and fps > 0 else float(processed - 1)
            )

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
                    "unified_count": unified_count,
                    "unified_mean_conf": round(unified_mean_conf, 6),
                    "unified_max_conf": round(float(unified_max_conf), 6),
                    "hybrid_count": hybrid_count,
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
                args.topk_preview,
            )

            if processed % 25 == 0:
                print(
                    f"[{video_path.name}] processed sampled frames: {processed}"
                )

    cap.release()

    sampled_frames = int(unified_acc["frames"])
    unified_summary = summarize_route(unified_acc)
    hybrid_summary = summarize_route(hybrid_acc)
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
            args.unified_conf,
            args.hybrid_conf,
        )
        preview_paths.append(str(preview_path.relative_to(output_root)))

    video_summary = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "sampled_frames": sampled_frames,
        "sample_stride": max(1, args.sample_stride),
        "fps": fps,
        "frame_size": [width, height],
        "total_frames": total_frames,
        "fog": {
            "beta": summarize_beta(beta_values),
            "probs": summarize_fog_probs(fog_prob_values),
        },
        "unified": unified_summary,
        "hybrid": hybrid_summary,
        "comparison": comparison_summary,
        "heuristic_recommendation": {},
        "artifacts": {
            "frame_metrics_csv": str(frame_csv_path.relative_to(output_root)),
            "preview_frames": preview_paths,
        },
    }
    video_summary["heuristic_recommendation"] = recommend_route(video_summary)
    return video_summary


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
        "",
        "## Video Summary",
        "",
    ]

    for video in summary["videos"]:
        recommendation = video["heuristic_recommendation"]
        lines.extend(
            [
                f"### {video['video_name']}",
                "",
                f"- Sampled frames: `{video['sampled_frames']}`",
                f"- Unified mean detections/frame: `{video['unified']['mean_count_per_frame']:.3f}`",
                f"- Hybrid mean detections/frame: `{video['hybrid']['mean_count_per_frame']:.3f}`",
                f"- Hybrid more-detection frames: `{video['comparison']['hybrid_more_detection_frames']}`",
                f"- Unified more-detection frames: `{video['comparison']['unified_more_detection_frames']}`",
                f"- Mean count gap (hybrid - unified): `{video['comparison']['mean_count_gap_hybrid_minus_unified']:.3f}`",
                f"- Beta mean/std: `{video['fog']['beta']['mean']:.5f} / {video['fog']['beta']['std']:.5f}`",
                f"- Recommendation: `{recommendation['route']}`",
                f"- Recommendation reason: {recommendation['reason']}",
                f"- Frame metrics CSV: `{video['artifacts']['frame_metrics_csv']}`",
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
    cfg = Config()

    videos = collect_videos(args)
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
    print(f"Videos to evaluate: {len(videos)}")

    fog_system = HighwayFogSystem(
        str(fog_weights),
        video_source=0,
        cfg=cfg,
    )
    yolo_model = YOLO(str(yolo_weights))

    video_summaries = []
    for video_path in videos:
        print(f"Evaluating video: {video_path}")
        video_summary = evaluate_video(
            video_path,
            fog_system,
            yolo_model,
            args,
            output_root,
        )
        recommendation = video_summary["heuristic_recommendation"]
        print(
            f"Finished {video_path.name}: recommendation={recommendation['route']}, "
            f"mean_gap={video_summary['comparison']['mean_count_gap_hybrid_minus_unified']:.3f}"
        )
        video_summaries.append(video_summary)

    summary = {
        "meta": {
            "device": cfg.DEVICE,
            "fog_weights": str(fog_weights),
            "yolo_weights": str(yolo_weights),
            "sample_stride": max(1, args.sample_stride),
            "max_frames": max(0, args.max_frames),
            "unified_conf": float(args.unified_conf),
            "hybrid_conf": float(args.hybrid_conf),
            "imgsz": int(args.imgsz),
            "output_dir": str(output_root),
        },
        "videos": video_summaries,
    }

    summary_json = output_root / "route_eval_summary.json"
    summary_md = output_root / "route_eval_summary.md"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(summary, summary_md)

    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
