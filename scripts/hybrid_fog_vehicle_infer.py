#!/usr/bin/env python3
"""
Hybrid video inference for vehicle detection + fog estimation.

This script combines:
1. `yolo11n.pt` for robust vehicle detection on real videos.
2. The project's `UnifiedMultiTaskModel` for fog class and beta estimation.

The intent is pragmatic: when the current unified model does not detect vehicles
reliably enough yet, we can still demo the full pipeline by decoupling the two
tasks at inference time.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.inference import HighwayFogSystem
from src.utils import resolve_model_weights


VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO vehicle detection and fog beta estimation together."
    )
    parser.add_argument(
        "--video",
        default=str(PROJECT_ROOT / "gettyimages-1353950094-640_adpp.mp4"),
        help="Input video path.",
    )
    parser.add_argument(
        "--yolo-weights",
        default=str(PROJECT_ROOT / "yolo11n.pt"),
        help="Vehicle detector weights.",
    )
    parser.add_argument(
        "--fog-weights",
        default=None,
        help="Optional unified fog model weights. Defaults to best available project weights.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path. Defaults to outputs/hybrid_infer/<video_stem>_hybrid.mp4",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="YOLO confidence threshold for vehicle detection.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=None,
        help="Fog EMA alpha. Smaller means stronger smoothing. Defaults to Config.EMA_ALPHA.",
    )
    return parser.parse_args()


def build_output_path(video_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    output_dir = PROJECT_ROOT / "outputs" / "hybrid_infer"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{video_path.stem}_hybrid.mp4"


def make_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer MP4, but fall back to AVI if the local OpenCV build lacks mp4 support.
    mp4_writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if mp4_writer.isOpened():
        return mp4_writer, output_path
    mp4_writer.release()

    fallback_path = output_path.with_suffix(".avi")
    avi_writer = cv2.VideoWriter(
        str(fallback_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        frame_size,
    )
    if not avi_writer.isOpened():
        raise RuntimeError("Failed to open video writer for both MP4 and AVI outputs.")
    return avi_writer, fallback_path


def draw_vehicle_detections(frame, result, conf_thres: float) -> tuple[object, int]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return frame, 0

    count = 0
    xyxy_list = boxes.xyxy.int().cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()
    cls_list = boxes.cls.int().cpu().tolist()

    for xyxy, score, cls_id in zip(xyxy_list, conf_list, cls_list):
        if float(score) < conf_thres or int(cls_id) not in VEHICLE_CLASS_IDS:
            continue

        count += 1
        x1, y1, x2, y2 = xyxy
        class_name = VEHICLE_CLASS_NAMES.get(int(cls_id), str(cls_id))
        color = (0, 255, 255) if class_name in {"bus", "truck"} else (80, 255, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {float(score):.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return frame, count


def compute_spatial_fog_map(
    frame: np.ndarray,
    beta: float,
    cfg: Config,
    map_size: tuple[int, int] = (168, 96),
) -> tuple[np.ndarray, float]:
    """
    Estimate a relative fog-density map from the current frame.

    The current fog model only predicts one global beta per frame, not a dense
    per-pixel map. This helper builds a pragmatic spatial approximation from
    image cues that usually correlate with haze:
    - brighter dark-channel response
    - lower saturation
    - lower local contrast
    Then it scales the map by the frame-level beta prediction.
    """
    frame_float = frame.astype(np.float32) / 255.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0

    dark_channel = np.min(frame_float, axis=2)
    local_mean = cv2.GaussianBlur(gray, (0, 0), 7)
    local_sq_mean = cv2.GaussianBlur(gray * gray, (0, 0), 7)
    local_std = np.sqrt(np.clip(local_sq_mean - local_mean * local_mean, 0.0, None))

    dark_cue = np.clip((dark_channel - 0.20) / 0.60, 0.0, 1.0)
    whiteness_cue = 1.0 - saturation
    low_contrast_cue = 1.0 - np.clip(local_std / 0.12, 0.0, 1.0)

    haze_score = (
        0.45 * dark_cue
        + 0.30 * whiteness_cue
        + 0.25 * low_contrast_cue
    )

    beta_strength = np.clip(beta / max(cfg.BETA_MAX, 1e-6), 0.0, 1.0)
    haze_score = np.clip(haze_score * (0.55 + 0.45 * beta_strength), 0.0, 1.0)

    small_map = cv2.resize(haze_score, map_size, interpolation=cv2.INTER_AREA)
    small_map_u8 = np.uint8(np.clip(small_map * 255.0, 0, 255))
    color_map = cv2.applyColorMap(small_map_u8, cv2.COLORMAP_TURBO)
    mean_density = float(np.mean(small_map))
    return color_map, mean_density


def density_level_label(mean_density: float) -> tuple[str, tuple[int, int, int]]:
    if mean_density < 0.33:
        return "LIGHT FOG", (80, 255, 80)
    if mean_density < 0.66:
        return "MEDIUM FOG", (0, 215, 255)
    return "DENSE FOG", (0, 80, 255)


def make_colorbar(width: int, height: int) -> np.ndarray:
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.tile(gradient, (height, 1))
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    return colorbar


def draw_fog_overlay(
    frame,
    fog_probs,
    beta: float,
    vehicle_count: int,
    frame_index: int,
    fog_map_panel: np.ndarray,
    fog_map_mean: float,
) -> object:
    fog_idx = int(fog_probs.argmax())
    fog_name = HighwayFogSystem.FOG_CLASS_NAMES[fog_idx]
    fog_color = HighwayFogSystem.FOG_COLORS[fog_idx]
    density_label, density_color = density_level_label(fog_map_mean)

    frame_h, frame_w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (frame_w, 68), (36, 36, 36), -1)
    cv2.putText(
        frame,
        f"FOG: {fog_name}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        fog_color,
        2,
    )
    cv2.putText(
        frame,
        (
            f"vehicles={vehicle_count} | frame={frame_index} | "
            f"clear={float(fog_probs[0]):.2f} uniform={float(fog_probs[1]):.2f} patchy={float(fog_probs[2]):.2f}"
        ),
        (20, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
    )

    # Put a spatial fog map in the bottom-right corner.
    map_h, map_w = fog_map_panel.shape[:2]
    colorbar = make_colorbar(map_w, 12)
    colorbar_h = colorbar.shape[0]
    panel_width = map_w + 20
    panel_height = map_h + colorbar_h + 88
    x2 = frame_w - 16
    y2 = frame_h - 16
    x1 = max(0, x2 - panel_width)
    y1 = max(0, y2 - panel_height)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (28, 28, 28), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), fog_color, 2)
    cv2.putText(
        frame,
        "FOG DENSITY MAP",
        (x1 + 10, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        frame,
        f"LEVEL: {density_label}",
        (x1 + 10, y1 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        density_color,
        1,
    )

    map_x1 = x1 + 10
    map_y1 = y1 + 50
    frame[map_y1:map_y1 + map_h, map_x1:map_x1 + map_w] = fog_map_panel
    cv2.rectangle(
        frame,
        (map_x1, map_y1),
        (map_x1 + map_w, map_y1 + map_h),
        (220, 220, 220),
        1,
    )

    colorbar_y1 = map_y1 + map_h + 8
    frame[colorbar_y1:colorbar_y1 + colorbar_h, map_x1:map_x1 + map_w] = colorbar
    cv2.rectangle(
        frame,
        (map_x1, colorbar_y1),
        (map_x1 + map_w, colorbar_y1 + colorbar_h),
        (220, 220, 220),
        1,
    )
    cv2.putText(
        frame,
        "LOW",
        (map_x1, colorbar_y1 + colorbar_h + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        frame,
        "MID",
        (map_x1 + map_w // 2 - 10, colorbar_y1 + colorbar_h + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        frame,
        "HIGH",
        (map_x1 + map_w - 24, colorbar_y1 + colorbar_h + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        frame,
        f"beta={beta:.4f}",
        (x1 + 10, y2 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        fog_color,
        1,
    )
    cv2.putText(
        frame,
        f"mean={fog_map_mean:.2f}",
        (x1 + 98, y2 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
    )
    return frame


def main() -> int:
    args = parse_args()
    cfg = Config()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video was not found: {video_path}")

    yolo_weights = Path(args.yolo_weights).resolve()
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights were not found: {yolo_weights}")

    fog_weights = args.fog_weights
    if not fog_weights:
        fog_weights = resolve_model_weights(
            cfg.OUTPUT_DIR,
            cfg.CHECKPOINT_DIR,
            preferred_files=["unified_model_best.pt", "unified_model.pt"],
        )

    output_path = build_output_path(video_path, args.output)

    print(f"Input video: {video_path}")
    print(f"YOLO weights: {yolo_weights}")
    print(f"Fog weights: {fog_weights}")
    print(f"Device: {cfg.DEVICE}")
    ema_alpha = float(args.ema_alpha if args.ema_alpha is not None else cfg.EMA_ALPHA)
    print(f"Fog EMA alpha: {ema_alpha:.3f}")

    fog_system = HighwayFogSystem(
        fog_weights,
        video_source=str(video_path),
        cfg=cfg,
    )
    yolo_model = YOLO(str(yolo_weights))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer, final_output_path = make_video_writer(output_path, fps, (width, height))
    print(
        f"Writing output: {final_output_path} "
        f"({width}x{height}, {fps:.2f} FPS, frames={total_frames})"
    )

    frame_index = 0
    ema_beta: float | None = None
    ema_probs: np.ndarray | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            fog_probs, beta, _ = fog_system.predict(frame)
            fog_probs = np.asarray(fog_probs, dtype=np.float32)
            if ema_beta is None:
                ema_beta = float(beta)
            else:
                ema_beta = ema_alpha * float(beta) + (1.0 - ema_alpha) * float(ema_beta)

            if ema_probs is None:
                ema_probs = fog_probs.copy()
            else:
                ema_probs = ema_alpha * fog_probs + (1.0 - ema_alpha) * ema_probs
                probs_sum = float(np.sum(ema_probs))
                if probs_sum > 0:
                    ema_probs = ema_probs / probs_sum

            yolo_result = yolo_model.predict(
                source=frame,
                verbose=False,
                conf=args.conf,
                iou=0.45,
                imgsz=args.imgsz,
                classes=VEHICLE_CLASS_IDS,
            )[0]

            draw_frame = frame.copy()
            draw_frame, vehicle_count = draw_vehicle_detections(
                draw_frame,
                yolo_result,
                args.conf,
            )
            fog_map_panel, fog_map_mean = compute_spatial_fog_map(
                frame,
                float(ema_beta),
                cfg,
            )
            draw_frame = draw_fog_overlay(
                draw_frame,
                ema_probs if ema_probs is not None else fog_probs,
                float(ema_beta),
                vehicle_count,
                frame_index,
                fog_map_panel,
                fog_map_mean,
            )
            writer.write(draw_frame)

            if frame_index % 50 == 0 or frame_index == total_frames:
                print(f"Processed {frame_index}/{total_frames} frames")
    finally:
        cap.release()
        writer.release()

    print(f"Saved hybrid inference video: {final_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
