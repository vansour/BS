#!/usr/bin/env python3
"""
Target-video adaptation for fog heads.

This script adapts the fog classifier/regressor to a specific video by:
1. Loading a source model as a frozen teacher.
2. Running teacher inference on sampled video frames.
3. Smoothing teacher outputs over time.
4. Fine-tuning only the fog heads of a student model to match the smoothed targets.

The goal is not generic retraining. It is to reduce time jitter on a known
deployment video without collapsing predictions to `clear`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.model import UnifiedMultiTaskModel
from src.utils import letterbox_tensor, load_model_weights, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt fog heads to a specific target video."
    )
    parser.add_argument(
        "--video",
        default=str(PROJECT_ROOT / "gettyimages-1353950094-640_adpp.mp4"),
        help="Target video path.",
    )
    parser.add_argument(
        "--source-weights",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "Fog_Detection_Project_fogbalance"
            / "unified_model_best.pt"
        ),
        help="Source fog model weights used as both teacher and student init.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "Fog_Detection_Project_videoadapt"),
        help="Directory to save adapted weights and summary.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=2,
        help="Use every N-th frame from the target video for adaptation.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum sampled frames to use. 0 means all sampled frames.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Adaptation batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of adaptation epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Adaptation learning rate.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Odd window size used to smooth teacher targets over time.",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.20,
        help="Weight for probability temporal-difference consistency.",
    )
    parser.add_argument(
        "--beta-consistency-weight",
        type=float,
        default=0.50,
        help="Weight for beta temporal-difference consistency.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def preprocess_frame(frame: np.ndarray, cfg: Config) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    img_tensor, _ = letterbox_tensor(img_tensor, cfg.IMG_SIZE)
    return img_tensor


def load_sampled_video_tensors(
    video_path: Path,
    cfg: Config,
    *,
    sample_stride: int,
    max_frames: int,
) -> tuple[torch.Tensor, list[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    tensors: list[torch.Tensor] = []
    frame_indices: list[int] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_stride == 0:
            tensors.append(preprocess_frame(frame, cfg).to(dtype=torch.float16))
            frame_indices.append(frame_idx)
            if max_frames > 0 and len(tensors) >= max_frames:
                break
        frame_idx += 1
    cap.release()

    if not tensors:
        raise RuntimeError("No sampled frames were collected for adaptation.")

    return torch.stack(tensors, dim=0), frame_indices


@torch.no_grad()
def infer_teacher_targets(
    model: UnifiedMultiTaskModel,
    frames_cpu: torch.Tensor,
    cfg: Config,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs_all: list[torch.Tensor] = []
    beta_all: list[torch.Tensor] = []
    for start in range(0, frames_cpu.shape[0], batch_size):
        batch = frames_cpu[start:start + batch_size].to(device=device, dtype=torch.float32)
        _, logits, beta = model(batch)
        probs_all.append(torch.softmax(logits, dim=1).cpu())
        beta_all.append((beta * cfg.BETA_MAX).cpu())
    return torch.cat(probs_all, dim=0), torch.cat(beta_all, dim=0)


def smooth_sequence(sequence: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1 or sequence.shape[0] <= 1:
        return sequence
    if window % 2 == 0:
        window += 1
    pad = window // 2
    if sequence.ndim == 1:
        work = sequence.view(1, 1, -1)
    else:
        work = sequence.transpose(0, 1).unsqueeze(0)
    padded = F.pad(work, (pad, pad), mode="replicate")
    kernel = torch.ones(1, 1, window, dtype=sequence.dtype) / float(window)
    smoothed = F.conv1d(
        padded.reshape(-1, 1, padded.shape[-1]),
        kernel,
        groups=1,
    )
    smoothed = smoothed.reshape(work.shape[1], -1).transpose(0, 1)
    if sequence.ndim == 1:
        return smoothed.squeeze(1)
    return smoothed


def batch_iter(
    frames_cpu: torch.Tensor,
    target_probs: torch.Tensor,
    target_beta: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for start in range(0, frames_cpu.shape[0], batch_size):
        batch_idx = slice(start, start + batch_size)
        yield (
            frames_cpu[batch_idx],
            target_probs[batch_idx],
            target_beta[batch_idx],
        )


def save_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = Config()
    set_seed(args.seed)

    video_path = Path(args.video).resolve()
    source_weights = Path(args.source_weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not source_weights.exists():
        raise FileNotFoundError(f"Source weights not found: {source_weights}")

    print(f"Video: {video_path}")
    print(f"Source weights: {source_weights}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {cfg.DEVICE}")

    frames_cpu, frame_indices = load_sampled_video_tensors(
        video_path,
        cfg,
        sample_stride=max(1, args.sample_stride),
        max_frames=max(0, args.max_frames),
    )
    print(
        f"Sampled frames: {frames_cpu.shape[0]} "
        f"(stride={max(1, args.sample_stride)})"
    )

    teacher = UnifiedMultiTaskModel(
        cfg.YOLO_BASE_MODEL,
        cfg.NUM_FOG_CLASSES,
        num_det_classes=cfg.NUM_DET_CLASSES,
    )
    load_model_weights(teacher, str(source_weights), map_location=cfg.DEVICE)
    teacher.to(cfg.DEVICE).eval()

    student = UnifiedMultiTaskModel(
        cfg.YOLO_BASE_MODEL,
        cfg.NUM_FOG_CLASSES,
        num_det_classes=cfg.NUM_DET_CLASSES,
    )
    load_model_weights(student, str(source_weights), map_location=cfg.DEVICE)
    student.to(cfg.DEVICE)
    for parameter in student.yolo.parameters():
        parameter.requires_grad = False
    student.yolo.eval()

    with torch.no_grad():
        teacher_probs, teacher_beta = infer_teacher_targets(
            teacher,
            frames_cpu,
            cfg,
            cfg.DEVICE,
            args.batch_size,
        )

    smooth_probs = smooth_sequence(teacher_probs, args.smooth_window)
    smooth_probs = smooth_probs / smooth_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    smooth_beta = smooth_sequence(teacher_beta, args.smooth_window)

    trainable_params = list(student.fog_classifier.parameters()) + list(
        student.fog_regressor.parameters()
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        student.train()
        student.yolo.eval()
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_reg = 0.0
        prob_cons_acc = 0.0
        beta_cons_acc = 0.0
        batches = 0

        for batch_frames, batch_probs, batch_beta in batch_iter(
            frames_cpu,
            smooth_probs,
            smooth_beta,
            args.batch_size,
        ):
            imgs = batch_frames.to(cfg.DEVICE, dtype=torch.float32)
            target_probs = batch_probs.to(cfg.DEVICE, dtype=torch.float32)
            target_beta = batch_beta.to(cfg.DEVICE, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            _, logits, pred_beta = student(imgs)
            pred_probs = torch.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            loss_cls = F.kl_div(log_probs, target_probs, reduction="batchmean")
            loss_reg = F.mse_loss(pred_beta * cfg.BETA_MAX, target_beta)
            loss_prob_cons = logits.new_zeros(())
            loss_beta_cons = logits.new_zeros(())
            if imgs.shape[0] > 1:
                target_prob_delta = target_probs[1:] - target_probs[:-1]
                pred_prob_delta = pred_probs[1:] - pred_probs[:-1]
                loss_prob_cons = F.mse_loss(pred_prob_delta, target_prob_delta)

                target_beta_delta = target_beta[1:] - target_beta[:-1]
                pred_beta_delta = (pred_beta[1:] - pred_beta[:-1]) * cfg.BETA_MAX
                loss_beta_cons = F.mse_loss(pred_beta_delta, target_beta_delta)

            loss = (
                loss_cls
                + 1.25 * loss_reg
                + args.consistency_weight * loss_prob_cons
                + args.beta_consistency_weight * loss_beta_cons
            )
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().item())
            epoch_cls += float(loss_cls.detach().item())
            epoch_reg += float(loss_reg.detach().item())
            if imgs.shape[0] > 1:
                epoch_prob_cons = float(loss_prob_cons.detach().item())
                epoch_beta_cons = float(loss_beta_cons.detach().item())
            else:
                epoch_prob_cons = 0.0
                epoch_beta_cons = 0.0
            batches += 1

            prob_cons_acc += epoch_prob_cons
            beta_cons_acc += epoch_beta_cons

        metrics = {
            "epoch": epoch,
            "loss": epoch_loss / max(batches, 1),
            "cls": epoch_cls / max(batches, 1),
            "reg": epoch_reg / max(batches, 1),
            "prob_cons": prob_cons_acc / max(batches, 1),
            "beta_cons": beta_cons_acc / max(batches, 1),
            "batches": batches,
        }
        history.append(metrics)
        print(
            f"Adapt epoch {epoch}/{args.epochs}: "
            f"loss={metrics['loss']:.4f}, cls={metrics['cls']:.4f}, "
            f"reg={metrics['reg']:.6f}, prob_cons={metrics['prob_cons']:.6f}, "
            f"beta_cons={metrics['beta_cons']:.6f}"
        )

    adapted_weights = output_dir / "unified_model_best.pt"
    torch.save(student.state_dict(), adapted_weights)
    print(f"Saved adapted weights: {adapted_weights}")

    summary = {
        "video": str(video_path),
        "source_weights": str(source_weights),
        "adapted_weights": str(adapted_weights),
        "sampled_frames": int(frames_cpu.shape[0]),
        "frame_indices_preview": frame_indices[:20],
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "smooth_window": args.smooth_window,
        "consistency_weight": args.consistency_weight,
        "beta_consistency_weight": args.beta_consistency_weight,
        "history": history,
        "teacher_target": {
            "beta_mean": float(teacher_beta.mean().item()),
            "beta_std": float(teacher_beta.std().item()),
            "smooth_beta_mean": float(smooth_beta.mean().item()),
            "smooth_beta_std": float(smooth_beta.std().item()),
        },
    }
    save_summary(output_dir / "adapt_summary.json", summary)
    print(f"Saved summary: {output_dir / 'adapt_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
