#!/usr/bin/env python3
"""
Build a thesis-ready figure that shows one representative false-negative case
and one representative false-positive case on the UA-DETRAC validation split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from tqdm import tqdm
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data.dataset import MultiTaskDataset
from src.model import UnifiedMultiTaskModel
from src.utils import load_model_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a representative FN/FP figure from UA-DETRAC validation data."
    )
    parser.add_argument(
        "--weights",
        default=str(
            ROOT / "outputs" / "Fog_Detection_Project_fogfocus_full" / "unified_model_best.pt"
        ),
        help="Unified model weights used to mine representative error cases.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config file (.json/.yaml/.yml).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2000,
        help="Maximum number of validation images to scan for representative cases.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "overleaf_thesis" / "assets" / "figures" / "fig_10_detection_error_cases.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.001,
        help="Detection confidence threshold used before NMS.",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.7,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Max detections per image after NMS.",
    )
    return parser.parse_args()


def configure_matplotlib_font():
    preferred = [
        "AR PL UMing CN",
        "AR PL UKai CN",
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "Source Han Sans SC",
        "Source Han Serif SC",
        "WenQuanYi Zen Hei",
        "SimHei",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def xywhn_to_xyxy_abs(boxes: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    x_c, y_c, w_n, h_n = boxes.unbind(dim=1)
    width = w_n * image_w
    height = h_n * image_h
    x1 = x_c * image_w - width / 2.0
    y1 = y_c * image_h - height / 2.0
    x2 = x_c * image_w + width / 2.0
    y2 = y_c * image_h + height / 2.0
    return torch.stack([x1, y1, x2, y2], dim=1)


def tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    return (image_tensor.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)


def greedy_match(iou_matrix: np.ndarray, threshold: float) -> tuple[set[int], set[int]]:
    matches = np.argwhere(iou_matrix >= threshold)
    if matches.shape[0] == 0:
        return set(), set()

    scores = iou_matrix[matches[:, 0], matches[:, 1]]
    order = np.argsort(-scores)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for idx in order:
        gt_idx, pred_idx = matches[idx]
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(int(gt_idx))
        matched_pred.add(int(pred_idx))
    return matched_gt, matched_pred


def crop_with_context(image: np.ndarray, focus_box: np.ndarray, *, min_size: int = 220, pad_scale: float = 1.2):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in focus_box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_extent = max(min_size / 2.0, max(bw, bh) * (0.5 + pad_scale))
    half_w = half_extent
    half_h = half_extent
    crop_x1 = int(max(0, round(cx - half_w)))
    crop_y1 = int(max(0, round(cy - half_h)))
    crop_x2 = int(min(w, round(cx + half_w)))
    crop_y2 = int(min(h, round(cy + half_h)))
    if crop_x2 <= crop_x1:
        crop_x2 = min(w, crop_x1 + min_size)
    if crop_y2 <= crop_y1:
        crop_y2 = min(h, crop_y1 + min_size)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    offset = np.array([crop_x1, crop_y1, crop_x1, crop_y1], dtype=np.float32)
    return crop, offset


def draw_boxes(ax, boxes: np.ndarray, color: str, *, linewidth: float = 1.8, linestyle: str = "-", labels: list[str] | None = None):
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        rect = mpatches.Rectangle(
            (x1, y1),
            max(1.0, x2 - x1),
            max(1.0, y2 - y1),
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
        )
        ax.add_patch(rect)
        if labels and idx < len(labels) and labels[idx]:
            ax.text(
                x1 + 2,
                max(12, y1 - 6),
                labels[idx],
                color=color,
                fontsize=9,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.35, pad=1.5, edgecolor="none"),
            )


def pick_representative_cases(cfg: Config, weights_path: Path, max_images: int, conf_thres: float, iou_thres: float, max_det: int):
    dataset = MultiTaskDataset(
        cfg.RAW_DATA_DIR,
        cfg.DEPTH_CACHE_DIR,
        xml_dir=cfg.XML_DIR,
        transform=None,
        is_train=False,
        frame_stride=cfg.FRAME_STRIDE,
        det_train_class_id=cfg.DET_TRAIN_CLASS_ID,
        img_size=cfg.IMG_SIZE,
        keep_ratio=True,
        train_ratio=cfg.TRAIN_RATIO,
        split_seed=cfg.SEED,
    )

    model = UnifiedMultiTaskModel(
        cfg.YOLO_BASE_MODEL,
        cfg.NUM_FOG_CLASSES,
        num_det_classes=cfg.NUM_DET_CLASSES,
        img_size=cfg.IMG_SIZE,
    )
    load_model_weights(model, str(weights_path), map_location=cfg.DEVICE)
    model.to(cfg.DEVICE).eval()

    best_fn = None
    best_fp = None
    best_fn_score = -1.0
    best_fp_score = -1.0

    scan_limit = min(max(1, max_images), len(dataset))
    for idx in tqdm(range(scan_limit), desc="Mining detection error cases"):
        image_tensor, _, det_cls, det_boxes = dataset[idx]
        image_rgb = tensor_to_rgb(image_tensor)
        gt_boxes = xywhn_to_xyxy_abs(
            det_boxes,
            image_h=int(image_tensor.shape[1]),
            image_w=int(image_tensor.shape[2]),
        )

        with torch.no_grad():
            det_out, _, _ = model(image_tensor.unsqueeze(0).to(cfg.DEVICE))
            det_batch = non_max_suppression(
                det_out,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                nc=cfg.NUM_DET_CLASSES,
                max_det=max_det,
            )[0].detach().cpu()

        pred_boxes = det_batch[:, :4].float() if det_batch.numel() > 0 else torch.zeros((0, 4), dtype=torch.float32)
        pred_conf = det_batch[:, 4].float() if det_batch.numel() > 0 else torch.zeros((0,), dtype=torch.float32)

        if gt_boxes.numel() > 0 and pred_boxes.numel() > 0:
            iou = box_iou(gt_boxes, pred_boxes).cpu().numpy()
        else:
            iou = np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32)

        matched_gt, matched_pred = greedy_match(iou, threshold=0.5)
        unmatched_gt = [gt_i for gt_i in range(gt_boxes.shape[0]) if gt_i not in matched_gt]
        unmatched_pred = [pred_i for pred_i in range(pred_boxes.shape[0]) if pred_i not in matched_pred]

        if unmatched_gt:
            areas = []
            for gt_i in unmatched_gt:
                box = gt_boxes[gt_i]
                area = float(max(1.0, (box[2] - box[0]).item()) * max(1.0, (box[3] - box[1]).item()))
                areas.append((area, gt_i))
            area, chosen_gt = max(areas, key=lambda item: item[0])
            score = area
            if score > best_fn_score:
                sample_path, seq_name, image_name = dataset.samples[idx]
                best_fn_score = score
                best_fn = {
                    "image": image_rgb,
                    "gt_boxes": gt_boxes.numpy(),
                    "pred_boxes": pred_boxes.numpy(),
                    "pred_conf": pred_conf.numpy(),
                    "focus_box": gt_boxes[chosen_gt].numpy(),
                    "sequence": seq_name,
                    "image_name": image_name,
                    "sample_path": sample_path,
                }

        if unmatched_pred:
            candidates = []
            for pred_i in unmatched_pred:
                box = pred_boxes[pred_i]
                conf = float(pred_conf[pred_i].item())
                area = float(max(1.0, (box[2] - box[0]).item()) * max(1.0, (box[3] - box[1]).item()))
                area_ratio = area / float(image_tensor.shape[1] * image_tensor.shape[2])
                score = conf + min(area_ratio * 4.0, 0.25)
                candidates.append((score, pred_i))
            score, chosen_pred = max(candidates, key=lambda item: item[0])
            if score > best_fp_score:
                sample_path, seq_name, image_name = dataset.samples[idx]
                best_fp_score = score
                best_fp = {
                    "image": image_rgb,
                    "gt_boxes": gt_boxes.numpy(),
                    "pred_boxes": pred_boxes.numpy(),
                    "pred_conf": pred_conf.numpy(),
                    "focus_box": pred_boxes[chosen_pred].numpy(),
                    "focus_conf": float(pred_conf[chosen_pred].item()),
                    "sequence": seq_name,
                    "image_name": image_name,
                    "sample_path": sample_path,
                }

        if best_fn is not None and best_fp is not None and idx > 400:
            break

    if best_fn is None or best_fp is None:
        raise RuntimeError("Failed to mine both a false-negative case and a false-positive case.")
    return best_fn, best_fp


def make_detection_error_figure(fn_case: dict, fp_case: dict, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.1), gridspec_kw={"wspace": 0.12})

    # False negative panel.
    fn_crop, fn_offset = crop_with_context(fn_case["image"], fn_case["focus_box"])
    axes[0].imshow(fn_crop)
    axes[0].axis("off")
    fn_gt = fn_case["gt_boxes"] - fn_offset
    fn_pred = fn_case["pred_boxes"] - fn_offset
    selected_fn = np.array([fn_case["focus_box"] - fn_offset], dtype=np.float32)
    draw_boxes(axes[0], fn_gt, "#74C476", linewidth=1.4)
    if fn_pred.size > 0:
        keep = fn_case["pred_conf"] >= 0.25
        draw_boxes(
            axes[0],
            fn_pred[keep] if np.any(keep) else fn_pred[: min(3, len(fn_pred))],
            "#6BAED6",
            linewidth=1.0,
            linestyle="--",
        )
    draw_boxes(axes[0], selected_fn, "#FF8C00", linewidth=2.8, labels=["漏检 GT"])
    axes[0].set_title(
        f"典型漏检案例\n{fn_case['sequence']} / {fn_case['image_name']}",
        fontsize=13,
        weight="bold",
    )

    # False positive panel.
    fp_crop, fp_offset = crop_with_context(fp_case["image"], fp_case["focus_box"])
    axes[1].imshow(fp_crop)
    axes[1].axis("off")
    fp_gt = fp_case["gt_boxes"] - fp_offset
    fp_pred = fp_case["pred_boxes"] - fp_offset
    selected_fp = np.array([fp_case["focus_box"] - fp_offset], dtype=np.float32)
    draw_boxes(axes[1], fp_gt, "#74C476", linewidth=1.4)
    if fp_pred.size > 0:
        keep = fp_case["pred_conf"] >= 0.25
        draw_boxes(
            axes[1],
            fp_pred[keep] if np.any(keep) else fp_pred[: min(3, len(fp_pred))],
            "#6BAED6",
            linewidth=1.0,
            linestyle="--",
        )
    draw_boxes(
        axes[1],
        selected_fp,
        "#D62728",
        linewidth=2.8,
        labels=[f"误检 Pred {fp_case['focus_conf']:.2f}"],
    )
    axes[1].set_title(
        f"典型误检案例\n{fp_case['sequence']} / {fp_case['image_name']}",
        fontsize=13,
        weight="bold",
    )

    legend_handles = [
        mpatches.Patch(color="#74C476", label="GT 车辆框"),
        mpatches.Patch(color="#6BAED6", label="模型保留检测框"),
        mpatches.Patch(color="#FF8C00", label="漏检目标"),
        mpatches.Patch(color="#D62728", label="误检目标"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.suptitle(
        "UA-DETRAC 验证集上的典型漏检与误检案例（fogfocus_full）",
        fontsize=17,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_matplotlib_font()
    cfg = Config(config_path=args.config)
    weights_path = Path(args.weights).resolve()
    output_path = Path(args.output).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    fn_case, fp_case = pick_representative_cases(
        cfg,
        weights_path,
        max_images=max(1, int(args.max_images)),
        conf_thres=float(args.conf_thres),
        iou_thres=float(args.iou_thres),
        max_det=max(1, int(args.max_det)),
    )
    make_detection_error_figure(fn_case, fp_case, output_path)

    print(f"Saved figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
