#!/usr/bin/env python3
"""
Evaluate detector performance on the UA-DETRAC validation split.

This script reports standard single-class detection metrics:
- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

Supported detector sources:
1. The detector branch of the project's UnifiedMultiTaskModel.
2. A plain YOLO detector such as `yolo11n.pt`, folded to a single `vehicle` class
   by keeping only COCO vehicle categories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou
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
        description="Evaluate single-class vehicle detection metrics on the UA-DETRAC validation split."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional runtime config file (.json/.yaml/.yml).",
    )
    parser.add_argument(
        "--model-type",
        choices=["unified", "yolo"],
        required=True,
        help="Whether to evaluate a unified multi-task checkpoint or a plain YOLO detector.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model weights.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Short label used in output filenames and summaries.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "UADetrac_Detection_Eval"),
        help="Directory for evaluation summaries.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for batched inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on the number of validation images to evaluate. 0 means full validation split.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.001,
        help="Confidence threshold used before AP computation.",
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
        help="Maximum number of detections kept after NMS per image.",
    )
    return parser.parse_args()


def xywhn_to_xyxy_abs(boxes: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
    """Convert normalized xywh boxes to absolute xyxy boxes."""
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


class UADetracDetectionEvalDataset(Dataset):
    """Validation split view that returns only image tensors and detection labels."""

    def __init__(self, cfg: Config, *, max_images: int = 0):
        self.dataset = MultiTaskDataset(
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
        self.max_images = max(0, int(max_images))

    def __len__(self) -> int:
        if self.max_images > 0:
            return min(self.max_images, len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _, det_cls, det_boxes = self.dataset[index]
        gt_boxes = xywhn_to_xyxy_abs(
            det_boxes,
            image_h=int(image.shape[1]),
            image_w=int(image.shape[2]),
        )
        gt_cls = det_cls.to(torch.int64)
        return image, gt_boxes, gt_cls


def collate_eval_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    gt_boxes = [item[1] for item in batch]
    gt_cls = [item[2] for item in batch]
    return images, gt_boxes, gt_cls


def extract_raw_det_tensor(model_output):
    """Extract the raw detector tensor from a YOLO DetectionModel output."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, (list, tuple)):
        for item in model_output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Unsupported detector output type: {type(model_output)!r}")


def match_predictions(
    pred_cls: torch.Tensor,
    true_cls: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
) -> torch.Tensor:
    """Ultralytics-style one-to-one matching across IoU thresholds."""
    correct = np.zeros((pred_cls.shape[0], iouv.shape[0]), dtype=bool)
    if pred_cls.numel() == 0 or true_cls.numel() == 0:
        return torch.zeros((pred_cls.shape[0], iouv.shape[0]), dtype=torch.bool)

    correct_class = true_cls[:, None] == pred_cls
    iou = (iou * correct_class).cpu().numpy()

    for idx, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0] == 0:
            continue
        if matches.shape[0] > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), idx] = True

    return torch.tensor(correct, dtype=torch.bool)


def process_batch(
    pred_boxes: torch.Tensor,
    pred_cls: torch.Tensor,
    true_boxes: torch.Tensor,
    true_cls: torch.Tensor,
    iouv: torch.Tensor,
) -> np.ndarray:
    if pred_boxes.shape[0] == 0:
        return np.zeros((0, iouv.numel()), dtype=bool)
    if true_boxes.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], iouv.numel()), dtype=bool)
    iou = box_iou(true_boxes, pred_boxes)
    return match_predictions(pred_cls, true_cls, iou, iouv).cpu().numpy()


def build_detector(args: argparse.Namespace, cfg: Config, device: str):
    weights_path = str(Path(args.weights).resolve())
    if args.model_type == "unified":
        model = UnifiedMultiTaskModel(
            cfg.YOLO_BASE_MODEL,
            cfg.NUM_FOG_CLASSES,
            num_det_classes=cfg.NUM_DET_CLASSES,
            img_size=cfg.IMG_SIZE,
        )
        report = load_model_weights(model, weights_path, map_location=device)
        model.to(device).eval()
        return model, report

    yolo_wrapper = YOLO(weights_path)
    model = yolo_wrapper.model.to(device).eval()
    report = {"source_type": "yolo_native", "weights": weights_path}
    return model, report


def run_model_predictions(
    model,
    model_type: str,
    images: torch.Tensor,
    cfg: Config,
    args: argparse.Namespace,
) -> list[torch.Tensor]:
    with torch.no_grad():
        if model_type == "unified":
            det_out, _, _ = model(images)
            raw_det = det_out
            det_batches = non_max_suppression(
                raw_det,
                conf_thres=float(args.conf_thres),
                iou_thres=float(args.iou_thres),
                nc=cfg.NUM_DET_CLASSES,
                max_det=int(args.max_det),
            )
            return [det.detach().cpu() for det in det_batches]

        raw_det = extract_raw_det_tensor(model(images))
        det_batches = non_max_suppression(
            raw_det,
            conf_thres=float(args.conf_thres),
            iou_thres=float(args.iou_thres),
            nc=80,
            max_det=int(args.max_det),
        )
        filtered_batches: list[torch.Tensor] = []
        vehicle_ids = set(int(class_id) for class_id in cfg.COCO_VEHICLE_CLASS_IDS)
        for det in det_batches:
            if det.numel() == 0:
                filtered_batches.append(torch.zeros((0, 6), dtype=torch.float32))
                continue
            cls_ids = det[:, 5].round().to(torch.int64)
            keep = torch.zeros_like(cls_ids, dtype=torch.bool)
            for class_id in vehicle_ids:
                keep |= cls_ids == class_id
            det = det[keep]
            if det.numel() == 0:
                filtered_batches.append(torch.zeros((0, 6), dtype=torch.float32))
                continue
            det = det.clone()
            det[:, 5] = 0.0
            filtered_batches.append(det.detach().cpu())
        return filtered_batches


def summarize_metrics(stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    if not stats:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "num_targets": 0,
            "num_predictions": 0,
            "num_images": 0,
        }

    tp, conf, pred_cls, target_cls = (
        np.concatenate(items, axis=0) if len(items) else np.zeros((0,))
        for items in zip(*stats)
    )

    if target_cls.size == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "num_targets": 0,
            "num_predictions": int(conf.shape[0]),
            "num_images": 0,
        }

    tp_count, fp_count, precision, recall, f1, ap, classes, *_ = ap_per_class(
        tp,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        names={0: "vehicle"},
    )

    precision_value = float(np.mean(precision)) if precision.size else 0.0
    recall_value = float(np.mean(recall)) if recall.size else 0.0
    map50_value = float(np.mean(ap[:, 0])) if ap.size else 0.0
    map_value = float(np.mean(ap)) if ap.size else 0.0

    return {
        "precision": precision_value,
        "recall": recall_value,
        "map50": map50_value,
        "map50_95": map_value,
        "num_targets": int(target_cls.shape[0]),
        "num_predictions": int(conf.shape[0]),
        "num_classes_evaluated": int(classes.shape[0]) if isinstance(classes, np.ndarray) else 0,
        "tp_at_best_f1": float(np.sum(tp_count)) if np.size(tp_count) else 0.0,
        "fp_at_best_f1": float(np.sum(fp_count)) if np.size(fp_count) else 0.0,
    }


def main() -> int:
    args = parse_args()
    cfg = Config(config_path=args.config)
    device = cfg.DEVICE

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = UADetracDetectionEvalDataset(cfg, max_images=args.max_images)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device == "cuda",
        collate_fn=collate_eval_batch,
    )

    model, load_report = build_detector(args, cfg, device)
    iouv = torch.linspace(0.5, 0.95, 10)

    stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    num_images = 0

    print(f"Evaluating label={args.label}")
    print(f"Model type: {args.model_type}")
    print(f"Weights: {Path(args.weights).resolve()}")
    print(f"Device: {device}")
    print(f"Validation images: {len(dataset)}")

    progress = tqdm(loader, desc=f"UA-DETRAC eval: {args.label}")
    for images, gt_boxes_list, gt_cls_list in progress:
        images = images.to(device, non_blocking=True)
        batch_predictions = run_model_predictions(
            model,
            args.model_type,
            images,
            cfg,
            args,
        )

        for preds, true_boxes, true_cls in zip(batch_predictions, gt_boxes_list, gt_cls_list):
            num_images += 1
            true_boxes = true_boxes.float()
            true_cls = true_cls.to(torch.int64)

            if preds.numel() == 0:
                correct = np.zeros((0, iouv.numel()), dtype=bool)
                conf = np.zeros((0,), dtype=np.float32)
                pred_cls = np.zeros((0,), dtype=np.float32)
            else:
                pred_boxes = preds[:, :4].float()
                conf = preds[:, 4].cpu().numpy()
                pred_cls_tensor = preds[:, 5].to(torch.int64)
                correct = process_batch(
                    pred_boxes,
                    pred_cls_tensor,
                    true_boxes,
                    true_cls,
                    iouv,
                )
                pred_cls = pred_cls_tensor.cpu().numpy()

            stats.append(
                (
                    correct,
                    conf,
                    pred_cls,
                    true_cls.cpu().numpy(),
                )
            )

    summary = summarize_metrics(stats)
    summary.update(
        {
            "label": args.label,
            "model_type": args.model_type,
            "weights": str(Path(args.weights).resolve()),
            "device": device,
            "config_file": cfg.CONFIG_FILE or "",
            "img_size": int(cfg.IMG_SIZE),
            "validation_images": len(dataset),
            "processed_images": num_images,
            "conf_thres": float(args.conf_thres),
            "iou_thres": float(args.iou_thres),
            "max_det": int(args.max_det),
            "load_report": load_report,
        }
    )

    output_json = output_dir / f"{args.label}_uadetrac_detection_metrics.json"
    output_md = output_dir / f"{args.label}_uadetrac_detection_metrics.md"
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_md.write_text(
        "\n".join(
            [
                f"# UA-DETRAC Detection Metrics: {args.label}",
                "",
                f"- Model type: `{args.model_type}`",
                f"- Weights: `{Path(args.weights).resolve()}`",
                f"- Precision: `{summary['precision']:.6f}`",
                f"- Recall: `{summary['recall']:.6f}`",
                f"- mAP@0.5: `{summary['map50']:.6f}`",
                f"- mAP@0.5:0.95: `{summary['map50_95']:.6f}`",
                f"- Targets: `{summary['num_targets']}`",
                f"- Predictions: `{summary['num_predictions']}`",
                f"- Images: `{summary['processed_images']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
