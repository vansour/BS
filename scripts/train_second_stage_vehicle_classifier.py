#!/usr/bin/env python3
"""
训练二次车辆/非车辆复核分类器。

目标：
- 用少量 bootstrap 负样本 + UA-DETRAC 车辆正样本，
  训练一个可直接回接 `TemporalVehicleFilter` 的二分类器；
- 输出的权重格式与 `ImageNetVehiclePatchVerifier` 兼容。
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_DIR = (
    ROOT / "data" / "UA-DETRAC" / "DETRAC-train-data" / "Insight-MVT_Annotation_Train"
)
DEFAULT_XML_DIR = ROOT / "data" / "UA-DETRAC" / "DETRAC-Train-Annotations-XML"
DEFAULT_REVIEW_CSV = (
    ROOT
    / "outputs"
    / "Static_False_Positive_Review_temporal_smoke"
    / "review_checklist.csv"
)
DEFAULT_PATCH_ROOT = ROOT / "outputs" / "Static_False_Positive_Mine_temporal_smoke"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "Second_Stage_Vehicle_Classifier_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a bootstrap binary vehicle/non-vehicle patch classifier."
    )
    parser.add_argument(
        "--raw-data-dir",
        default=str(DEFAULT_RAW_DATA_DIR),
        help="UA-DETRAC 原始图像目录。",
    )
    parser.add_argument(
        "--xml-dir",
        default=str(DEFAULT_XML_DIR),
        help="UA-DETRAC XML 标注目录。",
    )
    parser.add_argument(
        "--review-csv",
        default=str(DEFAULT_REVIEW_CSV),
        help="人工复核清单 CSV 路径。",
    )
    parser.add_argument(
        "--patch-root",
        default=str(DEFAULT_PATCH_ROOT),
        help="静止误检 patch 根目录。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="分类器输出目录。",
    )
    parser.add_argument(
        "--positive-target-count",
        type=int,
        default=240,
        help="抽取多少个车辆正样本 patch。",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=12,
        help="从 UA-DETRAC 中每隔多少帧抽取一次车辆样本。",
    )
    parser.add_argument(
        "--max-positives-per-sequence",
        type=int,
        default=20,
        help="单个序列最多抽取多少个车辆正样本。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.25,
        help="验证集比例。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="训练 batch size。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数。",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率。",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=160,
        help="输入 patch 尺寸。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="训练设备：auto / cuda / cpu。",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def clip_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    context_pad: float = 0.08,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * max(0.0, context_pad)
    pad_y = bh * max(0.0, context_pad)
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def parse_xml_boxes(xml_path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    root = ET.parse(xml_path).getroot()
    frame_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    for frame in root.findall("frame"):
        frame_num = int(frame.get("num", "0"))
        boxes = []
        target_list = frame.find("target_list")
        if target_list is not None:
            for target in target_list.findall("target"):
                box = target.find("box")
                if box is None:
                    continue
                left = float(box.get("left", "0"))
                top = float(box.get("top", "0"))
                width = float(box.get("width", "0"))
                height = float(box.get("height", "0"))
                boxes.append((left, top, left + width, top + height))
        frame_boxes[frame_num] = boxes
    return frame_boxes


@dataclass
class SampleRecord:
    image_path: str
    label: int
    source: str
    meta: dict[str, object]


class PatchRecordDataset(Dataset):
    def __init__(self, records: list[SampleRecord], transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            patch = image.convert("RGB")
        return self.transform(patch), int(record.label)


def load_negative_records(
    review_csv: Path,
    patch_root: Path,
) -> list[SampleRecord]:
    import csv

    negative_records: list[SampleRecord] = []
    with review_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("heuristic_decision") != "likely_false_positive":
                continue
            patch_path = patch_root / row["crop_path"]
            if not patch_path.exists():
                continue
            negative_records.append(
                SampleRecord(
                    image_path=str(patch_path),
                    label=0,
                    source="bootstrap_negative",
                    meta={
                        "review_id": int(row["review_id"]),
                        "route": row["route"],
                        "video_label": row["video_label"],
                        "track_id": int(row["track_id"]),
                    },
                )
            )
    return negative_records


def sample_positive_records(
    raw_data_dir: Path,
    xml_dir: Path,
    *,
    positive_target_count: int,
    frame_stride: int,
    max_positives_per_sequence: int,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    sequence_dirs = sorted([path for path in raw_data_dir.iterdir() if path.is_dir()])
    for sequence_dir in sequence_dirs:
        xml_path = xml_dir / f"{sequence_dir.name}.xml"
        if not xml_path.exists():
            continue

        frame_boxes = parse_xml_boxes(xml_path)
        image_paths = sorted(
            [path for path in sequence_dir.iterdir() if path.suffix.lower() in {".jpg", ".png"}]
        )[:: max(1, frame_stride)]
        positives_from_sequence = 0
        for image_path in image_paths:
            frame_name = image_path.stem
            if not frame_name.startswith("img"):
                continue
            try:
                frame_num = int(frame_name.replace("img", ""))
            except ValueError:
                continue

            boxes = frame_boxes.get(frame_num, [])
            if not boxes:
                continue

            with Image.open(image_path) as image:
                image_rgb = image.convert("RGB")
                for box_index, box in enumerate(boxes):
                    crop_box = clip_box(box, image_rgb.size)
                    patch = image_rgb.crop(crop_box)
                    if patch.size[0] < 16 or patch.size[1] < 16:
                        continue
                    patch_dir = ROOT / "outputs" / "_temp_positive_patches"
                    patch_dir.mkdir(parents=True, exist_ok=True)
                    patch_path = patch_dir / (
                        f"{sequence_dir.name}_{frame_num:05d}_{box_index:02d}.jpg"
                    )
                    patch.save(patch_path)
                    records.append(
                        SampleRecord(
                            image_path=str(patch_path),
                            label=1,
                            source="uadetrac_positive",
                            meta={
                                "sequence": sequence_dir.name,
                                "frame_num": frame_num,
                                "box_index": box_index,
                            },
                        )
                    )
                    positives_from_sequence += 1
                    if positives_from_sequence >= max_positives_per_sequence:
                        break
            if positives_from_sequence >= max_positives_per_sequence:
                break
            if len(records) >= positive_target_count:
                break
        if len(records) >= positive_target_count:
            break
    return records[:positive_target_count]


def split_records(records: list[SampleRecord], val_ratio: float, seed: int):
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * max(0.0, min(1.0, val_ratio))))
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    if not train_records:
        train_records = shuffled
        val_records = []
    return train_records, val_records


def write_records_jsonl(path: Path, records: list[SampleRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "image_path": record.image_path,
                "label": int(record.label),
                "source": record.source,
                "meta": record.meta,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)

    raw_data_dir = Path(args.raw_data_dir).resolve()
    xml_dir = Path(args.xml_dir).resolve()
    review_csv = Path(args.review_csv).resolve()
    patch_root = Path(args.patch_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data dir not found: {raw_data_dir}")
    if not xml_dir.exists():
        raise FileNotFoundError(f"XML dir not found: {xml_dir}")
    if not review_csv.exists():
        raise FileNotFoundError(f"Review CSV not found: {review_csv}")
    if not patch_root.exists():
        raise FileNotFoundError(f"Patch root not found: {patch_root}")

    negative_records = load_negative_records(review_csv, patch_root)
    positive_records = sample_positive_records(
        raw_data_dir,
        xml_dir,
        positive_target_count=max(1, int(args.positive_target_count)),
        frame_stride=max(1, int(args.frame_stride)),
        max_positives_per_sequence=max(1, int(args.max_positives_per_sequence)),
    )
    if not negative_records:
        raise RuntimeError("No bootstrap negative records were loaded.")
    if not positive_records:
        raise RuntimeError("No positive vehicle records were sampled.")

    pos_train, pos_val = split_records(positive_records, args.val_ratio, args.seed)
    neg_train, neg_val = split_records(negative_records, args.val_ratio, args.seed + 1)
    train_records = pos_train + neg_train
    val_records = pos_val + neg_val
    random.Random(args.seed).shuffle(train_records)
    random.Random(args.seed + 2).shuffle(val_records)

    train_manifest = output_dir / "train_manifest.jsonl"
    val_manifest = output_dir / "val_manifest.jsonl"
    write_records_jsonl(train_manifest, train_records)
    write_records_jsonl(val_manifest, val_records)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = PatchRecordDataset(train_records, train_transform)
    val_dataset = PatchRecordDataset(val_records, eval_transform) if val_records else None

    class_counts = {0: 0, 1: 0}
    for record in train_records:
        class_counts[int(record.label)] += 1
    sample_weights = [
        1.0 / max(class_counts[int(record.label)], 1) for record in train_records
    ]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=max(len(train_records), 2 * max(class_counts.values())),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, int(args.batch_size)),
        sampler=sampler,
        num_workers=0,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=0,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    model = mobilenet_v3_small(weights=weights)
    in_features = int(model.classifier[-1].in_features)
    model.classifier[-1] = nn.Linear(in_features, 2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_acc = -1.0
    history: list[dict[str, float]] = []
    best_payload = None

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().item())
            train_acc_sum += compute_accuracy(logits.detach(), labels)
            train_batches += 1

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_batches, 1),
            "train_acc": train_acc_sum / max(train_batches, 1),
        }

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss_sum += float(loss.detach().item())
                    val_acc_sum += compute_accuracy(logits, labels)
                    val_batches += 1
            epoch_record["val_loss"] = val_loss_sum / max(val_batches, 1)
            epoch_record["val_acc"] = val_acc_sum / max(val_batches, 1)
            monitored_acc = epoch_record["val_acc"]
        else:
            monitored_acc = epoch_record["train_acc"]

        history.append(epoch_record)
        print(
            f"Epoch {epoch}: "
            f"train_loss={epoch_record['train_loss']:.4f}, "
            f"train_acc={epoch_record['train_acc']:.4f}, "
            f"val_acc={epoch_record.get('val_acc', 0.0):.4f}"
        )

        if monitored_acc >= best_val_acc:
            best_val_acc = monitored_acc
            best_payload = {
                "classifier_type": "binary_vehicle_classifier",
                "model_name": "mobilenet_v3_small",
                "class_names": ["non_vehicle", "vehicle"],
                "img_size": int(args.img_size),
                "seed": int(args.seed),
                "review_csv": str(review_csv),
                "patch_root": str(patch_root),
                "train_manifest": str(train_manifest),
                "val_manifest": str(val_manifest),
                "model_state_dict": model.state_dict(),
            }

    if best_payload is None:
        raise RuntimeError("Training did not produce any checkpoint payload.")

    weights_path = output_dir / "second_stage_vehicle_classifier_best.pt"
    torch.save(best_payload, weights_path)

    summary = {
        "status": "completed",
        "device": device,
        "positive_records_total": len(positive_records),
        "negative_records_total": len(negative_records),
        "train_records_total": len(train_records),
        "val_records_total": len(val_records),
        "train_class_counts": {
            "non_vehicle": sum(1 for record in train_records if record.label == 0),
            "vehicle": sum(1 for record in train_records if record.label == 1),
        },
        "val_class_counts": {
            "non_vehicle": sum(1 for record in val_records if record.label == 0),
            "vehicle": sum(1 for record in val_records if record.label == 1),
        },
        "best_val_acc": float(best_val_acc),
        "weights_path": str(weights_path),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved weights: {weights_path}")
    print(f"Saved summary: {output_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
