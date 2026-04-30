#!/usr/bin/env python3
# ruff: noqa: E402
"""
阶段二数据审计脚本。

该脚本用于对当前 UA-DETRAC 数据链路做系统性检查，并产出：
1. 结构化 JSON 报告；
2. 人类可读的 Markdown 摘要；
3. 若干带检测框与深度状态的可视化样例。

检查范围主要包括：
- 原始训练/测试数据目录结构；
- train/val 划分后的样本统计；
- XML 标注覆盖情况；
- 边界框有效性；
- 图像可读性抽查；
- 深度缓存覆盖情况；
- 可视化样例输出。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data import MultiTaskDataset
from src.utils import compute_letterbox_metadata


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_div(numerator: int | float, denominator: int | float) -> float:
    """安全除法。"""
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def rounded(value: float, ndigits: int = 4) -> float:
    """统一浮点输出格式。"""
    return round(float(value), ndigits)


def even_indices(total: int, count: int) -> list[int]:
    """从 `[0, total)` 中均匀选取若干索引。"""
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]

    step = (total - 1) / (count - 1)
    selected = []
    for idx in range(count):
        candidate = int(round(idx * step))
        if not selected or candidate != selected[-1]:
            selected.append(candidate)
    return selected


def sequence_dirs(root: Path) -> list[Path]:
    """返回按名字排序的序列目录。"""
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()])


def count_image_files(root: Path) -> tuple[int, dict[str, int]]:
    """统计目录下图像文件总数与扩展名分布。"""
    total = 0
    suffixes: Counter[str] = Counter()
    for sequence_dir in sequence_dirs(root):
        for path in sequence_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jpg", ".png"}:
                continue
            total += 1
            suffixes[path.suffix.lower()] += 1
    return total, dict(sorted(suffixes.items()))


def dataset_for_split(cfg: Config, is_train: bool) -> MultiTaskDataset:
    """构建一个与训练脚本一致的数据集实例。"""
    return MultiTaskDataset(
        cfg.RAW_DATA_DIR,
        cfg.DEPTH_CACHE_DIR,
        xml_dir=cfg.XML_DIR,
        transform=None,
        is_train=is_train,
        frame_stride=cfg.FRAME_STRIDE,
        det_train_class_id=cfg.DET_TRAIN_CLASS_ID,
        img_size=cfg.IMG_SIZE,
        keep_ratio=True,
        train_ratio=cfg.TRAIN_RATIO,
        split_seed=cfg.SEED,
    )


def frame_boxes(
    dataset: MultiTaskDataset, seq: str, img_name: str
) -> list[list[float]]:
    """获取某一帧对应的标注框。"""
    frame_num = dataset._extract_frame_num(img_name)
    if frame_num is None:
        return []
    return dataset.annotations.get(seq, {}).get(frame_num, [])


def verify_image(path: str) -> tuple[bool, str | None]:
    """检查图像是否可读。"""
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        return False, str(exc)


def box_stats_template() -> dict[str, int]:
    """初始化边界框统计模板。"""
    return {
        "total_boxes": 0,
        "non_finite": 0,
        "negative_origin": 0,
        "non_positive_size": 0,
        "exceeds_width": 0,
        "exceeds_height": 0,
        "requires_clipping": 0,
    }


def analyze_split(
    split_name: str,
    dataset: MultiTaskDataset,
    xml_dir: Path,
    image_scan_limit: int,
    full_image_scan: bool,
) -> dict[str, object]:
    """分析 train 或 val 划分的数据状态。"""
    seq_sample_counts = Counter(seq for _, seq, _ in dataset.samples)
    selected_sequences = sorted(seq_sample_counts.keys())
    missing_xml_sequences = sorted(
        [seq for seq in selected_sequences if not (xml_dir / f"{seq}.xml").exists()]
    )

    invalid_frame_name_samples: list[str] = []
    missing_frame_in_xml_examples: list[str] = []
    empty_annotation_examples: list[str] = []
    unreadable_examples: list[str] = []
    missing_depth_examples: list[str] = []
    missing_frame_by_sequence: Counter[str] = Counter()

    labeled_samples = 0
    unlabeled_samples = 0
    missing_frame_in_xml_count = 0
    invalid_frame_name_count = 0
    existing_depth_count = 0
    missing_depth_count = 0

    width_by_sequence: dict[str, int] = {}
    height_by_sequence: dict[str, int] = {}
    image_extensions: Counter[str] = Counter()
    image_scan_failures = 0
    box_stats = box_stats_template()
    boxes_per_labeled_sample: list[int] = []

    if full_image_scan:
        image_scan_indices = set(range(len(dataset.samples)))
    else:
        image_scan_indices = set(even_indices(len(dataset.samples), image_scan_limit))

    for sample_idx, (img_path, seq, img_name) in enumerate(dataset.samples):
        image_extensions[Path(img_path).suffix.lower()] += 1

        depth_name = f"{seq}_{img_name}.npy"
        depth_path = os.path.join(dataset.depth_cache_dir, depth_name)
        if os.path.exists(depth_path):
            existing_depth_count += 1
        else:
            missing_depth_count += 1
            if len(missing_depth_examples) < 5:
                missing_depth_examples.append(depth_name)

        if seq not in width_by_sequence:
            try:
                with Image.open(img_path) as image:
                    width_by_sequence[seq], height_by_sequence[seq] = image.size
            except Exception as exc:  # pragma: no cover - best effort diagnostics
                image_scan_failures += 1
                width_by_sequence[seq], height_by_sequence[seq] = 0, 0
                if len(unreadable_examples) < 5:
                    unreadable_examples.append(f"{seq}/{img_name}: {exc}")
                continue

        frame_num = dataset._extract_frame_num(img_name)
        if frame_num is None:
            invalid_frame_name_count += 1
            if len(invalid_frame_name_samples) < 5:
                invalid_frame_name_samples.append(f"{seq}/{img_name}")
            continue

        annotations = dataset.annotations.get(seq, {})
        if frame_num not in annotations:
            missing_frame_in_xml_count += 1
            missing_frame_by_sequence[seq] += 1
            unlabeled_samples += 1
            if len(missing_frame_in_xml_examples) < 5:
                missing_frame_in_xml_examples.append(f"{seq}/{img_name}")
            continue

        boxes = annotations.get(frame_num, [])
        if not boxes:
            unlabeled_samples += 1
            if len(empty_annotation_examples) < 5:
                empty_annotation_examples.append(f"{seq}/{img_name}")
            continue

        labeled_samples += 1
        boxes_per_labeled_sample.append(len(boxes))

        seq_w = width_by_sequence[seq]
        seq_h = height_by_sequence[seq]
        for box in boxes:
            box_stats["total_boxes"] += 1
            left, top, width, height = box
            if not all(math.isfinite(value) for value in box):
                box_stats["non_finite"] += 1
                continue
            requires_clipping = False
            if left < 0 or top < 0:
                box_stats["negative_origin"] += 1
                requires_clipping = True
            if width <= 0 or height <= 0:
                box_stats["non_positive_size"] += 1
            if left + width > seq_w:
                box_stats["exceeds_width"] += 1
                requires_clipping = True
            if top + height > seq_h:
                box_stats["exceeds_height"] += 1
                requires_clipping = True
            if requires_clipping:
                box_stats["requires_clipping"] += 1

        if sample_idx in image_scan_indices:
            ok, error = verify_image(img_path)
            if not ok:
                image_scan_failures += 1
                if len(unreadable_examples) < 5:
                    unreadable_examples.append(f"{seq}/{img_name}: {error}")

    samples_per_sequence = list(seq_sample_counts.values())
    return {
        "split": split_name,
        "sequence_count": len(selected_sequences),
        "sample_count": len(dataset.samples),
        "samples_per_sequence": {
            "min": min(samples_per_sequence) if samples_per_sequence else 0,
            "max": max(samples_per_sequence) if samples_per_sequence else 0,
            "mean": (
                rounded(statistics.mean(samples_per_sequence))
                if samples_per_sequence
                else 0.0
            ),
            "median": (
                rounded(statistics.median(samples_per_sequence))
                if samples_per_sequence
                else 0.0
            ),
        },
        "image_extensions": dict(sorted(image_extensions.items())),
        "xml": {
            "missing_sequence_count": len(missing_xml_sequences),
            "missing_sequences_preview": missing_xml_sequences[:10],
            "missing_frame_in_xml_count": missing_frame_in_xml_count,
            "missing_frame_in_xml_examples": missing_frame_in_xml_examples,
            "top_missing_frame_sequences": missing_frame_by_sequence.most_common(10),
            "invalid_frame_name_count": invalid_frame_name_count,
            "invalid_frame_name_examples": invalid_frame_name_samples,
        },
        "labels": {
            "labeled_sample_count": labeled_samples,
            "unlabeled_sample_count": unlabeled_samples,
            "labeled_sample_ratio": rounded(
                safe_div(labeled_samples, len(dataset.samples))
            ),
            "avg_boxes_per_labeled_sample": (
                rounded(statistics.mean(boxes_per_labeled_sample))
                if boxes_per_labeled_sample
                else 0.0
            ),
            "empty_annotation_examples": empty_annotation_examples,
        },
        "boxes": box_stats,
        "depth_cache": {
            "existing_count": existing_depth_count,
            "missing_count": missing_depth_count,
            "coverage_ratio": rounded(
                safe_div(existing_depth_count, len(dataset.samples))
            ),
            "missing_examples": missing_depth_examples,
        },
        "image_readability": {
            "checked_count": len(image_scan_indices),
            "failure_count": image_scan_failures,
            "failure_examples": unreadable_examples,
        },
    }


def normalize_depth_to_rgb(
    depth: np.ndarray, target_size: tuple[int, int]
) -> Image.Image:
    """将深度图转换为可视化 RGB 图。"""
    depth = depth.astype(np.float32)
    d_min = float(np.min(depth))
    d_max = float(np.max(depth))
    if d_max - d_min < 1e-6:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        normalized = (
            ((depth - d_min) / (d_max - d_min) * 255.0).clip(0, 255).astype(np.uint8)
        )

    # 用简化的伪彩色映射让深度更容易区分。
    rgb = np.stack(
        [
            normalized,
            np.flipud(normalized),
            255 - normalized,
        ],
        axis=-1,
    )
    image = Image.fromarray(rgb, mode="RGB")
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    return image


def placeholder_panel(
    size: tuple[int, int], title: str, lines: list[str]
) -> Image.Image:
    """创建占位面板。"""
    panel = Image.new("RGB", size, (40, 40, 40))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(120, 120, 120), width=2)
    draw.text((16, 16), title, fill=(255, 255, 255))
    y = 52
    for line in lines:
        draw.text((16, y), line, fill=(220, 220, 220))
        y += 24
    return panel


def draw_boxes(
    image: Image.Image, boxes: list[list[float]], color: tuple[int, int, int]
) -> Image.Image:
    """在图像上绘制边界框。"""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for left, top, width, height in boxes:
        draw.rectangle(
            (left, top, left + width, top + height),
            outline=color,
            width=3,
        )
    return canvas


def letterbox_preview(
    image: Image.Image,
    boxes: list[list[float]],
    target_size: int,
) -> Image.Image:
    """生成 letterbox 预览图。"""
    src_h = image.height
    src_w = image.width
    metadata = compute_letterbox_metadata((src_h, src_w), target_size)
    resized_h, resized_w = metadata["resized_shape"]
    pad_left, pad_top, pad_right, pad_bottom = metadata["pad"]
    canvas = Image.new(
        "RGB",
        (target_size, target_size),
        (
            int(114),
            int(114),
            int(114),
        ),
    )
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas.paste(resized, (pad_left, pad_top))
    draw = ImageDraw.Draw(canvas)
    scale = float(metadata["scale"])
    for left, top, width, height in boxes:
        x1 = left * scale + pad_left
        y1 = top * scale + pad_top
        x2 = (left + width) * scale + pad_left
        y2 = (top + height) * scale + pad_top
        draw.rectangle((x1, y1, x2, y2), outline=(255, 190, 0), width=3)
    return canvas


def visualization_candidates(
    dataset: MultiTaskDataset,
    count: int,
) -> list[int]:
    """优先选择有标注框的样本做可视化。"""
    labeled_indices: list[int] = []
    unlabeled_indices: list[int] = []
    for index, (_, seq, img_name) in enumerate(dataset.samples):
        boxes = frame_boxes(dataset, seq, img_name)
        if boxes:
            labeled_indices.append(index)
        else:
            unlabeled_indices.append(index)

    target_labeled = min(count, len(labeled_indices))
    selected = [
        labeled_indices[idx]
        for idx in even_indices(len(labeled_indices), target_labeled)
    ]
    if len(selected) < count:
        missing = count - len(selected)
        selected.extend(
            unlabeled_indices[idx]
            for idx in even_indices(
                len(unlabeled_indices), min(missing, len(unlabeled_indices))
            )
        )
    return selected


def save_visualizations(
    cfg: Config,
    split_name: str,
    dataset: MultiTaskDataset,
    output_dir: Path,
    count: int,
) -> list[str]:
    """输出若干带标注框和深度状态的可视化样例。"""
    saved: list[str] = []
    for sample_index in visualization_candidates(dataset, count):
        img_path, seq, img_name = dataset.samples[sample_index]
        boxes = frame_boxes(dataset, seq, img_name)
        depth_path = Path(dataset.depth_cache_dir) / f"{seq}_{img_name}.npy"

        try:
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                boxed_image = draw_boxes(image, boxes, (80, 255, 80))
                original_panel = boxed_image.resize(
                    (512, 512), Image.Resampling.BILINEAR
                )
                letterbox_panel = letterbox_preview(image, boxes, cfg.IMG_SIZE)

                if depth_path.exists():
                    depth = np.load(depth_path)
                    depth_panel = normalize_depth_to_rgb(depth, (512, 512))
                else:
                    depth_panel = placeholder_panel(
                        (512, 512),
                        "Depth Cache",
                        [
                            "missing",
                            depth_path.name,
                        ],
                    )
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            original_panel = placeholder_panel(
                (512, 512),
                "Image Read Error",
                [
                    f"{seq}/{img_name}",
                    str(exc),
                ],
            )
            letterbox_panel = placeholder_panel(
                (512, 512),
                "Letterbox Preview",
                ["unavailable"],
            )
            depth_panel = placeholder_panel(
                (512, 512),
                "Depth Cache",
                ["skipped because image failed"],
            )

        header_height = 64
        canvas = Image.new("RGB", (512 * 3, 512 + header_height), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        header = f"{split_name.upper()} | {seq}/{img_name} | boxes={len(boxes)}"
        draw.text((16, 16), header, fill=(255, 255, 255))
        canvas.paste(original_panel, (0, header_height))
        canvas.paste(letterbox_panel, (512, header_height))
        canvas.paste(depth_panel, (1024, header_height))
        draw.text((24, header_height + 16), "Original + Boxes", fill=(255, 255, 255))
        draw.text((536, header_height + 16), "Letterbox Preview", fill=(255, 255, 255))
        draw.text(
            (1048, header_height + 16), "Depth / Placeholder", fill=(255, 255, 255)
        )

        output_path = output_dir / f"{split_name}_{seq}_{Path(img_name).stem}.png"
        canvas.save(output_path)
        saved.append(str(output_path))
    return saved


def raw_dataset_summary(cfg: Config) -> dict[str, object]:
    """统计原始训练/测试目录和 XML 目录状态。"""
    train_root = Path(cfg.RAW_DATA_DIR)
    test_root = (
        train_root.parents[1] / "DETRAC-test-data" / "Insight-MVT_Annotation_Test"
    )
    xml_root = Path(cfg.XML_DIR)

    train_sequence_dirs = sequence_dirs(train_root)
    test_sequence_dirs = sequence_dirs(test_root)
    xml_files = sorted(xml_root.glob("*.xml")) if xml_root.exists() else []

    train_image_count, train_suffixes = count_image_files(train_root)
    test_image_count, test_suffixes = count_image_files(test_root)

    train_sequences = {path.name for path in train_sequence_dirs}
    xml_sequences = {path.stem for path in xml_files}

    return {
        "train_root": str(train_root),
        "test_root": str(test_root),
        "xml_root": str(xml_root),
        "train_sequence_count": len(train_sequence_dirs),
        "test_sequence_count": len(test_sequence_dirs),
        "xml_file_count": len(xml_files),
        "train_image_count": train_image_count,
        "test_image_count": test_image_count,
        "train_image_extensions": train_suffixes,
        "test_image_extensions": test_suffixes,
        "missing_xml_for_train_sequences": sorted(train_sequences - xml_sequences)[:20],
        "orphan_xml_sequences": sorted(xml_sequences - train_sequences)[:20],
    }


def write_markdown_report(report: dict[str, object], path: Path) -> None:
    """将 JSON 报告转换为简洁 Markdown。"""
    raw = report["raw"]
    train = report["splits"]["train"]
    val = report["splits"]["val"]
    visualizations = report["visualizations"]

    lines = [
        "# Dataset Audit Report",
        "",
        "## Raw Dataset",
        "",
        f"- Train sequences: `{raw['train_sequence_count']}`",
        f"- Train images: `{raw['train_image_count']}`",
        f"- Test sequences: `{raw['test_sequence_count']}`",
        f"- Test images: `{raw['test_image_count']}`",
        f"- XML files: `{raw['xml_file_count']}`",
        "",
        "## Split Summary",
        "",
        f"- Train samples: `{train['sample_count']}`",
        f"- Train labeled samples: `{train['labels']['labeled_sample_count']}`",
        f"- Train depth coverage: `{train['depth_cache']['existing_count']}/{train['sample_count']}` ({train['depth_cache']['coverage_ratio']:.2%})",
        f"- Val samples: `{val['sample_count']}`",
        f"- Val labeled samples: `{val['labels']['labeled_sample_count']}`",
        f"- Val depth coverage: `{val['depth_cache']['existing_count']}/{val['sample_count']}` ({val['depth_cache']['coverage_ratio']:.2%})",
        "",
        "## Box Validity",
        "",
        f"- Train total boxes: `{train['boxes']['total_boxes']}`",
        f"- Train boxes requiring clipping: `{train['boxes']['requires_clipping']}`",
        f"- Val total boxes: `{val['boxes']['total_boxes']}`",
        f"- Val boxes requiring clipping: `{val['boxes']['requires_clipping']}`",
        "",
        "## XML Coverage",
        "",
        f"- Train missing frames in XML: `{train['xml']['missing_frame_in_xml_count']}`",
        f"- Val missing frames in XML: `{val['xml']['missing_frame_in_xml_count']}`",
        "",
        "## Visualizations",
        "",
    ]
    for rel_path in visualizations:
        lines.append(f"- `{rel_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the UA-DETRAC data pipeline.")
    parser.add_argument(
        "--config",
        default=None,
        help="可选配置文件路径（.json/.yaml/.yml）。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="报告输出目录，默认写入 outputs/Data_Audit。",
    )
    parser.add_argument(
        "--visualizations-per-split",
        type=int,
        default=3,
        help="每个 split 输出多少张可视化样例图。",
    )
    parser.add_argument(
        "--image-scan-limit",
        type=int,
        default=512,
        help="默认模式下每个 split 抽查多少张图像可读性。",
    )
    parser.add_argument(
        "--full-image-scan",
        action="store_true",
        help="对全部 train/val 图像执行可读性检查。",
    )
    args = parser.parse_args()

    cfg = Config(config_path=args.config)
    output_root = ensure_dir(
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "Data_Audit"
    )
    visualization_root = ensure_dir(output_root / "visualizations")

    print("== Building Datasets ==")
    train_dataset = dataset_for_split(cfg, is_train=True)
    val_dataset = dataset_for_split(cfg, is_train=False)
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")

    print("\n== Analyzing Raw Dataset ==")
    raw = raw_dataset_summary(cfg)
    print(
        f"raw train sequences={raw['train_sequence_count']}, "
        f"raw train images={raw['train_image_count']}, "
        f"xml={raw['xml_file_count']}"
    )

    print("\n== Analyzing Train Split ==")
    train_report = analyze_split(
        "train",
        train_dataset,
        Path(cfg.XML_DIR),
        image_scan_limit=args.image_scan_limit,
        full_image_scan=args.full_image_scan,
    )
    print(
        f"train labeled={train_report['labels']['labeled_sample_count']}, "
        f"depth coverage={train_report['depth_cache']['existing_count']}/{train_report['sample_count']}"
    )

    print("\n== Analyzing Validation Split ==")
    val_report = analyze_split(
        "val",
        val_dataset,
        Path(cfg.XML_DIR),
        image_scan_limit=args.image_scan_limit,
        full_image_scan=args.full_image_scan,
    )
    print(
        f"val labeled={val_report['labels']['labeled_sample_count']}, "
        f"depth coverage={val_report['depth_cache']['existing_count']}/{val_report['sample_count']}"
    )

    print("\n== Generating Visualizations ==")
    visualization_paths: list[str] = []
    for split_name, dataset in [("train", train_dataset), ("val", val_dataset)]:
        saved = save_visualizations(
            cfg,
            split_name,
            dataset,
            visualization_root,
            args.visualizations_per_split,
        )
        visualization_paths.extend(
            [str(Path(path).relative_to(output_root)) for path in saved]
        )
        print(f"{split_name} visualizations: {len(saved)}")

    report = {
        "config": {
            "raw_data_dir": cfg.RAW_DATA_DIR,
            "xml_dir": cfg.XML_DIR,
            "depth_cache_dir": cfg.DEPTH_CACHE_DIR,
            "output_dir": str(output_root),
            "img_size": cfg.IMG_SIZE,
            "frame_stride": cfg.FRAME_STRIDE,
            "train_ratio": cfg.TRAIN_RATIO,
            "seed": cfg.SEED,
            "device": cfg.DEVICE,
        },
        "raw": raw,
        "splits": {
            "train": train_report,
            "val": val_report,
        },
        "visualizations": visualization_paths,
    }

    json_path = output_root / "dataset_audit_report.json"
    md_path = output_root / "dataset_audit_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(report, md_path)

    print("\n== Outputs ==")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"Visualizations: {visualization_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
