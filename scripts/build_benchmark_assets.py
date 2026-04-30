#!/usr/bin/env python3
"""
Build runnable benchmark video assets from local frame sequences.

Current primary use:
- generate benchmark_v1 clear-weather control clips from UA-DETRAC sequences
- write a compact asset build report for later experiment tracking
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_CONFIG = ROOT / "configs" / "benchmark_videos.json"
RAW_SEQUENCE_ROOT = (
    ROOT
    / "data"
    / "UA-DETRAC"
    / "DETRAC-train-data"
    / "Insight-MVT_Annotation_Train"
)
XML_ROOT = ROOT / "data" / "UA-DETRAC" / "DETRAC-Train-Annotations-XML"
DEFAULT_REPORT_DIR = ROOT / "outputs" / "Benchmark_Assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark video assets from local frame sequences."
    )
    parser.add_argument(
        "--benchmark-config",
        default=str(DEFAULT_BENCHMARK_CONFIG),
        help="Benchmark 配置文件路径。",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="资产构建报告输出目录。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标视频已存在则覆盖重建。",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def video_entries_to_build(payload: dict) -> list[dict]:
    entries = []
    for entry in payload.get("videos", []):
        if not isinstance(entry, dict):
            continue
        if not entry.get("source_sequence"):
            continue
        if str(entry.get("status", "active")).strip().lower() != "active":
            continue
        entries.append(entry)
    return entries


def frame_paths_for_entry(entry: dict) -> list[Path]:
    sequence_name = str(entry["source_sequence"])
    start_frame = int(entry.get("clip_start_frame", 1))
    clip_num_frames = int(entry.get("clip_num_frames", 300))

    sequence_dir = RAW_SEQUENCE_ROOT / sequence_name
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")

    all_frames = sorted(
        [path for path in sequence_dir.iterdir() if path.suffix.lower() in {".jpg", ".png"}]
    )
    if not all_frames:
        raise RuntimeError(f"No image frames were found in sequence: {sequence_name}")

    start_index = max(0, start_frame - 1)
    selected = all_frames[start_index:start_index + clip_num_frames]
    if not selected:
        raise RuntimeError(
            f"No frames selected for {sequence_name}: start_frame={start_frame}, "
            f"clip_num_frames={clip_num_frames}"
        )
    return selected


def create_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    return writer


def parse_box_counts(sequence_name: str, frame_numbers: list[int]) -> dict[str, float]:
    xml_path = XML_ROOT / f"{sequence_name}.xml"
    if not xml_path.exists():
        return {"mean_vehicle_count": 0.0, "peak_vehicle_count": 0}

    frame_number_set = set(frame_numbers)
    counts: list[int] = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for frame in root.findall("frame"):
        number = int(frame.get("num", "0"))
        if number not in frame_number_set:
            continue
        target_list = frame.find("target_list")
        counts.append(len(target_list.findall("target")) if target_list is not None else 0)

    if not counts:
        return {"mean_vehicle_count": 0.0, "peak_vehicle_count": 0}
    return {
        "mean_vehicle_count": round(sum(counts) / len(counts), 4),
        "peak_vehicle_count": max(counts),
    }


def build_entry_asset(entry: dict, *, overwrite: bool) -> dict:
    output_path = resolve_path(str(entry["path"]))
    frame_paths = frame_paths_for_entry(entry)
    frame_numbers = [
        int(path.stem.replace("img", "")) for path in frame_paths if path.stem.startswith("img")
    ]

    if output_path.exists() and not overwrite:
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise RuntimeError(f"Failed to read first frame for metadata: {frame_paths[0]}")
        height, width = first_frame.shape[:2]
        stats = parse_box_counts(str(entry["source_sequence"]), frame_numbers)
        return {
            "label": entry["label"],
            "path": str(output_path),
            "sequence": str(entry["source_sequence"]),
            "frames_written": len(frame_paths),
            "fps": float(entry.get("fps", 25)),
            "frame_size": [width, height],
            "skipped_existing": True,
            **stats,
        }

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    writer = create_video_writer(output_path, float(entry.get("fps", 25)), (width, height))
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()

    stats = parse_box_counts(str(entry["source_sequence"]), frame_numbers)
    return {
        "label": entry["label"],
        "path": str(output_path),
        "sequence": str(entry["source_sequence"]),
        "frames_written": len(frame_paths),
        "fps": float(entry.get("fps", 25)),
        "frame_size": [width, height],
        "skipped_existing": False,
        **stats,
    }


def write_markdown_report(
    benchmark_id: str,
    assets: list[dict],
    output_path: Path,
):
    lines = [
        "# Benchmark Asset Build Report",
        "",
        f"- Benchmark ID: `{benchmark_id or 'N/A'}`",
        f"- Assets built: `{len(assets)}`",
        "",
    ]
    for asset in assets:
        lines.extend(
            [
                f"## {asset['label']}",
                "",
                f"- Output path: `{asset['path']}`",
                f"- Source sequence: `{asset['sequence']}`",
                f"- Frames written: `{asset['frames_written']}`",
                f"- FPS: `{asset['fps']}`",
                f"- Frame size: `{asset['frame_size']}`",
                f"- Mean vehicle count: `{asset['mean_vehicle_count']}`",
                f"- Peak vehicle count: `{asset['peak_vehicle_count']}`",
                f"- Skipped existing: `{asset['skipped_existing']}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    benchmark_config = resolve_path(args.benchmark_config)
    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(benchmark_config)
    benchmark_id = str(payload.get("benchmark_id", "") or "").strip()
    entries = video_entries_to_build(payload)
    if not entries:
        raise RuntimeError("No active benchmark entries with source_sequence were found.")

    built_assets = [
        build_entry_asset(entry, overwrite=args.overwrite)
        for entry in entries
    ]

    report_json = report_dir / "benchmark_asset_build_report.json"
    report_md = report_dir / "benchmark_asset_build_report.md"
    report_payload = {
        "benchmark_id": benchmark_id,
        "asset_count": len(built_assets),
        "assets": built_assets,
    }
    report_json.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(benchmark_id, built_assets, report_md)

    print(f"Benchmark asset build report JSON: {report_json}")
    print(f"Benchmark asset build report Markdown: {report_md}")
    for asset in built_assets:
        print(
            f"Built {asset['label']}: path={asset['path']}, "
            f"frames={asset['frames_written']}, skipped_existing={asset['skipped_existing']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
