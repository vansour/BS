#!/usr/bin/env python3
"""
基于人工复核包构建第一版 bootstrap hard negative 集。

注意：
- 该脚本不会假装替代人工定稿；
- 它只把“视觉上明显不像车”的高置信误检 patch 收口成第一版负样本；
- 其余样本继续保留在人工复核清单中，等待人工确认。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_CSV = (
    ROOT
    / "outputs"
    / "Static_False_Positive_Review_temporal_smoke"
    / "review_checklist.csv"
)
DEFAULT_PATCH_ROOT = ROOT / "outputs" / "Static_False_Positive_Mine_temporal_smoke"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "Hard_Negatives_static_fp_v1"

# 这是一份“极保守”的首轮收口名单。
# 只包含当前已人工视觉抽查后，可以高置信判断为非车辆的 patch。
DEFAULT_ALLOWLIST_REVIEW_IDS = {3, 4, 8, 9}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a conservative bootstrap hard-negative dataset from review artifacts."
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
        help="hard negative 数据集输出目录。",
    )
    parser.add_argument(
        "--review-ids",
        default="",
        help="逗号分隔的 review_id 白名单；为空时使用脚本内置首轮保守名单。",
    )
    return parser.parse_args()


def parse_review_ids(value: str) -> set[int]:
    if not value.strip():
        return set(DEFAULT_ALLOWLIST_REVIEW_IDS)
    result = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        result.add(int(part))
    return result


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_markdown(path: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "# Bootstrap Hard Negative Set v1",
        "",
        "## 说明",
        "",
        "- 本集合只包含首轮人工视觉抽查后可高置信判定为误检的 patch。",
        "- 这是 bootstrap 负样本，不等同于最终人工定稿全集。",
        "- 其目的在于先打通 hard negative 回灌链路，而不是一次性收满所有误检样本。",
        "",
        "## 统计",
        "",
        f"- 总样本数：`{summary['total_samples']}`",
        f"- 来源路线统计：`{summary['by_route']}`",
        f"- 来源视频统计：`{summary['by_video']}`",
        "",
        "## 样本清单",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"`review_id={row['review_id']}` / "
            f"`{row['route']}` / "
            f"`{row['video_label']}` / "
            f"`track={row['track_id']}` / "
            f"`frame={row['frame_index']}` / "
            f"`{row['negative_path']}`"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    review_csv = Path(args.review_csv).resolve()
    patch_root = Path(args.patch_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not review_csv.exists():
        raise FileNotFoundError(f"Review CSV not found: {review_csv}")
    if not patch_root.exists():
        raise FileNotFoundError(f"Patch root not found: {patch_root}")

    allowlist = parse_review_ids(args.review_ids)
    rows = load_rows(review_csv)

    selected = []
    by_route: dict[str, int] = {}
    by_video: dict[str, int] = {}
    negatives_dir = output_dir / "images"
    negatives_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        review_id = int(row["review_id"])
        if review_id not in allowlist:
            continue
        src_path = patch_root / row["crop_path"]
        if not src_path.exists():
            raise FileNotFoundError(f"Patch file not found: {src_path}")

        route_name = row["route"]
        video_label = row["video_label"]
        dst_dir = negatives_dir / route_name / video_label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_name = (
            f"review_{review_id:03d}_track_{int(row['track_id']):04d}_"
            f"frame_{int(row['frame_index']):06d}{src_path.suffix.lower()}"
        )
        dst_path = dst_dir / dst_name
        shutil.copy2(src_path, dst_path)

        exported = dict(row)
        exported["assistant_prelabel"] = "false_positive"
        exported["assistant_confidence"] = "high"
        exported["selection_basis"] = "visual_obvious_non_vehicle"
        exported["negative_path"] = str(dst_path.relative_to(output_dir))
        selected.append(exported)

        by_route[route_name] = by_route.get(route_name, 0) + 1
        by_video[video_label] = by_video.get(video_label, 0) + 1

    manifest_path = output_dir / "bootstrap_hard_negative_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "review_csv": str(review_csv),
        "patch_root": str(patch_root),
        "allowlist_review_ids": sorted(allowlist),
        "total_samples": len(selected),
        "by_route": by_route,
        "by_video": by_video,
        "manifest_jsonl": str(manifest_path),
    }
    (output_dir / "bootstrap_hard_negative_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(
        output_dir / "bootstrap_hard_negative_summary.md",
        selected,
        summary,
    )

    print(f"Bootstrap negatives: {len(selected)}")
    print(f"Manifest JSONL: {manifest_path}")
    print(f"Summary JSON: {output_dir / 'bootstrap_hard_negative_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
