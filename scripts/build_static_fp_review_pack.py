#!/usr/bin/env python3
"""
构建静止假车人工复核包。

输入：
- `scripts/mine_static_false_positives.py` 生成的 `static_false_positive_manifest.jsonl`

输出：
- `review_checklist.csv`
- `review_checklist.md`
- 若干联系图 `contact_sheet_*.jpg`

目标：
- 把 hard negative 候选 patch 整理成一份“可快速人工复核”的材料包；
- 给每个 patch 附带启发式初判与复核建议；
- 为后续 hard negative 回灌建立稳定的人审入口。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "outputs"
    / "Static_False_Positive_Mine_temporal_smoke"
    / "static_false_positive_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "Static_False_Positive_Review_temporal_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manual review pack from mined static false-positive patches."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="`static_false_positive_manifest.jsonl` 路径。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="人工复核包输出目录。",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=220,
        help="联系图中单张 patch 的显示宽度。",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="联系图列数。",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def heuristic_review_decision(row: dict) -> tuple[str, str]:
    """
    给 patch 一个启发式初判与人工关注建议。

    这不是最终标签，只是为了让人工先从最可能的真误检开始审。
    """
    appearance_prob = row.get("appearance_vehicle_prob")
    appearance_prob = (
        float(appearance_prob) if appearance_prob is not None else None
    )
    conf = float(row.get("conf", 0.0))
    road_score = float(row.get("road_score", 0.0))
    motion = float(row.get("motion_intensity", 0.0))
    reason = str(row.get("suppression_reason", "") or "")

    if (
        appearance_prob is not None
        and appearance_prob < 0.02
        and conf < 0.45
        and motion < 0.03
    ):
        return "likely_false_positive", "优先复核"
    if "appearance_non_vehicle" in reason and road_score < 0.20:
        return "likely_false_positive", "优先复核"
    if conf >= 0.45 and road_score >= 0.25:
        return "needs_careful_review", "可能误伤真车"
    return "review_required", "常规复核"


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            heuristic_review_decision(row)[0] != "likely_false_positive",
            heuristic_review_decision(row)[0] == "needs_careful_review",
            -float(row.get("conf", 0.0)),
            str(row.get("route", "")),
            str(row.get("video_label", "")),
        ),
    )


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "review_id",
        "route",
        "video_label",
        "track_id",
        "frame_index",
        "conf",
        "motion_intensity",
        "center_shift",
        "road_score",
        "appearance_vehicle_prob",
        "suppression_reason",
        "persistent_static_candidate",
        "crop_path",
        "heuristic_decision",
        "review_priority",
        "manual_label",
        "final_decision",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review_id, row in enumerate(rows, start=1):
            heuristic_decision, review_priority = heuristic_review_decision(row)
            writer.writerow(
                {
                    "review_id": review_id,
                    "route": row["route"],
                    "video_label": row["video_label"],
                    "track_id": row["track_id"],
                    "frame_index": row["frame_index"],
                    "conf": round(float(row.get("conf", 0.0)), 6),
                    "motion_intensity": round(
                        float(row.get("motion_intensity", 0.0)), 6
                    ),
                    "center_shift": round(float(row.get("center_shift", 0.0)), 6),
                    "road_score": round(float(row.get("road_score", 0.0)), 6),
                    "appearance_vehicle_prob": (
                        ""
                        if row.get("appearance_vehicle_prob") is None
                        else round(float(row["appearance_vehicle_prob"]), 6)
                    ),
                    "suppression_reason": row.get("suppression_reason", ""),
                    "persistent_static_candidate": bool(
                        row.get("persistent_static_candidate", False)
                    ),
                    "crop_path": row["crop_path"],
                    "heuristic_decision": heuristic_decision,
                    "review_priority": review_priority,
                    "manual_label": "",
                    "final_decision": "",
                    "notes": "",
                }
            )


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = []
    current_len = 0
    for word in words:
        if current_len + len(word) + len(current) > width and current:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word)
    if current:
        lines.append(" ".join(current))
    return lines


def make_contact_sheet(
    rows: list[dict],
    *,
    root_dir: Path,
    output_path: Path,
    thumb_width: int,
    columns: int,
) -> None:
    if not rows:
        return

    prepared = []
    max_thumb_height = 0
    for review_id, row in enumerate(rows, start=1):
        image_path = root_dir / row["crop_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        src_h, src_w = image.shape[:2]
        scale = thumb_width / max(src_w, 1)
        thumb_height = max(1, int(round(src_h * scale)))
        thumbnail = cv2.resize(image, (thumb_width, thumb_height))
        max_thumb_height = max(max_thumb_height, thumb_height)
        heuristic_decision, review_priority = heuristic_review_decision(row)
        prepared.append(
            {
                "review_id": review_id,
                "row": row,
                "thumbnail": thumbnail,
                "thumb_height": thumb_height,
                "heuristic_decision": heuristic_decision,
                "review_priority": review_priority,
            }
        )

    if not prepared:
        return

    columns = max(1, int(columns))
    cell_height = max_thumb_height + 92
    margin = 18
    header_height = 64
    rows_count = int(math.ceil(len(prepared) / columns))
    canvas_w = columns * (thumb_width + margin) + margin
    canvas_h = header_height + rows_count * (cell_height + margin) + margin
    canvas = np.full((canvas_h, canvas_w, 3), 248, dtype=np.uint8)

    title = output_path.stem.replace("_", " ")
    cv2.putText(
        canvas,
        title[:70],
        (margin, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (22, 32, 44),
        2,
    )
    cv2.putText(
        canvas,
        f"patches={len(prepared)}",
        (margin, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (74, 92, 108),
        1,
    )

    for index, item in enumerate(prepared):
        row_idx = index // columns
        col_idx = index % columns
        x = margin + col_idx * (thumb_width + margin)
        y = header_height + row_idx * (cell_height + margin)

        thumb = item["thumbnail"]
        thumb_h = item["thumb_height"]
        canvas[y:y + thumb_h, x:x + thumb_width] = thumb
        cv2.rectangle(
            canvas,
            (x, y),
            (x + thumb_width, y + thumb_h),
            (180, 188, 196),
            1,
        )

        row = item["row"]
        decision_color = (
            (50, 115, 60)
            if item["heuristic_decision"] == "likely_false_positive"
            else (0, 105, 170)
            if item["heuristic_decision"] == "needs_careful_review"
            else (95, 75, 45)
        )
        meta_lines = [
            f"#{item['review_id']} {row['route']} | {row['video_label']}",
            f"track={row['track_id']} frame={row['frame_index']} conf={float(row.get('conf', 0.0)):.2f}",
            f"reason={row.get('suppression_reason', '') or 'candidate_only'}",
            f"decision={item['heuristic_decision']}",
        ]
        for extra in wrap_text(meta_lines[2], 34)[1:]:
            meta_lines.insert(3, extra)

        text_y = y + max_thumb_height + 18
        for line_index, line in enumerate(meta_lines[:5]):
            cv2.putText(
                canvas,
                line[:48],
                (x, text_y + line_index * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                decision_color if line_index == 3 else (36, 48, 60),
                1,
            )

    cv2.imwrite(str(output_path), canvas)


def write_markdown(rows: list[dict], output_path: Path) -> None:
    lines = [
        "# 静止假车人工复核清单",
        "",
        f"- 总 patch 数：`{len(rows)}`",
        "",
        "## 复核说明",
        "",
        "- `manual_label` 建议填写：`true_vehicle` / `false_positive` / `uncertain`",
        "- `final_decision` 建议填写：`keep_suppressed` / `restore_vehicle` / `need_more_data`",
        "- 优先从 `heuristic_decision = likely_false_positive` 的样本开始看。",
        "",
        "## 样本摘要",
        "",
    ]

    by_route: dict[str, int] = defaultdict(int)
    by_decision: dict[str, int] = defaultdict(int)
    for row in rows:
        by_route[str(row["route"])] += 1
        by_decision[heuristic_review_decision(row)[0]] += 1

    for route_name, count in sorted(by_route.items()):
        lines.append(f"- 路线 `{route_name}`：`{count}`")
    lines.append("")
    for decision, count in sorted(by_decision.items()):
        lines.append(f"- 初判 `{decision}`：`{count}`")
    lines.append("")
    lines.append("## 联系图")
    lines.append("")
    lines.append("- `contact_sheet_all.jpg`")
    lines.append("- `contact_sheet_route_unified.jpg`")
    lines.append("- `contact_sheet_route_hybrid.jpg`")
    lines.append("")
    lines.append("## 前 20 条复核项")
    lines.append("")

    for review_id, row in enumerate(rows[:20], start=1):
        heuristic_decision, review_priority = heuristic_review_decision(row)
        lines.append(
            f"- `#{review_id}` `{row['route']}` / `{row['video_label']}` / "
            f"`track={row['track_id']}` / `frame={row['frame_index']}` / "
            f"`{heuristic_decision}` / `{review_priority}` / `{row['crop_path']}`"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    root_dir = manifest_path.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = sort_rows(load_manifest(manifest_path))

    write_csv(rows, output_dir / "review_checklist.csv")
    write_markdown(rows, output_dir / "review_checklist.md")

    make_contact_sheet(
        rows,
        root_dir=root_dir,
        output_path=output_dir / "contact_sheet_all.jpg",
        thumb_width=max(80, int(args.thumb_width)),
        columns=max(1, int(args.columns)),
    )
    for route_name in ("unified", "hybrid"):
        route_rows = [row for row in rows if row["route"] == route_name]
        make_contact_sheet(
            route_rows,
            root_dir=root_dir,
            output_path=output_dir / f"contact_sheet_route_{route_name}.jpg",
            thumb_width=max(80, int(args.thumb_width)),
            columns=max(1, int(args.columns)),
        )

    summary = {
        "manifest": str(manifest_path),
        "total_rows": len(rows),
        "review_csv": str(output_dir / "review_checklist.csv"),
        "review_markdown": str(output_dir / "review_checklist.md"),
        "contact_sheets": [
            str(output_dir / "contact_sheet_all.jpg"),
            str(output_dir / "contact_sheet_route_unified.jpg"),
            str(output_dir / "contact_sheet_route_hybrid.jpg"),
        ],
    }
    (output_dir / "review_pack_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Review rows: {len(rows)}")
    print(f"Review CSV: {output_dir / 'review_checklist.csv'}")
    print(f"Review Markdown: {output_dir / 'review_checklist.md'}")
    print(f"Contact sheets: {summary['contact_sheets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
