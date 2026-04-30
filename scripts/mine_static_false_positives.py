#!/usr/bin/env python3
"""
从路线评估结果中挖掘静止假车 hard negatives。

当前脚本依赖 `scripts/evaluate_inference_routes.py` 已经输出的轨迹日志：
- `unified_track_log.jsonl`
- `hybrid_track_log.jsonl`

主要用途：
1. 从被时序过滤器判为静止可疑/已抑制的轨迹中导出 patch；
2. 为后续 hard negative 回灌准备样本；
3. 给人工快速复查“到底抑制了哪些目标”提供直接素材。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTE_EVAL_SUMMARY = ROOT / "outputs" / "Route_Eval" / "route_eval_summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "Static_False_Positive_Mine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine static false-positive patches from route-eval track logs."
    )
    parser.add_argument(
        "--route-eval-summary",
        default=str(DEFAULT_ROUTE_EVAL_SUMMARY),
        help="route_eval_summary.json 路径。",
    )
    parser.add_argument(
        "--route",
        choices=["unified", "hybrid", "both"],
        default="both",
        help="导出哪条路线的静止误检 patch。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="patch 导出目录。",
    )
    parser.add_argument(
        "--only-suppressed",
        action="store_true",
        help="只导出已被抑制的静止轨迹；不加时则也导出 persistent_static_candidate。",
    )
    parser.add_argument(
        "--max-crops-per-video",
        type=int,
        default=40,
        help="每个视频每条路线最多导出多少个 patch。",
    )
    parser.add_argument(
        "--context-pad",
        type=float,
        default=0.10,
        help="在检测框周围额外保留的上下文比例。",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def clip_with_context(
    bbox: list[float],
    frame_shape: tuple[int, int],
    context_pad: float,
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * max(0.0, context_pad)
    pad_y = height * max(0.0, context_pad)
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(frame_w, int(round(x2 + pad_x)))
    y2 = min(frame_h, int(round(y2 + pad_y)))
    if x2 <= x1:
        x2 = min(frame_w, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_h, y1 + 1)
    return x1, y1, x2, y2


def read_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(
                f"Failed to read frame {frame_index} from video: {video_path}"
            )
        return frame
    finally:
        cap.release()


def select_track_candidates(
    events: list[dict],
    *,
    only_suppressed: bool,
) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        grouped[int(event["track_id"])].append(event)

    selected = []
    for track_id, track_events in grouped.items():
        chosen = None
        for event in reversed(track_events):
            suppressed = bool(event.get("suppression_reason"))
            persistent = bool(event.get("persistent_static_candidate"))
            if only_suppressed:
                if suppressed:
                    chosen = event
                    break
            else:
                if suppressed or persistent:
                    chosen = event
                    break
        if chosen is None:
            continue

        chosen = dict(chosen)
        chosen["track_id"] = track_id
        selected.append(chosen)

    selected.sort(
        key=lambda item: (
            bool(item.get("suppression_reason")),
            bool(item.get("persistent_static_candidate")),
            float(item.get("conf", 0.0)),
            -int(item.get("frame_index", 0)),
        ),
        reverse=True,
    )
    return selected


def export_route_patches(
    *,
    route_name: str,
    video_summary: dict,
    route_eval_root: Path,
    output_dir: Path,
    only_suppressed: bool,
    max_crops_per_video: int,
    context_pad: float,
) -> list[dict]:
    video_path = Path(video_summary["video_path"])
    artifact_key = f"{route_name}_track_log_jsonl"
    artifact_path = route_eval_root / video_summary["artifacts"][artifact_key]
    events = load_jsonl(artifact_path)
    candidates = select_track_candidates(
        events,
        only_suppressed=only_suppressed,
    )[: max(0, max_crops_per_video)]
    if not candidates:
        return []

    route_output_dir = output_dir / route_name / video_summary["benchmark"]["label"]
    route_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    frame_cache: dict[int, object] = {}
    for rank, event in enumerate(candidates, start=1):
        frame_index = int(event["frame_index"])
        if frame_index not in frame_cache:
            frame_cache[frame_index] = read_frame(video_path, frame_index)
        frame = frame_cache[frame_index]
        x1, y1, x2, y2 = clip_with_context(
            list(event["bbox"]),
            frame.shape[:2],
            context_pad=context_pad,
        )
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        output_path = route_output_dir / (
            f"{rank:03d}_track_{int(event['track_id']):04d}_"
            f"frame_{frame_index:06d}.jpg"
        )
        cv2.imwrite(str(output_path), patch)
        manifest_rows.append(
            {
                "route": route_name,
                "video_label": video_summary["benchmark"]["label"],
                "video_path": str(video_path),
                "track_id": int(event["track_id"]),
                "frame_index": frame_index,
                "suppression_reason": str(event.get("suppression_reason", "")),
                "persistent_static_candidate": bool(
                    event.get("persistent_static_candidate", False)
                ),
                "conf": float(event.get("conf", 0.0)),
                "motion_intensity": float(event.get("motion_intensity", 0.0)),
                "center_shift": float(event.get("center_shift", 0.0)),
                "road_score": float(event.get("road_score", 0.0)),
                "appearance_vehicle_prob": event.get("appearance_vehicle_prob"),
                "crop_path": str(output_path.relative_to(output_dir)),
            }
        )
    return manifest_rows


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Static False-Positive Mining Summary",
        "",
        f"- Total exported patches: `{len(rows)}`",
        "",
    ]
    by_route: dict[str, int] = defaultdict(int)
    by_video: dict[str, int] = defaultdict(int)
    for row in rows:
        by_route[str(row["route"])] += 1
        by_video[str(row["video_label"])] += 1

    lines.append("## By Route")
    lines.append("")
    for route_name, count in sorted(by_route.items()):
        lines.append(f"- `{route_name}`: `{count}`")
    lines.append("")
    lines.append("## By Video")
    lines.append("")
    for video_label, count in sorted(by_video.items()):
        lines.append(f"- `{video_label}`: `{count}`")
    lines.append("")
    lines.append("## Sample Entries")
    lines.append("")
    for row in rows[:20]:
        lines.append(
            "- "
            f"`{row['route']}` / `{row['video_label']}` / "
            f"`track={row['track_id']}` / "
            f"`frame={row['frame_index']}` / "
            f"`reason={row['suppression_reason'] or 'candidate_only'}` / "
            f"`{row['crop_path']}`"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    summary_path = Path(args.route_eval_summary).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Route eval summary not found: {summary_path}")

    route_eval_summary = load_json(summary_path)
    route_eval_root = summary_path.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    routes = ["unified", "hybrid"] if args.route == "both" else [args.route]
    manifest_rows: list[dict] = []
    for video_summary in route_eval_summary.get("videos", []):
        for route_name in routes:
            manifest_rows.extend(
                export_route_patches(
                    route_name=route_name,
                    video_summary=video_summary,
                    route_eval_root=route_eval_root,
                    output_dir=output_dir,
                    only_suppressed=bool(args.only_suppressed),
                    max_crops_per_video=max(0, args.max_crops_per_video),
                    context_pad=max(0.0, float(args.context_pad)),
                )
            )

    manifest_path = output_dir / "static_false_positive_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_json = output_dir / "static_false_positive_manifest_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "route_eval_summary": str(summary_path),
                "routes": routes,
                "only_suppressed": bool(args.only_suppressed),
                "total_patches": len(manifest_rows),
                "manifest_jsonl": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(
        output_dir / "static_false_positive_manifest_summary.md",
        manifest_rows,
    )

    print(f"Exported patches: {len(manifest_rows)}")
    print(f"Manifest JSONL: {manifest_path}")
    print(f"Summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
