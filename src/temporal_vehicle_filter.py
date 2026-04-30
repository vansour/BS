#!/usr/bin/env python3
"""
车辆检测时序过滤器
Temporal Vehicle Filter

该模块用于把“逐帧车辆检测结果”升级为“轨迹级车辆确认结果”，主要解决：
1. 静止背景物体在多帧中被持续误检为车；
2. 逐帧检测抖动导致的短暂误检和不稳定显示；
3. 统一推理路线、混合推理路线和 benchmark 评估逻辑不一致。

设计思路：
- 先对跨帧检测框进行轻量级 IoU 关联，形成轨迹；
- 再对每条轨迹统计位移、尺度变化、局部运动、置信度和可选外观复核结果；
- 最后把轨迹划分为 `tentative / confirmed / suspicious / suppressed`。

这里不追求完整 MOT，而是实现一套足够轻量、可解释、可配置的工程过滤层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch


def _iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """计算两个 `xyxy` 框的 IoU。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def _clip_bbox_to_frame(
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """把浮点框裁剪到图像有效范围并转为整数。"""
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    x1 = int(round(max(0.0, min(float(bbox[0]), frame_w - 1))))
    y1 = int(round(max(0.0, min(float(bbox[1]), frame_h - 1))))
    x2 = int(round(max(0.0, min(float(bbox[2]), frame_w))))
    y2 = int(round(max(0.0, min(float(bbox[3]), frame_h))))
    if x2 <= x1:
        x2 = min(frame_w, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_h, y1 + 1)
    return x1, y1, x2, y2


def _bbox_center_and_area(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """返回框中心点与面积。"""
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return center_x, center_y, width * height


def _road_score(
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, int],
) -> float:
    """
    基于几何先验估计当前框落在道路区域的相对可信度。

    这里只提供一个软先验，不直接代替检测逻辑：
    - 越靠近图像下半部分，越接近道路区域；
    - 越偏离横向中心区域，先验分数越低；
    """
    frame_h, frame_w = max(int(frame_shape[0]), 1), max(int(frame_shape[1]), 1)
    center_x, center_y, _ = _bbox_center_and_area(bbox)
    norm_x = center_x / frame_w
    norm_y = center_y / frame_h

    vertical = float(np.clip((norm_y - 0.12) / 0.60, 0.0, 1.0))
    horizontal = 1.0 - float(np.clip(abs(norm_x - 0.5) / 0.55, 0.0, 1.0))
    return float(np.clip(vertical * horizontal, 0.0, 1.0))


class ImageNetVehiclePatchVerifier:
    """
    使用轻量 ImageNet 预训练分类器做二次外观复核。

    注意：
    - 该分类器不是为本项目专门训练的车辆/非车辆模型；
    - 它的作用是给“长时间静止且可疑”的轨迹再提供一个外观侧信号，
      只在过滤链路的后半段触发；
    - 若初始化失败，会自动降级为关闭状态，而不阻塞主推理流程。
    """

    VEHICLE_KEYWORDS = (
        "ambulance",
        "cab",
        "car",
        "convertible",
        "jeep",
        "limousine",
        "minibus",
        "minivan",
        "model t",
        "motor scooter",
        "motorcycle",
        "pickup",
        "police van",
        "racer",
        "racing car",
        "sports car",
        "station wagon",
        "tow truck",
        "trailer truck",
        "truck",
        "van",
    )

    def __init__(
        self,
        *,
        enabled: bool,
        model_name: str,
        device: str,
        custom_weights: str | None = None,
    ):
        self.enabled = bool(enabled)
        self.model_name = str(model_name or "mobilenet_v3_small").strip()
        self.device = (
            "cuda"
            if device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        self.custom_weights = custom_weights
        self._model = None
        self._preprocess = None
        self._vehicle_indices: list[int] = []
        self._mode = "imagenet_multi"
        self._binary_vehicle_index = 1
        self._init_error: str | None = None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def _ensure_loaded(self) -> bool:
        if not self.enabled:
            return False
        if self._model is not None and self._preprocess is not None:
            return True
        if self._init_error is not None:
            return False

        try:
            from torchvision.models import (
                MobileNet_V3_Small_Weights,
                mobilenet_v3_small,
            )
        except Exception as exc:  # pragma: no cover - import failure is environment-dependent
            self._init_error = f"torchvision import failed: {exc}"
            self.enabled = False
            return False

        if self.model_name not in {"mobilenet_v3_small", ""}:
            self._init_error = (
                f"Unsupported second-stage model {self.model_name!r}. "
                "Currently only 'mobilenet_v3_small' is supported."
            )
            self.enabled = False
            return False

        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            payload = None
            if self.custom_weights:
                payload = torch.load(
                    self.custom_weights,
                    map_location="cpu",
                    weights_only=False,
                )

            if (
                isinstance(payload, dict)
                and payload.get("classifier_type") == "binary_vehicle_classifier"
            ):
                model = mobilenet_v3_small(weights=weights)
                in_features = int(model.classifier[-1].in_features)
                model.classifier[-1] = torch.nn.Linear(in_features, 2)
                state_dict = payload.get("model_state_dict", {})
                model.load_state_dict(state_dict, strict=True)
                self._mode = "binary_vehicle_classifier"
                class_names = payload.get("class_names", ["non_vehicle", "vehicle"])
                if (
                    isinstance(class_names, list)
                    and "vehicle" in class_names
                ):
                    self._binary_vehicle_index = int(class_names.index("vehicle"))
                else:
                    self._binary_vehicle_index = 1
                self._vehicle_indices = []
            else:
                model = mobilenet_v3_small(weights=weights)
                categories = [
                    str(item).strip().lower()
                    for item in weights.meta.get("categories", [])
                ]
                vehicle_indices = [
                    index
                    for index, name in enumerate(categories)
                    if any(keyword in name for keyword in self.VEHICLE_KEYWORDS)
                ]
                if not vehicle_indices:
                    raise RuntimeError("No ImageNet vehicle class indices were detected.")
                if payload is not None:
                    state_dict = (
                        payload["model_state_dict"]
                        if isinstance(payload, dict) and "model_state_dict" in payload
                        else payload
                    )
                    model.load_state_dict(state_dict, strict=False)
                self._mode = "imagenet_multi"
                self._vehicle_indices = vehicle_indices

            model.eval().to(self.device)
            self._model = model
            self._preprocess = weights.transforms()
            return True
        except Exception as exc:  # pragma: no cover - model download/init depends on env
            self._init_error = f"second-stage classifier init failed: {exc}"
            self.enabled = False
            return False

    def score_patch(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> float | None:
        """返回当前 patch 属于车辆类的概率和。"""
        if not self._ensure_loaded():
            return None

        x1, y1, x2, y2 = _clip_bbox_to_frame(bbox, frame_bgr.shape[:2])
        patch = frame_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        try:
            from PIL import Image
        except Exception:  # pragma: no cover - Pillow is expected in runtime deps
            return None

        rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_patch)
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        if self._mode == "binary_vehicle_classifier":
            vehicle_prob = float(probs[self._binary_vehicle_index].item())
        else:
            vehicle_prob = float(probs[self._vehicle_indices].sum().item())
        return vehicle_prob


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[float, float, float, float]
    cls_id: int
    label: str
    first_frame_index: int
    last_frame_index: int
    state: str = "tentative"
    hits: int = 1
    misses: int = 0
    observations: int = 1
    confirmed_frame_index: int | None = None
    suppression_reason: str = ""
    avg_conf: float = 0.0
    avg_motion: float = 0.0
    avg_center_shift: float = 0.0
    avg_area_change: float = 0.0
    avg_road_score: float = 0.0
    avg_appearance_vehicle_prob: float = 0.0
    appearance_evaluations: int = 0
    static_candidate_hits: int = 0
    output_hits: int = 0
    frame_records: list[dict[str, Any]] = field(default_factory=list)

    def update_running_average(self, attr_name: str, value: float):
        current = float(getattr(self, attr_name))
        count = max(self.observations - 1, 0)
        updated = ((current * count) + float(value)) / max(self.observations, 1)
        setattr(self, attr_name, updated)

    @property
    def confirmation_latency_frames(self) -> int:
        if self.confirmed_frame_index is None:
            return 0
        return max(0, int(self.confirmed_frame_index - self.first_frame_index))

    @property
    def dwell_frames(self) -> int:
        return max(1, int(self.last_frame_index - self.first_frame_index + 1))

    @property
    def is_persistent_static_candidate(self) -> bool:
        if self.static_candidate_hits <= 0:
            return False
        static_ratio = self.static_candidate_hits / max(self.observations, 1)
        return self.dwell_frames >= 2 and static_ratio >= 0.60

    @property
    def was_suppressed_as_static(self) -> bool:
        return self.state == "suppressed" and bool(self.suppression_reason)


class TemporalVehicleFilter:
    """
    车辆轨迹级过滤器。

    提供两类输入接口：
    - `filter_tensor_detections()`：面向统一模型的 `torch.Tensor` 检测结果
    - `filter_detection_dicts()`：面向混合推理和评估脚本的字典检测结果
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        route_name: str = "vehicle",
        min_hits: int = 3,
        max_missing: int = 5,
        iou_match_thres: float = 0.30,
        static_center_shift_thres: float = 0.015,
        static_area_change_thres: float = 0.12,
        static_motion_thres: float = 0.035,
        static_frame_limit: int = 8,
        low_conf_static_suppress: float = 0.45,
        enable_road_roi_prior: bool = False,
        road_roi_score_thres: float = 0.10,
        enable_second_stage_classifier: bool = True,
        second_stage_vehicle_prob_thres: float = 0.10,
        second_stage_model_name: str = "mobilenet_v3_small",
        second_stage_device: str = "cpu",
        second_stage_classifier_weights: str | None = None,
    ):
        self.enabled = bool(enabled)
        self.route_name = str(route_name or "vehicle").strip()
        self.min_hits = max(1, int(min_hits))
        self.max_missing = max(0, int(max_missing))
        self.iou_match_thres = max(0.0, float(iou_match_thres))
        self.static_center_shift_thres = max(0.0, float(static_center_shift_thres))
        self.static_area_change_thres = max(0.0, float(static_area_change_thres))
        self.static_motion_thres = max(0.0, float(static_motion_thres))
        self.static_frame_limit = max(1, int(static_frame_limit))
        self.low_conf_static_suppress = float(low_conf_static_suppress)
        self.enable_road_roi_prior = bool(enable_road_roi_prior)
        self.road_roi_score_thres = max(0.0, float(road_roi_score_thres))
        self.second_stage_vehicle_prob_thres = max(
            0.0, float(second_stage_vehicle_prob_thres)
        )
        self.patch_verifier = ImageNetVehiclePatchVerifier(
            enabled=enable_second_stage_classifier,
            model_name=second_stage_model_name,
            device=second_stage_device,
            custom_weights=second_stage_classifier_weights,
        )

        self._tracks: dict[int, TrackState] = {}
        self._completed_tracks: list[TrackState] = []
        self._next_track_id = 1
        self._frame_counter = 0
        self._prev_gray: np.ndarray | None = None
        self._event_log: list[dict[str, Any]] = []
        self._frame_reports: list[dict[str, Any]] = []
        self._last_report: dict[str, Any] = self._empty_report()
        self._raw_detection_total = 0
        self._filtered_detection_total = 0
        self._suppressed_detection_total = 0

    @classmethod
    def from_config(cls, cfg, *, route_name: str):
        """从 `Config` 构造过滤器实例。"""
        return cls(
            enabled=bool(getattr(cfg, "TEMPORAL_FILTER_ENABLED", True)),
            route_name=route_name,
            min_hits=int(getattr(cfg, "TEMPORAL_MIN_HITS", 3)),
            max_missing=int(getattr(cfg, "TEMPORAL_MAX_MISSING", 5)),
            iou_match_thres=float(getattr(cfg, "TEMPORAL_IOU_MATCH_THRES", 0.30)),
            static_center_shift_thres=float(
                getattr(cfg, "TEMPORAL_STATIC_CENTER_SHIFT_THRES", 0.015)
            ),
            static_area_change_thres=float(
                getattr(cfg, "TEMPORAL_STATIC_AREA_CHANGE_THRES", 0.12)
            ),
            static_motion_thres=float(
                getattr(cfg, "TEMPORAL_STATIC_MOTION_THRES", 0.035)
            ),
            static_frame_limit=int(
                getattr(cfg, "TEMPORAL_STATIC_FRAME_LIMIT", 8)
            ),
            low_conf_static_suppress=float(
                getattr(cfg, "TEMPORAL_LOW_CONF_STATIC_SUPPRESS", 0.45)
            ),
            enable_road_roi_prior=bool(
                getattr(cfg, "TEMPORAL_ENABLE_ROAD_ROI_PRIOR", False)
            ),
            road_roi_score_thres=float(
                getattr(cfg, "TEMPORAL_ROAD_ROI_SCORE_THRES", 0.10)
            ),
            enable_second_stage_classifier=bool(
                getattr(cfg, "TEMPORAL_ENABLE_SECOND_STAGE_CLASSIFIER", True)
            ),
            second_stage_vehicle_prob_thres=float(
                getattr(cfg, "TEMPORAL_SECOND_STAGE_VEHICLE_PROB_THRES", 0.10)
            ),
            second_stage_model_name=str(
                getattr(cfg, "TEMPORAL_SECOND_STAGE_MODEL_NAME", "mobilenet_v3_small")
            ),
            second_stage_device=str(getattr(cfg, "DEVICE", "cpu")),
            second_stage_classifier_weights=getattr(
                cfg, "TEMPORAL_SECOND_STAGE_CLASSIFIER_WEIGHTS", None
            ),
        )

    def _empty_report(self) -> dict[str, Any]:
        return {
            "route": self.route_name,
            "enabled": self.enabled,
            "frame_index": -1,
            "timestamp_sec": None,
            "input_count": 0,
            "output_count": 0,
            "suppressed_count": 0,
            "tentative_track_count": 0,
            "confirmed_track_count": 0,
            "suspicious_track_count": 0,
            "suppressed_track_count": 0,
            "persistent_static_candidate_track_count": 0,
            "output_track_ids": [],
            "suppressed_track_ids": [],
        }

    @property
    def last_report(self) -> dict[str, Any]:
        return dict(self._last_report)

    def reset(self):
        """清空所有时序状态。"""
        self._tracks.clear()
        self._completed_tracks.clear()
        self._next_track_id = 1
        self._frame_counter = 0
        self._prev_gray = None
        self._event_log.clear()
        self._frame_reports.clear()
        self._last_report = self._empty_report()
        self._raw_detection_total = 0
        self._filtered_detection_total = 0
        self._suppressed_detection_total = 0

    def flush(self):
        """把当前仍活跃的轨迹转入已完成轨迹列表。"""
        for track_id in sorted(self._tracks.keys()):
            self._archive_track(track_id)

    def export_event_log(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._event_log]

    def export_frame_reports(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._frame_reports]

    def _archive_track(self, track_id: int):
        track = self._tracks.pop(track_id, None)
        if track is None:
            return
        self._completed_tracks.append(track)

    def _build_observation(
        self,
        det: dict[str, Any],
        frame_shape: tuple[int, int],
    ) -> dict[str, float]:
        bbox = tuple(float(v) for v in det["xyxy"])
        center_x, center_y, area = _bbox_center_and_area(bbox)
        frame_h, frame_w = max(int(frame_shape[0]), 1), max(int(frame_shape[1]), 1)
        return {
            "center_x": center_x,
            "center_y": center_y,
            "norm_center_x": center_x / frame_w,
            "norm_center_y": center_y / frame_h,
            "area": area,
        }

    def _compute_motion_intensity(
        self,
        curr_gray: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> float:
        if self._prev_gray is None:
            return 1.0

        x1, y1, x2, y2 = _clip_bbox_to_frame(bbox, curr_gray.shape[:2])
        prev_patch = self._prev_gray[y1:y2, x1:x2]
        curr_patch = curr_gray[y1:y2, x1:x2]
        if prev_patch.size == 0 or curr_patch.size == 0:
            return 1.0
        diff = np.abs(curr_patch.astype(np.float32) - prev_patch.astype(np.float32))
        return float(diff.mean() / 255.0)

    def _match_tracks(
        self,
        detections: list[dict[str, Any]],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self._tracks or not detections:
            return [], list(self._tracks.keys()), list(range(len(detections)))

        candidate_pairs: list[tuple[float, int, int]] = []
        active_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.misses <= self.max_missing
        ]
        for track_id in active_track_ids:
            track = self._tracks[track_id]
            for det_index, det in enumerate(detections):
                iou = _iou_xyxy(track.bbox, tuple(float(v) for v in det["xyxy"]))
                if iou >= self.iou_match_thres:
                    candidate_pairs.append((iou, track_id, det_index))

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)
        matches: list[tuple[int, int]] = []
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        for _, track_id, det_index in candidate_pairs:
            if track_id in matched_track_ids or det_index in matched_det_indices:
                continue
            matches.append((track_id, det_index))
            matched_track_ids.add(track_id)
            matched_det_indices.add(det_index)

        unmatched_tracks = [
            track_id for track_id in active_track_ids if track_id not in matched_track_ids
        ]
        unmatched_detections = [
            det_index
            for det_index in range(len(detections))
            if det_index not in matched_det_indices
        ]
        return matches, unmatched_tracks, unmatched_detections

    def _create_track(
        self,
        det: dict[str, Any],
        frame_index: int,
        obs: dict[str, float],
        road_score_value: float,
        motion_intensity: float,
    ) -> TrackState:
        track = TrackState(
            track_id=self._next_track_id,
            bbox=tuple(float(v) for v in det["xyxy"]),
            cls_id=int(det["cls_id"]),
            label=str(det.get("name", "vehicle")),
            first_frame_index=frame_index,
            last_frame_index=frame_index,
            avg_conf=float(det["conf"]),
            avg_motion=float(motion_intensity),
            avg_center_shift=0.0,
            avg_area_change=0.0,
            avg_road_score=float(road_score_value),
        )
        if self.min_hits <= 1:
            track.confirmed_frame_index = int(frame_index)
            track.state = "confirmed"
            track.output_hits = 1
        track.frame_records.append(
            {
                "frame_index": frame_index,
                "bbox": [float(v) for v in track.bbox],
                "conf": float(det["conf"]),
                "state": track.state,
                "center_shift": 0.0,
                "area_change": 0.0,
                "motion_intensity": float(motion_intensity),
                "road_score": float(road_score_value),
                "appearance_vehicle_prob": None,
                "suppression_reason": "",
            }
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _update_track(
        self,
        track: TrackState,
        det: dict[str, Any],
        frame_bgr: np.ndarray,
        curr_gray: np.ndarray,
        frame_index: int,
        timestamp_sec: float | None,
    ) -> TrackState:
        prev_bbox = track.bbox
        prev_center_x, prev_center_y, prev_area = _bbox_center_and_area(prev_bbox)
        curr_bbox = tuple(float(v) for v in det["xyxy"])
        curr_center_x, curr_center_y, curr_area = _bbox_center_and_area(curr_bbox)
        frame_h, frame_w = frame_bgr.shape[:2]

        center_shift = float(
            np.hypot(
                (curr_center_x - prev_center_x) / max(frame_w, 1),
                (curr_center_y - prev_center_y) / max(frame_h, 1),
            )
        )
        area_change = float(abs(curr_area - prev_area) / max(prev_area, 1.0))
        motion_intensity = self._compute_motion_intensity(curr_gray, curr_bbox)
        road_score_value = _road_score(curr_bbox, frame_bgr.shape[:2])

        track.bbox = curr_bbox
        track.label = str(det.get("name", track.label))
        track.cls_id = int(det["cls_id"])
        track.hits += 1
        track.misses = 0
        track.observations += 1
        track.last_frame_index = int(frame_index)
        track.update_running_average("avg_conf", float(det["conf"]))
        track.update_running_average("avg_motion", motion_intensity)
        track.update_running_average("avg_center_shift", center_shift)
        track.update_running_average("avg_area_change", area_change)
        track.update_running_average("avg_road_score", road_score_value)

        static_like = (
            center_shift <= self.static_center_shift_thres
            and area_change <= self.static_area_change_thres
            and motion_intensity <= self.static_motion_thres
        )
        if static_like:
            track.static_candidate_hits += 1

        appearance_vehicle_prob = None
        suppression_reasons: list[str] = []
        if (
            track.hits >= self.min_hits
            and track.static_candidate_hits >= self.static_frame_limit
        ):
            suppression_reasons.append("persistent_static_candidate")
            if float(track.avg_conf) <= self.low_conf_static_suppress:
                suppression_reasons.append("low_conf_static")
            if (
                self.enable_road_roi_prior
                and float(track.avg_road_score) < self.road_roi_score_thres
            ):
                suppression_reasons.append("offroad_static")

            if self.patch_verifier.enabled:
                appearance_vehicle_prob = self.patch_verifier.score_patch(
                    frame_bgr,
                    curr_bbox,
                )
                if appearance_vehicle_prob is not None:
                    track.appearance_evaluations += 1
                    track.avg_appearance_vehicle_prob = (
                        (
                            track.avg_appearance_vehicle_prob
                            * max(track.appearance_evaluations - 1, 0)
                        )
                        + float(appearance_vehicle_prob)
                    ) / max(track.appearance_evaluations, 1)
                    if appearance_vehicle_prob < self.second_stage_vehicle_prob_thres:
                        suppression_reasons.append("appearance_non_vehicle")

        if track.confirmed_frame_index is None and track.hits >= self.min_hits:
            track.confirmed_frame_index = int(frame_index)
            track.state = "confirmed"

        persistent_candidate = "persistent_static_candidate" in suppression_reasons
        suppress_static = any(
            reason in suppression_reasons
            for reason in ("low_conf_static", "offroad_static", "appearance_non_vehicle")
        )
        if suppress_static:
            track.state = "suppressed"
            track.suppression_reason = ",".join(
                reason for reason in suppression_reasons if reason != "persistent_static_candidate"
            )
        elif persistent_candidate:
            track.state = "suspicious"
            track.suppression_reason = ""
        elif track.confirmed_frame_index is not None:
            track.state = "confirmed"
            track.suppression_reason = ""
        else:
            track.state = "tentative"
            track.suppression_reason = ""

        if track.state in {"confirmed", "suspicious"}:
            track.output_hits += 1

        record = {
            "route": self.route_name,
            "frame_index": int(frame_index),
            "timestamp_sec": timestamp_sec,
            "track_id": int(track.track_id),
            "bbox": [float(v) for v in curr_bbox],
            "conf": float(det["conf"]),
            "state": track.state,
            "center_shift": float(center_shift),
            "area_change": float(area_change),
            "motion_intensity": float(motion_intensity),
            "road_score": float(road_score_value),
            "appearance_vehicle_prob": (
                None if appearance_vehicle_prob is None else float(appearance_vehicle_prob)
            ),
            "persistent_static_candidate": bool(persistent_candidate),
            "suppression_reason": str(track.suppression_reason),
            "label": str(track.label),
        }
        track.frame_records.append(record)
        self._event_log.append(dict(record))
        return track

    def _touch_unmatched_tracks(self, unmatched_track_ids: list[int]):
        for track_id in unmatched_track_ids:
            track = self._tracks.get(track_id)
            if track is None:
                continue
            track.misses += 1
            if track.misses > self.max_missing:
                self._archive_track(track_id)

    def filter_detection_dicts(
        self,
        frame_bgr: np.ndarray,
        detections: list[dict[str, Any]],
        *,
        frame_index: int | None = None,
        timestamp_sec: float | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        过滤字典形式的检测结果。

        输入格式约定：
        - `xyxy`: `[x1, y1, x2, y2]`
        - `conf`: 置信度
        - `cls_id`: 类别 id
        - `name`: 可选类别名
        """
        if frame_index is None:
            frame_index = self._frame_counter
        self._frame_counter = max(self._frame_counter, int(frame_index) + 1)

        if not self.enabled:
            report = self._empty_report()
            report.update(
                {
                    "frame_index": int(frame_index),
                    "timestamp_sec": timestamp_sec,
                    "input_count": len(detections),
                    "output_count": len(detections),
                }
            )
            self._last_report = report
            self._frame_reports.append(dict(report))
            return list(detections), report

        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        normalized_detections = []
        for det in detections:
            normalized_detections.append(
                {
                    "xyxy": [float(v) for v in det["xyxy"]],
                    "conf": float(det["conf"]),
                    "cls_id": int(det.get("cls_id", 0)),
                    "name": str(det.get("name", "vehicle")),
                    "_payload": det,
                }
            )

        matches, unmatched_tracks, unmatched_detections = self._match_tracks(
            normalized_detections
        )

        output_detections: list[dict[str, Any]] = []
        suppressed_track_ids: list[int] = []
        for track_id, det_index in matches:
            track = self._tracks[track_id]
            updated_track = self._update_track(
                track,
                normalized_detections[det_index],
                frame_bgr,
                curr_gray,
                int(frame_index),
                timestamp_sec,
            )
            payload = dict(normalized_detections[det_index]["_payload"])
            payload["track_id"] = int(updated_track.track_id)
            payload["track_state"] = updated_track.state
            payload["suppression_reason"] = updated_track.suppression_reason
            if updated_track.state in {"confirmed", "suspicious"}:
                output_detections.append(payload)
            elif updated_track.state == "suppressed":
                suppressed_track_ids.append(int(updated_track.track_id))

        self._touch_unmatched_tracks(unmatched_tracks)

        for det_index in unmatched_detections:
            det = normalized_detections[det_index]
            obs = self._build_observation(det, frame_bgr.shape[:2])
            road_score_value = _road_score(tuple(float(v) for v in det["xyxy"]), frame_bgr.shape[:2])
            motion_intensity = self._compute_motion_intensity(
                curr_gray,
                tuple(float(v) for v in det["xyxy"]),
            )
            track = self._create_track(
                det,
                int(frame_index),
                obs,
                road_score_value,
                motion_intensity,
            )
            self._event_log.append(
                {
                    "route": self.route_name,
                    "frame_index": int(frame_index),
                    "timestamp_sec": timestamp_sec,
                    "track_id": int(track.track_id),
                    "bbox": [float(v) for v in track.bbox],
                    "conf": float(det["conf"]),
                    "state": track.state,
                    "center_shift": 1.0,
                    "area_change": 1.0,
                    "motion_intensity": float(motion_intensity),
                    "road_score": float(road_score_value),
                    "appearance_vehicle_prob": None,
                    "persistent_static_candidate": False,
                    "suppression_reason": "",
                    "label": str(track.label),
                }
            )

        self._prev_gray = curr_gray
        self._raw_detection_total += len(normalized_detections)
        self._filtered_detection_total += len(output_detections)
        suppressed_count = max(len(normalized_detections) - len(output_detections), 0)
        self._suppressed_detection_total += suppressed_count

        state_hist = {"tentative": 0, "confirmed": 0, "suspicious": 0, "suppressed": 0}
        persistent_candidate_count = 0
        for track in self._tracks.values():
            state_hist[track.state] = state_hist.get(track.state, 0) + 1
            if track.is_persistent_static_candidate:
                persistent_candidate_count += 1

        report = {
            "route": self.route_name,
            "enabled": True,
            "frame_index": int(frame_index),
            "timestamp_sec": timestamp_sec,
            "input_count": len(normalized_detections),
            "output_count": len(output_detections),
            "suppressed_count": suppressed_count,
            "tentative_track_count": state_hist.get("tentative", 0),
            "confirmed_track_count": state_hist.get("confirmed", 0),
            "suspicious_track_count": state_hist.get("suspicious", 0),
            "suppressed_track_count": state_hist.get("suppressed", 0),
            "persistent_static_candidate_track_count": persistent_candidate_count,
            "output_track_ids": [int(det["track_id"]) for det in output_detections],
            "suppressed_track_ids": suppressed_track_ids,
        }
        self._last_report = report
        self._frame_reports.append(dict(report))
        return output_detections, report

    def filter_tensor_detections(
        self,
        frame_bgr: np.ndarray,
        detections: torch.Tensor,
        *,
        frame_index: int | None = None,
        timestamp_sec: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """过滤 `torch.Tensor` 形式的检测结果。"""
        if detections.numel() == 0:
            filtered_dicts, report = self.filter_detection_dicts(
                frame_bgr,
                [],
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )
            width = int(detections.shape[1]) if detections.ndim == 2 else 6
            return detections.new_zeros((0, width)), report

        det_dicts: list[dict[str, Any]] = []
        for row in detections:
            cls_id = int(row[5].item()) if row.shape[0] > 5 else 0
            det_dicts.append(
                {
                    "xyxy": [float(v) for v in row[:4].tolist()],
                    "conf": float(row[4].item()),
                    "cls_id": cls_id,
                    "name": "vehicle",
                    "_tensor_row": row.detach().clone(),
                }
            )

        filtered_dicts, report = self.filter_detection_dicts(
            frame_bgr,
            det_dicts,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )

        if not filtered_dicts:
            return detections.new_zeros((0, detections.shape[1])), report

        rows = []
        for det in filtered_dicts:
            row = det.get("_tensor_row")
            if row is None:
                row = detections.new_tensor(
                    list(det["xyxy"]) + [float(det["conf"]), float(det.get("cls_id", 0))]
                )
            rows.append(row)
        return torch.stack(rows, dim=0), report

    def build_summary(self, fps: float | None = None) -> dict[str, Any]:
        """
        汇总当前过滤器的轨迹级统计。

        这些指标是启发式的，不等同于带人工标注的严格误检率，
        但足以作为“静止假车误检治理”过程中的工程对照量。
        """
        all_tracks = list(self._completed_tracks) + list(self._tracks.values())
        track_count_total = len(all_tracks)
        confirmed_tracks = [track for track in all_tracks if track.confirmed_frame_index is not None]
        suppressed_tracks = [track for track in all_tracks if track.state == "suppressed"]
        persistent_static_tracks = [
            track for track in all_tracks if track.is_persistent_static_candidate
        ]
        suppressed_static_tracks = [
            track for track in persistent_static_tracks if track.was_suppressed_as_static
        ]
        confirmation_latencies = [
            track.confirmation_latency_frames
            for track in confirmed_tracks
            if track.confirmation_latency_frames > 0
        ]
        dwell_frames = [track.dwell_frames for track in all_tracks]
        motion_values = [float(track.avg_motion) for track in all_tracks if track.observations > 0]

        total_frames = max(len(self._frame_reports), 1)
        evaluated_minutes = (
            (total_frames / max(float(fps), 1e-6)) / 60.0
            if fps and fps > 0
            else 0.0
        )

        return {
            "enabled": self.enabled,
            "route": self.route_name,
            "frames_seen": len(self._frame_reports),
            "raw_detection_count": int(self._raw_detection_total),
            "filtered_detection_count": int(self._filtered_detection_total),
            "suppressed_detection_count": int(self._suppressed_detection_total),
            "track_count_total": track_count_total,
            "track_count_confirmed": len(confirmed_tracks),
            "track_count_suppressed": len(suppressed_tracks),
            "heuristic_persistent_static_fp_count": len(persistent_static_tracks),
            "suppressed_heuristic_persistent_static_fp_count": len(suppressed_static_tracks),
            "heuristic_persistent_static_fp_per_min": (
                len(persistent_static_tracks) / evaluated_minutes
                if evaluated_minutes > 0
                else 0.0
            ),
            "suppressed_heuristic_persistent_static_fp_per_min": (
                len(suppressed_static_tracks) / evaluated_minutes
                if evaluated_minutes > 0
                else 0.0
            ),
            "mean_confirmation_latency_frames": (
                float(np.mean(confirmation_latencies)) if confirmation_latencies else 0.0
            ),
            "mean_confirmation_latency_sec": (
                float(np.mean(confirmation_latencies) / fps)
                if confirmation_latencies and fps and fps > 0
                else 0.0
            ),
            "mean_track_dwell_frames": (
                float(np.mean(dwell_frames)) if dwell_frames else 0.0
            ),
            "mean_motion_intensity": (
                float(np.mean(motion_values)) if motion_values else 0.0
            ),
            "second_stage_classifier_enabled": bool(self.patch_verifier.enabled),
            "second_stage_classifier_init_error": self.patch_verifier.init_error,
            "evaluated_minutes": float(evaluated_minutes),
        }
