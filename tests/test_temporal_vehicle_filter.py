from __future__ import annotations

import unittest

import numpy as np

from src.temporal_vehicle_filter import TemporalVehicleFilter


def make_detection(x1: float, y1: float, x2: float, y2: float, conf: float = 0.5):
    return {
        "xyxy": [x1, y1, x2, y2],
        "conf": conf,
        "cls_id": 0,
        "name": "vehicle",
    }


class TemporalVehicleFilterTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def test_track_becomes_confirmed_after_min_hits(self):
        filt = TemporalVehicleFilter(
            enabled=True,
            min_hits=2,
            max_missing=2,
            static_frame_limit=10,
            enable_second_stage_classifier=False,
        )

        detections, report = filt.filter_detection_dicts(
            self.frame,
            [make_detection(20, 20, 60, 60, conf=0.8)],
            frame_index=0,
        )
        self.assertEqual(report["output_count"], 0)
        self.assertEqual(report["tentative_track_count"], 1)

        detections, report = filt.filter_detection_dicts(
            self.frame,
            [make_detection(21, 20, 61, 60, conf=0.8)],
            frame_index=1,
        )
        self.assertEqual(report["output_count"], 1)
        self.assertEqual(report["confirmed_track_count"], 1)
        self.assertEqual(len(detections), 1)

    def test_persistent_static_low_conf_track_is_suppressed(self):
        filt = TemporalVehicleFilter(
            enabled=True,
            min_hits=2,
            max_missing=2,
            static_center_shift_thres=0.01,
            static_area_change_thres=0.05,
            static_motion_thres=0.01,
            static_frame_limit=3,
            low_conf_static_suppress=0.60,
            enable_second_stage_classifier=False,
        )

        states = []
        for frame_index in range(4):
            detections, report = filt.filter_detection_dicts(
                self.frame,
                [make_detection(30, 30, 70, 70, conf=0.35)],
                frame_index=frame_index,
            )
            states.append(
                {
                    "output_count": report["output_count"],
                    "suppressed_track_count": report["suppressed_track_count"],
                }
            )

        self.assertEqual(states[0]["output_count"], 0)
        self.assertEqual(states[1]["output_count"], 1)
        self.assertEqual(states[-1]["output_count"], 0)
        self.assertEqual(states[-1]["suppressed_track_count"], 1)

        summary = filt.build_summary(fps=10.0)
        self.assertEqual(summary["heuristic_persistent_static_fp_count"], 1)
        self.assertEqual(summary["suppressed_heuristic_persistent_static_fp_count"], 1)

    def test_moving_track_is_not_marked_as_static_candidate(self):
        filt = TemporalVehicleFilter(
            enabled=True,
            min_hits=2,
            max_missing=2,
            static_center_shift_thres=0.01,
            static_area_change_thres=0.05,
            static_motion_thres=0.01,
            static_frame_limit=2,
            low_conf_static_suppress=0.60,
            enable_second_stage_classifier=False,
        )

        for frame_index in range(4):
            x1 = 20 + frame_index * 8
            detections, _ = filt.filter_detection_dicts(
                self.frame,
                [make_detection(x1, 30, x1 + 40, 70, conf=0.40)],
                frame_index=frame_index,
            )

        summary = filt.build_summary(fps=10.0)
        self.assertEqual(summary["heuristic_persistent_static_fp_count"], 0)
        self.assertEqual(summary["track_count_confirmed"], 1)
        self.assertEqual(len(detections), 1)


if __name__ == "__main__":
    unittest.main()
