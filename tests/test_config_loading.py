from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.config import Config, get_default_config


class ConfigLoadingTests(unittest.TestCase):
    def test_yaml_config_overrides_nested_runtime_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            raw_dir = tmpdir_path / "raw"
            xml_dir = tmpdir_path / "xml"
            depth_dir = tmpdir_path / "depth"
            output_dir = tmpdir_path / "output"
            checkpoint_dir = tmpdir_path / "output" / "custom_checkpoints"
            raw_dir.mkdir()
            xml_dir.mkdir()

            payload = {
                "training": {
                    "batch_size": 4,
                    "img_size": 320,
                    "frame_stride": 3,
                },
                "paths": {
                    "raw_data_dir": str(raw_dir),
                    "xml_dir": str(xml_dir),
                    "depth_cache_dir": str(depth_dir),
                    "output_dir": str(output_dir),
                    "checkpoint_dir": str(checkpoint_dir),
                },
                "adaptive_threshold": {
                    "base_conf_thres": 0.4,
                    "ema_alpha": 0.2,
                    "beta_scale_factor": 1.5,
                    "min_conf_thres": 0.12,
                },
                "temporal_filter": {
                    "enabled": True,
                    "min_hits": 5,
                    "max_missing": 9,
                    "static_motion_thres": 0.08,
                    "enable_second_stage_classifier": False,
                },
                "display": {
                    "window_width": 1280,
                    "window_height": 720,
                    "status_bar_height": 96,
                },
                "unused_group": {
                    "foo": 1,
                },
            }
            config_path = tmpdir_path / "runtime.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )

            cfg = Config(config_path=str(config_path))

            self.assertEqual(cfg.BATCH_SIZE, 4)
            self.assertEqual(cfg.IMG_SIZE, 320)
            self.assertEqual(cfg.FRAME_STRIDE, 3)
            self.assertEqual(cfg.BASE_CONF_THRES, 0.4)
            self.assertEqual(cfg.EMA_ALPHA, 0.2)
            self.assertEqual(cfg.BETA_SCALE_FACTOR, 1.5)
            self.assertEqual(cfg.MIN_CONF_THRES, 0.12)
            self.assertEqual(cfg.DISPLAY_WINDOW_WIDTH, 1280)
            self.assertEqual(cfg.DISPLAY_WINDOW_HEIGHT, 720)
            self.assertEqual(cfg.STATUS_BAR_HEIGHT, 96)
            self.assertTrue(cfg.TEMPORAL_FILTER_ENABLED)
            self.assertEqual(cfg.TEMPORAL_MIN_HITS, 5)
            self.assertEqual(cfg.TEMPORAL_MAX_MISSING, 9)
            self.assertEqual(cfg.TEMPORAL_STATIC_MOTION_THRES, 0.08)
            self.assertFalse(cfg.TEMPORAL_ENABLE_SECOND_STAGE_CLASSIFIER)
            self.assertEqual(cfg.CONFIG_FILE, str(config_path.resolve()))
            self.assertIn("unused_group.foo", cfg.UNUSED_CONFIG_KEYS)
            self.assertTrue(depth_dir.exists())
            self.assertTrue(output_dir.exists())
            self.assertTrue(checkpoint_dir.exists())

    def test_json_config_supports_flat_and_alias_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            relative_weights_dir = tmpdir_path / "weights"
            relative_weights_dir.mkdir()
            weight_path = relative_weights_dir / "custom.pt"
            weight_path.write_text("stub", encoding="utf-8")
            output_dir = tmpdir_path / "out"
            checkpoint_dir = tmpdir_path / "out" / "ckpts"
            depth_dir = tmpdir_path / "depth"

            payload = {
                "batch_size": 2,
                "learning_rate": 0.0003,
                "img_size": 384,
                "yolo_weights": str(weight_path),
                "loss": {
                    "clear_det_loss_weight": 0.8,
                },
                "resume": {
                    "model_only": True,
                },
                "paths": {
                    "depth_cache_dir": str(depth_dir),
                    "output_dir": str(output_dir),
                    "checkpoint_dir": str(checkpoint_dir),
                },
            }
            config_path = tmpdir_path / "runtime.json"
            config_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            cfg = get_default_config(str(config_path))

            self.assertEqual(cfg.BATCH_SIZE, 2)
            self.assertEqual(cfg.LR, 0.0003)
            self.assertEqual(cfg.IMG_SIZE, 384)
            self.assertEqual(cfg.YOLO_BASE_MODEL, str(weight_path.resolve()))
            self.assertEqual(cfg.CLEAR_DET_LOSS_WEIGHT, 0.8)
            self.assertTrue(cfg.RESUME_MODEL_ONLY)
            self.assertEqual(cfg.DEPTH_CACHE_DIR, str(depth_dir.resolve()))
            self.assertEqual(cfg.OUTPUT_DIR, str(output_dir.resolve()))
            self.assertEqual(cfg.CHECKPOINT_DIR, str(checkpoint_dir.resolve()))
            self.assertEqual(cfg.UNUSED_CONFIG_KEYS, [])

    def test_det_head_mode_coco_vehicle_derives_detector_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_dir = tmpdir_path / "out"
            depth_dir = tmpdir_path / "depth"
            payload = {
                "model": {
                    "det_head_mode": "coco_vehicle",
                },
                "paths": {
                    "depth_cache_dir": str(depth_dir),
                    "output_dir": str(output_dir),
                },
            }
            config_path = tmpdir_path / "runtime.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )

            cfg = Config(config_path=str(config_path))

            self.assertEqual(cfg.DET_HEAD_MODE, "coco_vehicle")
            self.assertEqual(cfg.NUM_DET_CLASSES, 80)
            self.assertEqual(cfg.DET_TRAIN_CLASS_ID, cfg.COCO_VEHICLE_TRAIN_CLASS_ID)
            self.assertEqual(cfg.VEHICLE_CLASS_IDS, [2, 3, 5, 7])
            self.assertEqual(cfg.DET_CLASS_NAMES, ["vehicle"])

    def test_output_dir_override_recomputes_checkpoint_dir_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_dir = tmpdir_path / "custom_output"
            depth_dir = tmpdir_path / "depth"
            payload = {
                "paths": {
                    "output_dir": str(output_dir),
                    "depth_cache_dir": str(depth_dir),
                }
            }
            config_path = tmpdir_path / "runtime.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )

            cfg = Config(config_path=str(config_path))

            self.assertEqual(cfg.OUTPUT_DIR, str(output_dir.resolve()))
            self.assertEqual(
                cfg.CHECKPOINT_DIR,
                str((output_dir / "checkpoints").resolve()),
            )


if __name__ == "__main__":
    unittest.main()
