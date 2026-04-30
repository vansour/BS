from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate_mod = load_module(
    "benchmark_candidate_model",
    "scripts/benchmark_candidate_model.py",
)
multi_mod = load_module(
    "run_multi_model_benchmark",
    "scripts/run_multi_model_benchmark.py",
)
eval_mod = load_module(
    "evaluate_inference_routes",
    "scripts/evaluate_inference_routes.py",
)
asset_mod = load_module(
    "build_benchmark_assets",
    "scripts/build_benchmark_assets.py",
)
postprocess_mod = load_module(
    "postprocess_fogfocus_full",
    "scripts/postprocess_fogfocus_full.py",
)


class CandidateBenchmarkTests(unittest.TestCase):
    def test_merge_candidate_model_replaces_same_slug(self):
        base_payload = {
            "description": "base models",
            "models": [
                {
                    "label": "baseline_a",
                    "slug": "baseline_a",
                    "fog_weights": "/tmp/a.pt",
                },
                {
                    "label": "candidate_old",
                    "slug": "candidate_x",
                    "fog_weights": "/tmp/old.pt",
                },
            ],
        }
        candidate_entry = {
            "label": "candidate_new",
            "slug": "candidate_x",
            "fog_weights": "/tmp/new.pt",
            "tags": ["candidate"],
            "notes": "fresh",
        }

        merged = candidate_mod.merge_candidate_model(base_payload, candidate_entry)

        self.assertEqual(merged["description"], "base models")
        self.assertEqual(len(merged["models"]), 2)
        self.assertEqual(merged["models"][-1]["label"], "candidate_new")
        self.assertEqual(
            merged["candidate_merge"]["baseline_model_count_before"],
            2,
        )
        self.assertEqual(merged["candidate_merge"]["model_count_after"], 2)
        self.assertEqual(merged["candidate_merge"]["replaced_models"], ["candidate_old"])

    def test_rank_models_assigns_same_rank_on_ties(self):
        model_summaries = [
            {"slug": "a", "aggregate": {"score": 0.9}},
            {"slug": "b", "aggregate": {"score": 0.9}},
            {"slug": "c", "aggregate": {"score": 0.5}},
        ]

        ranks = candidate_mod.rank_models(
            model_summaries,
            "score",
            reverse=True,
        )

        self.assertEqual(ranks["a"], 1)
        self.assertEqual(ranks["b"], 1)
        self.assertEqual(ranks["c"], 3)

    def test_build_candidate_summary_computes_rankings_and_deltas(self):
        multi_model_summary = {
            "leaders": {
                "best_unified_mean_count_per_frame": {
                    "label": "candidate",
                    "value": 2.0,
                }
            },
            "artifacts": {
                "model_overview_csv": "model_overview.csv",
                "model_video_matrix_csv": "model_video_matrix.csv",
                "summary_json": "multi_model_benchmark_summary.json",
            },
            "models": [
                {
                    "label": "baseline_a",
                    "slug": "baseline_a",
                    "fog_weights": "/tmp/a.pt",
                    "tags": ["baseline"],
                    "notes": "",
                    "aggregate": {
                        "weighted_unified_mean_count_per_frame": 1.0,
                        "weighted_unified_frames_with_detections_ratio": 0.6,
                        "weighted_dominant_switch_rate": 0.2,
                        "weighted_beta_abs_delta_mean": 0.05,
                    },
                },
                {
                    "label": "candidate",
                    "slug": "candidate",
                    "fog_weights": "/tmp/candidate.pt",
                    "tags": ["candidate"],
                    "notes": "new",
                    "aggregate": {
                        "weighted_unified_mean_count_per_frame": 2.0,
                        "weighted_unified_frames_with_detections_ratio": 0.8,
                        "weighted_dominant_switch_rate": 0.1,
                        "weighted_beta_abs_delta_mean": 0.03,
                    },
                },
                {
                    "label": "baseline_b",
                    "slug": "baseline_b",
                    "fog_weights": "/tmp/b.pt",
                    "tags": ["baseline"],
                    "notes": "",
                    "aggregate": {
                        "weighted_unified_mean_count_per_frame": 2.0,
                        "weighted_unified_frames_with_detections_ratio": 0.7,
                        "weighted_dominant_switch_rate": 0.1,
                        "weighted_beta_abs_delta_mean": 0.03,
                    },
                },
            ],
        }

        summary = candidate_mod.build_candidate_summary(
            multi_model_summary,
            "candidate",
            {
                "allowed_unified_mean_drop": 0.0,
                "allowed_unified_ratio_drop": 0.0,
                "allowed_fog_switch_increase": 0.02,
                "allowed_beta_abs_delta_increase": 0.005,
            },
        )

        self.assertEqual(summary["candidate"]["label"], "candidate")
        self.assertEqual(
            summary["candidate"]["rankings"]["weighted_unified_mean_count_per_frame"],
            1,
        )
        self.assertEqual(
            summary["candidate"]["rankings"]["weighted_dominant_switch_rate"],
            1,
        )
        self.assertTrue(summary["overall_gate"]["pass"])
        self.assertEqual(len(summary["baseline_deltas"]), 2)
        deltas = {item["slug"]: item for item in summary["baseline_deltas"]}
        self.assertAlmostEqual(
            deltas["baseline_a"]["delta_unified_mean_count_per_frame"],
            1.0,
        )
        self.assertAlmostEqual(
            deltas["baseline_b"]["delta_unified_mean_count_per_frame"],
            0.0,
        )
        self.assertTrue(deltas["baseline_a"]["gate"]["pass"])
        self.assertTrue(deltas["baseline_b"]["gate"]["pass"])

    def test_evaluate_candidate_vs_baseline_fails_on_detection_regression(self):
        delta = {
            "delta_unified_mean_count_per_frame": -0.1,
            "delta_unified_frames_with_detections_ratio": 0.0,
            "delta_fog_switch_rate": 0.0,
            "delta_beta_abs_delta_mean": 0.0,
        }
        pass_rules = {
            "allowed_unified_mean_drop": 0.0,
            "allowed_unified_ratio_drop": 0.0,
            "allowed_fog_switch_increase": 0.02,
            "allowed_beta_abs_delta_increase": 0.005,
        }

        gate = candidate_mod.evaluate_candidate_vs_baseline(delta, pass_rules)

        self.assertFalse(gate["pass"])
        self.assertFalse(gate["checks"]["unified_mean_count_per_frame"])
        self.assertTrue(gate["checks"]["beta_abs_delta_mean"])


class MultiModelBenchmarkTests(unittest.TestCase):
    def test_video_entries_to_build_selects_active_sequence_entries(self):
        payload = {
            "videos": [
                {
                    "label": "build_me",
                    "status": "active",
                    "source_sequence": "MVI_20011",
                },
                {
                    "label": "skip_no_sequence",
                    "status": "active",
                },
                {
                    "label": "skip_planned",
                    "status": "planned",
                    "source_sequence": "MVI_20012",
                },
            ]
        }

        entries = asset_mod.video_entries_to_build(payload)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["label"], "build_me")

    def test_load_benchmark_entries_skips_planned_videos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            active_video = tmpdir_path / "active.mp4"
            active_video.write_text("stub", encoding="utf-8")
            config_path = tmpdir_path / "benchmark.json"
            config_payload = {
                "benchmark_id": "benchmark_v1",
                "description": "test benchmark",
                "default_runtime": {
                    "sample_stride": 7,
                    "max_frames": 40,
                },
                "videos": [
                    {
                        "label": "active_video",
                        "path": str(active_video),
                        "status": "active",
                    },
                    {
                        "label": "planned_video",
                        "path": str(tmpdir_path / "planned.mp4"),
                        "status": "planned",
                    },
                ],
            }
            config_path.write_text(
                json.dumps(config_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            entries, meta = eval_mod.load_benchmark_entries(config_path)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["label"], "active_video")
            self.assertEqual(entries[0]["sample_stride"], 7)
            self.assertEqual(entries[0]["max_frames"], 40)
            self.assertEqual(meta["benchmark_id"], "benchmark_v1")
            self.assertEqual(meta["benchmark_total_entries"], 2)
            self.assertEqual(meta["benchmark_active_entries"], 1)
            self.assertEqual(meta["benchmark_inactive_entries"], 1)
            self.assertEqual(meta["benchmark_inactive_labels"], ["planned_video"])

    def test_load_model_entries_reads_absolute_weight_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            weight_a = tmpdir_path / "a.pt"
            weight_b = tmpdir_path / "b.pt"
            model_cfg = tmpdir_path / "model.yaml"
            weight_a.write_text("stub", encoding="utf-8")
            weight_b.write_text("stub", encoding="utf-8")
            model_cfg.write_text("model:\n  det_head_mode: coco_vehicle\n", encoding="utf-8")

            config_path = tmpdir_path / "models.json"
            config_payload = {
                "benchmark_id": "benchmark_v1",
                "description": "test config",
                "selection_rules": [
                    "keep model_a",
                    "keep model_b",
                ],
                "models": [
                    {
                        "label": "model_a",
                        "fog_weights": str(weight_a),
                        "config": str(model_cfg),
                        "tags": ["a"],
                        "role": "historical_reference",
                        "status": "active",
                    },
                    {
                        "label": "model_b",
                        "slug": "custom_b",
                        "fog_weights": str(weight_b),
                        "tags": ["b"],
                        "notes": "second",
                        "role": "formal_training_reference",
                        "status": "active",
                    },
                ],
            }
            config_path.write_text(
                json.dumps(config_payload, ensure_ascii=False),
                encoding="utf-8",
            )

            entries, meta = multi_mod.load_model_entries(config_path)

            self.assertEqual(meta["model_config_description"], "test config")
            self.assertEqual(meta["benchmark_id"], "benchmark_v1")
            self.assertEqual(meta["model_selection_rules"], ["keep model_a", "keep model_b"])
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["slug"], "model_a")
            self.assertEqual(entries[0]["config_path"], str(model_cfg.resolve()))
            self.assertEqual(entries[0]["role"], "historical_reference")
            self.assertEqual(entries[0]["status"], "active")
            self.assertEqual(entries[1]["slug"], "custom_b")
            self.assertEqual(entries[1]["notes"], "second")
            self.assertEqual(entries[1]["role"], "formal_training_reference")
            self.assertEqual(entries[0]["fog_weights"], weight_a.resolve())

    def test_build_leaderboard_selects_expected_models(self):
        model_summaries = [
            {
                "label": "model_a",
                "aggregate": {
                    "weighted_unified_mean_count_per_frame": 0.8,
                    "weighted_unified_frames_with_detections_ratio": 0.5,
                    "weighted_dominant_switch_rate": 0.2,
                    "weighted_beta_abs_delta_mean": 0.04,
                },
            },
            {
                "label": "model_b",
                "aggregate": {
                    "weighted_unified_mean_count_per_frame": 1.2,
                    "weighted_unified_frames_with_detections_ratio": 0.6,
                    "weighted_dominant_switch_rate": 0.1,
                    "weighted_beta_abs_delta_mean": 0.03,
                },
            },
        ]

        leaders = multi_mod.build_leaderboard(model_summaries)

        self.assertEqual(
            leaders["best_unified_mean_count_per_frame"]["label"],
            "model_b",
        )
        self.assertEqual(
            leaders["lowest_fog_switch_rate"]["label"],
            "model_b",
        )
        self.assertEqual(
            leaders["lowest_beta_abs_delta_mean"]["value"],
            0.03,
        )


class RouteEvalAggregateTests(unittest.TestCase):
    def test_build_aggregate_summary_includes_temporal_filter_metrics(self):
        video_summaries = [
            {
                "sampled_frames": 10,
                "heuristic_recommendation": {"route": "hybrid"},
                "unified": {
                    "total_detections": 10,
                    "frames_with_detections": 6,
                    "temporal_filter": {
                        "heuristic_persistent_static_fp_count": 2,
                        "suppressed_heuristic_persistent_static_fp_count": 1,
                        "evaluated_minutes": 0.5,
                        "mean_confirmation_latency_frames": 2.0,
                    },
                },
                "hybrid": {
                    "total_detections": 14,
                    "frames_with_detections": 7,
                    "temporal_filter": {
                        "heuristic_persistent_static_fp_count": 1,
                        "suppressed_heuristic_persistent_static_fp_count": 1,
                        "evaluated_minutes": 0.5,
                        "mean_confirmation_latency_frames": 1.0,
                    },
                },
                "comparison": {"mean_abs_count_gap": 0.4},
                "fog": {
                    "beta": {"mean": 0.02},
                    "probs": {"dominant_hist": {"CLEAR": 7, "UNIFORM FOG": 3, "PATCHY FOG": 0}},
                    "stability": {
                        "dominant_switch_count": 2,
                        "beta_abs_delta_mean": 0.01,
                    },
                },
            }
        ]

        aggregate = eval_mod.build_aggregate_summary(video_summaries)

        self.assertAlmostEqual(
            aggregate["weighted_unified_heuristic_persistent_static_fp_per_min"],
            4.0,
        )
        self.assertAlmostEqual(
            aggregate["weighted_hybrid_heuristic_persistent_static_fp_per_min"],
            2.0,
        )
        self.assertAlmostEqual(
            aggregate["weighted_unified_confirmation_latency_frames"],
            2.0,
        )
        self.assertEqual(aggregate["recommendation_hist"], {"hybrid": 1})


class PostprocessSummaryTests(unittest.TestCase):
    def test_build_summary_includes_candidate_benchmark_artifacts(self):
        training_summary = {
            "status": "completed_fp32_only",
            "best_loss": 1.23,
            "best_qat_loss": None,
            "phase_summaries": [{"phase": "fp32", "epoch": 1}],
        }
        route_eval_summary = {
            "videos": [
                {
                    "video_name": "demo.mp4",
                    "sampled_frames": 10,
                    "unified": {"mean_count_per_frame": 1.0},
                    "hybrid": {"mean_count_per_frame": 1.5},
                    "comparison": {"mean_count_gap_hybrid_minus_unified": 0.5},
                    "fog": {"beta": {"mean": 0.02}, "probs": {"dominant_hist": {}}},
                    "heuristic_recommendation": {"route": "hybrid"},
                }
            ]
        }
        candidate_benchmark_summary = {
            "candidate": {
                "label": "candidate_x",
                "rankings": {
                    "weighted_unified_mean_count_per_frame": 1,
                    "weighted_unified_frames_with_detections_ratio": 1,
                    "weighted_dominant_switch_rate": 2,
                    "weighted_beta_abs_delta_mean": 2,
                },
            }
        }

        summary = postprocess_mod.build_summary(
            ROOT / "outputs" / "Fog_Detection_Project_fogfocus_full",
            ROOT / "outputs" / "Fog_Detection_Project_fogfocus_full" / "runs" / "demo_run",
            ROOT / "outputs" / "Route_Eval_demo",
            training_summary,
            route_eval_summary,
            candidate_benchmark_summary,
            ROOT / "outputs" / "Candidate_Benchmark_demo",
            None,
        )

        self.assertIn("candidate_benchmark", summary)
        self.assertIn("candidate_benchmark_summary_json", summary["artifacts"])
        self.assertEqual(summary["candidate_benchmark"]["candidate"]["label"], "candidate_x")


if __name__ == "__main__":
    unittest.main()
