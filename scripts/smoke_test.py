#!/usr/bin/env python3
"""
训练前 smoke test。

该脚本用于在正式训练前快速确认以下事项：
1. 当前 Python 环境是否装齐核心依赖；
2. `src/config.py` 的真实数据路径是否可用；
3. UA-DETRAC 训练/验证数据集能否正常索引；
4. 深度缓存覆盖情况是否至少符合当前预期；
5. 可选地执行一次随机张量前向，确认模型主链路可初始化。

默认模式尽量避免触发大规模下载或重计算，因此更适合作为“训练前检查”而不是
“完整集成测试”。如果需要进一步验证模型初始化，可显式加上 `--check-forward`。
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib import metadata, util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_DEPENDENCIES = [
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("tqdm", "tqdm"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("cv2", "opencv-python"),
    ("ultralytics", "ultralytics"),
    ("timm", "timm"),
]

OPTIONAL_DEPENDENCIES = [
    ("onnx", "onnx", "仅导出 ONNX 时需要"),
    ("tensorrt", "tensorrt", "仅 TensorRT 部署时需要"),
]


def dependency_report():
    """返回核心依赖与可选依赖的安装状态。"""
    missing: list[str] = []
    installed: dict[str, str] = {}
    optional: dict[str, str] = {}

    for import_name, dist_name in REQUIRED_DEPENDENCIES:
        if util.find_spec(import_name) is None:
            missing.append(dist_name)
            continue
        installed[dist_name] = metadata.version(dist_name)

    for import_name, dist_name, note in OPTIONAL_DEPENDENCIES:
        if util.find_spec(import_name) is None:
            optional[dist_name] = f"missing ({note})"
        else:
            optional[dist_name] = metadata.version(dist_name)

    return missing, installed, optional


def count_missing_depth_cache(
    dataset, limit: int | None = None
) -> tuple[int, list[str], int]:
    """统计数据集中缺失的深度缓存数量。"""
    missing_count = 0
    missing_examples: list[str] = []
    checked = 0

    for _, seq, img_name in dataset.samples:
        depth_name = f"{seq}_{img_name}.npy"
        depth_path = os.path.join(dataset.depth_cache_dir, depth_name)
        checked += 1
        if os.path.exists(depth_path):
            if limit is not None and checked >= limit:
                break
            continue

        missing_count += 1
        if len(missing_examples) < 5:
            missing_examples.append(depth_name)
        if limit is not None and checked >= limit:
            break

    return missing_count, missing_examples, checked


def fail(message: str, failures: list[str]) -> None:
    """记录失败项。"""
    print(f"[FAIL] {message}")
    failures.append(message)


def warn(message: str, warnings: list[str]) -> None:
    """记录警告项。"""
    print(f"[WARN] {message}")
    warnings.append(message)


def ok(message: str) -> None:
    """输出通过项。"""
    print(f"[ OK ] {message}")


def run_forward_check(cfg, device: str) -> None:
    """
    执行一次随机张量前向。

    该检查会初始化 `UnifiedMultiTaskModel`，若基础 YOLO 权重不在本地，
    Ultralytics 可能在第一次运行时尝试下载对应权重。
    """
    import torch

    from src.model import FogAugmentation, UnifiedMultiTaskModel

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Forward check requested CUDA, but torch.cuda.is_available() is False."
        )

    model_hint = cfg.YOLO_BASE_MODEL
    if not os.path.exists(model_hint):
        print(
            "[INFO] "
            f"Base model {model_hint!r} is not a local file. "
            "Ultralytics may download it on first initialization."
        )

    model = UnifiedMultiTaskModel(
        cfg.YOLO_BASE_MODEL,
        cfg.NUM_FOG_CLASSES,
        num_det_classes=cfg.NUM_DET_CLASSES,
    ).to(device)
    model.eval()

    fog_augmenter = FogAugmentation(cfg).to(device).eval()

    batch_size = 2
    dummy_images = torch.rand(batch_size, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device)
    dummy_depths = torch.rand(batch_size, 1, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device)

    with torch.no_grad():
        foggy_images, fog_labels, beta_labels = fog_augmenter(
            dummy_images, dummy_depths
        )
        det_out, fog_cls, fog_reg = model(foggy_images)

    if tuple(foggy_images.shape) != (batch_size, 3, cfg.IMG_SIZE, cfg.IMG_SIZE):
        raise RuntimeError(
            f"Unexpected fog augmentation output shape: {tuple(foggy_images.shape)}"
        )
    if tuple(fog_cls.shape) != (batch_size, cfg.NUM_FOG_CLASSES):
        raise RuntimeError(
            f"Unexpected fog classifier output shape: {tuple(fog_cls.shape)}"
        )
    if tuple(fog_reg.shape) != (batch_size,):
        raise RuntimeError(
            f"Unexpected fog regressor output shape: {tuple(fog_reg.shape)}"
        )

    det_shape = (
        tuple(det_out.shape) if hasattr(det_out, "shape") else type(det_out).__name__
    )
    print(
        f"[INFO] Forward check outputs: det={det_shape}, fog_cls={tuple(fog_cls.shape)}, fog_reg={tuple(fog_reg.shape)}"
    )
    print(
        f"[INFO] Random fog labels: {fog_labels.tolist()}, beta labels range=[{float(beta_labels.min()):.4f}, {float(beta_labels.max()):.4f}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a lightweight pre-training smoke test."
    )
    parser.add_argument(
        "--check-forward",
        action="store_true",
        help="初始化模型并执行一次随机张量前向；首次运行可能触发 YOLO 权重下载。",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="覆盖 smoke test 使用的设备；默认沿用 src.config.Config.DEVICE。",
    )
    parser.add_argument(
        "--full-depth-scan",
        action="store_true",
        help="扫描全部 train/val 样本的深度缓存覆盖情况；默认只抽样前 256 个样本。",
    )
    parser.add_argument(
        "--depth-scan-limit",
        type=int,
        default=256,
        help="未启用 --full-depth-scan 时，每个数据集最多检查多少个样本。",
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    print("== Dependency Check ==")
    missing_deps, installed_deps, optional_deps = dependency_report()
    if missing_deps:
        fail(
            "Missing required dependencies: "
            + ", ".join(sorted(missing_deps))
            + ". Install them with `python -m pip install -r requirements.txt`.",
            failures,
        )
    else:
        ok("All required runtime dependencies are installed.")
    for name, version in sorted(installed_deps.items()):
        print(f"  - {name}: {version}")
    for name, status in sorted(optional_deps.items()):
        print(f"  - {name}: {status}")

    if failures:
        print(
            "\nSmoke test stopped before project import because the runtime environment is incomplete."
        )
        return 1

    print("\n== Config Check ==")
    from src.config import Config
    from src.data import MultiTaskDataset

    cfg = Config()
    device = args.device or cfg.DEVICE
    print(f"  - device: {device}")
    print(f"  - yolo_base_model: {cfg.YOLO_BASE_MODEL}")
    print(f"  - img_size: {cfg.IMG_SIZE}")
    print(f"  - frame_stride: {cfg.FRAME_STRIDE}")
    print(f"  - train_ratio: {cfg.TRAIN_RATIO}")
    print(f"  - seed: {cfg.SEED}")
    for key, value in cfg.path_summary().items():
        print(f"  - {key}: {value}")

    path_checks = {
        "raw_data_dir": Path(cfg.RAW_DATA_DIR),
        "xml_dir": Path(cfg.XML_DIR),
        "depth_cache_dir": Path(cfg.DEPTH_CACHE_DIR),
        "output_dir": Path(cfg.OUTPUT_DIR),
        "checkpoint_dir": Path(cfg.CHECKPOINT_DIR),
    }
    for label, path in path_checks.items():
        if label in {"raw_data_dir", "xml_dir"}:
            if path.exists():
                ok(f"{label} exists: {path}")
            else:
                fail(f"{label} does not exist: {path}", failures)
        else:
            if path.exists():
                ok(f"{label} is ready: {path}")
            else:
                fail(f"{label} was not created: {path}", failures)

    print("\n== Dataset Index Check ==")
    if not failures:
        train_ds = MultiTaskDataset(
            cfg.RAW_DATA_DIR,
            cfg.DEPTH_CACHE_DIR,
            xml_dir=cfg.XML_DIR,
            transform=None,
            is_train=True,
            frame_stride=cfg.FRAME_STRIDE,
            det_train_class_id=cfg.DET_TRAIN_CLASS_ID,
            img_size=cfg.IMG_SIZE,
            keep_ratio=True,
            train_ratio=cfg.TRAIN_RATIO,
            split_seed=cfg.SEED,
        )
        val_ds = MultiTaskDataset(
            cfg.RAW_DATA_DIR,
            cfg.DEPTH_CACHE_DIR,
            xml_dir=cfg.XML_DIR,
            transform=None,
            is_train=False,
            frame_stride=cfg.FRAME_STRIDE,
            det_train_class_id=cfg.DET_TRAIN_CLASS_ID,
            img_size=cfg.IMG_SIZE,
            keep_ratio=True,
            train_ratio=cfg.TRAIN_RATIO,
            split_seed=cfg.SEED,
        )

        print(f"  - train samples: {len(train_ds)}")
        print(f"  - val samples: {len(val_ds)}")
        print(f"  - preloaded train XML sequences: {len(train_ds.annotations)}")
        print(f"  - preloaded val XML sequences: {len(val_ds.annotations)}")

        if len(train_ds) == 0:
            fail("Training dataset is empty after indexing.", failures)
        else:
            ok("Training dataset indexing succeeded.")
        if len(val_ds) == 0:
            warn("Validation dataset is empty after indexing.", warnings)
        else:
            ok("Validation dataset indexing succeeded.")

        scan_limit = None if args.full_depth_scan else max(1, args.depth_scan_limit)
        train_missing, train_examples, train_checked = count_missing_depth_cache(
            train_ds, limit=scan_limit
        )
        val_missing, val_examples, val_checked = count_missing_depth_cache(
            val_ds, limit=scan_limit
        )

        print(f"  - depth cache checked: train={train_checked}, val={val_checked}")
        if train_missing == 0:
            ok("No missing train depth cache was found in the checked sample range.")
        else:
            warn(
                f"Train depth cache is incomplete in checked samples: missing={train_missing}, examples={train_examples}",
                warnings,
            )
        if val_missing == 0:
            ok(
                "No missing validation depth cache was found in the checked sample range."
            )
        else:
            warn(
                f"Validation depth cache is incomplete in checked samples: missing={val_missing}, examples={val_examples}",
                warnings,
            )

    if args.check_forward and not failures:
        print("\n== Forward Check ==")
        try:
            run_forward_check(cfg, device)
            ok("Random-tensor forward check succeeded.")
        except Exception as exc:
            fail(f"Forward check failed: {exc}", failures)

    print("\n== Summary ==")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for item in warnings:
            print(f"  - {item}")
    else:
        print("Warnings: 0")

    if failures:
        print(f"Failures: {len(failures)}")
        for item in failures:
            print(f"  - {item}")
        print("Smoke test FAILED.")
        return 1

    print("Failures: 0")
    print("Smoke test PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
