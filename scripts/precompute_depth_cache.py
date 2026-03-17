#!/usr/bin/env python3
"""
离线深度缓存预计算脚本

本脚本用于一次性补齐训练集和验证集缺失的 MiDaS 深度缓存，从而避免在正式训练
或验证阶段再为缺失样本逐帧执行深度估计，减少中途因缓存缺失而中断的风险。

从工程定位上看，该脚本属于训练前准备工具，而非训练主流程的一部分。
其目标是把较重的深度估计工作前移到独立步骤中，以提升后续训练过程的稳定性
与可控性。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import Config
from src.data import MultiTaskDataset, precompute_depths


def count_depth_cache_files(depth_cache_dir: str) -> int:
    """
    统计当前深度缓存目录中的 `.npy` 文件数量。

    Args:
        depth_cache_dir: 深度缓存目录。

    Returns:
        int: 目录中实际存在的深度缓存文件数量。
    """
    return sum(1 for path in Path(depth_cache_dir).glob("*.npy") if path.is_file())


def count_missing_depth_cache_for_dataset(dataset: MultiTaskDataset) -> tuple[int, list[str]]:
    """
    统计指定数据集缺失的深度缓存数量，并返回少量样例文件名。
    """
    missing_count = 0
    missing_examples: list[str] = []

    for _, seq, img_name in dataset.samples:
        depth_name = f"{seq}_{img_name}.npy"
        depth_path = os.path.join(dataset.depth_cache_dir, depth_name)
        if os.path.exists(depth_path):
            continue

        missing_count += 1
        if len(missing_examples) < 5:
            missing_examples.append(depth_name)

    return missing_count, missing_examples


def main():
    """
    补全训练集和验证集所需的深度缓存。

    执行流程如下：
    1. 根据当前配置构建训练集和验证集索引；
    2. 统计已有深度缓存数量与 train/val 缺失数量；
    3. 对缺失样本执行深度预计算；
    4. 输出补算前后的缓存覆盖情况。
    """
    cfg = Config()

    train_dataset = MultiTaskDataset(
        cfg.RAW_DATA_DIR,
        cfg.DEPTH_CACHE_DIR,
        xml_dir=None,
        transform=None,
        is_train=True,
        frame_stride=cfg.FRAME_STRIDE,
    )
    val_dataset = MultiTaskDataset(
        cfg.RAW_DATA_DIR,
        cfg.DEPTH_CACHE_DIR,
        xml_dir=None,
        transform=None,
        is_train=False,
        frame_stride=cfg.FRAME_STRIDE,
    )

    train_samples = len(train_dataset)
    val_samples = len(val_dataset)
    total_samples = train_samples + val_samples
    cache_before = count_depth_cache_files(cfg.DEPTH_CACHE_DIR)
    train_missing_before, train_missing_examples = count_missing_depth_cache_for_dataset(train_dataset)
    val_missing_before, val_missing_examples = count_missing_depth_cache_for_dataset(val_dataset)
    missing_before = train_missing_before + val_missing_before

    print(f"Using device: {cfg.DEVICE}")
    print(f"Frame stride: {cfg.FRAME_STRIDE}")
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Total samples: {total_samples}")
    print(f"Depth cache before: {cache_before}")
    print(f"Missing train depth files before: {train_missing_before}")
    print(f"Missing val depth files before: {val_missing_before}")
    print(f"Missing depth files before: {missing_before}")
    if train_missing_examples:
        print(f"Sample missing train depth files: {train_missing_examples}")
    if val_missing_examples:
        print(f"Sample missing val depth files: {val_missing_examples}")

    if total_samples == 0:
        raise RuntimeError(f"Train/val datasets are empty: {cfg.RAW_DATA_DIR}")

    if missing_before == 0:
        print("Depth cache is already complete for both training and validation datasets. No work needed.")
        return

    if train_missing_before > 0:
        print("Precomputing missing training depth cache...")
        precompute_depths(train_dataset, cfg.DEVICE)
    if val_missing_before > 0:
        print("Precomputing missing validation depth cache...")
        precompute_depths(val_dataset, cfg.DEVICE)

    cache_after = count_depth_cache_files(cfg.DEPTH_CACHE_DIR)
    train_missing_after, _ = count_missing_depth_cache_for_dataset(train_dataset)
    val_missing_after, _ = count_missing_depth_cache_for_dataset(val_dataset)
    missing_after = train_missing_after + val_missing_after

    print(f"Depth cache after: {cache_after}")
    print(f"Missing train depth files after: {train_missing_after}")
    print(f"Missing val depth files after: {val_missing_after}")
    print(f"Missing depth files after: {missing_after}")

    if missing_after > 0:
        print("Some depth files are still missing. Review the log above for failed images.")
    else:
        print("Depth cache precomputation completed successfully for both training and validation datasets.")


if __name__ == "__main__":
    main()
