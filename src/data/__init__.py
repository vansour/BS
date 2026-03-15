#!/usr/bin/env python3
"""
数据子模块导出入口
Data Package Exports

该文件统一导出数据准备与深度估计相关组件，方便训练脚本和外部调用方直接导入：
- `MultiTaskDataset`
- `DepthEstimator`
- `precompute_depths`
- `DatasetPreparer`
"""

from .dataset import MultiTaskDataset
from .depth_estimator import DepthEstimator, precompute_depths
from .preparer import DatasetPreparer

# 仅暴露数据子包中最常用的核心符号。
__all__ = [
    "MultiTaskDataset",
    "DepthEstimator",
    "precompute_depths",
    "DatasetPreparer",
]


