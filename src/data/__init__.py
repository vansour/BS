#!/usr/bin/env python3
"""
数据子模块导出入口
Data Package Exports

该文件统一导出数据准备与深度估计相关组件，方便训练脚本和外部调用方直接导入：
- `MultiTaskDataset`
- `DepthEstimator`
- `precompute_depths`
- `DatasetPreparer`

本文件本身不承载业务逻辑，其作用在于为上层代码提供统一、稳定的数据层导入入口，
从而避免外部调用方直接依赖具体的文件拆分结构。
"""

from .dataset import MultiTaskDataset
from .depth_estimator import DepthEstimator, precompute_depths
from .preparer import DatasetPreparer

# 仅暴露数据子包中最常用的核心符号，避免把内部实现细节扩散到对外接口。
__all__ = [
    "MultiTaskDataset",
    "DepthEstimator",
    "precompute_depths",
    "DatasetPreparer",
]


