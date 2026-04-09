#!/usr/bin/env python3
"""
数据子模块导出入口
Data Package Exports

本文件统一导出数据准备与深度估计相关组件，方便训练脚本和外部调用方直接导入：
- `MultiTaskDataset`
- `DepthEstimator`
- `precompute_depths`
- `DatasetPreparer`

这里同样采用懒加载，避免只想读取某个数据类时就顺带导入全部数据子模块。
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "MultiTaskDataset",
    "DepthEstimator",
    "precompute_depths",
    "DatasetPreparer",
]

_EXPORTS = {
    "MultiTaskDataset": (".dataset", "MultiTaskDataset"),
    "DepthEstimator": (".depth_estimator", "DepthEstimator"),
    "precompute_depths": (".depth_estimator", "precompute_depths"),
    "DatasetPreparer": (".preparer", "DatasetPreparer"),
}


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))


if TYPE_CHECKING:
    from .dataset import MultiTaskDataset
    from .depth_estimator import DepthEstimator, precompute_depths
    from .preparer import DatasetPreparer
