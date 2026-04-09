#!/usr/bin/env python3
"""
高速公路团雾监测项目主包
Highway Fog Monitoring System

本文件是项目源码包 `src` 的统一导出入口，用于将配置、数据、模型、推理和
工具模块中的核心符号集中暴露给上层脚本调用。

当前实现使用懒加载导出，而不是在导入 `src` 时一次性导入全部子模块。
这样做有两个直接收益：
1. `from src.config import Config` 不再因为 `ultralytics` 等重依赖缺失而失败；
2. 配置检查、环境自检和轻量脚本可以只加载自己真正需要的模块。
"""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "1.0.0"
__author__ = "Roo"

__all__ = [
    "Config",
    "get_default_config",
    "UnifiedMultiTaskModel",
    "FogAugmentation",
    "MultiTaskDataset",
    "DepthEstimator",
    "precompute_depths",
    "DatasetPreparer",
    "HighwayFogSystem",
    "set_seed",
    "count_parameters",
    "format_time",
    "check_cuda_memory",
    "print_cuda_memory",
]

_EXPORTS = {
    "Config": (".config", "Config"),
    "get_default_config": (".config", "get_default_config"),
    "UnifiedMultiTaskModel": (".model", "UnifiedMultiTaskModel"),
    "FogAugmentation": (".model", "FogAugmentation"),
    "MultiTaskDataset": (".data", "MultiTaskDataset"),
    "DepthEstimator": (".data", "DepthEstimator"),
    "precompute_depths": (".data", "precompute_depths"),
    "DatasetPreparer": (".data", "DatasetPreparer"),
    "HighwayFogSystem": (".inference", "HighwayFogSystem"),
    "set_seed": (".utils", "set_seed"),
    "count_parameters": (".utils", "count_parameters"),
    "format_time": (".utils", "format_time"),
    "check_cuda_memory": (".utils", "check_cuda_memory"),
    "print_cuda_memory": (".utils", "print_cuda_memory"),
}


def __getattr__(name):
    """按需导入对外暴露的符号，避免包初始化阶段触发重依赖加载。"""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    """让交互式补全仍然看到懒加载导出的公共接口。"""
    return sorted(set(globals().keys()) | set(__all__))


if TYPE_CHECKING:
    from .config import Config, get_default_config
    from .data import (
        DatasetPreparer,
        DepthEstimator,
        MultiTaskDataset,
        precompute_depths,
    )
    from .inference import HighwayFogSystem
    from .model import FogAugmentation, UnifiedMultiTaskModel
    from .utils import (
        check_cuda_memory,
        count_parameters,
        format_time,
        print_cuda_memory,
        set_seed,
    )
