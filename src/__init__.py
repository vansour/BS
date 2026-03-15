#!/usr/bin/env python3
"""
高速公路团雾监测项目主包
Highway Fog Monitoring System

该文件作为 `src` 包的统一导出入口，负责把配置、数据、模型、推理和工具模块
中最常用的符号重新汇总，方便外部脚本直接通过 `src.xxx` 的形式导入。
"""

__version__ = "1.0.0"
__author__ = "Roo"

from .config import Config, get_default_config
from .model import UnifiedMultiTaskModel, FogAugmentation
from .data import MultiTaskDataset, DepthEstimator, precompute_depths, DatasetPreparer
from .inference import HighwayFogSystem
from .utils import set_seed, count_parameters, format_time, check_cuda_memory, print_cuda_memory

# __all__ 用于显式声明本包对外公开的核心接口。
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


