#!/usr/bin/env python3
"""
高速公路团雾监测项目主包
Highway Fog Monitoring System

本文件是项目源码包 `src` 的统一导出入口，用于将配置、数据、模型、推理和
工具模块中的核心符号集中暴露给上层脚本调用。

从工程组织角度看，这种设计具有以下价值：
1. 统一上层导入路径，降低脚本之间的耦合度；
2. 在内部模块重构时尽量维持对外接口稳定；
3. 便于在论文、答辩或项目说明材料中清晰界定系统对外公开的核心能力边界。
"""

__version__ = "1.0.0"
__author__ = "Roo"

from .config import Config, get_default_config
from .model import UnifiedMultiTaskModel, FogAugmentation
from .data import MultiTaskDataset, DepthEstimator, precompute_depths, DatasetPreparer
from .inference import HighwayFogSystem
from .utils import set_seed, count_parameters, format_time, check_cuda_memory, print_cuda_memory

# __all__ 用于显式声明本包对外公开的核心接口。
# 这样既能增强可读性，也能在 `from src import *` 时控制真正暴露的符号集合。
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


