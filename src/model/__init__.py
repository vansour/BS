#!/usr/bin/env python3
"""
模型子模块导出入口
Model Package Exports

该文件用于统一暴露 `src.model` 下最核心的两个组件：
1. `UnifiedMultiTaskModel`：统一多任务模型；
2. `FogAugmentation`：在线造雾增强算子。
"""

from .unified_model import UnifiedMultiTaskModel
from .fog_augmentation import FogAugmentation

# 通过 __all__ 显式声明对外公开的符号，便于 `from src.model import *`
# 这类导入方式只暴露项目真正希望外部使用的类。
__all__ = [
    "UnifiedMultiTaskModel",
    "FogAugmentation",
]



