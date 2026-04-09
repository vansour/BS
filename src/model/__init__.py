#!/usr/bin/env python3
"""
模型子模块导出入口
Model Package Exports

该文件用于统一暴露 `src.model` 下最核心的两个组件：
1. `UnifiedMultiTaskModel`：统一多任务模型；
2. `FogAugmentation`：在线造雾增强算子。

这里使用懒加载导出，避免只导入 `FogAugmentation` 时也强制加载整套 YOLO 依赖。
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "UnifiedMultiTaskModel",
    "FogAugmentation",
]

_EXPORTS = {
    "UnifiedMultiTaskModel": (".unified_model", "UnifiedMultiTaskModel"),
    "FogAugmentation": (".fog_augmentation", "FogAugmentation"),
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
    from .fog_augmentation import FogAugmentation
    from .unified_model import UnifiedMultiTaskModel
