#!/usr/bin/env python3
"""
模型子模块导出入口
Model Package Exports

该文件用于统一暴露 `src.model` 下最核心的两个组件：
1. `UnifiedMultiTaskModel`：统一多任务模型；
2. `FogAugmentation`：在线造雾增强算子。

保持该入口文件足够轻量，有利于：
1. 让训练、推理和导出脚本统一通过 `src.model` 导入核心模型组件；
2. 降低上层代码对底层文件组织方式的依赖；
3. 在论文和答辩材料中更清晰地说明模型层的公开接口。
"""

from .unified_model import UnifiedMultiTaskModel
from .fog_augmentation import FogAugmentation

# 通过 __all__ 显式声明对外公开的符号，便于 `from src.model import *`。
# 这里有意只导出两个稳定接口，不把内部辅助实现暴露出去，
# 这样后续即使重构文件内部结构，也能尽量保持对外导入路径不变。
__all__ = [
    "UnifiedMultiTaskModel",
    "FogAugmentation",
]



