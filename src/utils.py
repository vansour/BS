#!/usr/bin/env python3
"""
通用工具函数
Common Utility Functions

本模块收纳训练、推理和模型管理过程中常用的辅助逻辑，主要包括：
1. 随机种子设置，保证实验具有可复现性；
2. 模型参数统计与时间格式化；
3. CUDA 显存占用检查；
4. 权重文件定位、选择与加载。
"""

import os
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    设置随机种子，尽量保证实验结果可复现。

    该函数会同时固定以下随机源：
    - Python 内置 `random`
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA

    此外还会关闭 CuDNN 的自动 benchmark，并启用确定性模式，
    以减少不同运行之间的结果波动。

    Args:
        seed: 要设置的随机种子值，默认使用 42。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置为: {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型中需要训练的参数总量。

    这里只统计 `requires_grad=True` 的参数，因此返回值更适合用于衡量
    真正参与梯度更新的模型规模。

    Args:
        model: 需要统计参数量的 PyTorch 模型。

    Returns:
        int: 可训练参数的总数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    把秒数格式化为 `HH:MM:SS` 字符串。

    Args:
        seconds: 输入的秒数，可以是浮点数。

    Returns:
        str: 格式化后的时分秒字符串。
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def check_cuda_memory():
    """
    查询当前 CUDA 设备的显存使用情况。

    Returns:
        dict:
            - 如果当前环境没有可用 CUDA，返回 `{"error": "CUDA not available"}`；
            - 否则返回已分配显存、已保留显存、总显存、剩余显存和使用率。
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    # PyTorch 返回的显存单位是字节，这里统一转换为 GB，便于日志阅读。
    allocated = torch.cuda.memory_allocated() / 1024**3
    cached = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return {
        "allocated_gb": round(allocated, 2),
        "cached_gb": round(cached, 2),
        "total_gb": round(total, 2),
        "free_gb": round(total - allocated, 2),
        "usage_percent": round(allocated / total * 100, 1),
    }


def print_cuda_memory():
    """
    以人类可读的方式打印当前 CUDA 显存使用情况。

    这是 `check_cuda_memory()` 的轻量封装，适合在训练日志里直接调用。
    """
    mem_info = check_cuda_memory()
    if "error" in mem_info:
        print(mem_info["error"])
    else:
        print(
            f"当前 CUDA 显存: {mem_info['allocated_gb']:.2f}/{mem_info['total_gb']:.2f} GB "
            f"({mem_info['usage_percent']:.1f}%)"
        )


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    在指定目录中查找最新的 checkpoint 文件。

    查找规则：
    1. 只考虑扩展名为 `.pt` 的文件；
    2. 优先从文件名中提取 epoch 编号进行排序；
    3. 如果文件名中没有数字，则退化为按修改时间辅助排序。

    Args:
        checkpoint_dir: checkpoint 所在目录。

    Returns:
        Optional[str]: 找到时返回最新权重文件路径；否则返回 `None`。
    """
    path = Path(checkpoint_dir)
    if not path.exists() or not path.is_dir():
        return None

    checkpoint_files = [item for item in path.iterdir() if item.is_file() and item.suffix == ".pt"]
    if not checkpoint_files:
        return None

    def sort_key(item: Path):
        # 尝试从文件名中提取末尾数字作为 epoch，例如 `epoch_12.pt` -> 12。
        match = re.search(r"(\d+)(?=\.pt$)", item.name)
        epoch = int(match.group(1)) if match else -1
        return (epoch, item.stat().st_mtime)

    return str(sorted(checkpoint_files, key=sort_key)[-1])


def resolve_model_weights(
    output_dir: str,
    checkpoint_dir: Optional[str] = None,
    preferred_files: Optional[list[str]] = None,
) -> Optional[str]:
    """
    按优先级推断最适合加载的模型权重文件。

    优先级如下：
    1. 调用方显式给出的 `preferred_files`；
    2. 默认候选文件，例如 `unified_model.pt`、`unified_model_best.pt`；
    3. 如果以上都不存在，再尝试从 checkpoint 目录中寻找最新文件。

    Args:
        output_dir: 统一导出权重所在目录。
        checkpoint_dir: 训练过程中的 checkpoint 目录，可为空。
        preferred_files: 优先尝试的文件名列表，可为空。

    Returns:
        Optional[str]: 推断出的权重文件路径；若都不存在则返回 `None`。
    """
    preferred = preferred_files or [
        "unified_model.pt",
        "unified_model_best.pt",
    ]

    for filename in preferred:
        candidate = os.path.join(output_dir, filename)
        if os.path.exists(candidate):
            return candidate

    if checkpoint_dir:
        return find_latest_checkpoint(checkpoint_dir)

    return None


def load_model_weights(model: torch.nn.Module, weights_path: str, map_location: str = "cpu") -> dict:
    """
    把权重文件加载到指定模型中。

    支持两类输入格式：
    1. 纯 `state_dict`；
    2. 含有 `model_state_dict` 字段的训练 checkpoint。

    采用 `strict=False` 加载，因此即使存在少量缺失键或多余键，也会把信息
    返回给调用方，而不是直接终止程序。

    Args:
        model: 待加载权重的模型实例。
        weights_path: 权重文件路径。
        map_location: `torch.load()` 使用的设备映射，默认加载到 CPU。

    Returns:
        dict:
            - `source_type`：权重来源类型，取值为 `checkpoint` 或 `state_dict`；
            - `missing_keys`：模型中缺失但权重文件未提供的键；
            - `unexpected_keys`：权重文件中存在但模型未使用的键。
            - `skipped_mismatched_keys`：名称存在但张量形状不匹配，因此被跳过的键。
    """
    payload = torch.load(weights_path, map_location=map_location, weights_only=False)

    # 兼容训练保存的完整 checkpoint 与纯 state_dict 两种常见格式。
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        source_type = "checkpoint"
    else:
        state_dict = payload
        source_type = "state_dict"

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported weight payload type: {type(state_dict)!r}")

    # 兼容模型结构发生局部变化的情况，例如检测头类别数变化导致最后几层 shape 不一致。
    model_state = model.state_dict()
    compatible_state_dict = {}
    skipped_mismatched_keys = []

    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                compatible_state_dict[key] = value
            else:
                skipped_mismatched_keys.append(key)

    incompatible = model.load_state_dict(compatible_state_dict, strict=False)
    return {
        "source_type": source_type,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "skipped_mismatched_keys": skipped_mismatched_keys,
    }


if __name__ == "__main__":
    # 简单自检：确认工具函数模块可以被独立执行。
    set_seed(42)
    print("Utils module test passed.")



