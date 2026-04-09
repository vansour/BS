#!/usr/bin/env python3
"""
通用工具函数
Common Utility Functions

本模块收纳训练、推理和模型管理过程中常用的辅助逻辑，主要包括：
1. 随机种子设置，保证实验具有可复现性；
2. 模型参数统计与时间格式化；
3. CUDA 显存占用检查；
4. 权重文件定位、选择与加载。

这些函数在功能上彼此相对独立，但共同承担了“收口通用工程细节”的职责。
通过将随机性控制、设备状态查询、权重解析与兼容加载等逻辑集中到本模块，
训练、推理和导出脚本便可以更聚焦于各自主流程，从而提升整体代码结构的清晰度。
"""

import os
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


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

    需要说明的是，随机种子控制只能尽量降低实验结果波动，并不能在所有硬件、
    所有算子和所有第三方库组合下保证绝对逐位一致的结果。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # 允许通过环境变量在“确定性”和“性能优先”之间切换。
        # 这两个选项往往不能同时达到最优，需要根据实验目标取舍。
        deterministic = os.getenv("BS_CUDNN_DETERMINISTIC", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        benchmark = os.getenv(
            "BS_CUDNN_BENCHMARK",
            "0" if deterministic else "1",
        ).lower() in {"1", "true", "yes", "on"}

        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        print(
            "CuDNN config: "
            f"deterministic={torch.backends.cudnn.deterministic}, "
            f"benchmark={torch.backends.cudnn.benchmark}"
        )

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
    # 这里只统计 requires_grad=True 的参数，更符合“训练规模”的实际含义。
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
    # allocated 是已经实际分配给张量的显存；
    # reserved 是 CUDA caching allocator 已经向驱动申请但未必都在用的显存。
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


def split_sequence_names(
    sequence_names: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    按序列级执行可复现的 train/val 切分。

    Args:
        sequence_names: 待切分的序列名列表。
        train_ratio: 训练集比例，取值范围为 `[0, 1]`。
        seed: 随机种子，用于保证切分结果可复现。

    Returns:
        tuple[list[str], list[str]]: `(train_sequences, val_sequences)`。

    这里先按名字排序，再基于独立随机源打乱，确保：
    1. 同一输入集合在相同种子下结果稳定；
    2. 不依赖外部全局 `random` 状态；
    3. 训练主链和离线辅助链能共享完全一致的切分规则。
    """
    if not 0.0 <= float(train_ratio) <= 1.0:
        raise ValueError(f"train_ratio must be within [0, 1], got {train_ratio!r}")

    ordered_sequences = sorted(sequence_names)
    shuffled_sequences = list(ordered_sequences)
    random.Random(int(seed)).shuffle(shuffled_sequences)
    split_idx = int(len(shuffled_sequences) * float(train_ratio))
    return shuffled_sequences[:split_idx], shuffled_sequences[split_idx:]


def _normalize_target_shape(target_size: int | tuple[int, int]) -> tuple[int, int]:
    """
    将目标尺寸统一解析为 `(height, width)`。

    Args:
        target_size: 目标边长或显式的 `(height, width)`。

    Returns:
        tuple[int, int]: 规范化后的目标尺寸。
    """
    if isinstance(target_size, int):
        return target_size, target_size
    if len(target_size) != 2:
        raise ValueError(f"Unsupported target size: {target_size!r}")
    return int(target_size[0]), int(target_size[1])


def compute_letterbox_metadata(
    src_shape: tuple[int, int],
    target_size: int | tuple[int, int],
) -> dict[str, object]:
    """
    计算 letterbox 缩放与填充参数。

    Args:
        src_shape: 原始尺寸 `(height, width)`。
        target_size: 目标边长或 `(height, width)`。

    Returns:
        dict[str, object]: 包含缩放比例、padding 和输入输出尺寸的元数据。
    """
    src_h, src_w = int(src_shape[0]), int(src_shape[1])
    dst_h, dst_w = _normalize_target_shape(target_size)
    scale = min(dst_h / max(src_h, 1), dst_w / max(src_w, 1))
    resized_h = max(1, int(round(src_h * scale)))
    resized_w = max(1, int(round(src_w * scale)))
    pad_h = dst_h - resized_h
    pad_w = dst_w - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return {
        "scale": float(scale),
        "src_shape": (src_h, src_w),
        "target_shape": (dst_h, dst_w),
        "resized_shape": (resized_h, resized_w),
        "pad": (pad_left, pad_top, pad_right, pad_bottom),
    }


def letterbox_tensor(
    tensor: torch.Tensor,
    target_size: int | tuple[int, int],
    pad_value: float = 114.0 / 255.0,
) -> tuple[torch.Tensor, dict[str, object]]:
    """
    对 `CHW` 张量执行按比例缩放 + padding 的 letterbox。

    Args:
        tensor: 输入张量，形状为 `(C, H, W)`。
        target_size: 目标边长或 `(height, width)`。
        pad_value: padding 常量值。

    Returns:
        tuple[torch.Tensor, dict[str, object]]:
            处理后的张量，以及用于框坐标反变换的元数据。
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"letterbox_tensor expects a CHW tensor, got shape={tuple(tensor.shape)}"
        )

    metadata = compute_letterbox_metadata(tuple(tensor.shape[-2:]), target_size)
    resized_h, resized_w = metadata["resized_shape"]
    pad_left, pad_top, pad_right, pad_bottom = metadata["pad"]

    resized = F.interpolate(
        tensor.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    padded = F.pad(
        resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=pad_value,
    )
    return padded, metadata


def apply_letterbox_to_boxes_xyxy(
    boxes_xyxy: torch.Tensor,
    metadata: dict[str, object],
) -> torch.Tensor:
    """
    按 letterbox 元数据把绝对坐标 `xyxy` 框映射到目标尺寸。

    Args:
        boxes_xyxy: 原始图像坐标系下的 `xyxy` 框。
        metadata: `compute_letterbox_metadata()` 返回的元数据。

    Returns:
        torch.Tensor: 映射到目标尺寸后的 `xyxy` 框。
    """
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy

    scale = float(metadata["scale"])
    pad_left, pad_top, _, _ = metadata["pad"]
    transformed = boxes_xyxy.clone()
    transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale + pad_left
    transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale + pad_top
    return transformed


def invert_letterbox_boxes_xyxy(
    boxes_xyxy: torch.Tensor,
    metadata: dict[str, object],
) -> torch.Tensor:
    """
    将 letterbox 坐标系下的 `xyxy` 框映射回原始图像坐标。

    Args:
        boxes_xyxy: letterbox 输入尺寸上的 `xyxy` 框。
        metadata: `compute_letterbox_metadata()` 返回的元数据。

    Returns:
        torch.Tensor: 原始图像坐标系下的 `xyxy` 框。
    """
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy

    scale = float(metadata["scale"])
    src_h, src_w = metadata["src_shape"]
    pad_left, pad_top, _, _ = metadata["pad"]

    restored = boxes_xyxy.clone().float()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - pad_left) / max(scale, 1e-6)
    restored[:, [1, 3]] = (restored[:, [1, 3]] - pad_top) / max(scale, 1e-6)
    restored[:, [0, 2]] = restored[:, [0, 2]].clamp(0, src_w)
    restored[:, [1, 3]] = restored[:, [1, 3]].clamp(0, src_h)
    return restored


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

    该函数主要服务于断点续训与推理权重自动回退场景。
    """
    path = Path(checkpoint_dir)
    if not path.exists() or not path.is_dir():
        return None

    checkpoint_files = [
        item for item in path.iterdir() if item.is_file() and item.suffix == ".pt"
    ]
    if not checkpoint_files:
        return None

    def sort_key(item: Path):
        # 尝试从文件名中提取末尾数字作为 epoch，例如 `epoch_12.pt` -> 12。
        # 如果提取不到数字，则 epoch 记为 -1，再由修改时间辅助排序。
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

    该函数的目标不是返回“唯一正确”的文件，而是在当前项目约定下给出
    最合理的默认权重选择结果。
    """
    # 允许调用方显式指定优先候选文件名列表，
    # 没有指定时才使用项目内默认命名。
    preferred = preferred_files or [
        "unified_model.pt",
        "unified_model_best.pt",
    ]

    for filename in preferred:
        candidate = os.path.join(output_dir, filename)
        if os.path.exists(candidate):
            return candidate

    # 当正式导出文件不存在时，再回退到训练过程中的 checkpoint。
    if checkpoint_dir:
        return find_latest_checkpoint(checkpoint_dir)

    return None


def load_model_weights(
    model: torch.nn.Module,
    weights_path: str,
    map_location: str = "cpu",
    exclude_prefixes: Optional[list[str]] = None,
) -> dict:
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
        exclude_prefixes: 可选的参数名前缀黑名单；命中前缀的权重不会加载。

    Returns:
        dict:
            - `source_type`：权重来源类型，取值为 `checkpoint` 或 `state_dict`；
            - `missing_keys`：模型中缺失但权重文件未提供的键；
            - `unexpected_keys`：权重文件中存在但模型未使用的键。
            - `skipped_mismatched_keys`：名称存在但张量形状不匹配，因此被跳过的键。

    这个函数的重点不是“严格保证一模一样”，而是尽量提高模型版本演进时的可恢复性。
    在本项目里，最常见的兼容场景是检测头类别数变化，导致部分最后层的权重形状不匹配。
    这时如果使用严格加载，整个流程会直接失败；而这里会保留能复用的部分并报告被跳过的键。

    因此，该函数更适合作为研发阶段和结构演进阶段的权重恢复工具；
    若进入严格可复现的正式部署阶段，仍应优先使用结构完全一致的权重文件。
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

    excluded_prefixes = tuple(exclude_prefixes or [])

    for key, value in state_dict.items():
        if excluded_prefixes and key.startswith(excluded_prefixes):
            continue
        if key in model_state:
            if model_state[key].shape == value.shape:
                compatible_state_dict[key] = value
            else:
                # 只要名称相同但 shape 不同，就明确记录为“跳过的 shape 不匹配键”。
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
