#!/usr/bin/env python3
"""
深度估计器
Depth Estimator

本模块封装了基于 MiDaS 的深度估计流程，负责：
1. 加载预训练深度模型；
2. 计算单张图像的深度图；
3. 批量处理图像；
4. 为训练数据集预计算并缓存深度图。

在本项目中，深度并非最终研究目标，而是在线造雾模块所依赖的中间条件变量。
因此本模块的目标并非恢复绝对精确的度量深度，而是获得能够稳定反映场景相对远近
关系的深度先验，以支撑后续基于大气散射模型的雾化合成。
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _ensure_torch_hub_trusted(repo_owner_name: str) -> None:
    """
    确保指定仓库已写入 torch.hub 的信任名单。

    MiDaS_small 在加载过程中会递归依赖
    `rwightman/gen-efficientnet-pytorch`。在无交互脚本环境里，
    如果该仓库尚未出现在 `trusted_list` 中，`torch.hub` 会阻塞在
    `input()` 提示上并直接抛出 `EOFError`。这里在真正调用
    `torch.hub.load()` 之前主动补齐信任名单，避免训练前深度预计算被交互式提示卡死。
    """
    hub_dir = Path(torch.hub.get_dir())
    trusted_list = hub_dir / "trusted_list"
    trusted_list.parent.mkdir(parents=True, exist_ok=True)
    trusted_list.touch(exist_ok=True)

    existing = {
        line.strip()
        for line in trusted_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if repo_owner_name in existing:
        return

    with trusted_list.open("a", encoding="utf-8") as handle:
        handle.write(repo_owner_name + "\n")


class DepthEstimator:
    """
    基于 MiDaS 的深度估计器。

    该类把深度估计逻辑集中起来，便于在训练准备阶段批量生成深度缓存，
    从而避免训练期间重复执行较重的深度推理。

    当前实现依赖 `torch.hub` 加载 MiDaS 模型。该方案能够降低接入门槛，
    但也意味着首次运行可能需要联网下载或依赖本地缓存。
    """

    def __init__(self, model_name="MiDaS_small", device=None):
        """
        初始化深度估计器。

        Args:
            model_name: MiDaS 模型名称，例如 `MiDaS_small`。
            device: 指定推理设备；若为空则自动选择 CUDA 或 CPU。

        `MiDaS_small` 被默认选中，主要原因包括：
        - 更适合本项目这种需要批量预计算的场景；
        - 推理速度和显存压力相对更友好；
        - 深度图只作为雾化条件，不要求极致精度。
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"正在初始化 MiDaS 深度估计引擎，设备: {self.device}")

        try:
            # MiDaS_small 会级联加载 EfficientNet Lite 骨干；在脚本环境中提前写入
            # trusted_list，可避免 torch.hub 触发无法回答的交互式确认提示。
            _ensure_torch_hub_trusted("intel-isl_MiDaS")
            _ensure_torch_hub_trusted("rwightman_gen-efficientnet-pytorch")
            self.model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
            self.model.to(self.device).eval()
            self.transform = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
            ).small_transform
            print(f"MiDaS 模型加载成功: {model_name}")
        except Exception as e:
            print(f"MiDaS 模型加载失败: {e}")
            raise

    def compute_depth(self, image_rgb):
        """
        计算单张图像的深度图。

        Args:
            image_rgb: RGB 图像，`numpy` 数组格式，形状为 `(H, W, 3)`。

        Returns:
            np.ndarray: 归一化后的深度图，形状为 `(H, W)`，
            其中 `0` 表示近处，`1` 表示远处。

        当前输出并非 MiDaS 的原始预测值，而是经过以下处理后得到的项目内部深度定义：
        1. 插值回原图大小；
        2. 归一化到 `[0, 1]`；
        3. 方向翻转；

        最后的 `1.0 - depth` 很关键，因为项目后续约定是：
        - 0 表示近；
        - 1 表示远；
        这样更符合在线造雾时“远处受雾影响更明显”的直觉。
        """
        h, w = image_rgb.shape[:2]

        # 先执行 MiDaS 官方预处理，再送入模型推理。
        input_batch = self.transform(image_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # 将深度结果缩放到 [0, 1] 区间，并反转方向以符合本项目的远近约定。
        # 这里不保留原始物理量纲，因为后续造雾只需要相对深浅关系。
        depth = prediction.cpu().numpy()
        d_min, d_max = depth.min(), depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-6)
        depth = 1.0 - depth

        return depth

    def compute_depth_batch(self, images_rgb, batch_size=4):
        """
        批量计算多张图像的深度图。

        Args:
            images_rgb: RGB 图像列表，每个元素为 `(H, W, 3)` 的 `numpy` 数组。
            batch_size: 分批处理大小，用于平衡速度与显存占用。

        Returns:
            list[np.ndarray]: 深度图列表。

        该接口名称虽然包含 batch，但当前实现内部仍然逐张调用 `compute_depth()`。
        保留 `batch_size` 参数的目的，是为后续如需切换为真正批量推理时预留接口兼容性。
        """
        depths = []

        for i in range(0, len(images_rgb), batch_size):
            batch_imgs = images_rgb[i:i + batch_size]
            batch_depths = []

            for img in batch_imgs:
                depth = self.compute_depth(img)
                batch_depths.append(depth)

            depths.extend(batch_depths)

        return depths


def precompute_depths(dataset, device, batch_size=4):
    """
    为整个数据集预计算深度图并写入缓存。

    Args:
        dataset: `MultiTaskDataset` 实例。
        device: 当前计算设备。
        batch_size: 预留的批处理参数，便于后续扩展批量计算逻辑。

    本函数的职责是补齐缺失缓存，而不是重新覆盖全部缓存。
    因此在遍历过程中，已存在的 `.npy` 文件会被直接跳过。
    """
    print(f"开始深度图预计算，总样本数: {len(dataset)}")

    estimator = DepthEstimator(device=device)

    for i in tqdm(range(len(dataset)), desc="Depth Pre-computation"):
        img_path, seq, img_name = dataset.samples[i]
        depth_name = f"{seq}_{img_name}.npy"
        depth_path = os.path.join(dataset.depth_cache_dir, depth_name)

        # 已存在缓存时直接跳过，避免重复计算。
        if os.path.exists(depth_path):
            continue

        try:
            image_pil = Image.open(img_path).convert("RGB")
            img_rgb = np.array(image_pil)
            depth = estimator.compute_depth(img_rgb)
            # 保存为 `.npy`，便于后续在 Dataset 中零损耗读取。
            np.save(depth_path, depth)
        except Exception as e:
            print(f"预计算图像 {img_path} 失败: {e}")

    print("深度图预计算完成。")


if __name__ == "__main__":
    # 简单自检：随机生成一张图像，确认深度估计流程可执行。
    estimator = DepthEstimator()
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = estimator.compute_depth(test_img)
    print(f"Depth shape: {depth.shape}, range: [{depth.min():.4f}, {depth.max():.4f}]")

