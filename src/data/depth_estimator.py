#!/usr/bin/env python3
"""
深度估计器
Depth Estimator

本模块封装了基于 MiDaS 的深度估计流程，负责：
1. 加载预训练深度模型；
2. 计算单张图像的深度图；
3. 批量处理图像；
4. 为训练数据集预计算并缓存深度图。
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class DepthEstimator:
    """
    基于 MiDaS 的深度估计器。

    该类把深度估计逻辑集中起来，便于在训练准备阶段批量生成深度缓存，
    从而避免训练期间重复执行较重的深度推理。
    """

    def __init__(self, model_name="MiDaS_small", device=None):
        """
        初始化深度估计器。

        Args:
            model_name: MiDaS 模型名称，例如 `MiDaS_small`。
            device: 指定推理设备；若为空则自动选择 CUDA 或 CPU。
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"正在初始化 MiDaS 深度估计引擎，设备: {self.device}")

        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
            self.model.to(self.device).eval()
            self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
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


