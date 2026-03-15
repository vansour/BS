#!/usr/bin/env python3
"""
GPU 在线数据增强算子
GPU On-the-fly Augmentation Operator

基于大气散射模型的物理一致性造雾增强模块。

该模块在训练阶段直接在 GPU 上对清晰图像进行雾化合成，
避免离线生成大量雾天数据所带来的存储和数据管理成本。
增强完成后同时返回：
1. 合成后的雾天图像；
2. 雾类型标签；
3. 与当前样本对应的 beta 散射系数。
"""

import torch
import torch.nn as nn


class FogAugmentation(nn.Module):
    """
    基于大气散射模型的在线造雾模块。

    输入：
        - `images`：清晰图像批次，张量形状为 `(B, 3, H, W)`。
        - `depths`：深度图批次，张量形状为 `(B, 1, H, W)`，
          数值越大表示场景越远，受到雾影响通常越明显。

    输出：
        - `foggy_images`：合成后的雾天图像。
        - `fog_types`：每个样本的雾类型标签。
        - `final_betas`：每个样本最终使用的 beta 值。

    雾类型约定：
        - `0`：clear，无雾样本，beta 会被强制置为 0。
        - `1`：uniform，均匀雾，整幅图遵循统一散射强度。
        - `2`：patchy，团雾，通过低频噪声模拟局部浓淡不均。
    """

    def __init__(self, cfg):
        """
        初始化增强算子。

        Args:
            cfg: 配置对象，至少需要提供以下属性：
                - `BETA_MIN` / `BETA_MAX`：beta 采样范围；
                - `A_MIN` / `A_MAX`：大气光照强度范围。
        """
        super().__init__()
        self.cfg = cfg

    def forward(self, images, depths):
        """
        根据清晰图像与深度图实时生成雾天样本。

        Args:
            images: 清晰图像张量，形状为 `(B, 3, H, W)`，像素范围通常为 `[0, 1]`。
            depths: 深度图张量，形状为 `(B, 1, H, W)`，范围通常为 `[0, 1]`。
                其中 `0` 代表近距离区域，`1` 代表远距离区域。

        Returns:
            tuple:
                - `foggy_images`：合成后的雾天图像，形状为 `(B, 3, H, W)`；
                - `fog_types`：雾类型标签，形状为 `(B,)`，取值集合为 `{0, 1, 2}`；
                - `final_betas`：最终 beta 值，形状为 `(B,)`。
        """
        # 记录批次维度与当前设备，后续新建张量时保持和输入一致。
        B, C, H, W = images.shape
        device = images.device

        # 1. 为每张图像随机决定雾类型。
        # clear / uniform / patchy 三类按离散均匀分布随机采样。
        fog_types = torch.randint(0, 3, (B,), device=device)

        # 2. 为每个样本采样大气散射参数。
        # beta 控制雾的浓度，A 表示大气光照强度。
        betas = torch.rand(B, device=device) * (self.cfg.BETA_MAX - self.cfg.BETA_MIN) + self.cfg.BETA_MIN
        A = torch.rand(B, 1, 1, 1, device=device) * (self.cfg.A_MAX - self.cfg.A_MIN) + self.cfg.A_MIN

        # 从原图复制一份作为输出容器，避免直接覆盖输入。
        foggy_images = images.clone()

        # 3. 生成均匀雾。
        # 均匀雾假设整张图像上的散射强度只与深度相关，不引入空间随机纹理。
        uniform_mask = (fog_types == 1)
        if uniform_mask.any():
            idx = uniform_mask.nonzero().squeeze(-1)
            b_beta = betas[idx].view(-1, 1, 1, 1)
            b_depth = depths[idx]
            b_A = A[idx]

            # 通过放大深度范围增强远处区域的衰减效果。
            effective_depth = b_depth * 5.0
            transmission = torch.exp(-b_beta * effective_depth)

            # 对透射率进行裁剪，避免极端值导致图像过黑或过白。
            transmission = torch.clamp(transmission, 0.05, 0.95)
            foggy_images[idx] = images[idx] * transmission + b_A * (1 - transmission)

        # 4. 生成团雾。
        # 团雾在均匀雾基础上叠加低频噪声，使局部区域出现浓淡变化。
        patchy_mask = (fog_types == 2)
        if patchy_mask.any():
            idx = patchy_mask.nonzero().squeeze(-1)
            b_beta = betas[idx].view(-1, 1, 1, 1)
            b_depth = depths[idx]
            b_A = A[idx]
            num_patchy = idx.size(0)

            # 在 GPU 上先生成低分辨率噪声，再通过双三次插值放大到原图尺寸。
            # 这样得到的是低频、连续的雾密度分布，而不是高频随机雪花点。
            noise = torch.rand(num_patchy, 1, 16, 16, device=device)
            noise = torch.nn.functional.interpolate(
                noise, size=(H, W), mode="bicubic", align_corners=False
            )

            # 噪声与深度共同作用，形成局部浓淡不均的有效深度分布。
            effective_depth = b_depth * (noise * 8.0)
            transmission = torch.exp(-b_beta * effective_depth)
            transmission = torch.clamp(transmission, 0.05, 0.95)
            foggy_images[idx] = images[idx] * transmission + b_A * (1 - transmission)

        # 5. clear 类型不应携带雾参数，因此把对应 beta 显式归零。
        final_betas = betas.clone()
        final_betas[fog_types == 0] = 0.0

        return foggy_images, fog_types, final_betas

    def __repr__(self):
        """返回增强模块的关键信息，便于日志打印与调试。"""
        return (f"FogAugmentation(\n"
                f"  beta_range=[{self.cfg.BETA_MIN}, {self.cfg.BETA_MAX}],\n"
                f"  A_range=[{self.cfg.A_MIN}, {self.cfg.A_MAX}]\n"
                f")")



