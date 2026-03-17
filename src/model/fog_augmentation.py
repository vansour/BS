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

实现上采用了经典的大气散射表达式：
    I(x) = J(x) * t(x) + A * (1 - t(x))
其中：
    - J(x)：清晰图像；
    - I(x)：雾化后的图像；
    - A：大气光；
    - t(x)：透射率，通常由 `exp(-beta * depth)` 建模。

本项目并不追求严格的物理级仿真，而是基于上述公式构造一个
“具有物理启发、可用于监督学习”的在线雾化过程。该方案既保留了
深度与雾浓度之间的基本关系，又避免了离线生成海量雾图所带来的存储成本。
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

    雾类型定义如下：
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

        这里不提前创建固定参数张量，而是在 `forward()` 中按输入设备
        动态生成随机参数。这样做的原因是：
        1. 训练可能运行在 CPU 或 CUDA；
        2. DataParallel / DDP 场景下设备可能变化；
        3. beta 与 A 本身是每个 batch 随机采样的增强参数，不是可学习参数。
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

        整个增强流程的核心思想如下：
        1. 先随机决定当前样本属于哪一类天气；
        2. 再为每张图采样一个散射系数 beta 和大气光 A；
        3. 基于深度图计算透射率 transmission；
        4. 用散射公式把清晰图变成雾图；
        5. 把用于合成的天气标签和 beta 一并返回，供多任务训练监督使用。
        """
        # 记录输入张量的基本形状信息。
        # 这里只有 B、H、W 会被后续逻辑使用；保留 C 是为了代码可读性，
        # 明确表明输入假定为标准三通道图像。
        B, C, H, W = images.shape
        device = images.device

        # 1. 为每张图像随机决定雾类型。
        # 当前三类天气等概率采样，没有人为偏置哪一类更多。
        # 这意味着训练时模型会看到：
        # - 一部分保持原样的 clear；
        # - 一部分全局一致衰减的 uniform fog；
        # - 一部分局部浓淡变化的 patchy fog。
        fog_types = torch.randint(0, 3, (B,), device=device)

        # 2. 为每个样本采样大气散射参数。
        # beta 控制散射强度，数值越大通常意味着雾越浓；
        # A 表示空气光强度，决定图像被“抬白”的程度。
        #
        # betas: (B,)
        # A:     (B, 1, 1, 1)，便于后续广播到整张图像。
        betas = torch.rand(B, device=device) * (self.cfg.BETA_MAX - self.cfg.BETA_MIN) + self.cfg.BETA_MIN
        A = torch.rand(B, 1, 1, 1, device=device) * (self.cfg.A_MAX - self.cfg.A_MIN) + self.cfg.A_MIN

        # 从原图复制一份作为输出容器，避免直接修改输入张量。
        # 这样做能保证：
        # 1. clear 类样本无需额外处理时仍保持原值；
        # 2. 不同类型样本可以通过索引局部覆盖；
        # 3. 调试时可以明确区分输入与增强输出。
        foggy_images = images.clone()

        # 3. 生成均匀雾。
        # 均匀雾假设整张图像上的散射强度只与深度相关，不引入额外空间纹理。
        # 也就是说，同样深度的位置受到的雾影响近似一致。
        uniform_mask = (fog_types == 1)
        if uniform_mask.any():
            # idx 的形状是 (N_uniform,)，用于从 batch 中选出所有 uniform 样本。
            idx = uniform_mask.nonzero().squeeze(-1)
            b_beta = betas[idx].view(-1, 1, 1, 1)
            b_depth = depths[idx]
            b_A = A[idx]

            # 通过放大深度范围增强远处区域的衰减效果。
            # 这是一个经验系数，不是物理常数，作用是让合成结果在视觉上
            # 更明显，从而给分类与回归任务提供更可学习的差异。
            effective_depth = b_depth * 5.0
            transmission = torch.exp(-b_beta * effective_depth)

            # 对透射率进行裁剪，避免极端值导致：
            # - transmission 过小，图像几乎完全被空气光覆盖；
            # - transmission 过大，造雾效果过弱，近似看不出差别。
            transmission = torch.clamp(transmission, 0.05, 0.95)
            foggy_images[idx] = images[idx] * transmission + b_A * (1 - transmission)

        # 4. 生成团雾。
        # 团雾在均匀雾基础上叠加低频噪声，使局部区域出现浓淡变化。
        # 这部分是本项目“团雾”视觉特征的关键来源。
        patchy_mask = (fog_types == 2)
        if patchy_mask.any():
            idx = patchy_mask.nonzero().squeeze(-1)
            b_beta = betas[idx].view(-1, 1, 1, 1)
            b_depth = depths[idx]
            b_A = A[idx]
            num_patchy = idx.size(0)

            # 在 GPU 上先生成低分辨率噪声，再通过双三次插值放大到原图尺寸。
            # 这样得到的是低频、连续的雾密度分布，而不是高频随机雪花点。
            # 直观上可以把它理解为：先画一张粗糙的“雾浓度地图”，
            # 再把它平滑拉伸到全图。
            noise = torch.rand(num_patchy, 1, 16, 16, device=device)
            noise = torch.nn.functional.interpolate(
                noise, size=(H, W), mode="bicubic", align_corners=False
            )

            # 噪声与深度共同作用，形成局部浓淡不均的有效深度分布。
            # 这里没有直接把噪声当成最终雾图叠加，而是让噪声先影响
            # “有效深度”，再通过 transmission 进入散射公式。
            # 这样生成的团雾仍然保留“远处更容易被雾遮挡”的基本规律。
            effective_depth = b_depth * (noise * 8.0)
            transmission = torch.exp(-b_beta * effective_depth)
            transmission = torch.clamp(transmission, 0.05, 0.95)
            foggy_images[idx] = images[idx] * transmission + b_A * (1 - transmission)

        # 5. clear 类型不应携带雾参数，因此把对应 beta 显式归零。
        # 这样训练回归头时，clear 样本会自然形成 “beta = 0” 的监督信号。
        final_betas = betas.clone()
        final_betas[fog_types == 0] = 0.0

        return foggy_images, fog_types, final_betas

    def __repr__(self):
        """返回增强模块的关键信息，便于日志打印与调试。"""
        return (f"FogAugmentation(\n"
                f"  beta_range=[{self.cfg.BETA_MIN}, {self.cfg.BETA_MAX}],\n"
                f"  A_range=[{self.cfg.A_MIN}, {self.cfg.A_MAX}]\n"
                f")")



