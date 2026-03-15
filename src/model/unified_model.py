#!/usr/bin/env python3
"""
统一多任务模型
Unified Multi-Task Model

该模型以 YOLOv11s 作为共享主干网络，在高层语义特征上同时挂接：
1. 目标检测分支；
2. 雾类型分类分支；
3. 能见度 beta 回归分支。

整体思路是尽量复用检测骨干的表达能力，让检测、天气识别和能见度估计
在同一套特征图上联合学习，从而降低多模型并行部署的成本。
"""

from copy import deepcopy

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG


class UnifiedMultiTaskModel(nn.Module):
    """
    基于 YOLOv11s 主干的统一多任务模型。

    结构示意：
        输入图像
            -> YOLOv11s Backbone / Neck
            -> SPPF 高层特征图
                -> 检测分支（YOLO Detect Head）
                -> 雾类型分类头
                -> beta 回归头

    其中分类头负责判断 `clear / uniform / patchy` 三类天气，
    回归头负责预测与能见度相关的 beta 值。
    """

    def __init__(
        self,
        yolo_weights="yolo11s.pt",
        num_fog_classes=3,
        num_det_classes=1,
        in_features=None,
    ):
        """
        初始化统一多任务模型。

        Args:
            yolo_weights: YOLOv11s 预训练权重名称或路径。
            num_fog_classes: 雾分类类别数，默认值为 3。
            num_det_classes: 检测类别数，当前项目通常为 1（vehicle）。
            in_features: 输入到分类头和回归头的通道数。
                如果不手动指定，则通过一次虚拟前向自动探测。
        """
        super().__init__()

        self._num_det_classes = num_det_classes

        # 量化桩用于量化感知训练（QAT）。
        self.quant = QuantStub()
        self.dequant_det = DeQuantStub()
        self.dequant_cls = DeQuantStub()
        self.dequant_reg = DeQuantStub()

        self.yolo = self._build_detection_model(yolo_weights, num_det_classes)

        if in_features is None:
            in_features = self._detect_feature_dimension()

        self._in_features = in_features

        # 雾类型分类头：先做全局平均池化，再送入全连接分类器。
        self.fog_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_fog_classes),
        )

        # beta 回归头：结构更轻量，输出通过 Sigmoid 限制在 [0, 1]。
        self.fog_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def _build_detection_model(self, yolo_weights: str, num_det_classes: int) -> DetectionModel:
        """
        构建用于联合训练的检测模型。

        如果预训练权重的检测头类别数与当前任务不一致，则重建一个新的单类检测头，
        再尽可能加载能够对齐的预训练参数。
        """
        base_model = YOLO(yolo_weights).model
        base_nc = getattr(base_model.model[-1], "nc", None)

        if base_nc == num_det_classes:
            det_model = base_model
        else:
            det_model = DetectionModel(cfg=deepcopy(base_model.yaml), ch=3, nc=num_det_classes, verbose=False)
            base_state_dict = base_model.state_dict()
            new_state_dict = det_model.state_dict()
            intersect = {
                key: value
                for key, value in base_state_dict.items()
                if key in new_state_dict and new_state_dict[key].shape == value.shape
            }
            det_model.load_state_dict(intersect, strict=False)

        # Ultralytics 的检测损失依赖 args 中的 box/cls/dfl 等超参数，
        # 因此这里显式构造完整配置，而不是沿用从权重里读出来的精简 dict。
        det_model.args = get_cfg(
            DEFAULT_CFG,
            overrides={
                "task": "detect",
                "model": yolo_weights,
                "imgsz": 640,
                "single_cls": num_det_classes == 1,
            },
        )
        det_model.names = {idx: ("vehicle" if num_det_classes == 1 and idx == 0 else str(idx))
                           for idx in range(num_det_classes)}
        return det_model

    def _extract_detection_tensor(self, feat):
        """
        从 YOLO 的复杂输出结构中提取真正的检测张量。

        推理模式下 Ultralytics 可能返回：
        - 单个张量；
        - `(tensor, aux_dict)` 形式的元组。
        """
        if isinstance(feat, torch.Tensor):
            return feat

        if isinstance(feat, (list, tuple)):
            for item in feat:
                if isinstance(item, torch.Tensor):
                    return item

        raise TypeError(f"Unsupported detection output type: {type(feat)!r}")

    def _detect_feature_dimension(self):
        """
        自动探测 SPPF 特征图的通道数。

        实现方式是使用一张虚拟输入跑一遍 YOLO 主干，并在遍历过程中尽量捕获
        SPPF 层的输出。若无法精确识别，则退化到经验位置 `i == 9`，再不行则
        使用默认值 `512` 作为兜底。
        """
        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            y = []
            sppf_out = None
            feat = dummy
            for i, m in enumerate(self.yolo.model):
                if m.f != -1:
                    if isinstance(m.f, int):
                        feat = y[m.f] if len(y) > m.f else feat
                    else:
                        feat = [y[j] if j != -1 else feat for j in m.f] if isinstance(y, list) else feat

                feat = m(feat)
                y.append(feat)

                if hasattr(m, "named_children"):
                    for name, _ in m.named_children():
                        if "sppf" in name.lower():
                            sppf_out = feat
                            break

                if sppf_out is None and i == 9:
                    sppf_out = feat

            if sppf_out is not None:
                return sppf_out.shape[1]

            print("警告：无法自动检测特征维度，使用默认值 512")
            return 512

    def forward(self, x):
        """
        执行一次前向传播。

        Args:
            x: 输入图像张量，形状通常为 `(B, 3, H, W)`。

        Returns:
            tuple:
                - `det_out`：训练模式下为原始检测预测 dict，推理模式下为检测张量；
                - `fog_cls`：雾分类 logits，形状为 `(B, num_fog_classes)`；
                - `fog_reg`：beta 回归结果，形状为 `(B,)`，范围为 `[0, 1]`。
        """
        x = self.quant(x)
        y = []
        sppf_out = None
        feat = x

        # 遍历 YOLO 主干，按其拓扑逐层前向，同时捕获高层 SPPF 特征图。
        for i, m in enumerate(self.yolo.model):
            if m.f != -1:
                if isinstance(m.f, int):
                    feat = y[m.f] if len(y) > m.f else feat
                else:
                    feat = [y[j] if j != -1 else feat for j in m.f] if isinstance(y, list) else feat

            feat = m(feat)
            y.append(feat)

            if hasattr(m, "named_children"):
                for name, _ in m.named_children():
                    if "sppf" in name.lower():
                        sppf_out = feat
                        break

            if sppf_out is None and i == 9:
                sppf_out = feat

        if sppf_out is None:
            raise RuntimeError("无法提取 SPPF 特征图，请检查 YOLO 模型结构")

        fog_cls = self.fog_classifier(sppf_out)
        fog_reg = self.fog_regressor(sppf_out).squeeze(-1)

        # 训练时保留 YOLO 原始输出结构，供 detection loss 直接消费；
        # 推理时再抽取纯检测张量，兼容现有展示与导出逻辑。
        if self.training:
            det_out = feat
        else:
            det_out = self.dequant_det(self._extract_detection_tensor(feat))

        fog_cls = self.dequant_cls(fog_cls)
        fog_reg = self.dequant_reg(fog_reg)

        return det_out, fog_cls, fog_reg

    def fuse_model(self):
        """
        在量化感知训练前执行模块融合。

        当前实现只处理形如 `Linear + ReLU` 的简单可融合结构，
        这样可以减少量化误差并提升推理效率。
        """
        for m in self.modules():
            if type(m) is nn.Sequential:
                if len(m) >= 4 and isinstance(m[2], nn.Linear) and isinstance(m[3], nn.ReLU):
                    torch.ao.quantization.fuse_modules(m, ["2", "3"], inplace=True)

    def __repr__(self):
        """返回模型结构摘要，便于日志输出与交互式调试。"""
        return (f"UnifiedMultiTaskModel(\n"
                f"  backbone=YOLOv11s,\n"
                f"  num_det_classes={self._num_det_classes},\n"
                f"  in_features={self._in_features},\n"
                f"  num_fog_classes={self.fog_classifier[-1].out_features},\n"
                f"  device={next(self.parameters()).device}\n"
                f")")

