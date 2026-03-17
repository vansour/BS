#!/usr/bin/env python3
"""
统一多任务模型
Unified Multi-Task Model

该模型以 YOLO 系列检测模型作为共享主干网络，在高层语义特征上同时挂接：
1. 目标检测分支；
2. 雾类型分类分支；
3. 能见度 beta 回归分支。

整体思路是尽量复用检测骨干的表达能力，让检测、天气识别和能见度估计
在同一套特征图上联合学习，从而降低多模型并行部署的成本。

从模型设计角度看，本文件有两个关键点：
1. 检测分支并不是“另外再写一个检测头”，而是直接复用 Ultralytics 的检测模型；
2. 雾分类和 beta 回归不是从原图直接接出，而是从 YOLO 高层语义特征图上再接两个轻量头。

因此，该模型本质上属于“YOLO 共享特征 + 两个附加任务头”的多任务结构。
"""

from copy import deepcopy

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG

from src.config import Config


class UnifiedMultiTaskModel(nn.Module):
    """
    基于 YOLO 主干的统一多任务模型。

    结构示意：
        输入图像
            -> YOLO Backbone / Neck
            -> SPPF 高层特征图
                -> 检测分支（YOLO Detect Head）
                -> 雾类型分类头
                -> beta 回归头

    其中分类头负责判断 `clear / uniform / patchy` 三类天气，
    回归头负责预测与能见度相关的 beta 值。

    设计动机可以概括为：
    - 检测任务需要目标级局部细节；
    - 雾分类和 beta 估计更依赖整幅图的全局语义与退化程度；
    - 共享主干后，三项任务可以在同一特征空间中互相提供归纳偏置。
    """

    def __init__(
        self,
        yolo_weights="yolo11n.pt",
        num_fog_classes=3,
        num_det_classes=1,
        in_features=None,
    ):
        """
        初始化统一多任务模型。

        Args:
            yolo_weights: YOLO 预训练权重名称或路径。
            num_fog_classes: 雾分类类别数，默认值为 3。
            num_det_classes: 检测类别数，当前项目通常为 1（vehicle）。
            in_features: 输入到分类头和回归头的通道数。
                如果不手动指定，则通过一次虚拟前向自动探测。

        初始化顺序大致是：
        1. 先准备量化桩，保证后续 QAT 可以直接复用这份模型定义；
        2. 加载或重建 YOLO 检测模型；
        3. 自动探测共享特征通道数；
        4. 构建雾分类头和 beta 回归头。
        """
        super().__init__()

        self._num_det_classes = num_det_classes

        # 量化桩用于量化感知训练（QAT）。
        # 输入图像先经过 `QuantStub`，多任务输出再分别经过 `DeQuantStub`，
        # 这样后续在 prepare_qat / convert 时，PyTorch 才能正确插入量化逻辑。
        self.quant = QuantStub()
        self.dequant_det = DeQuantStub()
        self.dequant_cls = DeQuantStub()
        self.dequant_reg = DeQuantStub()

        # self.yolo 是真正承担主干特征提取和检测预测的主体。
        # 如果基础权重类别数与当前任务不一致，会在 `_build_detection_model()`
        # 中重建检测头，并尽量复用能对齐的预训练参数。
        self.yolo = self._build_detection_model(yolo_weights, num_det_classes)

        if in_features is None:
            in_features = self._detect_feature_dimension()

        self._in_features = in_features

        # 雾类型分类头：
        # - 先把空间维度做全局平均池化，得到整图语义描述；
        # - 再用两层全连接网络完成三分类。
        #
        # Dropout 的目的不是“必须有”，而是对这个轻量头做一点正则化，
        # 避免高层语义特征被分类头过拟合。
        self.fog_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_fog_classes),
        )

        # beta 回归头：
        # - 同样基于全局池化后的整图特征；
        # - 比分类头更轻；
        # - 最后一层用 Sigmoid 把输出压到 [0, 1]。
        #
        # 训练和推理阶段会再结合配置中的 `BETA_MAX` 做缩放，
        # 因此这里输出的是“归一化 beta”。
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

        这里的兼容逻辑很重要，因为本项目场景通常只保留一个检测类别 `vehicle`。
        而官方 YOLO 权重往往是在多类别检测任务上训练得到的，最后检测头 shape
        会和当前任务不一致。

        处理策略不是强行 strict 加载，也不是完全放弃预训练权重，而是：
        1. 若类别数一致，直接复用整套 base model；
        2. 若类别数不一致，重建当前类别数的 DetectionModel；
        3. 仅加载名字和 shape 同时匹配的参数。

        这样能最大限度保留 backbone / neck 的预训练能力，同时允许检测头按当前任务重建。
        """
        base_model = YOLO(yolo_weights).model
        base_nc = getattr(base_model.model[-1], "nc", None)
        base_names = getattr(base_model, "names", None)

        if base_nc == num_det_classes:
            det_model = base_model
        else:
            # 使用原始 yaml 结构重建一个“同架构、不同类别数”的检测模型。
            det_model = DetectionModel(cfg=deepcopy(base_model.yaml), ch=3, nc=num_det_classes, verbose=False)
            base_state_dict = base_model.state_dict()
            new_state_dict = det_model.state_dict()

            # 只保留形状严格兼容的参数。
            # 典型情况下，backbone / neck 大多可以复用，最终 detect head 的最后几层会被跳过。
            intersect = {
                key: value
                for key, value in base_state_dict.items()
                if key in new_state_dict and new_state_dict[key].shape == value.shape
            }
            det_model.load_state_dict(intersect, strict=False)

        # Ultralytics 的检测损失依赖 args 中的 box/cls/dfl 等超参数，
        # 因此这里显式构造完整配置，而不是沿用从权重里读出来的精简 dict。
        #
        # 这一段看起来像“只是补个属性”，实际上很关键。
        # 如果没有正确的 args，后面 `model.yolo.loss(...)` 可能无法正常工作。
        det_model.args = get_cfg(
            DEFAULT_CFG,
            overrides={
                "task": "detect",
                "model": yolo_weights,
                "imgsz": Config.IMG_SIZE,
                "single_cls": num_det_classes == 1,
            },
        )
        # names 主要供推理展示和后处理阶段使用。
        # 当类别数与官方预训练权重一致时，优先保留原始 COCO 类名；
        # 只有在自定义类别数场景下才退化为简单编号或单类名称。
        if base_nc == num_det_classes and isinstance(base_names, dict) and len(base_names) == num_det_classes:
            det_model.names = dict(base_names)
        else:
            det_model.names = {
                idx: ("vehicle" if num_det_classes == 1 and idx == 0 else str(idx))
                for idx in range(num_det_classes)
            }
        return det_model

    def _extract_detection_tensor(self, feat):
        """
        从 YOLO 的复杂输出结构中提取真正的检测张量。

        推理模式下 Ultralytics 可能返回：
        - 单个张量；
        - `(tensor, aux_dict)` 形式的元组。

        本项目的推理后处理只需要真正的检测张量，因此这里做一个小的适配层，
        把 Ultralytics 可能变化的输出包装统一成“纯张量”。
        """
        if isinstance(feat, torch.Tensor):
            return feat

        if isinstance(feat, (list, tuple)):
            for item in feat:
                if isinstance(item, torch.Tensor):
                    return item

        raise TypeError(f"Unsupported detection output type: {type(feat)!r}")

    @staticmethod
    def _capture_shared_feature(feat, module, fallback_feature, preferred_feature):
        """
        从当前层输出中提取供多任务头使用的单张特征图候选。

        优先级规则如下：
        1. 若当前模块本身就是 SPPF，则直接作为首选特征；
        2. 否则持续记录最近的 4D 张量作为回退候选；
        3. `Detect` 输出不参与共享特征候选，避免把预测张量误当特征图。
        """
        module_name = module.__class__.__name__.lower()
        if "detect" in module_name:
            return fallback_feature, preferred_feature

        candidate = None
        if isinstance(feat, torch.Tensor) and feat.ndim == 4:
            candidate = feat
        elif isinstance(feat, (list, tuple)):
            tensor_candidates = [
                item for item in feat
                if isinstance(item, torch.Tensor) and item.ndim == 4
            ]
            if tensor_candidates:
                candidate = tensor_candidates[-1]

        if candidate is not None:
            fallback_feature = candidate
            if "sppf" in module_name:
                preferred_feature = candidate

        return fallback_feature, preferred_feature

    def _detect_feature_dimension(self):
        """
        自动探测 SPPF 特征图的通道数。

        实现方式是使用一张虚拟输入跑一遍 YOLO 主干，并优先捕获真正的 SPPF 输出。
        若当前架构中没有显式的 SPPF 命名，则退化为“检测头之前最后一张 4D 特征图”。

        之所以做自动探测，而不是把通道数写死，是因为：
        - 不同 YOLO 规模的高层特征维度可能不同；
        - 如果后续切换 `yolo11n.pt`、`yolo11s.pt` 或别的配置，
          分类头和回归头的输入维度也要随之变化。
        """
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            y = []
            fallback_feature = None
            preferred_feature = None
            feat = dummy
            for m in self.yolo.model:
                # Ultralytics 每一层的 `m.f` 表示该层输入来自哪里：
                # -1 表示接上一步输出；
                # 整数表示从历史某层取单个输出；
                # 列表表示需要把多层输出一起送入当前层。
                if m.f != -1:
                    if isinstance(m.f, int):
                        feat = y[m.f] if len(y) > m.f else feat
                    else:
                        feat = [y[j] if j != -1 else feat for j in m.f] if isinstance(y, list) else feat

                feat = m(feat)
                y.append(feat)
                fallback_feature, preferred_feature = self._capture_shared_feature(
                    feat,
                    m,
                    fallback_feature,
                    preferred_feature,
                )

            shared_feature = preferred_feature if preferred_feature is not None else fallback_feature
            if shared_feature is not None:
                return shared_feature.shape[1]

            print("警告：无法自动检测特征维度，使用默认值 512")
            return 512

    def forward(self, x, return_raw_det: bool = False):
        """
        执行一次前向传播。

        Args:
            x: 输入图像张量，形状通常为 `(B, 3, H, W)`。
            return_raw_det: 是否强制返回原始检测预测结构。
                当模型处于 `eval()` 模式但仍需要计算 detection loss 时，
                例如验证阶段，应将其设为 `True`。

        Returns:
            tuple:
                - `det_out`：训练模式或 `return_raw_det=True` 时为原始检测预测结构，
                  推理模式下为检测张量；
                - `fog_cls`：雾分类 logits，形状为 `(B, num_fog_classes)`；
                - `fog_reg`：beta 回归结果，形状为 `(B,)`，范围为 `[0, 1]`。

        这里的前向流程故意没有调用 `self.yolo(x)` 这个高层封装，
        而是手动遍历 `self.yolo.model`。原因是我们不仅要拿最终检测输出，
        还要在中途截取高层共享特征图给两个附加任务头使用。
        """
        # 量化感知训练场景下，输入会在这里进入量化路径；
        # 普通 FP32 场景下这一步本质上是透明的。
        x = self.quant(x)
        y = []
        fallback_feature = None
        preferred_feature = None
        feat = x

        # 遍历 YOLO 主干，按其真实拓扑逐层前向，同时捕获高层 SPPF 特征图。
        # `y` 用来缓存历史层输出，因为后续层可能通过 `m.f` 回取它们。
        for m in self.yolo.model:
            if m.f != -1:
                if isinstance(m.f, int):
                    feat = y[m.f] if len(y) > m.f else feat
                else:
                    feat = [y[j] if j != -1 else feat for j in m.f] if isinstance(y, list) else feat

            feat = m(feat)
            y.append(feat)
            fallback_feature, preferred_feature = self._capture_shared_feature(
                feat,
                m,
                fallback_feature,
                preferred_feature,
            )

        sppf_out = preferred_feature if preferred_feature is not None else fallback_feature

        if sppf_out is None:
            raise RuntimeError("无法提取 SPPF 特征图，请检查 YOLO 模型结构")

        # 两个附加任务头都只消费高层共享特征，而不直接依赖检测输出张量。
        fog_cls = self.fog_classifier(sppf_out)
        fog_reg = self.fog_regressor(sppf_out).squeeze(-1)

        # 训练与“验证算损失”场景都需要原始检测输出结构，供 detection loss 直接消费；
        # 只有纯推理 / 导出路径才抽取纯检测张量。
        if self.training or return_raw_det:
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

        注意这里不会尝试“全模型自动融合”，因为 Ultralytics 内部模块较复杂，
        盲目融合反而更容易引入量化兼容问题。当前策略只处理我们自己新增的
        轻量头中最稳定、最明确的可融合结构。
        """
        for m in self.modules():
            if type(m) is nn.Sequential:
                if len(m) >= 4 and isinstance(m[2], nn.Linear) and isinstance(m[3], nn.ReLU):
                    torch.ao.quantization.fuse_modules(m, ["2", "3"], inplace=True)

    def __repr__(self):
        """返回模型结构摘要，便于日志输出与交互式调试。"""
        return (f"UnifiedMultiTaskModel(\n"
                f"  backbone={self.yolo.yaml.get('scale', 'yolo')},\n"
                f"  num_det_classes={self._num_det_classes},\n"
                f"  in_features={self._in_features},\n"
                f"  num_fog_classes={self.fog_classifier[-1].out_features},\n"
                f"  device={next(self.parameters()).device}\n"
                f")")

