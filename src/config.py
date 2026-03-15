#!/usr/bin/env python3
"""
全局配置模块
Centralized Configuration Module

本文件集中维护项目运行所需的关键常量和默认参数，包括：
1. 数据目录与输出目录；
2. 模型结构相关参数；
3. 训练超参数；
4. 检查点保存策略；
5. 物理散射模型参数；
6. 推理阶段使用的阈值和设备配置。
"""

import os
from pathlib import Path


class Config:
    """
    项目全局配置类。

    该类采用“类属性作为默认配置”的写法，优点是：
    - 所有模块可以通过统一入口读取配置；
    - 配置项集中定义，便于毕业设计答辩和后续维护说明；
    - 初始化时只负责确保必要目录存在，不引入复杂副作用。
    """

    # 项目源码目录的绝对路径，其他相对路径都从这里继续展开。
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ==================== 数据路径 ====================
    # 原始晴天数据目录，默认指向 UA-DETRAC 训练集图片目录。
    RAW_DATA_DIR = os.path.join(
        BASE_DIR, "..", "data", "UA-DETRAC", "DETRAC-train-data", "Insight-MVT_Annotation_Train"
    )
    # UA-DETRAC XML 标注目录，用于检测分支监督。
    XML_DIR = os.path.join(BASE_DIR, "..", "data", "UA-DETRAC", "DETRAC-Train-Annotations-XML")
    # 深度图缓存目录，用于存储 MiDaS 预计算结果，避免训练时重复推理。
    DEPTH_CACHE_DIR = os.path.join(BASE_DIR, "..", "outputs", "Depth_Cache")
    # 统一模型、日志和导出结果的输出目录。
    OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "Fog_Detection_Project")

    # ==================== 模型配置 ====================
    YOLO_BASE_MODEL = "yolo11s.pt"
    NUM_DET_CLASSES = 1
    DET_CLASS_NAMES = ["vehicle"]
    NUM_FOG_CLASSES = 3
    FOG_CLASS_NAMES = ["clear", "uniform", "patchy"]

    # ==================== 训练超参数 ====================
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-4
    QAT_EPOCHS = 5
    QAT_LR = 1e-5
    IMG_SIZE = 640  # 与 YOLO 默认输入尺度保持一致。

    # ==================== 检查点配置 ====================
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "..", "outputs", "Fog_Detection_Project", "checkpoints")
    CHECKPOINT_SAVE_INTERVAL = 1  # 每隔多少个 epoch 保存一次 checkpoint。
    CHECKPOINT_KEEP_MAX = 5  # 最多保留的历史 checkpoint 数量。

    # ==================== 物理参数（大气散射模型） ====================
    BETA_MIN = 0.02
    BETA_MAX = 0.1
    A_MIN = 0.7
    A_MAX = 0.95

    # ==================== 设备配置 ====================
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # ==================== 推理配置 ====================
    BASE_CONF_THRES = 0.25
    EMA_ALPHA = 0.1
    USE_IMAGENET_NORMALIZE = False

    # ==================== 多任务损失权重 ====================
    DET_LOSS_WEIGHT = 1.0
    FOG_CLS_LOSS_WEIGHT = 1.0
    FOG_REG_LOSS_WEIGHT = 1.0

    def __init__(self):
        """
        初始化配置实例。

        这里不做参数重写，只负责确保项目运行时依赖的输出目录真实存在，
        以避免训练、缓存或导出阶段因为路径不存在而中断。
        """
        os.makedirs(self.DEPTH_CACHE_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def __repr__(self):
        """返回配置摘要，便于调试和日志打印。"""
        return (f"Config(BASE_DIR={self.BASE_DIR}, "
                f"DEVICE={self.DEVICE}, "
                f"NUM_DET_CLASSES={self.NUM_DET_CLASSES}, "
                f"BETA_RANGE=[{self.BETA_MIN}, {self.BETA_MAX}])")


def get_default_config() -> Config:
    """
    获取默认配置实例。

    Returns:
        Config: 按当前源码默认值构造的配置对象。
    """
    return Config()


if __name__ == "__main__":
    cfg = Config()
    print(cfg)


