#!/usr/bin/env python3
"""
全局配置模块
Centralized Configuration Module

本模块集中维护项目运行所需的关键常量和默认参数，主要覆盖以下内容：
1. 数据目录、缓存目录与输出目录；
2. 模型结构相关参数；
3. 训练超参数与 DataLoader 参数；
4. 检查点保存策略；
5. 大气散射模型相关物理参数；
6. 推理阶段使用的阈值与设备配置。

在当前工程中，本模块承担“统一配置源”的职责。训练、推理和导出三条主链
默认均从本模块读取配置；`configs/*.yaml` 主要用于示例展示和文档说明，
并非主流程的实际配置入口。顶层 `config.py` 则仅作为旧版 checkpoint 的
兼容层存在。

本项目采用“类属性保存默认配置”的轻量设计，而未引入更复杂的配置管理框架。
这种实现方式更适合毕业设计场景中的代码阅读、运行维护与答辩展示。
"""

import os
from pathlib import Path


class Config:
    """
    项目全局配置类。

    该类以类属性形式组织默认配置，具有以下特点：
    - 所有模块可通过统一入口访问同一组关键参数；
    - 配置项定义集中，便于实验复现、论文撰写和后续维护；
    - 初始化过程仅负责创建必要目录，副作用较小。

    从实现语义上看，该类更接近“带少量初始化逻辑的配置命名空间”，
    而非需要频繁构造和动态修改的复杂对象。绝大多数参数均以类属性存在，
    因此既可通过 `cfg = Config()` 的实例形式访问，也可通过 `Config.XXX`
    的类属性形式访问。
    """

    # 项目源码目录的绝对路径，其他相对路径都从这里继续展开。
    # 注意这里是 `src/` 目录本身，不是仓库根目录，因此后面的路径大多会再拼接 `..`。
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
    YOLO_BASE_MODEL = "yolo11n.pt"
    # 当前检测任务明确收敛为单类 `vehicle`。
    # 训练、推理和导出都按单类检测头组织，不再保留 COCO 80 类的过渡逻辑。
    NUM_DET_CLASSES = 1
    DET_TRAIN_CLASS_ID = 0
    VEHICLE_CLASS_IDS = [0]
    DET_CLASS_NAMES = ["vehicle"]
    NUM_FOG_CLASSES = 3
    FOG_CLASS_NAMES = ["clear", "uniform", "patchy"]

    # ==================== 训练超参数 ====================
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-4
    QAT_EPOCHS = 5
    QAT_LR = 1e-5
    IMG_SIZE = 512  # 默认使用更轻量的训练输入尺度以提升迭代速度。
    # 允许通过环境变量在不改源码的情况下控制帧抽样步长。
    # 例如 BS_FRAME_STRIDE=5 表示每 5 帧取 1 帧，可用于快速烟雾测试或减轻训练压力。
    FRAME_STRIDE = max(1, int(os.getenv("BS_FRAME_STRIDE", "1")))

    # 是否在正式训练前强制执行深度缓存预计算流程。
    # 即使默认关闭，训练脚本现在也会在发现 train/val 任一侧缺失缓存时自动补齐，
    # 以避免训练或验证阶段因缺少 `.npy` 文件而中途崩溃。
    # 显式开启后则会无条件进入预计算流程（内部仍只对缺失文件真正写盘）。
    PRECOMPUTE_DEPTH_CACHE = os.getenv("BS_PRECOMPUTE_DEPTH_CACHE", "0").lower() in {
        "1", "true", "yes", "on"
    }

    # DataLoader 相关参数也允许通过环境变量覆盖，便于针对不同机器微调吞吐。
    CPU_COUNT = os.cpu_count() or 4
    DEFAULT_NUM_WORKERS = (
        min(8, max(4, CPU_COUNT // 2))
        if os.name == "nt"
        else min(8, max(4, CPU_COUNT // 2))
    )
    NUM_WORKERS = max(0, int(os.getenv("BS_NUM_WORKERS", str(DEFAULT_NUM_WORKERS))))
    DEFAULT_PREFETCH_FACTOR = 4 if os.name == "nt" else 2
    PREFETCH_FACTOR = max(2, int(os.getenv("BS_PREFETCH_FACTOR", str(DEFAULT_PREFETCH_FACTOR))))
    PERSISTENT_WORKERS = os.getenv(
        "BS_PERSISTENT_WORKERS",
        "1" if NUM_WORKERS > 0 else "0",
    ).lower() in {"1", "true", "yes", "on"}

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
    # 在导入配置时就根据当前环境探测默认设备。
    # 这样训练、推理和导出脚本都能共享同一套设备选择逻辑。
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # ==================== 推理配置 ====================
    BASE_CONF_THRES = 0.25
    EMA_ALPHA = 0.1

    # 是否对输入应用 ImageNet 均值方差归一化。
    # 当前默认关闭，意味着训练和推理都主要使用 `[0, 1]` 范围的原始张量。
    # 之所以保留这个开关，是为了后续做输入规范对比实验时不必改动多处代码。
    USE_IMAGENET_NORMALIZE = False

    # ==================== 多任务损失权重 ====================
    # 当前三项任务默认等权。
    # 若后续发现检测、分类或回归的尺度差异较大，可以从这里统一调整。
    DET_LOSS_WEIGHT = 1.0
    FOG_CLS_LOSS_WEIGHT = 1.0
    FOG_REG_LOSS_WEIGHT = 1.0

    def __init__(self):
        """
        初始化配置实例。

        初始化过程中不重写默认配置，仅确保项目运行所依赖的输出目录真实存在，
        以避免训练、缓存或导出阶段因路径缺失而中断。

        由于主要配置均已通过类属性定义，因此实例化动作本身较轻量；
        这里唯一保留的副作用是目录存在性检查与创建。
        """
        os.makedirs(self.DEPTH_CACHE_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def __repr__(self):
        """返回配置摘要，便于调试和日志打印。"""
        return (f"Config(BASE_DIR={self.BASE_DIR}, "
                f"DEVICE={self.DEVICE}, "
                f"NUM_DET_CLASSES={self.NUM_DET_CLASSES}, "
                f"BETA_RANGE=[{self.BETA_MIN}, {self.BETA_MAX}], "
                f"FRAME_STRIDE={self.FRAME_STRIDE}, "
                f"PRECOMPUTE_DEPTH_CACHE={self.PRECOMPUTE_DEPTH_CACHE}, "
                f"NUM_WORKERS={self.NUM_WORKERS})")


def get_default_config() -> Config:
    """
    获取默认配置实例。

    Returns:
        Config: 按当前源码默认值构造的配置对象。

    该工厂函数的存在主要是为了给外部调用方提供一个更清晰、稳定的默认配置入口。
    """
    return Config()


if __name__ == "__main__":
    cfg = Config()
    print(cfg)
