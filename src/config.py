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

import json
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency fallback
    yaml = None


def _project_root(base_dir: str) -> str:
    """返回仓库根目录绝对路径。"""
    return os.path.abspath(os.path.join(base_dir, ".."))


def _parse_bool(value) -> bool:
    """把常见的字符串/数值表示解析为布尔值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _resolve_path(base_dir: str, *parts: str, env_var: str | None = None) -> str:
    """
    解析配置路径。

    若传入了环境变量名且环境变量有值，则优先使用环境变量中的路径；
    否则回退到仓库内的默认相对路径。
    """
    override = os.getenv(env_var) if env_var else None
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.abspath(os.path.join(base_dir, *parts))


def _resolve_device() -> str:
    """
    解析默认运行设备。

    允许通过 `BS_DEVICE` 显式覆盖自动探测结果，便于在 CPU/CUDA 之间切换。
    """
    override = os.getenv("BS_DEVICE")
    if override:
        return override
    return "cuda" if __import__("torch").cuda.is_available() else "cpu"


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
    PROJECT_ROOT = _project_root(BASE_DIR)
    CONFIG_FILE_ENV = "BS_CONFIG_FILE"

    # ==================== 数据路径 ====================
    # 原始晴天数据目录，默认指向 UA-DETRAC 训练集图片目录。
    RAW_DATA_DIR = _resolve_path(
        BASE_DIR,
        "..",
        "data",
        "UA-DETRAC",
        "DETRAC-train-data",
        "Insight-MVT_Annotation_Train",
        env_var="BS_RAW_DATA_DIR",
    )
    # UA-DETRAC XML 标注目录，用于检测分支监督。
    XML_DIR = _resolve_path(
        BASE_DIR,
        "..",
        "data",
        "UA-DETRAC",
        "DETRAC-Train-Annotations-XML",
        env_var="BS_XML_DIR",
    )
    # 深度图缓存目录，用于存储 MiDaS 预计算结果，避免训练时重复推理。
    DEPTH_CACHE_DIR = _resolve_path(
        BASE_DIR,
        "..",
        "outputs",
        "Depth_Cache",
        env_var="BS_DEPTH_CACHE_DIR",
    )
    # 统一模型、日志和导出结果的输出目录。
    OUTPUT_DIR = _resolve_path(
        BASE_DIR,
        "..",
        "outputs",
        "Fog_Detection_Project",
        env_var="BS_OUTPUT_DIR",
    )

    # ==================== 模型配置 ====================
    YOLO_BASE_MODEL = os.getenv("BS_YOLO_BASE_MODEL", "yolo11n.pt")
    DET_HEAD_MODE = os.getenv("BS_DET_HEAD_MODE", "single_vehicle")
    COCO_NUM_DET_CLASSES = 80
    COCO_VEHICLE_TRAIN_CLASS_ID = int(
        os.getenv("BS_COCO_VEHICLE_TRAIN_CLASS_ID", "2")
    )
    COCO_VEHICLE_CLASS_IDS = [2, 3, 5, 7]
    # 当前默认检测路线仍然是单类 `vehicle`。
    # 当 `DET_HEAD_MODE = "coco_vehicle"` 时，会保留 COCO 80 类检测头，
    # 训练时把 UA-DETRAC 车辆统一映射到 `car` 类，推理时再把 COCO 车辆类折叠回 `vehicle`。
    NUM_DET_CLASSES = COCO_NUM_DET_CLASSES if DET_HEAD_MODE == "coco_vehicle" else 1
    DET_TRAIN_CLASS_ID = (
        COCO_VEHICLE_TRAIN_CLASS_ID if DET_HEAD_MODE == "coco_vehicle" else 0
    )
    VEHICLE_CLASS_IDS = (
        list(COCO_VEHICLE_CLASS_IDS) if DET_HEAD_MODE == "coco_vehicle" else [0]
    )
    DET_CLASS_NAMES = ["vehicle"]
    NUM_FOG_CLASSES = 3
    FOG_CLASS_NAMES = ["clear", "uniform", "patchy"]

    # ==================== 训练超参数 ====================
    BATCH_SIZE = max(1, int(os.getenv("BS_BATCH_SIZE", "16")))
    EPOCHS = max(1, int(os.getenv("BS_EPOCHS", "30")))
    LR = float(os.getenv("BS_LR", "1e-4"))
    QAT_EPOCHS = max(0, int(os.getenv("BS_QAT_EPOCHS", "5")))
    QAT_LR = float(os.getenv("BS_QAT_LR", "1e-5"))
    IMG_SIZE = max(
        32, int(os.getenv("BS_IMG_SIZE", "512"))
    )  # 默认使用更轻量的训练输入尺度以提升迭代速度。
    # 允许通过环境变量在不改源码的情况下控制帧抽样步长。
    # 例如 BS_FRAME_STRIDE=5 表示每 5 帧取 1 帧，可用于快速烟雾测试或减轻训练压力。
    FRAME_STRIDE = max(1, int(os.getenv("BS_FRAME_STRIDE", "1")))
    TRAIN_RATIO = min(1.0, max(0.0, float(os.getenv("BS_TRAIN_RATIO", "0.8"))))

    # 是否在正式训练前强制执行深度缓存预计算流程。
    # 即使默认关闭，训练脚本现在也会在发现 train/val 任一侧缺失缓存时自动补齐，
    # 以避免训练或验证阶段因缺少 `.npy` 文件而中途崩溃。
    # 显式开启后则会无条件进入预计算流程（内部仍只对缺失文件真正写盘）。
    PRECOMPUTE_DEPTH_CACHE = os.getenv("BS_PRECOMPUTE_DEPTH_CACHE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    SKIP_QAT = os.getenv("BS_SKIP_QAT", "0").lower() in {"1", "true", "yes", "on"}
    DISABLE_AMP = os.getenv("BS_DISABLE_AMP", "0").lower() in {"1", "true", "yes", "on"}
    MAX_TRAIN_BATCHES = max(0, int(os.getenv("BS_MAX_TRAIN_BATCHES", "0")))
    MAX_VAL_BATCHES = max(0, int(os.getenv("BS_MAX_VAL_BATCHES", "0")))
    GRAD_CLIP_NORM = max(0.0, float(os.getenv("BS_GRAD_CLIP_NORM", "0")))
    NONFINITE_GRAD_MIN_BATCHES = max(
        1, int(os.getenv("BS_NONFINITE_GRAD_MIN_BATCHES", "20"))
    )
    NONFINITE_GRAD_WARN_RATIO = max(
        0.0, float(os.getenv("BS_NONFINITE_GRAD_WARN_RATIO", "0.05"))
    )
    NONFINITE_GRAD_FAIL_RATIO = max(
        0.0, float(os.getenv("BS_NONFINITE_GRAD_FAIL_RATIO", "0.2"))
    )
    NONFINITE_GRAD_FAIL_STREAK = max(
        1, int(os.getenv("BS_NONFINITE_GRAD_FAIL_STREAK", "1"))
    )
    NONFINITE_GRAD_AUTO_DISABLE_AMP = os.getenv(
        "BS_NONFINITE_GRAD_AUTO_DISABLE_AMP", "1"
    ).lower() in {"1", "true", "yes", "on"}
    SEED = int(os.getenv("BS_SEED", "42"))

    # DataLoader 相关参数也允许通过环境变量覆盖，便于针对不同机器微调吞吐。
    CPU_COUNT = os.cpu_count() or 4
    DEFAULT_NUM_WORKERS = (
        min(8, max(4, CPU_COUNT // 2))
        if os.name == "nt"
        else min(8, max(4, CPU_COUNT // 2))
    )
    NUM_WORKERS = max(0, int(os.getenv("BS_NUM_WORKERS", str(DEFAULT_NUM_WORKERS))))
    DEFAULT_PREFETCH_FACTOR = 4 if os.name == "nt" else 2
    PREFETCH_FACTOR = max(
        2, int(os.getenv("BS_PREFETCH_FACTOR", str(DEFAULT_PREFETCH_FACTOR)))
    )
    PERSISTENT_WORKERS = os.getenv(
        "BS_PERSISTENT_WORKERS",
        "1" if NUM_WORKERS > 0 else "0",
    ).lower() in {"1", "true", "yes", "on"}

    # ==================== 检查点配置 ====================
    CHECKPOINT_DIR = _resolve_path(
        BASE_DIR,
        "..",
        "outputs",
        "Fog_Detection_Project",
        "checkpoints",
        env_var="BS_CHECKPOINT_DIR",
    )
    if os.getenv("BS_CHECKPOINT_DIR") is None:
        CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    CHECKPOINT_SAVE_INTERVAL = max(
        1, int(os.getenv("BS_CHECKPOINT_SAVE_INTERVAL", "1"))
    )  # 每隔多少个 epoch 保存一次 checkpoint。
    CHECKPOINT_KEEP_MAX = max(
        0, int(os.getenv("BS_CHECKPOINT_KEEP_MAX", "5"))
    )  # 最多保留的历史 checkpoint 数量。

    # ==================== 物理参数（大气散射模型） ====================
    BETA_MIN = 0.02
    BETA_MAX = 0.1
    A_MIN = 0.7
    A_MAX = 0.95
    # 真实雾视频验证表明，过多 clear 样本和过轻的雾增强会让天气头更容易
    # 向 `clear` 方向过度自信。下面这些参数用于把在线造雾做得更“偏雾天”。
    FOG_CLEAR_PROB = min(
        1.0, max(0.0, float(os.getenv("BS_FOG_CLEAR_PROB", "0.15")))
    )
    FOG_UNIFORM_PROB = min(
        1.0, max(0.0, float(os.getenv("BS_FOG_UNIFORM_PROB", "0.35")))
    )
    FOG_PATCHY_PROB = min(
        1.0, max(0.0, float(os.getenv("BS_FOG_PATCHY_PROB", "0.50")))
    )
    FOG_BETA_MIN = min(
        BETA_MAX,
        max(BETA_MIN, float(os.getenv("BS_FOG_BETA_MIN", "0.04"))),
    )
    UNIFORM_DEPTH_SCALE = max(
        1.0, float(os.getenv("BS_UNIFORM_DEPTH_SCALE", "7.0"))
    )
    PATCHY_DEPTH_BASE = max(
        0.0, float(os.getenv("BS_PATCHY_DEPTH_BASE", "2.0"))
    )
    PATCHY_DEPTH_NOISE_SCALE = max(
        0.0, float(os.getenv("BS_PATCHY_DEPTH_NOISE_SCALE", "8.0"))
    )

    # ==================== 设备配置 ====================
    # 在导入配置时就根据当前环境探测默认设备。
    # 这样训练、推理和导出脚本都能共享同一套设备选择逻辑。
    DEVICE = _resolve_device()

    # ==================== 推理配置 ====================
    BASE_CONF_THRES = 0.25
    EMA_ALPHA = float(os.getenv("BS_EMA_ALPHA", "0.05"))
    BETA_SCALE_FACTOR = float(os.getenv("BS_BETA_SCALE_FACTOR", "1.2"))
    MIN_CONF_THRES = float(os.getenv("BS_MIN_CONF_THRES", "0.15"))
    MODEL_PATH = os.getenv("BS_MODEL_PATH")
    VIDEO_SOURCE = os.getenv("BS_VIDEO_SOURCE")
    DISPLAY_WINDOW_WIDTH = max(
        1, int(os.getenv("BS_DISPLAY_WINDOW_WIDTH", "960"))
    )
    DISPLAY_WINDOW_HEIGHT = max(
        1, int(os.getenv("BS_DISPLAY_WINDOW_HEIGHT", "540"))
    )
    STATUS_BAR_HEIGHT = max(
        0, int(os.getenv("BS_STATUS_BAR_HEIGHT", "80"))
    )

    # ==================== 时序车辆过滤配置 ====================
    TEMPORAL_FILTER_ENABLED = os.getenv(
        "BS_TEMPORAL_FILTER_ENABLED",
        "1",
    ).lower() in {"1", "true", "yes", "on"}
    TEMPORAL_MIN_HITS = max(1, int(os.getenv("BS_TEMPORAL_MIN_HITS", "3")))
    TEMPORAL_MAX_MISSING = max(0, int(os.getenv("BS_TEMPORAL_MAX_MISSING", "4")))
    TEMPORAL_IOU_MATCH_THRES = max(
        0.0, float(os.getenv("BS_TEMPORAL_IOU_MATCH_THRES", "0.30"))
    )
    TEMPORAL_STATIC_CENTER_SHIFT_THRES = max(
        0.0,
        float(os.getenv("BS_TEMPORAL_STATIC_CENTER_SHIFT_THRES", "0.015")),
    )
    TEMPORAL_STATIC_AREA_CHANGE_THRES = max(
        0.0,
        float(os.getenv("BS_TEMPORAL_STATIC_AREA_CHANGE_THRES", "0.12")),
    )
    TEMPORAL_STATIC_MOTION_THRES = max(
        0.0,
        float(os.getenv("BS_TEMPORAL_STATIC_MOTION_THRES", "0.035")),
    )
    TEMPORAL_STATIC_FRAME_LIMIT = max(
        1, int(os.getenv("BS_TEMPORAL_STATIC_FRAME_LIMIT", "8"))
    )
    TEMPORAL_LOW_CONF_STATIC_SUPPRESS = float(
        os.getenv("BS_TEMPORAL_LOW_CONF_STATIC_SUPPRESS", "0.45")
    )
    TEMPORAL_ENABLE_ROAD_ROI_PRIOR = os.getenv(
        "BS_TEMPORAL_ENABLE_ROAD_ROI_PRIOR",
        "0",
    ).lower() in {"1", "true", "yes", "on"}
    TEMPORAL_ROAD_ROI_SCORE_THRES = max(
        0.0,
        float(os.getenv("BS_TEMPORAL_ROAD_ROI_SCORE_THRES", "0.10")),
    )
    TEMPORAL_ENABLE_SECOND_STAGE_CLASSIFIER = os.getenv(
        "BS_TEMPORAL_ENABLE_SECOND_STAGE_CLASSIFIER",
        "1",
    ).lower() in {"1", "true", "yes", "on"}
    TEMPORAL_SECOND_STAGE_VEHICLE_PROB_THRES = max(
        0.0,
        float(os.getenv("BS_TEMPORAL_SECOND_STAGE_VEHICLE_PROB_THRES", "0.10")),
    )
    TEMPORAL_SECOND_STAGE_MODEL_NAME = os.getenv(
        "BS_TEMPORAL_SECOND_STAGE_MODEL_NAME",
        "mobilenet_v3_small",
    )
    TEMPORAL_SECOND_STAGE_CLASSIFIER_WEIGHTS = os.getenv(
        "BS_TEMPORAL_SECOND_STAGE_CLASSIFIER_WEIGHTS"
    )

    # 是否对输入应用 ImageNet 均值方差归一化。
    # 当前默认关闭，意味着训练和推理都主要使用 `[0, 1]` 范围的原始张量。
    # 之所以保留这个开关，是为了后续做输入规范对比实验时不必改动多处代码。
    USE_IMAGENET_NORMALIZE = False

    # ==================== 多任务损失权重 ====================
    # 当前三项任务默认等权。
    # 若后续发现检测、分类或回归的尺度差异较大，可以从这里统一调整。
    DET_LOSS_WEIGHT = float(os.getenv("BS_DET_LOSS_WEIGHT", "1.0"))
    CLEAR_DET_LOSS_WEIGHT = float(os.getenv("BS_CLEAR_DET_LOSS_WEIGHT", "0.0"))
    FOG_CLS_LOSS_WEIGHT = float(os.getenv("BS_FOG_CLS_LOSS_WEIGHT", "1.5"))
    FOG_REG_LOSS_WEIGHT = float(os.getenv("BS_FOG_REG_LOSS_WEIGHT", "1.25"))
    FOG_LABEL_SMOOTHING = min(
        0.3, max(0.0, float(os.getenv("BS_FOG_LABEL_SMOOTHING", "0.05")))
    )
    FOG_CLS_CLEAR_WEIGHT = max(
        0.1, float(os.getenv("BS_FOG_CLS_CLEAR_WEIGHT", "0.75"))
    )
    FOG_CLS_UNIFORM_WEIGHT = max(
        0.1, float(os.getenv("BS_FOG_CLS_UNIFORM_WEIGHT", "1.0"))
    )
    FOG_CLS_PATCHY_WEIGHT = max(
        0.1, float(os.getenv("BS_FOG_CLS_PATCHY_WEIGHT", "1.1"))
    )
    RESUME_CHECKPOINT = os.getenv("BS_RESUME_CHECKPOINT")
    RESUME_MODEL_ONLY = os.getenv("BS_RESUME_MODEL_ONLY", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    RESUME_NONSTRICT_MODEL_ONLY = os.getenv(
        "BS_RESUME_NONSTRICT_MODEL_ONLY",
        "0",
    ).lower() in {"1", "true", "yes", "on"}
    RESUME_RESET_EPOCH = os.getenv("BS_RESUME_RESET_EPOCH", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    FREEZE_YOLO_FOR_FOG = os.getenv("BS_FREEZE_YOLO_FOR_FOG", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    _PATH_LIKE_ATTRS = {
        "RAW_DATA_DIR",
        "XML_DIR",
        "DEPTH_CACHE_DIR",
        "OUTPUT_DIR",
        "CHECKPOINT_DIR",
        "YOLO_BASE_MODEL",
        "MODEL_PATH",
        "RESUME_CHECKPOINT",
        "TEMPORAL_SECOND_STAGE_CLASSIFIER_WEIGHTS",
    }
    _EXPLICIT_KEY_ALIASES = {
        ("model", "yolo_weights"): "YOLO_BASE_MODEL",
        ("model", "det_head_mode"): "DET_HEAD_MODE",
        ("model", "num_fog_classes"): "NUM_FOG_CLASSES",
        ("model", "fog_class_names"): "FOG_CLASS_NAMES",
        ("model", "num_det_classes"): "NUM_DET_CLASSES",
        ("model", "det_class_names"): "DET_CLASS_NAMES",
        ("model", "det_train_class_id"): "DET_TRAIN_CLASS_ID",
        ("model", "coco_vehicle_train_class_id"): "COCO_VEHICLE_TRAIN_CLASS_ID",
        ("model", "vehicle_class_ids"): "VEHICLE_CLASS_IDS",
        ("training", "batch_size"): "BATCH_SIZE",
        ("training", "epochs"): "EPOCHS",
        ("training", "learning_rate"): "LR",
        ("training", "qat_epochs"): "QAT_EPOCHS",
        ("training", "qat_learning_rate"): "QAT_LR",
        ("training", "img_size"): "IMG_SIZE",
        ("training", "frame_stride"): "FRAME_STRIDE",
        ("training", "train_ratio"): "TRAIN_RATIO",
        ("physics", "beta_min"): "BETA_MIN",
        ("physics", "beta_max"): "BETA_MAX",
        ("physics", "a_min"): "A_MIN",
        ("physics", "a_max"): "A_MAX",
        ("paths", "raw_data_dir"): "RAW_DATA_DIR",
        ("paths", "xml_dir"): "XML_DIR",
        ("paths", "depth_cache_dir"): "DEPTH_CACHE_DIR",
        ("paths", "output_dir"): "OUTPUT_DIR",
        ("paths", "checkpoint_dir"): "CHECKPOINT_DIR",
        ("inference", "base_conf_thres"): "BASE_CONF_THRES",
        ("inference", "ema_alpha"): "EMA_ALPHA",
        ("display", "window_width"): "DISPLAY_WINDOW_WIDTH",
        ("display", "window_height"): "DISPLAY_WINDOW_HEIGHT",
        ("display", "status_bar_height"): "STATUS_BAR_HEIGHT",
        ("adaptive_threshold", "base_conf_thres"): "BASE_CONF_THRES",
        ("adaptive_threshold", "ema_alpha"): "EMA_ALPHA",
        ("adaptive_threshold", "beta_scale_factor"): "BETA_SCALE_FACTOR",
        ("adaptive_threshold", "min_conf_thres"): "MIN_CONF_THRES",
        ("temporal_filter", "enabled"): "TEMPORAL_FILTER_ENABLED",
        ("temporal_filter", "min_hits"): "TEMPORAL_MIN_HITS",
        ("temporal_filter", "max_missing"): "TEMPORAL_MAX_MISSING",
        ("temporal_filter", "iou_match_thres"): "TEMPORAL_IOU_MATCH_THRES",
        ("temporal_filter", "static_center_shift_thres"): "TEMPORAL_STATIC_CENTER_SHIFT_THRES",
        ("temporal_filter", "static_area_change_thres"): "TEMPORAL_STATIC_AREA_CHANGE_THRES",
        ("temporal_filter", "static_motion_thres"): "TEMPORAL_STATIC_MOTION_THRES",
        ("temporal_filter", "static_frame_limit"): "TEMPORAL_STATIC_FRAME_LIMIT",
        ("temporal_filter", "low_conf_static_suppress"): "TEMPORAL_LOW_CONF_STATIC_SUPPRESS",
        ("temporal_filter", "enable_road_roi_prior"): "TEMPORAL_ENABLE_ROAD_ROI_PRIOR",
        ("temporal_filter", "road_roi_score_thres"): "TEMPORAL_ROAD_ROI_SCORE_THRES",
        ("temporal_filter", "enable_second_stage_classifier"): "TEMPORAL_ENABLE_SECOND_STAGE_CLASSIFIER",
        ("temporal_filter", "second_stage_vehicle_prob_thres"): "TEMPORAL_SECOND_STAGE_VEHICLE_PROB_THRES",
        ("temporal_filter", "second_stage_model_name"): "TEMPORAL_SECOND_STAGE_MODEL_NAME",
        ("temporal_filter", "second_stage_classifier_weights"): "TEMPORAL_SECOND_STAGE_CLASSIFIER_WEIGHTS",
        ("loss", "det_loss_weight"): "DET_LOSS_WEIGHT",
        ("loss", "clear_det_loss_weight"): "CLEAR_DET_LOSS_WEIGHT",
        ("loss", "fog_cls_loss_weight"): "FOG_CLS_LOSS_WEIGHT",
        ("loss", "fog_reg_loss_weight"): "FOG_REG_LOSS_WEIGHT",
        ("resume", "checkpoint"): "RESUME_CHECKPOINT",
        ("resume", "model_only"): "RESUME_MODEL_ONLY",
        ("resume", "nonstrict_model_only"): "RESUME_NONSTRICT_MODEL_ONLY",
        ("resume", "reset_epoch"): "RESUME_RESET_EPOCH",
    }
    _GENERIC_KEY_ALIASES = {
        "yolo_weights": "YOLO_BASE_MODEL",
        "learning_rate": "LR",
        "qat_learning_rate": "QAT_LR",
        "window_width": "DISPLAY_WINDOW_WIDTH",
        "window_height": "DISPLAY_WINDOW_HEIGHT",
        "status_bar_height": "STATUS_BAR_HEIGHT",
        "beta_scale_factor": "BETA_SCALE_FACTOR",
        "min_conf_thres": "MIN_CONF_THRES",
        "model_only": "RESUME_MODEL_ONLY",
        "checkpoint": "RESUME_CHECKPOINT",
    }

    def __init__(self, config_path: str | None = None):
        """
        初始化配置实例。

        初始化过程中不重写默认配置，仅确保项目运行所依赖的输出目录真实存在，
        以避免训练、缓存或导出阶段因路径缺失而中断。

        由于主要配置均已通过类属性定义，因此实例化动作本身较轻量；
        这里唯一保留的副作用是目录存在性检查与创建。
        """
        self.CONFIG_FILE = self._resolve_config_file(config_path)
        self.UNUSED_CONFIG_KEYS: list[str] = []
        overridden_attrs: set[str] = set()
        if self.CONFIG_FILE is not None:
            payload = self._load_config_payload(Path(self.CONFIG_FILE))
            overridden_attrs = self._apply_config_payload(
                payload,
                config_dir=Path(self.CONFIG_FILE).resolve().parent,
            )
            if self.UNUSED_CONFIG_KEYS:
                preview = ", ".join(self.UNUSED_CONFIG_KEYS[:5])
                if len(self.UNUSED_CONFIG_KEYS) > 5:
                    preview += ", ..."
                print(
                    "Warning: some config keys were not consumed by Config: "
                    f"{preview}"
                )
        self._finalize_derived_fields(overridden_attrs)

        os.makedirs(self.DEPTH_CACHE_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

    @classmethod
    def _resolve_config_file(cls, config_path: str | None) -> str | None:
        """
        解析当前配置实例要加载的配置文件路径。

        优先级：
        1. 调用方显式传入的 `config_path`
        2. 环境变量 `BS_CONFIG_FILE`
        """
        candidate = config_path or os.getenv(cls.CONFIG_FILE_ENV)
        if not candidate:
            return None

        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (Path(cls.PROJECT_ROOT) / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return str(path)

    @staticmethod
    def _load_config_payload(path: Path) -> dict:
        """
        从 JSON 或 YAML 文件中读取配置覆盖项。
        """
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")

        if suffix == ".json":
            payload = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError(
                    "YAML config loading requires PyYAML to be installed."
                )
            payload = yaml.safe_load(text)
        else:
            raise ValueError(
                f"Unsupported config file type: {path.name}. Use .json/.yaml/.yml."
            )

        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise TypeError("Config file root must be a mapping/object.")
        return payload

    @classmethod
    def _iter_config_entries(
        cls,
        payload: dict,
        prefix: tuple[str, ...] = (),
    ) -> list[tuple[tuple[str, ...], object]]:
        entries: list[tuple[tuple[str, ...], object]] = []
        for key, value in payload.items():
            if not isinstance(key, str):
                raise TypeError(f"Config key must be a string, got {type(key)!r}")
            path = prefix + (key,)
            if isinstance(value, dict):
                entries.extend(cls._iter_config_entries(value, path))
            else:
                entries.append((path, value))
        return entries

    @classmethod
    def _resolve_attr_name(cls, key_path: tuple[str, ...]) -> str | None:
        """
        把配置文件中的键路径映射到 `Config` 属性名。
        """
        lowered = tuple(part.strip().lower().replace("-", "_") for part in key_path)
        if lowered in cls._EXPLICIT_KEY_ALIASES:
            return cls._EXPLICIT_KEY_ALIASES[lowered]

        leaf = lowered[-1]
        if leaf in cls._GENERIC_KEY_ALIASES:
            return cls._GENERIC_KEY_ALIASES[leaf]

        candidate = leaf.upper()
        if hasattr(cls, candidate):
            return candidate
        return None

    @classmethod
    def _resolve_override_path(cls, value: str, config_dir: Path) -> str:
        """
        解析配置文件中的相对路径。

        优先尝试：
        1. 绝对路径
        2. 相对配置文件目录
        3. 相对仓库根目录
        """
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return str(candidate.resolve())

        config_relative = (config_dir / candidate).resolve()
        if config_relative.exists():
            return str(config_relative)

        project_relative = (Path(cls.PROJECT_ROOT) / candidate).resolve()
        return str(project_relative)

    @classmethod
    def _coerce_override_value(
        cls,
        attr_name: str,
        raw_value,
        config_dir: Path,
    ):
        """
        根据默认类型把配置文件里的值转换成运行时值。
        """
        default_value = getattr(cls, attr_name)

        if raw_value is None:
            return None

        if attr_name in cls._PATH_LIKE_ATTRS and isinstance(raw_value, str):
            return cls._resolve_override_path(raw_value, config_dir)

        if isinstance(default_value, bool):
            return _parse_bool(raw_value)
        if isinstance(default_value, int) and not isinstance(default_value, bool):
            return int(raw_value)
        if isinstance(default_value, float):
            return float(raw_value)
        if isinstance(default_value, list):
            if not isinstance(raw_value, list):
                raise TypeError(
                    f"Config field {attr_name} expects a list, got {type(raw_value)!r}"
                )
            return list(raw_value)
        if isinstance(default_value, tuple):
            if not isinstance(raw_value, (list, tuple)):
                raise TypeError(
                    f"Config field {attr_name} expects a tuple-like value, got {type(raw_value)!r}"
                )
            return tuple(raw_value)
        if default_value is None:
            return raw_value
        return raw_value

    def _apply_config_payload(self, payload: dict, config_dir: Path) -> set[str]:
        """
        把配置文件内容映射并覆盖到当前配置实例上。
        """
        unused_keys: list[str] = []
        overridden_attrs: set[str] = set()
        for key_path, raw_value in self._iter_config_entries(payload):
            attr_name = self._resolve_attr_name(key_path)
            if attr_name is None:
                unused_keys.append(".".join(key_path))
                continue

            coerced = self._coerce_override_value(attr_name, raw_value, config_dir)
            setattr(self, attr_name, coerced)
            overridden_attrs.add(attr_name)

        if "OUTPUT_DIR" in overridden_attrs and "CHECKPOINT_DIR" not in overridden_attrs:
            self.CHECKPOINT_DIR = os.path.join(self.OUTPUT_DIR, "checkpoints")

        self.UNUSED_CONFIG_KEYS = unused_keys
        return overridden_attrs

    def _finalize_derived_fields(self, overridden_attrs: set[str]):
        """
        根据高层配置模式补齐派生字段。

        例如：
        - `DET_HEAD_MODE` 会影响 `NUM_DET_CLASSES`、`DET_TRAIN_CLASS_ID`
          和 `VEHICLE_CLASS_IDS` 的默认语义；
        - 但如果调用方已经显式覆盖这些字段，则保留显式值。
        """
        det_head_mode = str(getattr(self, "DET_HEAD_MODE", "single_vehicle")).strip().lower()
        if det_head_mode not in {"single_vehicle", "coco_vehicle"}:
            raise ValueError(f"Unsupported DET_HEAD_MODE: {det_head_mode!r}")

        self.DET_HEAD_MODE = det_head_mode
        if det_head_mode == "coco_vehicle":
            if "NUM_DET_CLASSES" not in overridden_attrs:
                self.NUM_DET_CLASSES = int(self.COCO_NUM_DET_CLASSES)
            if "DET_TRAIN_CLASS_ID" not in overridden_attrs:
                self.DET_TRAIN_CLASS_ID = int(self.COCO_VEHICLE_TRAIN_CLASS_ID)
            if "VEHICLE_CLASS_IDS" not in overridden_attrs:
                self.VEHICLE_CLASS_IDS = list(self.COCO_VEHICLE_CLASS_IDS)
        else:
            if "NUM_DET_CLASSES" not in overridden_attrs:
                self.NUM_DET_CLASSES = 1
            if "DET_TRAIN_CLASS_ID" not in overridden_attrs:
                self.DET_TRAIN_CLASS_ID = 0
            if "VEHICLE_CLASS_IDS" not in overridden_attrs:
                self.VEHICLE_CLASS_IDS = [0]

        if "DET_CLASS_NAMES" not in overridden_attrs:
            self.DET_CLASS_NAMES = ["vehicle"]

    def path_summary(self) -> dict[str, str]:
        """
        返回当前运行时最关键的路径配置。

        该方法主要服务于自检脚本、日志打印和问题排查。
        """
        return {
            "config_file": self.CONFIG_FILE or "",
            "raw_data_dir": self.RAW_DATA_DIR,
            "xml_dir": self.XML_DIR,
            "depth_cache_dir": self.DEPTH_CACHE_DIR,
            "output_dir": self.OUTPUT_DIR,
            "checkpoint_dir": self.CHECKPOINT_DIR,
            "model_path": self.MODEL_PATH or "",
            "temporal_second_stage_classifier_weights": (
                self.TEMPORAL_SECOND_STAGE_CLASSIFIER_WEIGHTS or ""
            ),
        }

    def training_controls(self) -> dict[str, int | float | bool]:
        """
        返回训练阶段的关键控制参数。

        该方法主要服务于日志记录、smoke run 排查和实验复现。
        """
        return {
            "config_file": self.CONFIG_FILE or "",
            "batch_size": self.BATCH_SIZE,
            "det_head_mode": self.DET_HEAD_MODE,
            "num_det_classes": self.NUM_DET_CLASSES,
            "det_train_class_id": self.DET_TRAIN_CLASS_ID,
            "epochs": self.EPOCHS,
            "lr": self.LR,
            "qat_epochs": self.QAT_EPOCHS,
            "qat_lr": self.QAT_LR,
            "img_size": self.IMG_SIZE,
            "frame_stride": self.FRAME_STRIDE,
            "train_ratio": self.TRAIN_RATIO,
            "max_train_batches": self.MAX_TRAIN_BATCHES,
            "max_val_batches": self.MAX_VAL_BATCHES,
            "grad_clip_norm": self.GRAD_CLIP_NORM,
            "nonfinite_grad_min_batches": self.NONFINITE_GRAD_MIN_BATCHES,
            "nonfinite_grad_warn_ratio": self.NONFINITE_GRAD_WARN_RATIO,
            "nonfinite_grad_fail_ratio": self.NONFINITE_GRAD_FAIL_RATIO,
            "nonfinite_grad_fail_streak": self.NONFINITE_GRAD_FAIL_STREAK,
            "nonfinite_grad_auto_disable_amp": self.NONFINITE_GRAD_AUTO_DISABLE_AMP,
            "skip_qat": self.SKIP_QAT,
            "disable_amp": self.DISABLE_AMP,
            "fog_clear_prob": self.FOG_CLEAR_PROB,
            "fog_uniform_prob": self.FOG_UNIFORM_PROB,
            "fog_patchy_prob": self.FOG_PATCHY_PROB,
            "fog_beta_min": self.FOG_BETA_MIN,
            "uniform_depth_scale": self.UNIFORM_DEPTH_SCALE,
            "patchy_depth_base": self.PATCHY_DEPTH_BASE,
            "patchy_depth_noise_scale": self.PATCHY_DEPTH_NOISE_SCALE,
            "fog_label_smoothing": self.FOG_LABEL_SMOOTHING,
            "fog_cls_clear_weight": self.FOG_CLS_CLEAR_WEIGHT,
            "fog_cls_uniform_weight": self.FOG_CLS_UNIFORM_WEIGHT,
            "fog_cls_patchy_weight": self.FOG_CLS_PATCHY_WEIGHT,
            "det_loss_weight": self.DET_LOSS_WEIGHT,
            "clear_det_loss_weight": self.CLEAR_DET_LOSS_WEIGHT,
            "fog_reg_loss_weight": self.FOG_REG_LOSS_WEIGHT,
            "resume_model_only": self.RESUME_MODEL_ONLY,
            "resume_nonstrict_model_only": self.RESUME_NONSTRICT_MODEL_ONLY,
            "resume_reset_epoch": self.RESUME_RESET_EPOCH,
            "freeze_yolo_for_fog": self.FREEZE_YOLO_FOR_FOG,
            "temporal_filter_enabled": self.TEMPORAL_FILTER_ENABLED,
            "temporal_min_hits": self.TEMPORAL_MIN_HITS,
            "temporal_max_missing": self.TEMPORAL_MAX_MISSING,
            "temporal_iou_match_thres": self.TEMPORAL_IOU_MATCH_THRES,
            "temporal_static_center_shift_thres": self.TEMPORAL_STATIC_CENTER_SHIFT_THRES,
            "temporal_static_area_change_thres": self.TEMPORAL_STATIC_AREA_CHANGE_THRES,
            "temporal_static_motion_thres": self.TEMPORAL_STATIC_MOTION_THRES,
            "temporal_static_frame_limit": self.TEMPORAL_STATIC_FRAME_LIMIT,
            "temporal_low_conf_static_suppress": self.TEMPORAL_LOW_CONF_STATIC_SUPPRESS,
            "temporal_enable_road_roi_prior": self.TEMPORAL_ENABLE_ROAD_ROI_PRIOR,
            "temporal_road_roi_score_thres": self.TEMPORAL_ROAD_ROI_SCORE_THRES,
            "temporal_enable_second_stage_classifier": self.TEMPORAL_ENABLE_SECOND_STAGE_CLASSIFIER,
            "temporal_second_stage_vehicle_prob_thres": self.TEMPORAL_SECOND_STAGE_VEHICLE_PROB_THRES,
            "seed": self.SEED,
        }

    def __repr__(self):
        """返回配置摘要，便于调试和日志打印。"""
        return (
            f"Config(BASE_DIR={self.BASE_DIR}, "
            f"CONFIG_FILE={self.CONFIG_FILE}, "
            f"DEVICE={self.DEVICE}, "
            f"NUM_DET_CLASSES={self.NUM_DET_CLASSES}, "
            f"BETA_RANGE=[{self.BETA_MIN}, {self.BETA_MAX}], "
            f"FRAME_STRIDE={self.FRAME_STRIDE}, "
            f"TRAIN_RATIO={self.TRAIN_RATIO}, "
            f"PRECOMPUTE_DEPTH_CACHE={self.PRECOMPUTE_DEPTH_CACHE}, "
            f"NUM_WORKERS={self.NUM_WORKERS}, "
            f"BATCH_SIZE={self.BATCH_SIZE}, "
            f"EPOCHS={self.EPOCHS}, "
            f"QAT_EPOCHS={self.QAT_EPOCHS}, "
            f"MAX_TRAIN_BATCHES={self.MAX_TRAIN_BATCHES}, "
            f"MAX_VAL_BATCHES={self.MAX_VAL_BATCHES}, "
            f"TEMPORAL_FILTER_ENABLED={self.TEMPORAL_FILTER_ENABLED}, "
            f"TEMPORAL_MIN_HITS={self.TEMPORAL_MIN_HITS}, "
            f"NONFINITE_GRAD_WARN_RATIO={self.NONFINITE_GRAD_WARN_RATIO}, "
            f"NONFINITE_GRAD_FAIL_RATIO={self.NONFINITE_GRAD_FAIL_RATIO}, "
            f"NONFINITE_GRAD_FAIL_STREAK={self.NONFINITE_GRAD_FAIL_STREAK}, "
            f"NONFINITE_GRAD_AUTO_DISABLE_AMP={self.NONFINITE_GRAD_AUTO_DISABLE_AMP}, "
            f"SKIP_QAT={self.SKIP_QAT}, "
            f"DISABLE_AMP={self.DISABLE_AMP}, "
            f"UNUSED_CONFIG_KEYS={len(getattr(self, 'UNUSED_CONFIG_KEYS', []))})"
        )


def get_default_config(config_path: str | None = None) -> Config:
    """
    获取默认配置实例。

    Returns:
        Config: 按当前源码默认值构造的配置对象。

    该工厂函数的存在主要是为了给外部调用方提供一个更清晰、稳定的默认配置入口。
    """
    return Config(config_path=config_path)


if __name__ == "__main__":
    cfg = Config()
    print(cfg)
    print(cfg.path_summary())
