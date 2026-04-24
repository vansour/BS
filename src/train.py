#!/usr/bin/env python3
# ruff: noqa: E402
"""
统一多任务训练脚本
Unified Multi-Task Training Script

本模块负责整个训练闭环，主要包括：
1. 数据集与深度缓存准备；
2. 在线造雾增强；
3. FP32 主训练阶段；
4. QAT 量化感知训练阶段；
5. INT8 模型转换与保存；
6. 检查点续训与历史检查点清理。

本模块是当前项目训练主线的真实入口。与许多“先离线生成增强数据再训练”的方案不同，
本项目采用如下训练路径：
清晰图像 + 深度缓存 + XML 检测框  ->  在线造雾  ->  多任务联合优化

因此，本文件所组织的不仅是模型训练过程本身，还包括数据准备约定、损失函数组织方式、
断点续训机制、QAT 量化训练以及最终 INT8 转换前的必要步骤。
"""

import json
import math
import multiprocessing
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import Config
from src.data import MultiTaskDataset, precompute_depths
from src.model import FogAugmentation, UnifiedMultiTaskModel
from src.utils import find_latest_checkpoint, set_seed


def build_cfg_snapshot(cfg: Config) -> dict:
    """
    把配置对象转成可序列化字典，写入 checkpoint。

    Args:
        cfg: 当前训练使用的配置对象。

    Returns:
        dict: 适合保存到 checkpoint 中的配置快照。

    把关键训练配置写入 checkpoint 的好处是：
    - 续训时更容易核对当前权重对应的训练条件；
    - 后续追溯模型来源时，不必完全依赖 README 或人工记录；
    - 即使配置类后续发生变化，也能保留当时训练时的主要超参数摘要。
    """
    return {
        "yolo_base_model": cfg.YOLO_BASE_MODEL,
        "num_det_classes": cfg.NUM_DET_CLASSES,
        "det_train_class_id": cfg.DET_TRAIN_CLASS_ID,
        "vehicle_class_ids": list(cfg.VEHICLE_CLASS_IDS),
        "num_fog_classes": cfg.NUM_FOG_CLASSES,
        "det_loss_weight": cfg.DET_LOSS_WEIGHT,
        "fog_cls_loss_weight": cfg.FOG_CLS_LOSS_WEIGHT,
        "fog_reg_loss_weight": cfg.FOG_REG_LOSS_WEIGHT,
        "batch_size": cfg.BATCH_SIZE,
        "epochs": cfg.EPOCHS,
        "qat_epochs": cfg.QAT_EPOCHS,
        "lr": cfg.LR,
        "qat_lr": cfg.QAT_LR,
        "img_size": cfg.IMG_SIZE,
        "frame_stride": cfg.FRAME_STRIDE,
        "train_ratio": cfg.TRAIN_RATIO,
        "precompute_depth_cache": cfg.PRECOMPUTE_DEPTH_CACHE,
        "skip_qat": cfg.SKIP_QAT,
        "max_train_batches": cfg.MAX_TRAIN_BATCHES,
        "max_val_batches": cfg.MAX_VAL_BATCHES,
        "grad_clip_norm": cfg.GRAD_CLIP_NORM,
        "nonfinite_grad_min_batches": cfg.NONFINITE_GRAD_MIN_BATCHES,
        "nonfinite_grad_warn_ratio": cfg.NONFINITE_GRAD_WARN_RATIO,
        "nonfinite_grad_fail_ratio": cfg.NONFINITE_GRAD_FAIL_RATIO,
        "nonfinite_grad_fail_streak": cfg.NONFINITE_GRAD_FAIL_STREAK,
        "nonfinite_grad_auto_disable_amp": cfg.NONFINITE_GRAD_AUTO_DISABLE_AMP,
        "seed": cfg.SEED,
        "num_workers": cfg.NUM_WORKERS,
        "prefetch_factor": cfg.PREFETCH_FACTOR,
        "persistent_workers": cfg.PERSISTENT_WORKERS,
        "beta_min": cfg.BETA_MIN,
        "beta_max": cfg.BETA_MAX,
        "a_min": cfg.A_MIN,
        "a_max": cfg.A_MAX,
        "fog_clear_prob": cfg.FOG_CLEAR_PROB,
        "fog_uniform_prob": cfg.FOG_UNIFORM_PROB,
        "fog_patchy_prob": cfg.FOG_PATCHY_PROB,
        "fog_beta_min": cfg.FOG_BETA_MIN,
        "uniform_depth_scale": cfg.UNIFORM_DEPTH_SCALE,
        "patchy_depth_base": cfg.PATCHY_DEPTH_BASE,
        "patchy_depth_noise_scale": cfg.PATCHY_DEPTH_NOISE_SCALE,
        "fog_label_smoothing": cfg.FOG_LABEL_SMOOTHING,
        "fog_cls_clear_weight": cfg.FOG_CLS_CLEAR_WEIGHT,
        "fog_cls_uniform_weight": cfg.FOG_CLS_UNIFORM_WEIGHT,
        "fog_cls_patchy_weight": cfg.FOG_CLS_PATCHY_WEIGHT,
        "resume_checkpoint": cfg.RESUME_CHECKPOINT,
        "resume_model_only": cfg.RESUME_MODEL_ONLY,
        "freeze_yolo_for_fog": cfg.FREEZE_YOLO_FOR_FOG,
    }


def make_run_dir(cfg: Config) -> str:
    """
    为当前训练任务创建独立运行目录。

    该目录用于保存：
    - 配置快照；
    - 逐 epoch 指标日志；
    - 本次训练的最终摘要。
    """
    mode = (
        "smoke"
        if cfg.MAX_TRAIN_BATCHES > 0 or cfg.MAX_VAL_BATCHES > 0 or cfg.SKIP_QAT
        else "train"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.OUTPUT_DIR, "runs", f"{mode}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_json(path: str, payload: dict):
    """以 UTF-8 JSON 格式写入结构化信息。"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: str, payload: dict):
    """向 JSONL 文件追加一条记录。"""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def finite_scalar(name: str, value: torch.Tensor) -> float:
    """
    检查标量张量是否为有限值。

    训练阶段一旦出现 NaN/Inf，就应立即终止并暴露问题，而不是继续污染后续状态。
    """
    scalar = float(value.detach().item())
    if not math.isfinite(scalar):
        raise RuntimeError(f"Non-finite {name} detected: {scalar}")
    return scalar


def init_epoch_meter() -> dict[str, float]:
    """初始化单个 epoch 的指标累加器。"""
    return {
        "loss": 0.0,
        "det": 0.0,
        "fog_cls": 0.0,
        "fog_reg": 0.0,
        "grad_norm": 0.0,
        "nonfinite_grad_batches": 0.0,
        "batches": 0,
    }


def finalize_epoch_meter(meter: dict[str, float]) -> dict[str, float]:
    """将累加器转换为平均指标。"""
    batches = int(meter["batches"])
    if batches <= 0:
        raise RuntimeError("No batches were processed in the epoch.")

    result = {
        "loss": meter["loss"] / batches,
        "det": meter["det"] / batches,
        "fog_cls": meter["fog_cls"] / batches,
        "fog_reg": meter["fog_reg"] / batches,
        "batches": batches,
    }
    if meter["grad_norm"] > 0:
        result["grad_norm"] = meter["grad_norm"] / batches
    if meter["nonfinite_grad_batches"] > 0:
        result["nonfinite_grad_batches"] = int(meter["nonfinite_grad_batches"])
        result["nonfinite_grad_ratio"] = meter["nonfinite_grad_batches"] / batches
    return result


def clip_gradients(
    model,
    max_norm: float,
    *,
    allow_nonfinite: bool,
) -> tuple[float, bool]:
    """
    执行梯度裁剪，并返回梯度范数与是否出现非有限值。

    在 AMP 路径下，`GradScaler` 会负责处理非有限梯度并在必要时跳过更新，
    因此这里只记录该状态而不立即中断训练；在纯 FP32 路径下则保持严格失败。
    """
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        error_if_nonfinite=not allow_nonfinite,
    )
    grad_norm_value = float(grad_norm.detach().item())
    if math.isfinite(grad_norm_value):
        return grad_norm_value, False
    if allow_nonfinite:
        return 0.0, True
    raise RuntimeError(f"Non-finite grad_norm detected: {grad_norm_value}")


def evaluate_nonfinite_grad_health(
    metrics: dict[str, float],
    cfg: Config,
    *,
    phase: str,
    epoch: int,
    consecutive_fail_epochs: int,
    amp_enabled: bool,
) -> tuple[str | None, int, bool]:
    """
    评估当前 epoch 中非有限梯度的频率是否异常。

    返回值为：
    - 可打印的状态消息；
    - 更新后的连续 fail 计数。
    - 是否应触发“关闭 AMP 并继续训练”的恢复动作。

    只有当“达到最小 batch 数”且“连续 N 个 epoch 超过 fail 阈值”时才中止训练。
    """
    nonfinite_batches = int(metrics.get("nonfinite_grad_batches", 0))
    total_batches = int(metrics.get("batches", 0))
    if nonfinite_batches <= 0 or total_batches <= 0:
        return None, 0, False

    ratio = float(
        metrics.get("nonfinite_grad_ratio", nonfinite_batches / total_batches)
    )
    descriptor = (
        f"{phase} epoch {epoch}: non-finite grad batches="
        f"{nonfinite_batches}/{total_batches} ({ratio:.2%})"
    )
    if total_batches < cfg.NONFINITE_GRAD_MIN_BATCHES:
        return (
            (
                f"Info: {descriptor}. "
                f"Below BS_NONFINITE_GRAD_MIN_BATCHES={cfg.NONFINITE_GRAD_MIN_BATCHES}, "
                "so warn/fail thresholds are not enforced."
            ),
            0,
            False,
        )

    if cfg.NONFINITE_GRAD_FAIL_RATIO > 0 and ratio >= cfg.NONFINITE_GRAD_FAIL_RATIO:
        next_streak = consecutive_fail_epochs + 1
        should_disable_amp = (
            amp_enabled
            and cfg.NONFINITE_GRAD_AUTO_DISABLE_AMP
            and next_streak >= cfg.NONFINITE_GRAD_FAIL_STREAK
        )
        if should_disable_amp:
            return (
                f"Warning: {descriptor}, which exceeds "
                f"BS_NONFINITE_GRAD_FAIL_RATIO={cfg.NONFINITE_GRAD_FAIL_RATIO:.2%} "
                f"for {next_streak} consecutive epoch(s). "
                "AMP recovery will be triggered and subsequent epochs will run with AMP disabled.",
                0,
                True,
            )
        if next_streak >= cfg.NONFINITE_GRAD_FAIL_STREAK:
            raise RuntimeError(
                f"{descriptor}, which exceeds "
                f"BS_NONFINITE_GRAD_FAIL_RATIO={cfg.NONFINITE_GRAD_FAIL_RATIO:.2%} "
                f"for {next_streak} consecutive epoch(s)."
            )
        return (
            f"Warning: {descriptor}, which exceeds "
            f"BS_NONFINITE_GRAD_FAIL_RATIO={cfg.NONFINITE_GRAD_FAIL_RATIO:.2%}. "
            f"Current fail streak: {next_streak}/{cfg.NONFINITE_GRAD_FAIL_STREAK}.",
            next_streak,
            False,
        )

    if cfg.NONFINITE_GRAD_WARN_RATIO > 0 and ratio >= cfg.NONFINITE_GRAD_WARN_RATIO:
        return (
            (
                f"Warning: {descriptor}, which exceeds "
                f"BS_NONFINITE_GRAD_WARN_RATIO={cfg.NONFINITE_GRAD_WARN_RATIO:.2%}."
            ),
            0,
            False,
        )

    return None, 0, False


def multitask_collate_fn(batch):
    """
    组装多任务训练所需的 batch。

    检测框数量在不同图像之间并不固定，因此这里把检测标签整理为
    Ultralytics 检测损失需要的扁平结构：
    - `batch_idx`
    - `cls`
    - `bboxes`

    之所以不能简单 `torch.stack(det_boxes)`，是因为每张图的目标数量不同。
    Ultralytics 的检测损失接口期望看到的是“所有目标拼在一起，再用 batch_idx
    指明每个目标属于哪张图”的格式，这个函数正是在做这层适配。
    """
    imgs, depths, det_cls_list, det_box_list = zip(*batch)

    # 图像和深度本身是固定尺寸张量，可以直接按 batch 维堆叠。
    imgs = torch.stack(imgs, 0)
    depths = torch.stack(depths, 0)

    batch_idx_all = []
    cls_all = []
    bboxes_all = []

    for sample_idx, (det_cls, det_boxes) in enumerate(zip(det_cls_list, det_box_list)):
        if det_boxes.numel() == 0:
            continue
        # 为当前图像中的每个目标都记录一个所属样本索引，
        # 供 Ultralytics loss 在 batch 内回溯目标归属。
        batch_idx_all.append(
            torch.full((det_boxes.shape[0],), sample_idx, dtype=torch.int64)
        )
        cls_all.append(det_cls)
        bboxes_all.append(det_boxes)

    if batch_idx_all:
        det_targets = {
            "batch_idx": torch.cat(batch_idx_all, 0),
            "cls": torch.cat(cls_all, 0),
            "bboxes": torch.cat(bboxes_all, 0),
        }
    else:
        # 整个 batch 都没有目标时，仍然返回结构完整的空张量，
        # 避免下游检测损失接口因为键缺失或 shape 异常而报错。
        det_targets = {
            "batch_idx": torch.zeros((0,), dtype=torch.int64),
            "cls": torch.zeros((0,), dtype=torch.float32),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32),
        }

    return imgs, depths, det_targets


def prune_old_checkpoints(checkpoint_dir: str, keep_max: int):
    """
    删除过旧 checkpoint，限制目录体积。

    Args:
        checkpoint_dir: checkpoint 目录。
        keep_max: 最多保留多少个最新文件。

    训练过程中如果每个 epoch 都保存 checkpoint，而从不清理，很快就会把磁盘写满。
    这个函数用一个简单的“保留最近 N 个”策略控制目录体积。
    """
    if keep_max <= 0 or not os.path.isdir(checkpoint_dir):
        return

    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if not filename.endswith(".pt"):
            continue
        full_path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(full_path):
            checkpoint_files.append(full_path)

    if len(checkpoint_files) <= keep_max:
        return

    # 按修改时间从新到旧排序，再删除多余的旧文件。
    checkpoint_files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    for stale_path in checkpoint_files[keep_max:]:
        try:
            os.remove(stale_path)
            print(f"Removed old checkpoint: {stale_path}")
        except OSError as exc:
            print(f"Failed to remove old checkpoint {stale_path}: {exc}")


def save_checkpoint(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    train_loss,
    best_loss,
    cfg,
    val_loss=None,
):
    """
    保存训练 checkpoint。

    Args:
        checkpoint_path: checkpoint 输出路径。
        epoch: 当前 epoch 编号。
        model: 当前模型。
        optimizer: 优化器。
        scheduler: 学习率调度器。
        scaler: AMP 梯度缩放器，可为空。
        train_loss: 当前 epoch 平均训练损失。
        best_loss: 当前历史最佳损失。
        cfg: 当前配置对象。

    这里保存的是“可恢复训练状态”的 checkpoint，而不是仅供推理使用的纯权重文件。
    因此除了模型参数，还包含优化器、调度器、AMP scaler 和配置快照。
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "best_loss": best_loss,
        "cfg_snapshot": build_cfg_snapshot(cfg),
    }
    if val_loss is not None:
        checkpoint["val_loss"] = val_loss

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, scaler=None
):
    """
    加载训练 checkpoint，并恢复训练状态。

    Args:
        checkpoint_path: checkpoint 路径。
        model: 待恢复的模型。
        optimizer: 可选优化器。
        scheduler: 可选学习率调度器。
        scaler: 可选 AMP 梯度缩放器。

    Returns:
        tuple: `(start_epoch, train_loss, best_loss)`。

    这里默认把 checkpoint 先加载到 CPU，再恢复到调用方的模型和优化器中。
    这样做更稳妥，避免直接在 CUDA 上反序列化时受到设备环境影响。
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float("inf"), float("inf")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    train_loss = checkpoint.get("train_loss", float("inf"))
    best_loss = checkpoint.get("best_loss", float("inf"))

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous train loss: {train_loss:.4f}")
    if "val_loss" in checkpoint and checkpoint["val_loss"] is not None:
        print(f"Previous val loss: {checkpoint['val_loss']:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    return start_epoch, train_loss, best_loss


def summarize_missing_depth_cache(dataset: MultiTaskDataset) -> tuple[int, list[str]]:
    """
    统计数据集中缺失的深度缓存数量，并返回少量样例文件名。

    训练与验证都依赖深度缓存。若只检查训练集而忽略验证集，就会出现
    “训练能跑完，第一个验证 batch 直接因缺缓存崩溃”的情况。
    这里在构建 DataLoader 前统一盘点两边的缓存覆盖情况。
    """
    missing_count = 0
    missing_examples: list[str] = []

    for _, seq, img_name in dataset.samples:
        depth_name = f"{seq}_{img_name}.npy"
        depth_path = os.path.join(dataset.depth_cache_dir, depth_name)
        if os.path.exists(depth_path):
            continue

        missing_count += 1
        if len(missing_examples) < 5:
            missing_examples.append(depth_name)

    return missing_count, missing_examples


def build_train_components(cfg: Config, device: str):
    """
    构建训练所需的数据与损失组件。

    Args:
        cfg: 当前配置对象。
        device: 当前运行设备。

    Returns:
        tuple: 归一化变换、训练 DataLoader、验证 DataLoader、分类损失、回归损失。

    这个函数把“训练前固定准备工作”集中起来，避免 `train()` 主体里同时混杂：
    - 变换定义；
    - 数据集构建；
    - 深度缓存补算；
    - DataLoader 参数组织；
    - 损失函数创建。
    """
    if cfg.USE_IMAGENET_NORMALIZE:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        def normalize(batch):
            return (batch - mean.to(batch.device, batch.dtype)) / std.to(
                batch.device, batch.dtype
            )

    else:
        # 保持接口一致：即便不做归一化，也返回一个可调用对象。
        normalize = nn.Identity()
    if not os.path.isdir(cfg.XML_DIR):
        raise RuntimeError(
            f"Detection annotation directory was not found: {cfg.XML_DIR}"
        )

    train_ds = MultiTaskDataset(
        cfg.RAW_DATA_DIR,
        cfg.DEPTH_CACHE_DIR,
        xml_dir=cfg.XML_DIR,
        transform=None,
        is_train=True,
        frame_stride=cfg.FRAME_STRIDE,
        det_train_class_id=cfg.DET_TRAIN_CLASS_ID,
        img_size=cfg.IMG_SIZE,
        keep_ratio=True,
        train_ratio=cfg.TRAIN_RATIO,
        split_seed=cfg.SEED,
    )
    val_ds = MultiTaskDataset(
        cfg.RAW_DATA_DIR,
        cfg.DEPTH_CACHE_DIR,
        xml_dir=cfg.XML_DIR,
        transform=None,
        is_train=False,
        frame_stride=cfg.FRAME_STRIDE,
        det_train_class_id=cfg.DET_TRAIN_CLASS_ID,
        img_size=cfg.IMG_SIZE,
        keep_ratio=True,
        train_ratio=cfg.TRAIN_RATIO,
        split_seed=cfg.SEED,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"Training dataset is empty: {cfg.RAW_DATA_DIR}")

    print(f"Training samples: {len(train_ds)} (frame_stride={cfg.FRAME_STRIDE})")
    print(f"Validation samples: {len(val_ds)} (frame_stride={cfg.FRAME_STRIDE})")

    train_missing_count, train_missing_examples = summarize_missing_depth_cache(
        train_ds
    )
    val_missing_count, val_missing_examples = (
        summarize_missing_depth_cache(val_ds) if len(val_ds) > 0 else (0, [])
    )

    print(
        "Depth cache coverage: "
        f"train missing={train_missing_count}, "
        f"val missing={val_missing_count}"
    )
    if train_missing_examples:
        print(f"Sample missing train depth files: {train_missing_examples}")
    if val_missing_examples:
        print(f"Sample missing val depth files: {val_missing_examples}")

    # 当前策略改为：
    # 1. 显式开启 BS_PRECOMPUTE_DEPTH_CACHE=1 时，总是先跑预计算流程；
    # 2. 即使未显式开启，只要检测到 train/val 任一侧存在缺失缓存，也会自动补齐；
    # 3. 只有两边缓存都完整时，才真正跳过预计算。
    if cfg.PRECOMPUTE_DEPTH_CACHE or train_missing_count > 0 or val_missing_count > 0:
        if cfg.PRECOMPUTE_DEPTH_CACHE:
            print("Depth cache precomputation is forced by BS_PRECOMPUTE_DEPTH_CACHE.")
        else:
            print(
                "Missing depth cache files were detected. Auto-precomputing train/val depth cache."
            )

        if train_missing_count > 0 or cfg.PRECOMPUTE_DEPTH_CACHE:
            precompute_depths(train_ds, device)
        if len(val_ds) > 0 and (val_missing_count > 0 or cfg.PRECOMPUTE_DEPTH_CACHE):
            precompute_depths(val_ds, device)
    else:
        print(
            "Depth cache is already complete for both training and validation datasets."
        )

    # DataLoader 的几个关键选项：
    # - shuffle=True：训练阶段打乱样本顺序；
    # - pin_memory：CUDA 下加快主机到设备的数据传输；
    # - collate_fn：把不定长检测框整理成 Ultralytics 期望的格式。
    loader_kwargs = {
        "batch_size": cfg.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": device == "cuda",
        "collate_fn": multitask_collate_fn,
    }
    if cfg.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = cfg.PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = cfg.PREFETCH_FACTOR

    print(
        "DataLoader config: "
        f"num_workers={cfg.NUM_WORKERS}, "
        f"persistent_workers={loader_kwargs.get('persistent_workers', False)}, "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')}"
    )

    train_loader = DataLoader(train_ds, **loader_kwargs)

    val_loader = None
    if len(val_ds) > 0:
        val_loader_kwargs = dict(loader_kwargs)
        val_loader_kwargs["shuffle"] = False
        val_loader = DataLoader(val_ds, **val_loader_kwargs)

    cls_weight = torch.tensor(
        [
            cfg.FOG_CLS_CLEAR_WEIGHT,
            cfg.FOG_CLS_UNIFORM_WEIGHT,
            cfg.FOG_CLS_PATCHY_WEIGHT,
        ],
        dtype=torch.float32,
    )
    # 给 clear 更低的损失权重，并加少量 label smoothing，
    # 以降低天气头在真实视频上“高置信度塌到 clear”的风险。
    criterion_cls = nn.CrossEntropyLoss(
        weight=cls_weight,
        label_smoothing=cfg.FOG_LABEL_SMOOTHING,
    ).to(device)
    criterion_reg = nn.MSELoss().to(device)
    return normalize, train_loader, val_loader, criterion_cls, criterion_reg


def compute_multitask_losses(
    model,
    imgs_norm,
    det_targets,
    cls_labels,
    reg_labels,
    criterion_cls,
    criterion_reg,
    cfg,
    device,
):
    """
    统一计算检测、雾分类和 beta 回归损失。
    """
    # 验证阶段模型处于 eval()，但 detection loss 仍需要训练态那类原始检测输出结构。
    # 因此这里显式要求模型返回 raw detection preds，而不是推理用的后处理张量。
    det_preds, pred_cls, pred_reg = model(imgs_norm, return_raw_det=True)
    det_batch = {
        "img": imgs_norm,
        "batch_idx": det_targets["batch_idx"].to(device),
        "cls": det_targets["cls"].to(device),
        "bboxes": det_targets["bboxes"].to(device),
    }

    if cfg.DET_LOSS_WEIGHT > 0:
        det_loss_components, _ = model.yolo.loss(det_batch, preds=det_preds)
        loss_det = det_loss_components.sum() / max(imgs_norm.size(0), 1)
    else:
        loss_det = pred_cls.new_zeros(())
    loss_cls = criterion_cls(pred_cls, cls_labels)
    loss_reg = criterion_reg(pred_reg * cfg.BETA_MAX, reg_labels)
    loss = (
        cfg.DET_LOSS_WEIGHT * loss_det
        + cfg.FOG_CLS_LOSS_WEIGHT * loss_cls
        + cfg.FOG_REG_LOSS_WEIGHT * loss_reg
    )
    return loss, loss_det, loss_cls, loss_reg


def train_epoch(
    model,
    fog_augmenter,
    train_loader,
    normalize,
    optimizer,
    scaler,
    criterion_cls,
    criterion_reg,
    cfg,
    device,
    desc,
):
    """
    执行单个 epoch 的训练并返回平均损失。

    Args:
        model: 统一多任务模型。
        fog_augmenter: 在线造雾增强模块。
        train_loader: 训练数据加载器。
        normalize: 输入归一化变换。
        optimizer: 优化器。
        scaler: AMP 梯度缩放器，可为空。
        criterion_cls: 分类损失函数。
        criterion_reg: 回归损失函数。
        cfg: 配置对象。
        device: 当前设备。
        desc: 进度条描述文案。

    Returns:
        float: 当前 epoch 的平均训练损失。

    单个 batch 的真实计算链路是：
    1. 把清晰图和深度图搬到目标设备；
    2. 用 FogAugmentation 在线生成雾图、雾类别标签和 beta 标签；
    3. 把雾图送入统一模型，得到检测输出、分类输出和回归输出；
    4. 分别计算检测、分类和回归损失；
    5. 按权重求和后反向传播。
    """
    model.train()
    if cfg.FREEZE_YOLO_FOR_FOG:
        # Freeze detector BN/dropout behavior as well; this mode is intended for
        # short weather-head-focused adaptation rather than detector optimization.
        model.yolo.eval()
    meter = init_epoch_meter()
    pbar = tqdm(train_loader, desc=desc)
    warned_nonfinite_amp_grad = False

    for batch_index, (imgs, depths, det_targets) in enumerate(pbar, start=1):
        if cfg.MAX_TRAIN_BATCHES > 0 and batch_index > cfg.MAX_TRAIN_BATCHES:
            break

        imgs = imgs.to(device)
        depths = depths.to(device)

        # 在线生成雾天样本，得到分类标签和回归标签。
        # 这里放在 no_grad() 中，是因为增强参数不是可学习参数，不需要保留梯度图。
        with torch.no_grad():
            imgs_foggy, cls_labels, reg_labels = fog_augmenter(imgs, depths)
            imgs_norm = normalize(imgs_foggy)

        optimizer.zero_grad(set_to_none=True)
        grad_norm_value = 0.0

        # 在 CUDA 环境下优先启用 AMP，以降低显存压力并提升吞吐。
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, loss_det, loss_cls, loss_reg = compute_multitask_losses(
                    model,
                    imgs_norm,
                    det_targets,
                    cls_labels,
                    reg_labels,
                    criterion_cls,
                    criterion_reg,
                    cfg,
                    device,
                )
            loss_value = finite_scalar("loss", loss)
            det_value = finite_scalar("det_loss", loss_det)
            cls_value = finite_scalar("fog_cls_loss", loss_cls)
            reg_value = finite_scalar("fog_reg_loss", loss_reg)
            scaler.scale(loss).backward()
            if cfg.GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                grad_norm_value, nonfinite_grad = clip_gradients(
                    model,
                    cfg.GRAD_CLIP_NORM,
                    allow_nonfinite=True,
                )
                if nonfinite_grad:
                    meter["nonfinite_grad_batches"] += 1
                    if not warned_nonfinite_amp_grad:
                        print(
                            "Warning: non-finite grad norm detected after AMP unscale+clip; "
                            "letting GradScaler decide whether to skip the optimizer step."
                        )
                        warned_nonfinite_amp_grad = True
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_det, loss_cls, loss_reg = compute_multitask_losses(
                model,
                imgs_norm,
                det_targets,
                cls_labels,
                reg_labels,
                criterion_cls,
                criterion_reg,
                cfg,
                device,
            )
            loss_value = finite_scalar("loss", loss)
            det_value = finite_scalar("det_loss", loss_det)
            cls_value = finite_scalar("fog_cls_loss", loss_cls)
            reg_value = finite_scalar("fog_reg_loss", loss_reg)
            loss.backward()
            if cfg.GRAD_CLIP_NORM > 0:
                grad_norm_value, _ = clip_gradients(
                    model,
                    cfg.GRAD_CLIP_NORM,
                    allow_nonfinite=False,
                )
            optimizer.step()

        meter["loss"] += loss_value
        meter["det"] += det_value
        meter["fog_cls"] += cls_value
        meter["fog_reg"] += reg_value
        meter["grad_norm"] += grad_norm_value
        meter["batches"] += 1
        # 进度条里同时展示总损失和三项子损失，便于快速判断哪一项在主导训练。
        postfix = {
            "loss": f"{loss_value:.4f}",
            "det": f"{det_value:.4f}",
            "fog_cls": f"{cls_value:.4f}",
            "fog_reg": f"{reg_value:.4f}",
        }
        if cfg.GRAD_CLIP_NORM > 0:
            postfix["grad"] = (
                f"{grad_norm_value:.4f}"
                if grad_norm_value > 0
                else (
                    "nonfinite"
                    if meter["nonfinite_grad_batches"] > 0
                    else f"{grad_norm_value:.4f}"
                )
            )
        pbar.set_postfix(postfix)

    return finalize_epoch_meter(meter)


def validate_epoch(
    model,
    fog_augmenter,
    val_loader,
    normalize,
    criterion_cls,
    criterion_reg,
    cfg,
    device,
    amp_enabled,
    desc,
):
    """
    执行单个验证 epoch，并使用固定随机种子稳定在线造雾结果。
    """
    if val_loader is None:
        return None

    model.eval()
    meter = init_epoch_meter()
    pbar = tqdm(val_loader, desc=desc)
    fork_devices = [torch.cuda.current_device()] if device == "cuda" else []

    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(1234)
        if device == "cuda":
            torch.cuda.manual_seed_all(1234)

        with torch.no_grad():
            for batch_index, (imgs, depths, det_targets) in enumerate(pbar, start=1):
                if cfg.MAX_VAL_BATCHES > 0 and batch_index > cfg.MAX_VAL_BATCHES:
                    break

                imgs = imgs.to(device)
                depths = depths.to(device)

                imgs_foggy, cls_labels, reg_labels = fog_augmenter(imgs, depths)
                imgs_norm = normalize(imgs_foggy)

                if device == "cuda" and amp_enabled:
                    with torch.amp.autocast("cuda"):
                        loss, loss_det, loss_cls, loss_reg = compute_multitask_losses(
                            model,
                            imgs_norm,
                            det_targets,
                            cls_labels,
                            reg_labels,
                            criterion_cls,
                            criterion_reg,
                            cfg,
                            device,
                        )
                else:
                    loss, loss_det, loss_cls, loss_reg = compute_multitask_losses(
                        model,
                        imgs_norm,
                        det_targets,
                        cls_labels,
                        reg_labels,
                        criterion_cls,
                        criterion_reg,
                        cfg,
                        device,
                    )

                loss_value = finite_scalar("val_loss", loss)
                det_value = finite_scalar("val_det_loss", loss_det)
                cls_value = finite_scalar("val_fog_cls_loss", loss_cls)
                reg_value = finite_scalar("val_fog_reg_loss", loss_reg)

                meter["loss"] += loss_value
                meter["det"] += det_value
                meter["fog_cls"] += cls_value
                meter["fog_reg"] += reg_value
                meter["batches"] += 1
                pbar.set_postfix(
                    {
                        "loss": f"{loss_value:.4f}",
                        "det": f"{det_value:.4f}",
                        "fog_cls": f"{cls_value:.4f}",
                        "fog_reg": f"{reg_value:.4f}",
                    }
                )

    return finalize_epoch_meter(meter)


def train():
    """
    训练主入口。

    训练流程分为两个阶段：
    1. FP32 常规训练；
    2. QAT 量化感知训练与最终 INT8 转换。

    整体执行顺序如下：
    - 初始化配置、随机种子、模型和增强器；
    - 构建训练集与 DataLoader；
    - 尝试从最近 checkpoint 续训；
    - 完成 FP32 主训练；
    - 保存最终 FP32 权重；
    - 切入 QAT；
    - 做少量校准并尝试转换成 INT8。
    """
    try:
        multiprocessing.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass

    cfg = Config()
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Using device: {device}")
    print(f"Training controls: {cfg.training_controls()}")

    run_dir = make_run_dir(cfg)
    metrics_jsonl = os.path.join(run_dir, "metrics.jsonl")
    summary_json = os.path.join(run_dir, "summary.json")
    config_json = os.path.join(run_dir, "config_snapshot.json")
    write_json(
        config_json,
        {
            "config_snapshot": build_cfg_snapshot(cfg),
            "paths": cfg.path_summary(),
            "training_controls": cfg.training_controls(),
            "device": device,
        },
    )
    print(f"Run directory: {run_dir}")

    print(f"Loading base model: {cfg.YOLO_BASE_MODEL}")
    model = UnifiedMultiTaskModel(
        cfg.YOLO_BASE_MODEL,
        cfg.NUM_FOG_CLASSES,
        num_det_classes=cfg.NUM_DET_CLASSES,
    ).to(device)
    fog_augmenter = FogAugmentation(cfg).to(device)

    if cfg.FREEZE_YOLO_FOR_FOG:
        for parameter in model.yolo.parameters():
            parameter.requires_grad = False
        print("YOLO detector parameters are frozen for fog-focused fine-tuning.")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    normalize, train_loader, val_loader, criterion_cls, criterion_reg = (
        build_train_components(cfg, device)
    )

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying freeze settings.")
    optimizer = optim.AdamW(trainable_parameters, lr=cfg.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    amp_enabled = device == "cuda" and not cfg.DISABLE_AMP
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    if amp_enabled:
        print("AMP is enabled.")
    elif device == "cuda":
        print("AMP is disabled.")

    latest_checkpoint = (
        cfg.RESUME_CHECKPOINT
        if cfg.RESUME_CHECKPOINT
        else find_latest_checkpoint(cfg.CHECKPOINT_DIR)
    )
    start_epoch = 0
    best_loss = float("inf")
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "unified_model_best.pt")

    # 若发现已有 checkpoint，则优先续训，避免重复浪费训练时间。
    if latest_checkpoint:
        print(f"Resume checkpoint: {latest_checkpoint}")
        try:
            if cfg.FREEZE_YOLO_FOR_FOG and not cfg.RESUME_MODEL_ONLY:
                print(
                    "FREEZE_YOLO_FOR_FOG is enabled, so optimizer/scheduler state "
                    "recovery is skipped automatically to avoid mismatched param groups."
                )
            resume_model_only = cfg.RESUME_MODEL_ONLY or cfg.FREEZE_YOLO_FOR_FOG
            start_epoch, _, best_loss = load_checkpoint(
                latest_checkpoint,
                model,
                None if resume_model_only else optimizer,
                None if resume_model_only else scheduler,
                None if resume_model_only else scaler,
            )
        except Exception as exc:
            print(
                f"Checkpoint is incompatible with the current model, restarting from scratch: {exc}"
            )
            start_epoch = 0
            best_loss = float("inf")
        if start_epoch >= cfg.EPOCHS:
            print(
                "FP32 training already reached the configured epoch count, skipping FP32 stage."
            )
            start_epoch = cfg.EPOCHS
    else:
        print("Starting training from scratch.")

    print(f"Starting FP32 training for {cfg.EPOCHS} epochs.")
    phase_summaries: list[dict] = []
    amp_recovery_events: list[dict] = []
    nonfinite_fail_streaks = {"fp32": 0, "qat": 0}
    for epoch in range(start_epoch, cfg.EPOCHS):
        train_metrics = train_epoch(
            model,
            fog_augmenter,
            train_loader,
            normalize,
            optimizer,
            scaler,
            criterion_cls,
            criterion_reg,
            cfg,
            device,
            desc=f"Epoch {epoch + 1}/{cfg.EPOCHS}",
        )
        print(f"Epoch {epoch + 1} average loss: {train_metrics['loss']:.4f}")
        (
            nonfinite_message,
            nonfinite_fail_streaks["fp32"],
            should_disable_amp,
        ) = evaluate_nonfinite_grad_health(
            train_metrics,
            cfg,
            phase="fp32",
            epoch=epoch + 1,
            consecutive_fail_epochs=nonfinite_fail_streaks["fp32"],
            amp_enabled=amp_enabled,
        )
        if nonfinite_message:
            print(nonfinite_message)
        if should_disable_amp:
            amp_enabled = False
            scaler = None
            recovery_event = {
                "phase": "fp32",
                "epoch": epoch + 1,
                "reason": "nonfinite_grad_fail_streak",
                "new_amp_enabled": amp_enabled,
            }
            amp_recovery_events.append(recovery_event)
            print(
                "AMP recovery: disabling AMP for subsequent training/validation epochs."
            )
        val_metrics = validate_epoch(
            model,
            fog_augmenter,
            val_loader,
            normalize,
            criterion_cls,
            criterion_reg,
            cfg,
            device,
            amp_enabled,
            desc=f"Val {epoch + 1}/{cfg.EPOCHS}",
        )
        if val_metrics is not None:
            print(f"Epoch {epoch + 1} validation loss: {val_metrics['loss']:.4f}")

        monitored_loss = (
            val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        )
        if monitored_loss < best_loss:
            best_loss = monitored_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model: {best_model_path}")

        epoch_record = {
            "phase": "fp32",
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "best_loss": best_loss,
            "nonfinite_fail_streak": nonfinite_fail_streaks["fp32"],
            "amp_enabled": amp_enabled,
        }
        append_jsonl(metrics_jsonl, epoch_record)
        phase_summaries.append(epoch_record)

        # 按固定间隔保存 checkpoint，并清理更旧的历史文件。
        if (epoch + 1) % cfg.CHECKPOINT_SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                cfg.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1:04d}.pt"
            )
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                train_metrics["loss"],
                best_loss,
                cfg,
                val_loss=val_metrics["loss"] if val_metrics is not None else None,
            )
            prune_old_checkpoints(cfg.CHECKPOINT_DIR, cfg.CHECKPOINT_KEEP_MAX)

        scheduler.step()

    # 保存 FP32 最终权重，作为常规部署或后续导出的基础版本。
    fp32_path = os.path.join(cfg.OUTPUT_DIR, "unified_model.pt")
    torch.save(model.state_dict(), fp32_path)
    print(f"Saved FP32 model: {fp32_path}")

    if cfg.SKIP_QAT or cfg.QAT_EPOCHS <= 0:
        print("Skipping QAT/INT8 stage.")
        write_json(
            summary_json,
            {
                "status": "completed_fp32_only",
                "run_dir": run_dir,
                "fp32_model": fp32_path,
                "best_fp32_model": best_model_path,
                "checkpoint_dir": cfg.CHECKPOINT_DIR,
                "best_loss": best_loss,
                "amp_recovery_events": amp_recovery_events,
                "nonfinite_fail_streaks": nonfinite_fail_streaks,
                "phase_summaries": phase_summaries,
            },
        )
        print(f"Run summary: {summary_json}")
        return

    print(f"Starting QAT training for {cfg.QAT_EPOCHS} epochs.")
    best_loss_qat = float("inf")
    qat_checkpoint_dir = os.path.join(cfg.CHECKPOINT_DIR, "qat")
    os.makedirs(qat_checkpoint_dir, exist_ok=True)

    # QAT 前先切回 CPU 进行融合与量化配置，再迁回目标设备。
    # PyTorch 的某些量化准备步骤在 CPU 上更稳妥。
    model.to("cpu")
    model.fuse_model()

    backend = (
        "fbgemm"
        if torch.backends.quantized.supported_engines.count("fbgemm")
        else "qnnpack"
    )
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(model, inplace=True)
    model.to(device)

    optimizer_qat = optim.AdamW(model.parameters(), lr=cfg.QAT_LR)
    scheduler_qat = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_qat, T_max=cfg.QAT_EPOCHS
    )
    # QAT 不与 AMP 混用。
    # fake quant / observer 通常要求 float32 路径，和 autocast 的半精度前向容易产生 dtype 冲突。
    scaler_qat = None
    if device == "cuda":
        print("AMP is disabled during QAT to avoid fake-quant dtype mismatch.")

    qat_latest_checkpoint = find_latest_checkpoint(qat_checkpoint_dir)
    qat_start_epoch = 0
    if qat_latest_checkpoint:
        try:
            qat_start_epoch, _, best_loss_qat = load_checkpoint(
                qat_latest_checkpoint,
                model,
                optimizer_qat,
                scheduler_qat,
                scaler_qat,
            )
            if qat_start_epoch >= cfg.QAT_EPOCHS:
                print("QAT already reached the configured epoch count, skipping QAT.")
                qat_start_epoch = cfg.QAT_EPOCHS
        except Exception as exc:
            print(
                f"QAT checkpoint is incompatible with the current model, restarting QAT: {exc}"
            )
            qat_start_epoch = 0
            best_loss_qat = float("inf")
    else:
        print("Starting QAT from scratch.")

    for epoch in range(qat_start_epoch, cfg.QAT_EPOCHS):
        train_metrics = train_epoch(
            model,
            fog_augmenter,
            train_loader,
            normalize,
            optimizer_qat,
            scaler_qat,
            criterion_cls,
            criterion_reg,
            cfg,
            device,
            desc=f"QAT Epoch {epoch + 1}/{cfg.QAT_EPOCHS}",
        )
        print(f"QAT epoch {epoch + 1} average loss: {train_metrics['loss']:.4f}")
        (
            nonfinite_message,
            nonfinite_fail_streaks["qat"],
            _,
        ) = evaluate_nonfinite_grad_health(
            train_metrics,
            cfg,
            phase="qat",
            epoch=epoch + 1,
            consecutive_fail_epochs=nonfinite_fail_streaks["qat"],
            amp_enabled=False,
        )
        if nonfinite_message:
            print(nonfinite_message)
        val_metrics = validate_epoch(
            model,
            fog_augmenter,
            val_loader,
            normalize,
            criterion_cls,
            criterion_reg,
            cfg,
            device,
            False,
            desc=f"QAT Val {epoch + 1}/{cfg.QAT_EPOCHS}",
        )
        if val_metrics is not None:
            print(f"QAT epoch {epoch + 1} validation loss: {val_metrics['loss']:.4f}")

        monitored_loss = (
            val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        )
        if monitored_loss < best_loss_qat:
            best_loss_qat = monitored_loss

        epoch_record = {
            "phase": "qat",
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer_qat.param_groups[0]["lr"],
            "best_loss": best_loss_qat,
            "nonfinite_fail_streak": nonfinite_fail_streaks["qat"],
            "amp_enabled": False,
        }
        append_jsonl(metrics_jsonl, epoch_record)
        phase_summaries.append(epoch_record)

        qat_checkpoint_path = os.path.join(
            qat_checkpoint_dir, f"checkpoint_epoch_{epoch + 1:04d}.pt"
        )
        save_checkpoint(
            qat_checkpoint_path,
            epoch,
            model,
            optimizer_qat,
            scheduler_qat,
            scaler_qat,
            train_metrics["loss"],
            best_loss_qat,
            cfg,
            val_loss=val_metrics["loss"] if val_metrics is not None else None,
        )
        prune_old_checkpoints(qat_checkpoint_dir, cfg.CHECKPOINT_KEEP_MAX)

        # QAT 后半程逐步冻结 observer 和 fake quant，稳定量化参数。
        if epoch > 2:
            model.apply(torch.ao.quantization.disable_observer)
        if epoch > 3:
            model.apply(torch.ao.quantization.disable_fake_quant)

        scheduler_qat.step()

    print("Preparing INT8 conversion.")
    model.to(device)
    model.apply(torch.ao.quantization.enable_observer)

    # 进行少量校准前向，尽量让 observer 拿到更合理的激活范围。
    # 这里只取少量 batch 做快速校准，目的不是重新训练，而是补全量化统计信息。
    print("Running QAT calibration.")
    model.eval()
    with torch.no_grad():
        for i, (imgs, depths, _) in enumerate(train_loader):
            if i >= 5:
                break
            imgs = imgs.to(device)
            depths = depths.to(device)
            imgs_foggy, _, _ = fog_augmenter(imgs, depths)
            imgs_norm = normalize(imgs_foggy)
            model(imgs_norm)

    model.to("cpu")
    model.eval()

    # 若 observer 中仍存在非法极值，则用 Identity 替换，避免 convert 直接失败。
    repaired_observers = 0
    for _, module in model.named_modules():
        if hasattr(module, "activation_post_process"):
            observer = module.activation_post_process
            if hasattr(observer, "min_val") and hasattr(observer, "max_val"):
                if (observer.min_val == float("inf")).any() or (
                    observer.max_val == float("-inf")
                ).any():
                    module.activation_post_process = nn.Identity()
                    repaired_observers += 1
    if repaired_observers > 0:
        print(f"Repaired {repaired_observers} invalid observers before conversion.")

    try:
        qat_model_int8 = torch.ao.quantization.convert(model, inplace=False)
        qat_path = os.path.join(cfg.OUTPUT_DIR, "unified_model_qat_int8.pt")
        torch.save(qat_model_int8.state_dict(), qat_path)
        print(f"Saved INT8 model: {qat_path}")
    except Exception as exc:
        print(f"INT8 conversion failed: {exc}")
        return

    print("Training finished.")
    print(f"FP32 model: {fp32_path}")
    print(f"INT8 model: {qat_path}")
    print(f"Best FP32 model: {best_model_path}")
    print(f"Checkpoint dir: {cfg.CHECKPOINT_DIR}")
    write_json(
        summary_json,
        {
            "status": "completed_with_qat",
            "run_dir": run_dir,
            "fp32_model": fp32_path,
            "int8_model": qat_path,
            "best_fp32_model": best_model_path,
            "checkpoint_dir": cfg.CHECKPOINT_DIR,
            "best_loss": best_loss,
            "best_qat_loss": best_loss_qat,
            "amp_recovery_events": amp_recovery_events,
            "nonfinite_fail_streaks": nonfinite_fail_streaks,
            "phase_summaries": phase_summaries,
        },
    )
    print(f"Run summary: {summary_json}")


if __name__ == "__main__":
    train()
