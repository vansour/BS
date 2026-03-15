#!/usr/bin/env python3
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
"""

import multiprocessing
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
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
    """
    return {
        "yolo_base_model": cfg.YOLO_BASE_MODEL,
        "num_det_classes": cfg.NUM_DET_CLASSES,
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
        "beta_min": cfg.BETA_MIN,
        "beta_max": cfg.BETA_MAX,
        "a_min": cfg.A_MIN,
        "a_max": cfg.A_MAX,
    }


def multitask_collate_fn(batch):
    """
    组装多任务训练所需的 batch。

    检测框数量在不同图像之间并不固定，因此这里把检测标签整理为
    Ultralytics 检测损失需要的扁平结构：
    - `batch_idx`
    - `cls`
    - `bboxes`
    """
    imgs, depths, det_cls_list, det_box_list = zip(*batch)

    imgs = torch.stack(imgs, 0)
    depths = torch.stack(depths, 0)

    batch_idx_all = []
    cls_all = []
    bboxes_all = []

    for sample_idx, (det_cls, det_boxes) in enumerate(zip(det_cls_list, det_box_list)):
        if det_boxes.numel() == 0:
            continue
        batch_idx_all.append(torch.full((det_boxes.shape[0],), sample_idx, dtype=torch.int64))
        cls_all.append(det_cls)
        bboxes_all.append(det_boxes)

    if batch_idx_all:
        det_targets = {
            "batch_idx": torch.cat(batch_idx_all, 0),
            "cls": torch.cat(cls_all, 0),
            "bboxes": torch.cat(bboxes_all, 0),
        }
    else:
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

    checkpoint_files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    for stale_path in checkpoint_files[keep_max:]:
        try:
            os.remove(stale_path)
            print(f"Removed old checkpoint: {stale_path}")
        except OSError as exc:
            print(f"Failed to remove old checkpoint {stale_path}: {exc}")


def save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, scaler, train_loss, best_loss, cfg):
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
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "best_loss": best_loss,
        "cfg_snapshot": build_cfg_snapshot(cfg),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
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
    print(f"Best loss: {best_loss:.4f}")
    return start_epoch, train_loss, best_loss


def build_train_components(cfg: Config, device: str):
    """
    构建训练所需的数据与损失组件。

    Args:
        cfg: 当前配置对象。
        device: 当前运行设备。

    Returns:
        tuple: 归一化变换、训练 DataLoader、分类损失、回归损失。
    """
    if cfg.USE_IMAGENET_NORMALIZE:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        def normalize(batch):
            return (batch - mean.to(batch.device, batch.dtype)) / std.to(batch.device, batch.dtype)
    else:
        normalize = nn.Identity()
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    if not os.path.isdir(cfg.XML_DIR):
        raise RuntimeError(f"Detection annotation directory was not found: {cfg.XML_DIR}")

    train_ds = MultiTaskDataset(
        cfg.RAW_DATA_DIR,
        cfg.DEPTH_CACHE_DIR,
        xml_dir=cfg.XML_DIR,
        transform=transform,
        is_train=True,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"Training dataset is empty: {cfg.RAW_DATA_DIR}")

    print(f"Training samples: {len(train_ds)}")

    # 训练正式开始前先确保深度缓存齐备，避免 batch 过程中频繁缺缓存。
    precompute_depths(train_ds, device)

    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        collate_fn=multitask_collate_fn,
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    return normalize, train_loader, criterion_cls, criterion_reg


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
    """
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=desc)

    for imgs, depths, det_targets in pbar:
        imgs = imgs.to(device)
        depths = depths.to(device)

        # 在线生成雾天样本，得到分类标签和回归标签。
        with torch.no_grad():
            imgs_foggy, cls_labels, reg_labels = fog_augmenter(imgs, depths)
            imgs_norm = normalize(imgs_foggy)

        optimizer.zero_grad()

        # 在 CUDA 环境下优先启用 AMP，以降低显存压力并提升吞吐。
        if scaler is not None:
            with torch.cuda.amp.autocast():
                det_preds, pred_cls, pred_reg = model(imgs_norm)
                det_batch = {
                    "img": imgs_norm,
                    "batch_idx": det_targets["batch_idx"].to(device),
                    "cls": det_targets["cls"].to(device),
                    "bboxes": det_targets["bboxes"].to(device),
                }
                det_loss_components, _ = model.yolo.loss(det_batch, preds=det_preds)
                loss_det = det_loss_components.sum() / max(imgs_norm.size(0), 1)
                loss_cls = criterion_cls(pred_cls, cls_labels)
                loss_reg = criterion_reg(pred_reg * cfg.BETA_MAX, reg_labels)
                loss = (
                    cfg.DET_LOSS_WEIGHT * loss_det
                    + cfg.FOG_CLS_LOSS_WEIGHT * loss_cls
                    + cfg.FOG_REG_LOSS_WEIGHT * loss_reg
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            det_preds, pred_cls, pred_reg = model(imgs_norm)
            det_batch = {
                "img": imgs_norm,
                "batch_idx": det_targets["batch_idx"].to(device),
                "cls": det_targets["cls"].to(device),
                "bboxes": det_targets["bboxes"].to(device),
            }
            det_loss_components, _ = model.yolo.loss(det_batch, preds=det_preds)
            loss_det = det_loss_components.sum() / max(imgs_norm.size(0), 1)
            loss_cls = criterion_cls(pred_cls, cls_labels)
            loss_reg = criterion_reg(pred_reg * cfg.BETA_MAX, reg_labels)
            loss = (
                cfg.DET_LOSS_WEIGHT * loss_det
                + cfg.FOG_CLS_LOSS_WEIGHT * loss_cls
                + cfg.FOG_REG_LOSS_WEIGHT * loss_reg
            )
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "det": f"{loss_det.item():.4f}",
            "fog_cls": f"{loss_cls.item():.4f}",
            "fog_reg": f"{loss_reg.item():.4f}",
        })

    return total_loss / max(len(train_loader), 1)


def train():
    """
    训练主入口。

    训练流程分为两个阶段：
    1. FP32 常规训练；
    2. QAT 量化感知训练与最终 INT8 转换。
    """
    try:
        multiprocessing.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass

    cfg = Config()
    set_seed(42)
    device = cfg.DEVICE
    print(f"Using device: {device}")

    print(f"Loading base model: {cfg.YOLO_BASE_MODEL}")
    model = UnifiedMultiTaskModel(
        cfg.YOLO_BASE_MODEL,
        cfg.NUM_FOG_CLASSES,
        num_det_classes=cfg.NUM_DET_CLASSES,
    ).to(device)
    fog_augmenter = FogAugmentation(cfg).to(device)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    normalize, train_loader, criterion_cls, criterion_reg = build_train_components(cfg, device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    if scaler is not None:
        print("AMP is enabled.")

    latest_checkpoint = find_latest_checkpoint(cfg.CHECKPOINT_DIR)
    start_epoch = 0
    best_loss = float("inf")
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "unified_model_best.pt")

    # 若发现已有 checkpoint，则优先续训，避免重复浪费训练时间。
    if latest_checkpoint:
        try:
            start_epoch, _, best_loss = load_checkpoint(
                latest_checkpoint,
                model,
                optimizer,
                scheduler,
                scaler,
            )
        except Exception as exc:
            print(f"Checkpoint is incompatible with the current model, restarting from scratch: {exc}")
            start_epoch = 0
            best_loss = float("inf")
        if start_epoch >= cfg.EPOCHS:
            print("Training already reached the configured epoch count, restarting from epoch 0.")
            start_epoch = 0
            best_loss = float("inf")
    else:
        print("Starting training from scratch.")

    print(f"Starting FP32 training for {cfg.EPOCHS} epochs.")
    for epoch in range(start_epoch, cfg.EPOCHS):
        avg_loss = train_epoch(
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
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # 按固定间隔保存 checkpoint，并清理更旧的历史文件。
        if (epoch + 1) % cfg.CHECKPOINT_SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1:04d}.pt")
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                avg_loss,
                best_loss,
                cfg,
            )
            prune_old_checkpoints(cfg.CHECKPOINT_DIR, cfg.CHECKPOINT_KEEP_MAX)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model: {best_model_path}")

        scheduler.step()

    # 保存 FP32 最终权重，作为常规部署或后续导出的基础版本。
    fp32_path = os.path.join(cfg.OUTPUT_DIR, "unified_model.pt")
    torch.save(model.state_dict(), fp32_path)
    print(f"Saved FP32 model: {fp32_path}")

    print(f"Starting QAT training for {cfg.QAT_EPOCHS} epochs.")
    best_loss_qat = float("inf")
    qat_checkpoint_dir = os.path.join(cfg.CHECKPOINT_DIR, "qat")
    os.makedirs(qat_checkpoint_dir, exist_ok=True)

    # QAT 前先切回 CPU 进行融合与量化配置，再迁回目标设备。
    model.to("cpu")
    model.fuse_model()

    backend = "fbgemm" if torch.backends.quantized.supported_engines.count("fbgemm") else "qnnpack"
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(model, inplace=True)
    model.to(device)

    optimizer_qat = optim.AdamW(model.parameters(), lr=cfg.QAT_LR)
    scheduler_qat = optim.lr_scheduler.CosineAnnealingLR(optimizer_qat, T_max=cfg.QAT_EPOCHS)
    scaler_qat = torch.cuda.amp.GradScaler() if device == "cuda" else None

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
            print(f"QAT checkpoint is incompatible with the current model, restarting QAT: {exc}")
            qat_start_epoch = 0
            best_loss_qat = float("inf")
    else:
        print("Starting QAT from scratch.")

    for epoch in range(qat_start_epoch, cfg.QAT_EPOCHS):
        avg_loss = train_epoch(
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
        print(f"QAT epoch {epoch + 1} average loss: {avg_loss:.4f}")

        qat_checkpoint_path = os.path.join(qat_checkpoint_dir, f"checkpoint_epoch_{epoch + 1:04d}.pt")
        save_checkpoint(
            qat_checkpoint_path,
            epoch,
            model,
            optimizer_qat,
            scheduler_qat,
            scaler_qat,
            avg_loss,
            best_loss_qat,
            cfg,
        )
        prune_old_checkpoints(qat_checkpoint_dir, cfg.CHECKPOINT_KEEP_MAX)

        if avg_loss < best_loss_qat:
            best_loss_qat = avg_loss

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
                if (observer.min_val == float("inf")).any() or (observer.max_val == float("-inf")).any():
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


if __name__ == "__main__":
    train()


