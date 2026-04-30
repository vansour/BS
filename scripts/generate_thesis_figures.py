#!/usr/bin/env python3
"""
Generate thesis-ready figures and tables from current project artifacts.

Outputs are written to:
    outputs/Thesis_Final_Figures/

The script focuses on figures that can be reproduced from the current
workspace state without inventing unavailable experimental labels.
"""

from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data.dataset import MultiTaskDataset
from src.inference import HighwayFogSystem
from src.model.fog_augmentation import FogAugmentation
from src.model.unified_model import UnifiedMultiTaskModel
from src.utils import load_model_weights


OUTPUT_DIR = ROOT / "outputs" / "Thesis_Final_Figures"
TARGET_VIDEO = ROOT / "gettyimages-1353950094-640_adpp.mp4"
FINAL_VIDEO = (
    ROOT / "outputs" / "hybrid_infer" / "gettyimages-1353950094-640_adpp_fogfocus_final.mp4"
)
FOGFOCUS_WEIGHTS = (
    ROOT / "outputs" / "Fog_Detection_Project_fogfocus" / "unified_model_best.pt"
)
DEFAULT_ROUTE_EVAL = ROOT / "outputs" / "Route_Eval_formal" / "route_eval_summary.json"
VIDEADAPT_ROUTE_EVAL = (
    ROOT / "outputs" / "Route_Eval_videoadapt_formal" / "route_eval_summary.json"
)
FOGFOCUS_ROUTE_EVAL = (
    ROOT / "outputs" / "Route_Eval_fogfocus_final" / "route_eval_summary.json"
)
FORMAL5_SUMMARY = (
    ROOT
    / "outputs"
    / "Fog_Detection_Project_formal5"
    / "runs"
    / "smoke_20260411_152317"
    / "summary.json"
)
FOGFOCUS_SUMMARY = (
    ROOT
    / "outputs"
    / "Fog_Detection_Project_fogfocus"
    / "runs"
    / "smoke_20260411_145220"
    / "summary.json"
)
DATA_AUDIT_JSON = ROOT / "outputs" / "Data_Audit" / "dataset_audit_report.json"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_matplotlib_font():
    preferred = [
        "AR PL UMing CN",
        "AR PL UKai CN",
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "Source Han Sans SC",
        "Source Han Serif SC",
        "WenQuanYi Zen Hei",
        "SimHei",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(fig, path: Path, dpi: int = 220):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, xy, w, h, text, fc="#F4F7FB", ec="#2F4F6F", fontsize=11):
    rect = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#142433",
        wrap=True,
    )


def draw_arrow(ax, start, end, color="#556B7D"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.6, color=color),
    )


def fig_system_overview(path: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.06, 0.72), 0.2, 0.12, "UA-DETRAC\n晴天交通图像")
    draw_box(ax, (0.06, 0.48), 0.2, 0.12, "MiDaS 深度估计\n离线缓存 .npy")
    draw_box(ax, (0.06, 0.24), 0.2, 0.12, "XML 车辆标注\n序列级划分")
    draw_box(ax, (0.35, 0.42), 0.22, 0.16, "MultiTaskDataset\n图像 + 深度 + 检测框")
    draw_box(ax, (0.66, 0.42), 0.22, 0.16, "FogAugmentation\nclear / uniform / patchy")

    draw_box(ax, (0.35, 0.14), 0.22, 0.16, "UnifiedMultiTaskModel\nYOLO 骨干 + 分类头 + beta头")
    draw_box(ax, (0.66, 0.14), 0.22, 0.16, "视频推理系统\nNMS + EMA beta + 动态阈值")

    ax.text(0.77, 0.80, "输出", fontsize=13, weight="bold")
    draw_box(ax, (0.82, 0.70), 0.14, 0.08, "车辆检测")
    draw_box(ax, (0.82, 0.58), 0.14, 0.08, "雾类型分类")
    draw_box(ax, (0.82, 0.46), 0.14, 0.08, "beta 回归")

    draw_arrow(ax, (0.26, 0.78), (0.35, 0.52))
    draw_arrow(ax, (0.26, 0.54), (0.35, 0.50))
    draw_arrow(ax, (0.26, 0.30), (0.35, 0.48))
    draw_arrow(ax, (0.57, 0.50), (0.66, 0.50))
    draw_arrow(ax, (0.57, 0.22), (0.66, 0.22))
    draw_arrow(ax, (0.77, 0.42), (0.82, 0.74))
    draw_arrow(ax, (0.77, 0.42), (0.82, 0.62))
    draw_arrow(ax, (0.77, 0.42), (0.82, 0.50))
    draw_arrow(ax, (0.77, 0.22), (0.88, 0.22))
    draw_arrow(ax, (0.77, 0.42), (0.46, 0.30))

    ax.text(0.5, 0.95, "系统总体框架图", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(
        0.5,
        0.03,
        "数据层通过深度缓存与 XML 标注构建训练样本，在线造雾为分类与回归任务生成监督，统一模型完成三任务联合建模。",
        ha="center",
        va="center",
        fontsize=10,
        color="#32495A",
    )
    save_figure(fig, path)


def load_sample_frame_and_depth(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    dataset = MultiTaskDataset(
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
    sample_index = min(120, max(0, len(dataset) - 1))
    image_tensor, depth_tensor, _, _ = dataset[sample_index]
    image = (image_tensor.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    depth = depth_tensor.squeeze(0).numpy()
    return image, depth


def synthesize_fog_modes(cfg: Config, image_rgb: np.ndarray, depth: np.ndarray) -> dict[str, np.ndarray]:
    image = image_rgb.astype(np.float32) / 255.0
    depth = depth.astype(np.float32)
    depth = np.clip(depth, 0.0, 1.0)

    beta_uniform = 0.055
    beta_patchy = 0.075
    atmosphere = 0.88

    transmission_uniform = np.exp(-beta_uniform * depth * cfg.UNIFORM_DEPTH_SCALE)
    transmission_uniform = np.clip(transmission_uniform, 0.05, 0.95)

    rng = np.random.default_rng(42)
    noise = rng.random((16, 16), dtype=np.float32)
    noise = cv2.resize(noise, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
    effective_depth = depth * (cfg.PATCHY_DEPTH_BASE + noise * cfg.PATCHY_DEPTH_NOISE_SCALE)
    transmission_patchy = np.exp(-beta_patchy * effective_depth)
    transmission_patchy = np.clip(transmission_patchy, 0.05, 0.95)

    clear = image
    uniform = image * transmission_uniform[..., None] + atmosphere * (1 - transmission_uniform[..., None])
    patchy = image * transmission_patchy[..., None] + atmosphere * (1 - transmission_patchy[..., None])

    outputs = {
        "clear": (clear * 255).astype(np.uint8),
        "uniform": (uniform.clip(0, 1) * 255).astype(np.uint8),
        "patchy": (patchy.clip(0, 1) * 255).astype(np.uint8),
    }
    return outputs


def fig_fog_augmentation_examples(cfg: Config, path: Path):
    image_rgb, depth = load_sample_frame_and_depth(cfg)
    fog_outputs = synthesize_fog_modes(cfg, image_rgb, depth)

    depth_norm = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
    depth_vis = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    panels = [
        ("清晰原图", image_rgb),
        ("深度图（伪彩）", depth_vis),
        ("clear 样本", fog_outputs["clear"]),
        ("uniform 样本", fog_outputs["uniform"]),
        ("patchy 样本", fog_outputs["patchy"]),
        ("大气散射模型", None),
    ]

    for ax, (title, panel) in zip(axes, panels):
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        if panel is not None:
            ax.imshow(panel)
        else:
            ax.text(
                0.5,
                0.62,
                r"$I(x)=J(x)\cdot t(x)+A\cdot (1-t(x))$",
                ha="center",
                va="center",
                fontsize=18,
            )
            ax.text(
                0.5,
                0.42,
                r"$t(x)=e^{-\beta d(x)}$",
                ha="center",
                va="center",
                fontsize=18,
            )
            ax.text(
                0.5,
                0.20,
                "patchy 模式通过低频噪声调制有效深度，\n从而模拟局地浓淡不均的团雾。",
                ha="center",
                va="center",
                fontsize=11,
            )

    fig.suptitle("在线造雾原理图与三类天气样例图", fontsize=18, weight="bold")
    save_figure(fig, path)


def fig_model_structure(path: Path):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.05, 0.40), 0.12, 0.18, "输入图像\n3×512×512", fc="#F7F4EA")
    draw_box(ax, (0.23, 0.40), 0.18, 0.18, "YOLO Backbone\n多尺度特征提取", fc="#EAF4FF")
    draw_box(ax, (0.47, 0.40), 0.18, 0.18, "YOLO Neck / SPPF\n高层共享特征", fc="#EAF4FF")

    draw_box(ax, (0.74, 0.64), 0.18, 0.14, "检测分支\nYOLO Detect Head", fc="#E9F7EF")
    draw_box(ax, (0.74, 0.40), 0.18, 0.14, "雾分类头\nGAP + FC + Softmax", fc="#FFF4E5")
    draw_box(ax, (0.74, 0.16), 0.18, 0.14, "beta 回归头\nGAP + FC + Sigmoid", fc="#FDEDEC")

    draw_arrow(ax, (0.17, 0.49), (0.23, 0.49))
    draw_arrow(ax, (0.41, 0.49), (0.47, 0.49))
    draw_arrow(ax, (0.65, 0.53), (0.74, 0.71))
    draw_arrow(ax, (0.65, 0.49), (0.74, 0.47))
    draw_arrow(ax, (0.65, 0.45), (0.74, 0.23))

    ax.text(0.83, 0.84, "输出", fontsize=13, weight="bold", ha="center")
    ax.text(0.83, 0.60, "vehicle 检测框", fontsize=11, ha="center")
    ax.text(0.83, 0.36, "clear / uniform / patchy", fontsize=11, ha="center")
    ax.text(0.83, 0.12, "归一化 beta", fontsize=11, ha="center")

    ax.text(0.5, 0.94, "统一多任务模型结构图", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(
        0.5,
        0.05,
        "统一模型复用 YOLO 主干特征，并在高层共享特征上挂接雾分类头与 beta 回归头。",
        ha="center",
        va="center",
        fontsize=10,
        color="#32495A",
    )
    save_figure(fig, path)


def fig_dataset_statistics(path: Path):
    report = json.loads(DATA_AUDIT_JSON.read_text(encoding="utf-8"))
    raw = report["raw"]
    train = report["splits"]["train"]
    val = report["splits"]["val"]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.bar(
        ["训练序列", "测试序列", "XML 文件"],
        [raw["train_sequence_count"], raw["test_sequence_count"], raw["xml_file_count"]],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    ax1.set_title("原始数据规模")
    ax1.set_ylabel("数量")

    ax2.bar(
        ["训练样本", "验证样本", "训练检测框", "验证检测框"],
        [
            train["sample_count"],
            val["sample_count"],
            train["boxes"]["total_boxes"],
            val["boxes"]["total_boxes"],
        ],
        color=["#4C78A8", "#72B7B2", "#E45756", "#F2CF5B"],
    )
    ax2.set_title("训练/验证样本与检测框规模")
    ax2.tick_params(axis="x", rotation=15)

    ax3.axis("off")
    table_lines = [
        ["划分", "序列数", "样本数", "有标注样本", "检测框总数", "深度覆盖率"],
        [
            "训练集",
            str(train["sequence_count"]),
            str(train["sample_count"]),
            str(train["labels"]["labeled_sample_count"]),
            str(train["boxes"]["total_boxes"]),
            f"{train['depth_cache']['coverage_ratio']*100:.1f}%",
        ],
        [
            "验证集",
            str(val["sequence_count"]),
            str(val["sample_count"]),
            str(val["labels"]["labeled_sample_count"]),
            str(val["boxes"]["total_boxes"]),
            f"{val['depth_cache']['coverage_ratio']*100:.1f}%",
        ],
    ]
    table = ax3.table(cellText=table_lines[1:], colLabels=table_lines[0], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    fig.suptitle("数据集构成统计图与统计表", fontsize=18, weight="bold")
    save_figure(fig, path)


def fig_training_validation_loss(path: Path):
    summary = json.loads(FORMAL5_SUMMARY.read_text(encoding="utf-8"))
    epochs = []
    train_loss = []
    val_loss = []
    det_loss = []
    fog_cls_loss = []
    fog_reg_loss = []
    for item in summary["phase_summaries"]:
        if item.get("phase") != "fp32":
            continue
        epochs.append(item["epoch"])
        train_loss.append(item["train"]["loss"])
        val_loss.append(item["val"]["loss"] if item["val"] else math.nan)
        det_loss.append(item["train"]["det"])
        fog_cls_loss.append(item["train"]["fog_cls"])
        fog_reg_loss.append(item["train"]["fog_reg"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(epochs, train_loss, marker="o", label="训练总损失")
    axes[0].plot(epochs, val_loss, marker="s", label="验证总损失")
    axes[0].set_title("总损失曲线")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, det_loss, marker="o", label="检测损失")
    axes[1].plot(epochs, fog_cls_loss, marker="s", label="雾分类损失")
    axes[1].plot(epochs, fog_reg_loss, marker="^", label="beta 回归损失")
    axes[1].set_title("训练阶段子损失曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("训练与验证损失曲线图", fontsize=18, weight="bold")
    save_figure(fig, path)


def fig_fog_confusion_matrix(cfg: Config, path: Path, sample_limit: int = 240, batch_size: int = 16):
    dataset = MultiTaskDataset(
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
    device = cfg.DEVICE
    model = UnifiedMultiTaskModel(
        cfg.YOLO_BASE_MODEL,
        cfg.NUM_FOG_CLASSES,
        num_det_classes=cfg.NUM_DET_CLASSES,
        img_size=cfg.IMG_SIZE,
    ).to(device)
    load_model_weights(model, str(FOGFOCUS_WEIGHTS), map_location=device)
    model.eval()
    fog_aug = FogAugmentation(cfg).to(device).eval()

    confusion = np.zeros((cfg.NUM_FOG_CLASSES, cfg.NUM_FOG_CLASSES), dtype=np.int32)
    limit = min(sample_limit, len(dataset))
    with torch.no_grad():
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
            torch.manual_seed(2026)
            if device == "cuda":
                torch.cuda.manual_seed_all(2026)
            for start in range(0, limit, batch_size):
                items = [dataset[idx] for idx in range(start, min(start + batch_size, limit))]
                imgs = torch.stack([item[0] for item in items]).to(device)
                depths = torch.stack([item[1] for item in items]).to(device)
                foggy, cls_labels, _ = fog_aug(imgs, depths)
                _, logits, _ = model(foggy)
                preds = logits.argmax(dim=1)
                for true_label, pred_label in zip(cls_labels.cpu().tolist(), preds.cpu().tolist()):
                    confusion[int(true_label), int(pred_label)] += 1

    row_sums = confusion.sum(axis=1, keepdims=True).clip(min=1)
    normalized = confusion / row_sums

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = ["clear", "uniform", "patchy"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("雾分类混淆矩阵（合成验证样本）")

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{confusion[i, j]}\n{normalized[i, j]*100:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if normalized[i, j] > 0.45 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("雾分类混淆矩阵", fontsize=18, weight="bold")
    save_figure(fig, path)


def load_sampled_frames(video_path: Path, stride: int = 10) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    frame_index = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % stride == 0:
            frames.append(frame)
    cap.release()
    return frames


def collect_beta_values(weight_path: Path, frames: list[np.ndarray], cfg: Config) -> list[float]:
    system = HighwayFogSystem(str(weight_path), video_source=0, cfg=cfg)
    values = []
    for frame in frames:
        _, beta, _ = system.predict(frame)
        values.append(float(beta))
    return values


def fig_beta_distribution_by_epoch(cfg: Config, path: Path):
    formal_dir = ROOT / "outputs" / "Fog_Detection_Project_formal5" / "checkpoints"
    checkpoints = sorted(formal_dir.glob("checkpoint_epoch_*.pt"))
    selected = checkpoints[-5:]
    frames = load_sampled_frames(TARGET_VIDEO, stride=10)
    beta_series = []
    labels = []
    for ckpt in selected:
        beta_series.append(collect_beta_values(ckpt, frames, cfg))
        labels.append(ckpt.stem.replace("checkpoint_epoch_", "Epoch "))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(beta_series, labels=labels, patch_artist=True)
    ax.set_ylabel("beta")
    ax.set_title("不同训练轮次下 beta 分布图")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def frame_from_video(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def fig_inference_visualization(path: Path):
    frame = frame_from_video(FINAL_VIDEO, 250)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(frame)
    ax.axis("off")
    ax.set_title("推理结果可视化图（最终演示视频帧）", fontsize=16, weight="bold")
    save_figure(fig, path)


def fig_fog_map_example(path: Path):
    frame = frame_from_video(FINAL_VIDEO, 250)
    h, w = frame.shape[:2]
    crop = frame[int(h * 0.52) : h - 10, int(w * 0.63) : w - 8]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.imshow(crop)
    ax.axis("off")
    ax.set_title("混合推理右下角“雾浓度分布图”示例", fontsize=14, weight="bold")
    save_figure(fig, path)


def route_eval_video(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))["videos"][0]


def table_method_comparison(path_png: Path, path_csv: Path):
    default_video = route_eval_video(DEFAULT_ROUTE_EVAL)
    videoadapt_video = route_eval_video(VIDEADAPT_ROUTE_EVAL)
    fogfocus_video = route_eval_video(FOGFOCUS_ROUTE_EVAL)

    rows = [
        [
            "默认统一模型",
            "默认权重",
            f"{default_video['unified']['mean_count_per_frame']:.3f}",
            f"{default_video['unified']['frames_with_detections_ratio']:.3f}",
            f"{default_video['fog']['beta']['mean']:.5f}",
            f"{default_video['fog']['probs']['mean_probs'][0]:.3f}",
            "统一检测 + 雾估计",
        ],
        [
            "视频适配统一模型",
            "videoadapt",
            f"{videoadapt_video['unified']['mean_count_per_frame']:.3f}",
            f"{videoadapt_video['unified']['frames_with_detections_ratio']:.3f}",
            f"{videoadapt_video['fog']['beta']['mean']:.5f}",
            f"{videoadapt_video['fog']['probs']['mean_probs'][0]:.3f}",
            "统一检测 + 雾估计",
        ],
        [
            "偏雾天统一模型",
            "fogfocus",
            f"{fogfocus_video['unified']['mean_count_per_frame']:.3f}",
            f"{fogfocus_video['unified']['frames_with_detections_ratio']:.3f}",
            f"{fogfocus_video['fog']['beta']['mean']:.5f}",
            f"{fogfocus_video['fog']['probs']['mean_probs'][0]:.3f}",
            "统一检测 + 雾估计",
        ],
        [
            "最终演示混合方案",
            "fogfocus + yolo11n",
            f"{fogfocus_video['hybrid']['mean_count_per_frame']:.3f}",
            f"{fogfocus_video['hybrid']['frames_with_detections_ratio']:.3f}",
            f"{fogfocus_video['fog']['beta']['mean']:.5f}",
            f"{fogfocus_video['fog']['probs']['mean_probs'][0]:.3f}",
            "独立检测 + 雾估计",
        ],
    ]
    headers = [
        "方案",
        "权重/路线",
        "平均检测数/帧",
        "非零检测帧占比",
        "beta均值",
        "CLEAR均值概率",
        "说明",
    ]

    with path_csv.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(row) + "\n")

    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.6)
    fig.suptitle("当前可用方案对比表", fontsize=18, weight="bold")
    save_figure(fig, path_png)


def write_summary(index_path: Path, figures: list[tuple[str, str, str]]):
    lines = [
        "# 终稿阶段建议补充图表",
        "",
        "本目录中的图表由当前工作区的真实训练产物、数据审计结果、路线评估结果和最终演示视频自动生成。",
        "",
        "| 文件 | 图表名称 | 说明 |",
        "|---|---|---|",
    ]
    for filename, title, desc in figures:
        lines.append(f"| `{filename}` | {title} | {desc} |")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ensure_dir(OUTPUT_DIR)
    configure_matplotlib_font()
    cfg = Config()

    figures: list[tuple[str, str, str]] = []

    items = [
        ("fig_01_system_overview.png", "系统总体框架图", "从数据、造雾、统一模型到视频推理的整体链路。", fig_system_overview),
        ("fig_02_fog_augmentation_examples.png", "在线造雾原理图与三类天气样例图", "展示清晰图、深度图、clear/uniform/patchy 生成示例。", lambda p: fig_fog_augmentation_examples(cfg, p)),
        ("fig_03_model_structure.png", "统一多任务模型结构图", "展示 YOLO 主干、分类头和 beta 回归头之间的关系。", fig_model_structure),
        ("fig_04_dataset_statistics.png", "数据集构成统计图与统计表", "依据 Data Audit 结果生成的样本规模统计图表。", fig_dataset_statistics),
        ("fig_05_training_validation_loss.png", "训练与验证损失曲线图", "依据 formal5 训练摘要生成总损失和子损失曲线。", fig_training_validation_loss),
        ("fig_06_fog_class_confusion_matrix.png", "雾分类混淆矩阵", "在合成验证样本上统计最终 fogfocus 权重的分类表现。", lambda p: fig_fog_confusion_matrix(cfg, p)),
        ("fig_07_beta_distribution_by_epoch.png", "不同训练轮次下 beta 分布图", "使用 formal5 多个 checkpoint 在同一视频上生成 beta 分布。", lambda p: fig_beta_distribution_by_epoch(cfg, p)),
        ("fig_08_inference_visualization.png", "推理结果可视化图", "从最终演示视频中截取代表帧。", fig_inference_visualization),
        ("fig_09_fog_map_example.png", "雾浓度分布图示例", "截取最终演示视频右下角雾浓度面板。", fig_fog_map_example),
    ]

    for filename, title, desc, fn in items:
        out_path = OUTPUT_DIR / filename
        print(f"Generating {filename} ...")
        fn(out_path)
        figures.append((filename, title, desc))

    comparison_png = OUTPUT_DIR / "table_01_method_comparison.png"
    comparison_csv = OUTPUT_DIR / "table_01_method_comparison.csv"
    print("Generating table_01_method_comparison ...")
    table_method_comparison(comparison_png, comparison_csv)
    figures.append(
        (
            comparison_png.name,
            "当前可用方案对比表",
            "对比默认统一模型、视频适配模型、fogfocus 模型与最终混合方案。",
        )
    )
    figures.append(
        (
            comparison_csv.name,
            "当前可用方案对比表（CSV）",
            "与 PNG 表格对应的源数据，便于论文后续排版。",
        )
    )

    write_summary(OUTPUT_DIR / "README.md", figures)
    print(f"Generated thesis figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
