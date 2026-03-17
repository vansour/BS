#!/usr/bin/env python3
"""
高速公路团雾监测推理脚本
Highway Fog Monitoring Inference Script

本模块提供一个面向运行时的封装类 `HighwayFogSystem`，用于：
1. 加载统一多任务模型；
2. 对摄像头或视频文件中的帧进行预处理；
3. 执行雾类型分类与 beta 推理；
4. 结合 EMA 平滑结果动态调整检测置信度提示；
5. 以可视化界面的方式展示运行状态。

与训练脚本相比，本模块更强调运行时工程组织能力，主要解决以下问题：
- 如何解析并加载最合适的模型权重；
- 如何对视频帧进行规范化预处理；
- 如何完成检测后处理与动态阈值调节；
- 如何将结果组织为可演示的实时可视化界面。
"""

import os
import sys
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import Config
from src.model import UnifiedMultiTaskModel
from src.utils import (
    invert_letterbox_boxes_xyxy,
    letterbox_tensor,
    load_model_weights,
    resolve_model_weights,
)


class HighwayFogSystem:
    """
    统一多任务模型的运行时封装。

    该类的职责不是重新定义模型，而是把模型推理链路串起来，包括：
    - 权重解析与加载；
    - 图像预处理；
    - GPU 多流推理；
    - 输出后处理；
    - 实时界面显示。

    从设计定位上看，该类更接近“运行时系统封装”，而非单纯的 `predict()` 包装器。
    """

    FOG_CLASS_NAMES = ["CLEAR", "UNIFORM FOG", "PATCHY FOG"]
    FOG_COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 0)]
    DETECTION_COLORS = [(80, 255, 80), (0, 255, 255), (0, 200, 255)]

    def __init__(self, model_path: Optional[str], video_source: Union[str, int] = 0, cfg: Optional[Config] = None):
        """
        初始化推理系统。

        Args:
            model_path: 指定权重路径；若为空或不存在，则自动回退到默认导出路径。
            video_source: 视频源，可以是摄像头编号，也可以是视频文件路径。
            cfg: 可选配置对象；若不提供，则创建默认配置。

        初始化时除了模型本身，还会准备：
        - 图像变换流程；
        - CUDA 三段异步流；
        - 视频读取器；
        - beta 平滑状态。
        """
        self.cfg = cfg or Config()
        self.device = self.cfg.DEVICE
        self.model_path = self._resolve_model_path(model_path)

        # 初始化模型并加载权重。
        self.model = UnifiedMultiTaskModel(
            self.cfg.YOLO_BASE_MODEL,
            self.cfg.NUM_FOG_CLASSES,
            num_det_classes=self.cfg.NUM_DET_CLASSES,
        )
        self._load_model()
        self.model.to(self.device).eval()

        # 推理阶段直接在张量层执行 letterbox，与训练阶段保持同样的几何约束。
        self.input_mean = None
        self.input_std = None
        if self.cfg.USE_IMAGENET_NORMALIZE:
            self.input_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
            self.input_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        # 若 GPU 可用，则启用预处理 / 推理 / 后处理三阶段流并行。
        # 这并不意味着每一步都完全并行，但可以让不同阶段尽量形成流水线。
        if self.device == "cuda":
            print("CUDA detected, enabling asynchronous pipeline streams.")
            self.preprocess_stream = torch.cuda.Stream()
            self.inference_stream = torch.cuda.Stream()
            self.postprocess_stream = torch.cuda.Stream()
        else:
            print("CUDA is not available, using synchronous execution.")
            self.preprocess_stream = None
            self.inference_stream = None
            self.postprocess_stream = None

        self.cap = cv2.VideoCapture(video_source)
        self.ema_beta = 0.0
        self.smooth_alpha = self.cfg.EMA_ALPHA
        self.base_conf_thres = self.cfg.BASE_CONF_THRES
        self.det_class_names = getattr(self.cfg, "DET_CLASS_NAMES", ["vehicle"])

    def _resolve_model_path(self, requested_path: Optional[str]) -> Optional[str]:
        """
        解析最终要加载的模型路径。

        Args:
            requested_path: 调用方传入的权重路径。

        Returns:
            Optional[str]: 可用权重路径；若都找不到，则返回原始请求值或 `None`。

        路径解析优先级是：
        1. 调用方显式给出的存在路径；
        2. `OUTPUT_DIR` 中的正式导出权重；
        3. checkpoint 目录里的最新权重。
        """
        if requested_path and os.path.exists(requested_path):
            return requested_path

        fallback = resolve_model_weights(
            self.cfg.OUTPUT_DIR,
            self.cfg.CHECKPOINT_DIR,
            preferred_files=["unified_model_best.pt", "unified_model.pt"],
        )
        if fallback and fallback != requested_path:
            print(f"Requested weights were not found, falling back to: {fallback}")
        return fallback or requested_path

    def _load_model(self):
        """
        将权重加载到当前模型实例中。

        找不到可用权重时不会直接报错退出，而是保留随机初始化权重，
        这样至少可以进行链路级联调或空跑验证。

        这在项目中期开发阶段很有用，因为即使训练权重还未完全准备好，
        推理链路、UI、后处理逻辑仍然可以先做烟雾测试。
        """
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading model weights from: {self.model_path}")
            try:
                report = load_model_weights(self.model, self.model_path, map_location=self.device)
                skipped_yolo_keys = [
                    key for key in report.get("skipped_mismatched_keys", []) if key.startswith("yolo.")
                ]
                print(f"Weights loaded successfully ({report['source_type']}).")
                if report["missing_keys"] or report["unexpected_keys"]:
                    print(
                        f"Non-strict load summary: missing={len(report['missing_keys'])}, "
                        f"unexpected={len(report['unexpected_keys'])}"
                    )
                if report.get("skipped_mismatched_keys"):
                    print(f"Skipped mismatched keys: {len(report['skipped_mismatched_keys'])}")
                if skipped_yolo_keys:
                    print(
                        "Detection head class-count mismatch detected. "
                        "The current checkpoint was trained with a different `NUM_DET_CLASSES`; "
                        "single-class inference will need a re-trained single-class checkpoint."
                    )
                return
            except Exception as exc:
                print(f"Failed to load weights ({exc}), using randomly initialized weights.")

        print("No usable weights were found, using randomly initialized weights.")

    def _preprocess_async(
        self,
        frame: np.ndarray,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """
        异步或同步地完成图像预处理。

        Args:
            frame: OpenCV 读取得到的 BGR 图像。
            stream: 可选 CUDA 流；若提供则在该流上执行预处理。

        Returns:
            tuple[torch.Tensor, dict[str, object]]: 模型输入张量与当前帧的 letterbox 元数据。

        输入视频帧来自 OpenCV，默认是 BGR；而 PIL / torchvision 习惯使用 RGB，
        因此这里首先会做一次颜色空间转换。
        """
        def process():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
            img_tensor, letterbox_meta = letterbox_tensor(img_tensor, self.cfg.IMG_SIZE)
            if self.input_mean is not None and self.input_std is not None:
                img_tensor = (img_tensor - self.input_mean) / self.input_std
            img_tensor = img_tensor.unsqueeze(0)
            if self.device == "cuda":
                return img_tensor.to(self.device, non_blocking=True), letterbox_meta
            return img_tensor, letterbox_meta

        # 如果提供了 CUDA stream，就在该 stream 上提交预处理；
        # 否则退化为同步执行。
        if stream is not None:
            with torch.cuda.stream(stream):
                return process()
        return process()

    def _inference_async(
        self,
        img_tensor: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        在指定 CUDA 流或默认执行流上执行模型推理。

        Args:
            img_tensor: 预处理后的输入张量。
            stream: 可选 CUDA 流。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                检测输出、雾分类 logits 和 beta 回归结果。

        模型本身已经被置为 `eval()`，这里再配合 `torch.no_grad()`，
        以避免推理阶段构建不必要的梯度图。
        """
        if stream is not None:
            with torch.cuda.stream(stream):
                with torch.no_grad():
                    return self.model(img_tensor)

        with torch.no_grad():
            return self.model(img_tensor)

    def _postprocess_async(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[np.ndarray, float, torch.Tensor]:
        """
        对模型输出做后处理。

        Args:
            outputs: 模型的三路输出结果。
            stream: 可选 CUDA 流。

        Returns:
            Tuple[np.ndarray, float, torch.Tensor]:
                雾分类概率、映射后的 beta 值以及经过 NMS 的检测结果。

        这里完成的是“模型输出到业务可消费结果”的转换：
        - logits -> softmax 概率；
        - 归一化 beta -> 实际 beta；
        - 原始检测输出 -> NMS 后候选框。
        """
        def process():
            det_out, logits, beta_tensor = outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            beta = float(beta_tensor.item() * self.cfg.BETA_MAX)

            # 推理阶段先用较低阈值做一次 NMS，尽量保留候选框，
            # 后续再根据平滑后的 beta 动态过滤显示阈值。
            # 这相当于把“几何去重”和“显示保留策略”分成两个阶段。
            det_batches = non_max_suppression(
                det_out,
                conf_thres=0.05,
                iou_thres=0.45,
                nc=self.cfg.NUM_DET_CLASSES,
                max_det=100,
            )
            detections = det_batches[0].detach().cpu() if det_batches else torch.zeros((0, 6))
            detections = self._merge_vehicle_detections(detections)
            return probs, beta, detections

        if stream is not None:
            with torch.cuda.stream(stream):
                return process()
        return process()

    def _merge_vehicle_detections(self, detections: torch.Tensor) -> torch.Tensor:
        """
        将检测结果统一整理为单类 `vehicle`。

        在当前单类配置下，NMS 后的类别 id 理论上就只有 `0`。
        这里保留一个轻量归一化步骤，是为了兼容旧结果或边界情况，
        确保绘制逻辑始终看到统一的 `vehicle` 类别编号。
        """
        if detections.numel() == 0:
            return detections

        vehicle_detections = detections.clone()
        vehicle_detections[:, 5] = 0.0
        return vehicle_detections

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, float, torch.Tensor]:
        """
        对单帧图像执行同步推理。

        Args:
            frame: 输入帧，格式为 OpenCV 的 BGR 图像。

        Returns:
            Tuple[np.ndarray, float, torch.Tensor]:
                概率向量、beta 值以及经过 NMS 的检测结果。

        这是最适合被外部脚本直接调用的同步接口。
        如果不需要完整视频循环和 UI，只需要单帧结果，就走这里。
        """
        img_tensor, _ = self._preprocess_async(frame)
        outputs = self._inference_async(img_tensor)
        return self._postprocess_async(outputs)

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: torch.Tensor,
        adaptive_conf: float,
        fog_idx: int,
        preprocess_meta: dict[str, object],
    ) -> tuple[np.ndarray, int]:
        """
        在显示帧上绘制检测框。

        Args:
            frame: 待绘制的显示帧。
            detections: NMS 后检测结果，格式为 `[x1, y1, x2, y2, conf, cls]`。
            adaptive_conf: 动态置信度阈值。
            fog_idx: 当前雾类型索引，用于选择显示颜色。
            preprocess_meta: 当前帧输入阶段的 letterbox 元数据。

        Returns:
            tuple:
                - 绘制后的帧
                - 最终被展示的检测框数量

        注意这里绘制的是“已经经过第二次动态阈值过滤”的框，不是 NMS 后的全部候选框。
        因此最终展示数量会随着雾浓度估计而变化。
        """
        if detections.numel() == 0:
            return frame, 0

        if preprocess_meta is None:
            return frame, 0

        display_boxes = detections[detections[:, 4] >= adaptive_conf]
        if display_boxes.numel() == 0:
            return frame, 0

        # 检测框当前还处于 letterbox 输入坐标系，需要先还原到原始帧，
        # 再根据展示分辨率做一次线性映射。
        boxes = invert_letterbox_boxes_xyxy(display_boxes[:, :4], preprocess_meta)
        src_h, src_w = preprocess_meta["src_shape"]
        frame_h, frame_w = frame.shape[:2]
        boxes[:, [0, 2]] *= frame_w / max(src_w, 1)
        boxes[:, [1, 3]] *= frame_h / max(src_h, 1)

        box_color = self.DETECTION_COLORS[min(fog_idx, len(self.DETECTION_COLORS) - 1)]
        for box, det in zip(boxes, display_boxes):
            x1, y1, x2, y2 = box.round().int().tolist()
            conf = float(det[4].item())
            cls_idx = int(det[5].item()) if det.shape[0] > 5 else 0
            cls_name = self.det_class_names[cls_idx] if cls_idx < len(self.det_class_names) else str(cls_idx)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"{cls_name} {conf:.2f}"
            label_y = max(20, y1 - 8)
            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        return frame, int(display_boxes.shape[0])

    def run(self):
        """
        启动实时演示主循环。

        按 `Q` 键可以退出。若设备支持 CUDA，则主循环会尽量让
        预处理、推理和后处理三个阶段形成轻量级流水线。

        运行时大致是一个“当前帧在推理、下一帧先预处理、上一帧做显示”的节奏，
        以尽量减少 GPU 和 CPU 的空闲等待。
        """
        print("Realtime pipeline is running. Press Q to exit.")

        ret, frame_n = self.cap.read()
        if not ret:
            print("Failed to read from the video source.")
            return

        tensor_n, preprocess_meta_n = self._preprocess_async(frame_n, self.preprocess_stream)
        prev_outputs: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        prev_frame: Optional[np.ndarray] = None
        prev_preprocess_meta: Optional[dict[str, object]] = None

        while True:
            # 确保当前帧的推理要等待上一阶段的预处理完成。
            if self.preprocess_stream and self.inference_stream:
                self.inference_stream.wait_stream(self.preprocess_stream)

            curr_outputs = self._inference_async(tensor_n, self.inference_stream)

            # 尽早读取下一帧并启动预处理，形成轻量级流水线。
            ret, frame_next = self.cap.read()
            if ret:
                tensor_next, preprocess_meta_next = self._preprocess_async(frame_next, self.preprocess_stream)

            if prev_outputs is not None and prev_frame is not None:
                if self.inference_stream and self.postprocess_stream:
                    self.postprocess_stream.wait_stream(self.inference_stream)

                probs, beta, detections = self._postprocess_async(prev_outputs, self.postprocess_stream)

                if self.postprocess_stream:
                    self.postprocess_stream.synchronize()

                # 使用 EMA 平滑 beta，降低单帧预测抖动。
                # 后续动态阈值直接基于这个平滑结果，而不是单帧瞬时值。
                self.ema_beta = self.smooth_alpha * beta + (1 - self.smooth_alpha) * self.ema_beta
                adaptive_conf = max(0.15, self.base_conf_thres - self.ema_beta * 1.2)

                draw_frame = cv2.resize(prev_frame, (960, 540))
                fog_idx = int(np.argmax(probs))
                status = self.FOG_CLASS_NAMES[fog_idx]
                color = self.FOG_COLORS[fog_idx]

                # 叠加状态栏与 beta 信息。
                cv2.rectangle(draw_frame, (0, 0), (960, 80), (45, 45, 45), -1)
                cv2.putText(draw_frame, f"STATUS: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                info_line = (
                    f"Beta: {self.ema_beta:.4f} | Base Conf: {self.base_conf_thres:.2f} "
                    f"-> Adaptive Conf: {adaptive_conf:.2f}"
                )
                cv2.putText(draw_frame, info_line, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                draw_frame, det_count = self._draw_detections(
                    draw_frame,
                    detections,
                    adaptive_conf,
                    fog_idx,
                    prev_preprocess_meta,
                )
                det_line = f"Visible Detections: {det_count}"
                cv2.putText(draw_frame, det_line, (720, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

                # 雾天条件下给出提示性文案，表示系统倾向于放宽低置信度目标的保留。
                if fog_idx > 0:
                    cv2.putText(
                        draw_frame,
                        "LOW CONF OBJECTS RECOVERED",
                        (300, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                cv2.imshow("Adaptive Fog System (Pipelined)", draw_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            prev_frame = frame_n.copy() if frame_n is not None else None
            prev_outputs = curr_outputs
            prev_preprocess_meta = preprocess_meta_n

            if not ret:
                break

            frame_n = frame_next
            tensor_n = tensor_next
            preprocess_meta_n = preprocess_meta_next

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """推理脚本入口函数。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = Config()
    model_path = resolve_model_weights(cfg.OUTPUT_DIR, cfg.CHECKPOINT_DIR)
    video_source = os.path.join(base_dir, "..", "test_video.mp4")
    if not os.path.exists(video_source):
        video_source = 0

    system = HighwayFogSystem(model_path, video_source=video_source, cfg=cfg)

    # Windows 通常没有 DISPLAY 环境变量，因此只在类 Unix 环境下据此判断是否无头运行。
    # 无头环境下不弹 UI，而是跑一次最小烟雾推理，确认链路可执行。
    if os.name != "nt" and "DISPLAY" not in os.environ:
        print("Headless environment detected. Running a smoke inference.")
        dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
        probs, beta, detections = system.predict(dummy_frame)
        print(f"Inference test: beta={beta:.4f}, prob={probs}, detections={len(detections)}")
    else:
        system.run()


if __name__ == "__main__":
    main()


