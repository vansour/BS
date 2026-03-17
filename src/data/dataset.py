#!/usr/bin/env python3
"""
多任务数据集
Multi-Task Dataset

该模块负责同时加载：
1. 原始清晰图像；
2. 与图像对应的深度图缓存；
3. 原始 XML 检测标注；
4. 训练阶段所需的基础图像变换。

最终每个样本会同时返回：
- 图像张量；
- 深度图张量；
- 检测类别张量；
- 检测框张量。

本模块对应的是当前联合训练主链路的数据入口，而非离线 YOLO 数据集读取器。
训练阶段并不直接消费已经雾化好的离线图像，而是读取：
1. 清晰原图；
2. 对应的深度缓存；
3. 对应帧的 XML 检测框；
随后将这些信息交给在线造雾模块与统一多任务模型共同完成训练。
"""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import apply_letterbox_to_boxes_xyxy, letterbox_tensor


class MultiTaskDataset(Dataset):
    """
    面向统一多任务训练的数据集类。

    每个样本返回：
        - `image_tensor`：清晰图像张量；
        - `depth_tensor`：与图像空间对齐的深度图张量；
        - `det_cls_tensor`：当前图像中所有检测目标的类别；
        - `det_box_tensor`：当前图像中所有检测目标的归一化 `xywh` 框。

    其中深度图用于在线造雾，检测框则用于给 YOLO 检测头提供联合训练监督。

    数据组织的关键约束如下：
    - 原始图像默认按 UA-DETRAC 的“序列目录/帧图像”结构组织；
    - 深度缓存按 `序列名_图像名.npy` 命名；
    - 检测监督来自同名序列的 XML；
    - 训练/验证划分按序列切分，而不是按单帧随机切分。
    """

    def __init__(
        self,
        raw_data_dir,
        depth_cache_dir,
        xml_dir=None,
        transform=None,
        is_train=True,
        frame_stride=1,
        det_train_class_id=0,
        img_size=None,
        keep_ratio=True,
    ):
        """
        初始化数据集。

        Args:
            raw_data_dir: 原始图像目录，默认遵循 UA-DETRAC 的序列组织方式。
            depth_cache_dir: 深度图缓存目录，内部存储 `.npy` 文件。
            xml_dir: XML 标注目录；若为空则返回空检测标签。
            transform: 图像变换，例如 `Resize`、`ToTensor` 等。
            is_train: 是否构建训练集；若为 False，则构建验证子集。
            frame_stride: 帧抽样步长。`1` 表示使用全部帧，`N` 表示每隔 N 帧取 1 帧。
            det_train_class_id: 检测训练标签在当前检测头类别空间中的类别编号。
            img_size: 训练输入尺寸；指定后会在数据集内部执行统一的空间变换。
            keep_ratio: 是否使用 letterbox 保持纵横比。

        初始化阶段主要完成三项工作：
        1. 扫描原始图像目录，建立样本索引；
        2. 按序列名切分训练/验证集合；
        3. 预加载对应序列的 XML 标注到内存，减少 `__getitem__()` 的磁盘解析开销。
        """
        self.raw_data_dir = raw_data_dir
        self.depth_cache_dir = depth_cache_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.frame_stride = max(1, int(frame_stride))
        self.det_train_class_id = int(det_train_class_id)
        self.img_size = img_size
        self.keep_ratio = bool(keep_ratio)

        if not os.path.exists(depth_cache_dir):
            os.makedirs(depth_cache_dir, exist_ok=True)

        self.samples = []
        self.annotations = {}

        if not os.path.exists(raw_data_dir):
            print(f"警告：原始数据目录不存在: {raw_data_dir}")
            return

        # 按视频序列划分训练/验证集，避免同一序列中的相邻帧同时出现在不同集合中。
        # 这是视频类数据处理中很重要的约束，否则验证集会和训练集高度相似，指标失真。
        seq_folders = sorted([
            d for d in os.listdir(raw_data_dir)
            if os.path.isdir(os.path.join(raw_data_dir, d))
        ])

        split_idx = int(len(seq_folders) * 0.8)
        selected_seqs = seq_folders[:split_idx] if is_train else seq_folders[split_idx:]

        for seq in selected_seqs:
            seq_path = os.path.join(raw_data_dir, seq)
            # 对每个序列中的图像文件按文件名排序，再按 frame_stride 做稀疏采样。
            files = sorted([
                f for f in os.listdir(seq_path)
                if f.lower().endswith((".jpg", ".png"))
            ])[::self.frame_stride]
            for img_name in files:
                self.samples.append((os.path.join(seq_path, img_name), seq, img_name))

        # 预加载所选序列的 XML 标注，避免 __getitem__ 中反复解析磁盘文件。
        # 这里建立的是 `sequence -> {frame_num -> boxes}` 的内存映射。
        if self.xml_dir and os.path.exists(self.xml_dir):
            for seq in selected_seqs:
                xml_path = os.path.join(self.xml_dir, f"{seq}.xml")
                if os.path.exists(xml_path):
                    self.annotations[seq] = self._parse_xml_sequence(xml_path)
                else:
                    self.annotations[seq] = {}
        elif self.xml_dir:
            print(f"警告：XML 标注目录不存在，将返回空检测标签: {self.xml_dir}")

    def __len__(self):
        """返回样本总数。"""
        return len(self.samples)

    @staticmethod
    def _extract_frame_num(img_name: str) -> int | None:
        """
        从图像文件名中提取帧号。

        Args:
            img_name: 图像文件名，例如 `img00001.jpg`。

        Returns:
            int | None: 提取出的帧号；若匹配失败则返回 `None`。
        """
        # UA-DETRAC 图像名通常形如 img00001.jpg，这里提取数字部分作为帧号键。
        match = re.search(r"img(\d+)", img_name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_xml_sequence(xml_file: str) -> dict[int, list[list[float]]]:
        """
        解析单个视频序列的 XML 标注文件。

        Returns:
            dict[int, list[list[float]]]:
                键为帧号，值为该帧所有目标框的绝对坐标列表 `[left, top, width, height]`。

        这里只保留训练真正需要的信息，即 bbox 坐标。
        其他潜在字段例如目标属性、遮挡信息等，当前并未纳入联合训练流程。
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        sequence_data: dict[int, list[list[float]]] = {}

        for frame in root.findall("frame"):
            num_str = frame.get("num")
            if num_str is None:
                continue

            frame_num = int(num_str)
            boxes: list[list[float]] = []
            target_list = frame.find("target_list")
            if target_list is not None:
                for target in target_list.findall("target"):
                    box = target.find("box")
                    if box is None:
                        continue
                    boxes.append([
                        float(box.get("left")),
                        float(box.get("top")),
                        float(box.get("width")),
                        float(box.get("height")),
                    ])

            sequence_data[frame_num] = boxes

        return sequence_data

    @staticmethod
    def _convert_box(size: tuple[int, int], box: list[float]) -> list[float]:
        """
        把绝对坐标框转换为 YOLO 所需的归一化 `xywh`。

        Args:
            size: 图像尺寸，格式为 `(width, height)`。
            box: 原始绝对坐标框 `[left, top, width, height]`。

        Returns:
            list[float]: 归一化后的 `[x_center, y_center, width, height]`。

        训练时图像会被 Resize，但检测框这里使用的是相对比例坐标，
        因此不需要再根据训练输入尺寸额外重算，只要原始宽高正确即可。
        """
        width, height = size
        x_center = box[0] + box[2] / 2.0
        y_center = box[1] + box[3] / 2.0
        return [
            x_center / width,
            y_center / height,
            box[2] / width,
            box[3] / height,
        ]

    @staticmethod
    def _boxes_to_xyxy(frame_boxes: list[list[float]]) -> torch.Tensor:
        """
        把 `[left, top, width, height]` 绝对框转换为 `xyxy` 张量。
        """
        if not frame_boxes:
            return torch.zeros((0, 4), dtype=torch.float32)

        return torch.tensor(
            [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in frame_boxes],
            dtype=torch.float32,
        )

    @staticmethod
    def _xyxy_to_xywh_norm(boxes_xyxy: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
        """
        把绝对坐标 `xyxy` 框转换为归一化 `xywh`。
        """
        if boxes_xyxy.numel() == 0:
            return torch.zeros((0, 4), dtype=torch.float32)

        height, width = image_shape
        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
        converted = torch.stack(
            [
                ((x1 + x2) / 2.0) / width,
                ((y1 + y2) / 2.0) / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            ],
            dim=1,
        )
        return converted.clamp_(0.0, 1.0)

    def __getitem__(self, idx):
        """
        读取指定索引的样本。

        Args:
            idx: 样本索引。

        Returns:
            tuple:
                - `image_tensor`
                - `depth_tensor`
                - `det_cls_tensor`
                - `det_box_tensor`

        单个样本的处理流程如下：
        1. 读原图；
        2. 读同名深度缓存；
        3. 把图像按 transform 处理成训练输入；
        4. 把深度图插值到和训练图像一致的空间尺寸；
        5. 读取当前帧对应的检测框并转成归一化 xywh。
        """
        img_path, seq, img_name = self.samples[idx]

        try:
            image_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            image_pil = Image.new("RGB", (640, 640), (0, 0, 0))

        orig_w, orig_h = image_pil.size

        # 深度缓存命名规则必须和预计算脚本保持一致，否则这里无法对齐读取。
        depth_name = f"{seq}_{img_name}.npy"
        depth_path = os.path.join(self.depth_cache_dir, depth_name)

        if os.path.exists(depth_path):
            try:
                depth = np.load(depth_path)
            except Exception as e:
                raise RuntimeError(
                    f"无法加载深度图缓存 {depth_path}: {e}。"
                    f"请检查深度预计算流程是否已经正确完成。"
                )
        else:
            raise FileNotFoundError(
                f"缺少深度图缓存: {depth_path}\n"
                f"请先运行深度预计算流程，或手动生成对应的 `.npy` 缓存文件。"
            )

        # 检测标签仍然使用原图坐标，只是转换为相对比例，因此对后续 Resize 保持不变。
        frame_num = self._extract_frame_num(img_name)
        seq_annotations = self.annotations.get(seq, {})
        frame_boxes = seq_annotations.get(frame_num, []) if frame_num is not None else []

        image_tensor = transforms.ToTensor()(image_pil)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        boxes_xyxy = self._boxes_to_xyxy(frame_boxes)

        if self.img_size is not None:
            if self.keep_ratio:
                image_tensor, letterbox_meta = letterbox_tensor(image_tensor, self.img_size)
                depth_tensor, _ = letterbox_tensor(depth_tensor, self.img_size, pad_value=0.0)
                boxes_xyxy = apply_letterbox_to_boxes_xyxy(boxes_xyxy, letterbox_meta)
            else:
                if isinstance(self.img_size, int):
                    target_shape = (self.img_size, self.img_size)
                else:
                    target_shape = (int(self.img_size[0]), int(self.img_size[1]))
                scale_w = target_shape[1] / max(orig_w, 1)
                scale_h = target_shape[0] / max(orig_h, 1)
                image_tensor = torch.nn.functional.interpolate(
                    image_tensor.unsqueeze(0),
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                depth_tensor = torch.nn.functional.interpolate(
                    depth_tensor.unsqueeze(0),
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if boxes_xyxy.numel() > 0:
                    boxes_xyxy[:, [0, 2]] *= scale_w
                    boxes_xyxy[:, [1, 3]] *= scale_h
        else:
            if self.transform:
                image_tensor = self.transform(image_pil)
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor.unsqueeze(0),
                size=image_tensor.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            if boxes_xyxy.numel() > 0:
                scale_w = image_tensor.shape[2] / max(orig_w, 1)
                scale_h = image_tensor.shape[1] / max(orig_h, 1)
                boxes_xyxy[:, [0, 2]] *= scale_w
                boxes_xyxy[:, [1, 3]] *= scale_h

        if boxes_xyxy.numel() > 0:
            # UA-DETRAC 当前只提供“车辆”这一粗粒度目标。
            # 当检测头保留 COCO 80 类时，这里把所有车辆统一映射到一个 COCO 车辆类上，
            # 从而继续利用预训练检测头，而不是从零开始学习一个新头。
            det_box_tensor = self._xyxy_to_xywh_norm(
                boxes_xyxy,
                (image_tensor.shape[1], image_tensor.shape[2]),
            )
            det_cls_tensor = torch.full(
                (det_box_tensor.shape[0],),
                float(self.det_train_class_id),
                dtype=torch.float32,
            )
        else:
            det_box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            det_cls_tensor = torch.zeros((0,), dtype=torch.float32)

        return image_tensor, depth_tensor, det_cls_tensor, det_box_tensor

