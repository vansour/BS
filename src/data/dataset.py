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
"""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiTaskDataset(Dataset):
    """
    面向统一多任务训练的数据集类。

    每个样本返回：
        - `image_tensor`：清晰图像张量；
        - `depth_tensor`：与图像空间对齐的深度图张量；
        - `det_cls_tensor`：当前图像中所有检测目标的类别；
        - `det_box_tensor`：当前图像中所有检测目标的归一化 `xywh` 框。

    其中深度图用于在线造雾，检测框则用于给 YOLO 检测头提供联合训练监督。
    """

    def __init__(self, raw_data_dir, depth_cache_dir, xml_dir=None, transform=None, is_train=True):
        """
        初始化数据集。

        Args:
            raw_data_dir: 原始图像目录，默认遵循 UA-DETRAC 的序列组织方式。
            depth_cache_dir: 深度图缓存目录，内部存储 `.npy` 文件。
            xml_dir: XML 标注目录；若为空则返回空检测标签。
            transform: 图像变换，例如 `Resize`、`ToTensor` 等。
            is_train: 是否构建训练集；若为 False，则构建验证子集。
        """
        self.raw_data_dir = raw_data_dir
        self.depth_cache_dir = depth_cache_dir
        self.xml_dir = xml_dir
        self.transform = transform

        if not os.path.exists(depth_cache_dir):
            os.makedirs(depth_cache_dir, exist_ok=True)

        self.samples = []
        self.annotations = {}

        if not os.path.exists(raw_data_dir):
            print(f"警告：原始数据目录不存在: {raw_data_dir}")
            return

        # 按视频序列划分训练/验证集，避免同一序列中的相邻帧同时出现在不同集合中。
        seq_folders = sorted([
            d for d in os.listdir(raw_data_dir)
            if os.path.isdir(os.path.join(raw_data_dir, d))
        ])

        split_idx = int(len(seq_folders) * 0.8)
        selected_seqs = seq_folders[:split_idx] if is_train else seq_folders[split_idx:]

        for seq in selected_seqs:
            seq_path = os.path.join(raw_data_dir, seq)
            files = sorted([
                f for f in os.listdir(seq_path)
                if f.lower().endswith((".jpg", ".png"))
            ])
            for img_name in files:
                self.samples.append((os.path.join(seq_path, img_name), seq, img_name))

        # 预加载所选序列的 XML 标注，避免 __getitem__ 中反复解析磁盘文件。
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
        match = re.search(r"img(\d+)", img_name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_xml_sequence(xml_file: str) -> dict[int, list[list[float]]]:
        """
        解析单个视频序列的 XML 标注文件。

        Returns:
            dict[int, list[list[float]]]:
                键为帧号，值为该帧所有目标框的绝对坐标列表 `[left, top, width, height]`。
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
        """
        img_path, seq, img_name = self.samples[idx]

        try:
            image_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            image_pil = Image.new("RGB", (640, 640), (0, 0, 0))

        orig_w, orig_h = image_pil.size

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

        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)

        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(0),
            size=image_tensor.shape[1:],
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # 检测标签仍然使用原图坐标，只是转换为相对比例，因此对后续 Resize 保持不变。
        frame_num = self._extract_frame_num(img_name)
        seq_annotations = self.annotations.get(seq, {})
        frame_boxes = seq_annotations.get(frame_num, []) if frame_num is not None else []

        if frame_boxes:
            det_box_tensor = torch.tensor(
                [self._convert_box((orig_w, orig_h), box) for box in frame_boxes],
                dtype=torch.float32,
            )
            det_cls_tensor = torch.zeros((det_box_tensor.shape[0],), dtype=torch.float32)
        else:
            det_box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            det_cls_tensor = torch.zeros((0,), dtype=torch.float32)

        return image_tensor, depth_tensor, det_cls_tensor, det_box_tensor

