#!/usr/bin/env python3
"""
数据集准备器
Dataset Preparer

本模块用于把原始 UA-DETRAC 数据和已经生成好的雾天图像整理成适合 YOLO
训练的标准数据集目录结构，主要步骤包括：
1. 解析 UA-DETRAC XML 标注；
2. 转换为 YOLO 所需的归一化框坐标；
3. 按视频序列划分训练集和验证集；
4. 拷贝对应图像并生成标签文件；
5. 生成 `data.yaml` 配置文件。

本模块属于离线辅助链路，与当前“在线造雾 + 多任务联合训练”的主流程不同。
其主要价值在于：
1. 构造独立 YOLO 检测实验所需的数据集；
2. 快速核查离线雾图与标注是否对齐；
3. 为只做检测任务的对比实验提供标准目录结构。
"""

import os
import re
import shutil
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm

from src.utils import split_sequence_names


class DatasetPreparer:
    """
    YOLO 数据集构建工具类。

    该类把“从原始标注到训练目录”的完整流程集中管理，便于在离线数据准备阶段
    一次性执行全部转换步骤，并统计最终的训练/验证样本规模。

    当前实现默认将所有目标统一映射为单一类别 `vehicle`，与项目现阶段的检测设置保持一致。
    """

    def __init__(
        self,
        xml_dir,
        foggy_image_root,
        output_dataset_dir,
        train_ratio=0.8,
        split_seed=42,
    ):
        """
        初始化数据集准备器。

        Args:
            xml_dir: UA-DETRAC XML 标注目录。
            foggy_image_root: 已生成雾天图像的根目录，内部包含多个 `*_Foggy` 子目录。
            output_dataset_dir: 输出的 YOLO 数据集根目录。
            train_ratio: 训练集所占的视频序列比例。
            split_seed: 序列级切分的随机种子。

        `stats` 用于累计最终转换结果，以便在流程结束后输出清晰的样本统计摘要。
        """
        self.xml_dir = xml_dir
        self.foggy_image_root = foggy_image_root
        self.output_dir = output_dataset_dir
        self.train_ratio = train_ratio
        self.split_seed = int(split_seed)

        self.stats = {
            "train_images": 0,
            "val_images": 0,
            "train_labels": 0,
            "val_labels": 0,
            "train_sequences": 0,
            "val_sequences": 0,
        }

    def convert_box(self, size, box):
        """
        把绝对坐标框转换为 YOLO 归一化中心点格式。

        Args:
            size: 图像尺寸，格式为 `(width, height)`。
            box: 原始框，格式为 `[left, top, width, height]`。

        Returns:
            tuple: `(x_center, y_center, w, h)`，都已经归一化到 `[0, 1]`。

        这是把 UA-DETRAC 风格的左上角坐标框转成 YOLO 风格中心点框的标准步骤。
        """
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x = box[0] + box[2] / 2.0
        y = box[1] + box[3] / 2.0
        w = box[2]
        h = box[3]
        return (x * dw, y * dh, w * dw, h * dh)

    @staticmethod
    def build_output_image_name(sequence_name, img_file):
        """
        为导出的 YOLO 样本生成全局唯一文件名。

        UA-DETRAC 不同序列会重复使用 `img00001.jpg` 这类帧名。
        若直接把所有图像写入扁平化的 `images/train` 或 `images/val`，
        后处理序列会覆盖前面的样本。这里统一加上序列名前缀，
        保持当前目录结构不变的同时消除命名冲突。
        """
        return f"{sequence_name}_{img_file}"

    def parse_xml_sequence(self, xml_file):
        """
        解析单个视频序列的 XML 标注文件。

        Args:
            xml_file: XML 文件路径。

        Returns:
            dict: 结构为 `{frame_num: [objects]}`，其中每个 object 至少包含 `bbox` 字段。

        这里将每帧解析为“对象列表”而非仅保留 bbox 数组，
        主要是为了给未来扩展目标属性字段保留结构空间。
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        sequence_data = {}

        for frame in root.findall("frame"):
            num_str = frame.get("num")
            if num_str is None:
                continue
            frame_num = int(num_str)

            objects = []
            target_list = frame.find("target_list")
            if target_list is not None:
                for target in target_list.findall("target"):
                    box = target.find("box")
                    if box is None:
                        continue
                    objects.append(
                        {
                            "bbox": [
                                float(box.get("left")),
                                float(box.get("top")),
                                float(box.get("width")),
                                float(box.get("height")),
                            ]
                        }
                    )
            sequence_data[frame_num] = objects

        return sequence_data

    def create_structure(self):
        """
        创建 YOLO 标准目录结构。

        最终会生成：
            - `images/train`
            - `images/val`
            - `labels/train`
            - `labels/val`

        这是 Ultralytics YOLO 最常见的数据集目录布局之一，
        后续可以直接配合生成的 `data.yaml` 使用。
        """
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

    def process(self):
        """
        执行完整的数据集转换流程。

        流程包括目录构建、序列划分、图像拷贝、标签写入和统计输出。

        这里采用“序列级”而非“帧级”划分 train/val，
        以避免同一视频中的相邻帧同时进入训练集和验证集，造成评估泄漏。
        """
        self.create_structure()

        if not os.path.exists(self.foggy_image_root):
            print(f"错误：找不到雾天图像目录: {self.foggy_image_root}")
            return

        # 收集所有已经生成好的雾天序列目录。
        # 这些目录名默认形如 `MVI_xxx_Foggy`，后面会通过去掉 `_Foggy` 找到原 XML 名称。
        foggy_folders = [
            d
            for d in os.listdir(self.foggy_image_root)
            if os.path.isdir(os.path.join(self.foggy_image_root, d))
        ]
        print(f"发现 {len(foggy_folders)} 个已处理序列，开始转换。")

        if not os.path.exists(self.xml_dir):
            print(f"\n严重错误：XML 标注目录不存在: {self.xml_dir}")
            print("请确认已下载并放置 `DETRAC-Train-Annotations-XML` 数据。")
            return

        # 按序列做有种子的随机切分，避免单帧级划分造成泄漏，
        # 同时保证多次运行可复现。
        train_sequence_list, val_sequence_list = split_sequence_names(
            foggy_folders,
            train_ratio=self.train_ratio,
            seed=self.split_seed,
        )
        train_sequences = set(train_sequence_list)
        val_sequences = set(val_sequence_list)

        self.stats["train_sequences"] = len(train_sequences)
        self.stats["val_sequences"] = len(val_sequences)

        print(
            f"序列划分完成：训练集 {len(train_sequences)} 个序列，"
            f"验证集 {len(val_sequences)} 个序列。"
        )

        for folder_name in tqdm(foggy_folders, desc="处理序列"):
            sequence_name = folder_name.replace("_Foggy", "")
            xml_path = os.path.join(self.xml_dir, f"{sequence_name}.xml")

            if not os.path.exists(xml_path):
                continue

            foggy_seq_path = os.path.join(self.foggy_image_root, folder_name)
            foggy_images = [f for f in os.listdir(foggy_seq_path) if f.endswith(".jpg")]
            if not foggy_images:
                continue

            # 通过第一张图像读取原始尺寸，后续所有标签都按这个尺寸归一化。
            # 这里假定同一序列中图像尺寸一致，这与常见视频序列数据组织相符。
            test_img = cv2.imread(os.path.join(foggy_seq_path, foggy_images[0]))
            if test_img is None:
                continue
            h, w = test_img.shape[:2]

            annotations = self.parse_xml_sequence(xml_path)
            subset = "train" if folder_name in train_sequences else "val"

            for img_file in foggy_images:
                try:
                    match = re.search(r"img(\d+)", img_file)
                    if not match:
                        continue
                    frame_num = int(match.group(1))
                except Exception:
                    continue

                if frame_num not in annotations:
                    continue

                # 该项目目前只保留车辆检测类别，因此类别编号固定为 0。
                label_lines = []
                for obj in annotations[frame_num]:
                    yolo_box = self.convert_box((w, h), obj["bbox"])
                    label_lines.append(
                        f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}"
                    )

                if not label_lines:
                    continue

                output_image_name = self.build_output_image_name(
                    sequence_name, img_file
                )

                # 图像和标签分别写入标准 YOLO 目录。
                shutil.copy(
                    os.path.join(foggy_seq_path, img_file),
                    os.path.join(self.output_dir, "images", subset, output_image_name),
                )

                label_save_path = os.path.join(
                    self.output_dir,
                    "labels",
                    subset,
                    os.path.splitext(output_image_name)[0] + ".txt",
                )
                with open(label_save_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(label_lines))

                self.stats[f"{subset}_images"] += 1
                self.stats[f"{subset}_labels"] += len(label_lines)

        self.create_yaml()
        self._print_stats()
        print("\n数据集构建完成。")

    def _print_stats(self):
        """打印最终数据集统计信息。"""
        print("\n" + "=" * 50)
        print("数据集统计信息")
        print("=" * 50)
        print(
            f"训练集: {self.stats['train_sequences']} 个序列，{self.stats['train_images']} 张图像"
        )
        print(
            f"验证集: {self.stats['val_sequences']} 个序列，{self.stats['val_images']} 张图像"
        )
        denominator = self.stats["train_images"] + self.stats["val_images"]
        if denominator > 0:
            train_p = self.stats["train_images"] / denominator * 100
            val_p = self.stats["val_images"] / denominator * 100
            print(f"训练/验证比例: {train_p:.1f}% / {val_p:.1f}%")
        print("=" * 50)

    def create_yaml(self):
        """
        生成 YOLO 训练所需的 `data.yaml` 配置文件。

        当前任务只保留一个检测类别：`vehicle`。
        """
        # path 使用绝对路径，避免从不同工作目录启动 YOLO 训练时路径解析混乱。
        yaml_content = (
            "\n".join(
                [
                    f"path: {os.path.abspath(self.output_dir)}",
                    "train: images/train",
                    "val: images/val",
                    "nc: 1",
                    "names: ['vehicle']",
                ]
            )
            + "\n"
        )
        with open(
            os.path.join(self.output_dir, "data.yaml"), "w", encoding="utf-8"
        ) as f:
            f.write(yaml_content)
