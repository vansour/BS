#!/usr/bin/env python3
"""
生成项目 Python 源码逐行说明文档。

本脚本用于自动生成“项目 Python 源码逐行说明文档”，产出包括：
1. 根目录总文档：`PYTHON_LINE_BY_LINE_EXPLANATION.md`；
2. 拆分文档目录：`docs/python-line-by-line/`。

该脚本的核心目标不是生成面向机器的 API 文档，而是生成面向人工阅读的
教学型说明文档，使项目源码能够以“文件职责 + 关键定义 + 逐行解释”的形式
被系统化整理，便于课程答辩、项目交接与代码讲解。

注意事项如下：
- 仅覆盖项目自写 Python 文件；
- 不覆盖 `python/` 目录中的解释器、标准库与第三方包。
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMBINED_PATH = ROOT / "PYTHON_LINE_BY_LINE_EXPLANATION.md"
SPLIT_ROOT = ROOT / "docs" / "python-line-by-line"

# TARGET_FILES 定义需要纳入文档生成的源码范围。
# 当前策略是：
# - 显式纳入根目录下的重要脚本；
# - 递归纳入 `src/` 目录中全部 Python 源文件。
TARGET_FILES = [
    ROOT / "config.py",
    *sorted((ROOT / "src").rglob("*.py")),
]


MODULE_DESCRIPTIONS = {
    "config.py": "顶层兼容模块。它本身不承载业务逻辑，而是为了兼容旧版 checkpoint 中的模块路径 `config.Config`，把真正的配置类从 `src.config` 重新导出。",
    "src/__init__.py": "项目顶层包初始化文件。它统一暴露最常用的配置、数据、模型、推理与工具接口，方便外部直接从 `src` 导入。",
    "src/config.py": "项目的全局配置中心，集中保存路径、超参数、物理参数、设备配置和输出目录，是训练、推理、导出三条主链共用的配置源。",
    "src/data/__init__.py": "数据子包初始化文件，负责把数据集、深度估计器和数据准备器统一导出。",
    "src/data/dataset.py": "多任务训练数据集定义。它负责读取晴天图像和对应的深度缓存，并把二者整理成训练时需要的张量对。",
    "src/data/depth_estimator.py": "深度估计模块。它基于 MiDaS 估计图像深度，并支持批量预计算与磁盘缓存。",
    "src/data/preparer.py": "检测数据准备模块。它把 UA-DETRAC 的 XML 标注和离线雾图整理成 YOLO 可用的数据集目录。",
    "src/export.py": "模型导出模块。它负责选择权重、构建模型、导出 ONNX，并附带 TensorRT/Jetson 部署提示。",
    "src/inference.py": "推理运行模块。它封装了视频输入、模型加载、图像预处理、前向推理、结果平滑与可视化展示流程。",
    "src/model/__init__.py": "模型子包初始化文件，统一导出雾增强模块和统一多任务模型。",
    "src/model/fog_augmentation.py": "在线造雾增强模块。它根据深度图和大气散射思想，在训练时即时构造 clear、uniform、patchy 三类样本。",
    "src/model/unified_model.py": "统一多任务模型定义。它使用 YOLOv11s 作为共享主干，在高层特征上挂接雾分类头和 beta 回归头，并保留检测输出接口。",
    "src/train.py": "训练入口脚本。它负责训练组件构建、深度缓存预计算、FP32 训练、QAT 训练、checkpoint 管理和最终模型保存。",
    "src/utils.py": "通用工具函数集合，包含随机种子设置、参数统计、时间格式化、CUDA 内存检查、权重定位与加载等辅助逻辑。",
}


IMPORT_HINTS = {
    "os": "处理操作系统路径、文件和目录操作",
    "sys": "处理 Python 运行时路径与解释器环境",
    "multiprocessing": "处理多进程启动方式设置",
    "random": "生成随机数，用于数据划分或随机增强",
    "re": "处理正则表达式匹配",
    "shutil": "处理文件复制等高级文件操作",
    "xml": "解析 XML 标注文件",
    "Path": "以面向对象方式处理路径",
    "torch": "提供 PyTorch 张量、模型和训练能力",
    "nn": "提供神经网络层与损失函数",
    "optim": "提供优化器实现",
    "DataLoader": "按批次加载数据",
    "Dataset": "定义自定义数据集基类",
    "transforms": "定义图像预处理和数据变换流程",
    "Image": "负责图像读取与图像对象转换",
    "np": "负责数组计算与深度图缓存处理",
    "numpy": "负责数组计算与数值处理",
    "cv2": "负责图像和视频读写、可视化绘制",
    "tqdm": "显示训练或预处理进度条",
    "YOLO": "加载 Ultralytics 的 YOLO 模型",
    "QuantStub": "量化感知训练中用于插入量化桩",
    "DeQuantStub": "量化感知训练中用于插入反量化桩",
    "Presentation": "创建和保存 PPT 演示文稿",
    "ChartData": "构建图表数据源",
    "RGBColor": "表示 PPT 中的 RGB 颜色值",
    "XL_CHART_TYPE": "指定图表类型",
    "XL_LEGEND_POSITION": "指定图表图例位置",
    "MSO_AUTO_SHAPE_TYPE": "指定 PPT 自选图形类型",
    "MSO_ANCHOR": "指定文本框垂直对齐方式",
    "PP_ALIGN": "指定段落水平对齐方式",
    "Inches": "把英寸换算为 PPT 坐标单位",
    "Pt": "把字号点数换算为 PPT 文本大小",
    "Optional": "类型注解，表示值可以为空",
    "Tuple": "类型注解，表示固定结构元组",
    "Union": "类型注解，表示联合类型",
}


VAR_HINTS = {
    "ROOT": "表示当前脚本所在的项目根目录路径对象",
    "OUTPUT_PATH": "表示最终生成的 PPT 输出文件路径",
    "SLIDE_WIDTH": "表示整份 PPT 的页面宽度",
    "SLIDE_HEIGHT": "表示整份 PPT 的页面高度",
    "FONT_HEAD": "表示标题字体名称",
    "FONT_BODY": "表示正文字体名称",
    "FONT_CODE": "表示代码字体名称",
    "COLORS": "保存整套 PPT 的颜色主题字典",
    "FACTS": "保存用于答辩展示的统计事实和产物路径",
    "project_root": "表示项目根目录，用来确保从脚本运行时也能正确导入 src 包",
    "cfg": "表示配置对象实例，集中携带路径与超参数",
    "device": "表示模型或张量运行的设备，例如 cuda 或 cpu",
    "model": "表示当前构建或加载的模型对象",
    "fog_augmenter": "表示在线造雾增强模块实例",
    "train_loader": "表示训练数据加载器",
    "criterion_cls": "表示分类损失函数",
    "criterion_reg": "表示回归损失函数",
    "optimizer": "表示优化器对象",
    "scheduler": "表示学习率调度器对象",
    "scaler": "表示 AMP 混合精度梯度缩放器",
    "best_loss": "表示历史最优损失值",
    "best_model_path": "表示最佳模型保存路径",
    "img_tensor": "表示经过预处理后的图像张量",
    "outputs": "表示模型输出元组或后处理输入",
    "probs": "表示雾分类概率分布",
    "beta": "表示当前或平滑后的雾浓度估计值",
    "adaptive_conf": "表示根据雾浓度自适应调整后的检测阈值",
    "fog_types": "表示每个样本的雾类别标签",
    "betas": "表示每个样本对应的 beta 参数",
    "foggy_images": "表示经过雾化增强后的图像批次",
}


SYMBOL_DESCRIPTIONS = {
    "config.py::Config": "兼容导出的配置类名，真正定义位于 `src.config.Config`。",
    "config.py::get_default_config": "兼容导出的默认配置工厂函数，真正定义位于 `src.config.get_default_config`。",
    "src/config.py::Config": "项目全局配置类，用类属性保存路径、超参数、物理参数和设备设置。",
    "src/config.py::Config.__repr__": "定义配置对象的字符串表现形式，便于打印调试。",
    "src/config.py::get_default_config": "返回默认配置实例，作为便捷工厂函数。",
    "src/data/dataset.py::MultiTaskDataset": "训练数据集类，负责把原图和深度缓存拼装成模型输入。",
    "src/data/dataset.py::MultiTaskDataset.__init__": "初始化数据集，扫描序列目录并按训练/验证划分样本。",
    "src/data/dataset.py::MultiTaskDataset.__len__": "返回样本总数，供 DataLoader 查询数据集长度。",
    "src/data/dataset.py::MultiTaskDataset.__getitem__": "读取单个样本，返回图像张量和深度张量。",
    "src/data/depth_estimator.py::DepthEstimator": "MiDaS 深度估计器类。",
    "src/data/depth_estimator.py::DepthEstimator.__init__": "加载 MiDaS 模型与配套预处理变换。",
    "src/data/depth_estimator.py::DepthEstimator.compute_depth": "对单张 RGB 图像计算深度图。",
    "src/data/depth_estimator.py::DepthEstimator.compute_depth_batch": "按批处理多张图像的深度估计。",
    "src/data/depth_estimator.py::precompute_depths": "遍历整个数据集，预生成并缓存深度图文件。",
    "src/data/preparer.py::DatasetPreparer": "检测数据集准备类。",
    "src/data/preparer.py::DatasetPreparer.__init__": "初始化 XML、雾图目录和输出目录等参数。",
    "src/data/preparer.py::DatasetPreparer.convert_box": "把绝对坐标框转换成 YOLO 归一化坐标格式。",
    "src/data/preparer.py::DatasetPreparer.parse_xml_sequence": "解析单个序列的 XML 标注。",
    "src/data/preparer.py::DatasetPreparer.create_structure": "创建 YOLO 数据集所需目录结构。",
    "src/data/preparer.py::DatasetPreparer.process": "执行完整的数据整理流程。",
    "src/data/preparer.py::DatasetPreparer._print_stats": "打印数据集转换后的统计信息。",
    "src/data/preparer.py::DatasetPreparer.create_yaml": "生成 YOLO 训练所需的 data.yaml。",
    "src/model/fog_augmentation.py::FogAugmentation": "在线雾增强模块类。",
    "src/model/fog_augmentation.py::FogAugmentation.__init__": "保存配置对象，供增强时读取 beta 与 A 范围。",
    "src/model/fog_augmentation.py::FogAugmentation.forward": "根据深度图实时生成三类雾化样本与标签。",
    "src/model/fog_augmentation.py::FogAugmentation.__repr__": "返回增强模块的字符串表示，方便打印参数范围。",
    "src/model/unified_model.py::UnifiedMultiTaskModel": "统一多任务模型类。",
    "src/model/unified_model.py::UnifiedMultiTaskModel.__init__": "加载 YOLO 主干并构建雾分类头和回归头。",
    "src/model/unified_model.py::UnifiedMultiTaskModel._extract_detection_tensor": "从 YOLO 输出结构中提取真正的检测张量。",
    "src/model/unified_model.py::UnifiedMultiTaskModel._detect_feature_dimension": "通过一次虚拟前向自动探测高层特征维度。",
    "src/model/unified_model.py::UnifiedMultiTaskModel.forward": "执行前向传播，输出检测特征、分类结果和回归结果。",
    "src/model/unified_model.py::UnifiedMultiTaskModel.fuse_model": "在 QAT 前融合可融合层。",
    "src/model/unified_model.py::UnifiedMultiTaskModel.__repr__": "返回模型的简要结构说明字符串。",
    "src/train.py::build_cfg_snapshot": "把配置对象压缩为可序列化字典，写入 checkpoint。",
    "src/train.py::prune_old_checkpoints": "清理过旧的 checkpoint 文件，控制磁盘占用。",
    "src/train.py::save_checkpoint": "保存训练过程中的 checkpoint。",
    "src/train.py::load_checkpoint": "加载已有 checkpoint，以便断点续训。",
    "src/train.py::build_train_components": "构建训练所需的数据集、变换和损失函数。",
    "src/train.py::train_epoch": "执行一个 epoch 的训练循环。",
    "src/train.py::train": "训练主入口，串起 FP32、QAT 与 INT8 转换流程。",
    "src/inference.py::HighwayFogSystem": "推理系统封装类。",
    "src/inference.py::HighwayFogSystem.__init__": "初始化配置、模型、视频源和推理状态。",
    "src/inference.py::HighwayFogSystem._resolve_model_path": "解析实际可用的权重路径。",
    "src/inference.py::HighwayFogSystem._load_model": "加载模型权重，必要时回退到随机初始化。",
    "src/inference.py::HighwayFogSystem._preprocess_async": "执行图像预处理并可绑定到 CUDA 流。",
    "src/inference.py::HighwayFogSystem._inference_async": "执行模型前向推理并可绑定到 CUDA 流。",
    "src/inference.py::HighwayFogSystem._postprocess_async": "把模型输出转成概率、beta 等更易消费的结果。",
    "src/inference.py::HighwayFogSystem.predict": "对单帧执行同步预测。",
    "src/inference.py::HighwayFogSystem.run": "执行实时视频推理和窗口显示。",
    "src/inference.py::main": "推理脚本主入口。",
    "src/export.py::export_qat_onnx": "将当前模型导出为 ONNX 文件。",
    "src/export.py::get_trt_int8_config_example": "返回 TensorRT INT8 引擎构建示例代码。",
    "src/export.py::build_int8_engine": "该名字出现在示例代码字符串中，用来示意如何构建 TensorRT 引擎。",
    "src/export.py::print_jetson_deployment_tips": "打印 Jetson 部署建议。",
    "src/export.py::main": "导出脚本主入口。",
    "src/utils.py::set_seed": "设置多种随机种子，保证实验可复现。",
    "src/utils.py::count_parameters": "统计模型可训练参数总量。",
    "src/utils.py::format_time": "把秒数格式化为时分秒字符串。",
    "src/utils.py::check_cuda_memory": "查询当前 CUDA 显存使用情况。",
    "src/utils.py::print_cuda_memory": "以可读格式打印显存信息。",
    "src/utils.py::find_latest_checkpoint": "在目录中查找最新的 checkpoint。",
    "src/utils.py::resolve_model_weights": "按优先级推断最适合使用的权重文件。",
    "src/utils.py::load_model_weights": "把 state dict 或 checkpoint 加载进模型。",
}


def rel_key(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def escape_md(text: str) -> str:
    if text == "":
        return "（空行）"
    return text.replace("`", "\\`").replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


class BlockCollector(ast.NodeVisitor):
    """
    基于 AST 的代码块收集器。

    该类用于在源码抽象语法树中提取类、函数及其文档字符串范围，
    为后续“逐行解释”阶段提供结构化上下文信息。
    """

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.blocks: list[dict[str, object]] = []
        self.doc_ranges: list[tuple[int, int, str]] = []

    def record_doc(self, owner: ast.AST, qualname: str) -> None:
        body = getattr(owner, "body", None) or []
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                self.doc_ranges.append((body[0].lineno, body[0].end_lineno, qualname))

    def visit_Module(self, node: ast.Module) -> None:
        self.record_doc(node, "<module>")
        self.generic_visit(node)

    def _visit_block(self, node: ast.AST, kind: str, name: str) -> None:
        qualname = ".".join(self.stack + [name]) if self.stack else name
        self.blocks.append(
            {
                "name": name,
                "qualname": qualname,
                "type": kind,
                "lineno": node.lineno,
                "end_lineno": node.end_lineno,
            }
        )
        self.record_doc(node, qualname)
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_block(node, "class", node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_block(node, "function", node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_block(node, "function", node.name)


def collect_metadata(path: Path):
    """
    读取单个源码文件并提取结构化元数据。

    Args:
        path: 目标源码文件路径。

    Returns:
        tuple: 源码行列表、代码块信息和文档字符串范围。
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    collector = BlockCollector()
    collector.visit(tree)
    return source.splitlines(), collector.blocks, collector.doc_ranges


def describe_symbol(path_key: str, block: dict[str, object]) -> str:
    key = f"{path_key}::{block['qualname']}"
    if key in SYMBOL_DESCRIPTIONS:
        return SYMBOL_DESCRIPTIONS[key]

    name = str(block["name"])
    block_type = str(block["type"])

    if block_type == "class":
        return f"定义类 `{name}`，用于封装与 `{name}` 名称相关的状态和方法。"
    if name == "__init__":
        return "定义初始化方法，负责在对象创建时建立基础属性和初始状态。"
    if name.startswith("build_"):
        return f"定义函数 `{name}`，用于构建一页或一组与 `{name[6:]}` 相关的 PPT 内容。"
    if name.startswith("add_"):
        return f"定义函数 `{name}`，用于向当前幻灯片添加某种视觉元素或文本元素。"
    if name.startswith("get_"):
        return f"定义函数 `{name}`，用于获取或推导某项结果。"
    if name.startswith("load_"):
        return f"定义函数 `{name}`，用于加载外部资源、配置或权重。"
    if name.startswith("save_"):
        return f"定义函数 `{name}`，用于保存文件、模型或中间状态。"
    if name.startswith("print_"):
        return f"定义函数 `{name}`，用于向终端打印说明信息。"
    if name.startswith("resolve_"):
        return f"定义函数 `{name}`，用于解析并确定某个最终可用值。"
    if name.startswith("format_"):
        return f"定义函数 `{name}`，用于格式化某个输入结果。"
    if name.startswith("count_"):
        return f"定义函数 `{name}`，用于统计数量或规模。"
    if name.startswith("precompute_"):
        return f"定义函数 `{name}`，用于提前批量计算并缓存结果。"
    if name.startswith("check_"):
        return f"定义函数 `{name}`，用于检查某个状态或运行条件。"
    if name == "main":
        return "定义脚本主入口函数，用于在直接运行该文件时启动主流程。"
    return f"定义函数 `{name}`，承担当前模块中的一个独立子任务。"


def get_innermost_block(line_no: int, blocks: list[dict[str, object]]):
    candidates = [block for block in blocks if block["lineno"] <= line_no <= block["end_lineno"]]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item["end_lineno"] - item["lineno"], item["lineno"]))[0]


def doc_range_for_line(line_no: int, doc_ranges: list[tuple[int, int, str]]):
    for start, end, qualname in doc_ranges:
        if start <= line_no <= end:
            return start, end, qualname
    return None


def block_label(block) -> str:
    if block is None:
        return "模块级逻辑"
    kind = "类" if block["type"] == "class" else "函数"
    return f"{kind} `{block['qualname']}`"


def explain_import(stripped: str) -> str:
    if stripped.startswith("from "):
        match = re.match(r"from\s+([^\s]+)\s+import\s+(.+)", stripped)
        if match:
            module = match.group(1)
            names = match.group(2)
            base = module.split(".")[0]
            hint = IMPORT_HINTS.get(base, IMPORT_HINTS.get(module, "为后续代码引入外部名称"))
            return f"这是 `from ... import ...` 导入语句，从模块 `{module}` 中引入 `{names}`；它的目的通常是{hint}。"
    else:
        match = re.match(r"import\s+(.+)", stripped)
        if match:
            names = match.group(1)
            base = names.split(",")[0].strip().split(" as ")[0].split(".")[0]
            hint = IMPORT_HINTS.get(base, "为后续逻辑引入外部模块")
            return f"这是普通 `import` 导入语句，引入 `{names}`；它的用途是{hint}。"
    return "这是导入语句，用来把外部模块或名称引入当前文件作用域。"


def explain_assignment(stripped: str, block) -> str:
    left = stripped.split("=", 1)[0].strip()
    hint = VAR_HINTS.get(left)
    prefix = f"这行位于{block_label(block)}内部，" if block else "这是一条模块级赋值语句，"
    if hint:
        return prefix + f"它给 `{left}` 赋值，{hint}。"
    if left.startswith("self."):
        return prefix + f"它给实例属性 `{left}` 赋值，用来把初始化或计算结果保存到对象状态里。"
    if "," in left:
        return prefix + f"它执行多变量解包赋值，把右侧表达式拆分后分别写入 `{left}`。"
    return prefix + f"它给变量 `{left}` 赋值，用于保存后续步骤需要复用的中间结果或配置值。"


def explain_control(stripped: str, block) -> str:
    where = f"在{block_label(block)}中" if block else "在模块级逻辑中"
    if stripped.startswith("if "):
        return f"{where}开始一个 `if` 条件分支，只有条件成立时下面缩进的语句才会执行。"
    if stripped.startswith("elif "):
        return f"{where}追加一个 `elif` 分支，用来在前面的条件不满足时继续判断新的条件。"
    if stripped == "else:":
        return f"{where}进入 `else` 分支，表示当前面条件都不满足时执行备用逻辑。"
    if stripped.startswith("for "):
        return f"{where}开始一个 `for` 循环，用来按顺序遍历集合中的元素并重复执行下面的代码块。"
    if stripped.startswith("while "):
        return f"{where}开始一个 `while` 循环，只要条件保持为真就持续重复执行。"
    if stripped.startswith("with "):
        return f"{where}开始一个 `with` 上下文块，用来安全管理资源、设备状态或临时执行环境。"
    if stripped == "try:":
        return f"{where}开始异常捕获流程中的 `try` 代码块，用来尝试执行可能失败的逻辑。"
    if stripped.startswith("except"):
        return f"{where}定义异常处理分支；当前面 `try` 中抛出匹配异常时，会转而执行这里的处理逻辑。"
    if stripped == "finally:":
        return f"{where}定义 `finally` 分支，表示无论前面成功或失败，最终都会执行这里的收尾逻辑。"
    return f"{where}使用控制流语句组织执行路径。"


def explain_line(path_key: str, line_no: int, line: str, blocks, doc_ranges):
    stripped = line.strip()
    block = get_innermost_block(line_no, blocks)
    doc = doc_range_for_line(line_no, doc_ranges)

    if stripped == "":
        return "空行", "空行本身不执行任何逻辑，它的作用是把相邻代码块分开，帮助读者按功能理解文件结构。"

    if stripped.startswith("#!"):
        return "Shebang", "这是一行 Shebang，说明该文件可以被当作脚本直接执行，并优先使用环境中的 `python3` 解释器。"

    if doc:
        start, end, qualname = doc
        if start == end:
            pos = "单行文档字符串"
        elif line_no == start:
            pos = "文档字符串起始行"
        elif line_no == end:
            pos = "文档字符串结束行"
        else:
            pos = "文档字符串内容行"

        owner = "模块" if qualname == "<module>" else f"定义 `{qualname}`"
        return pos, f"这行属于{owner}的文档字符串，用自然语言说明该模块或定义的用途、输入输出、设计意图或使用方式。"

    if stripped.startswith("#"):
        return "注释", "这是一行注释，不参与程序执行，主要用于给开发者解释下面代码块的目的、阶段或注意事项。"

    for item in blocks:
        if item["lineno"] == line_no:
            if item["type"] == "class":
                return "类定义", describe_symbol(path_key, item)
            return "函数定义", describe_symbol(path_key, item)

    if stripped.startswith("from ") or stripped.startswith("import "):
        return "导入语句", explain_import(stripped)

    if stripped.startswith("@"):
        return "装饰器", "这是一行装饰器语法，用于在不改动函数主体的情况下，为下面定义的函数或方法附加额外行为。"

    if re.match(r"^[A-Za-z_][A-Za-z0-9_\.,\s]*\s*=\s*.+", stripped):
        return "赋值语句", explain_assignment(stripped, block)

    if stripped.startswith("return"):
        if block:
            return "返回语句", f"这行位于{block_label(block)}内部，用 `return` 把当前计算结果返回给调用方，并结束当前函数执行。"
        return "返回语句", "这是一条返回语句，用于把结果交回调用方。"

    if stripped.startswith("raise "):
        return "抛出异常", "这行主动抛出异常；当运行条件不满足或关键资源缺失时，用它中断当前流程并向上层报告错误。"

    if any(stripped.startswith(item) for item in ["if ", "elif ", "else:", "for ", "while ", "with ", "try:", "except", "finally:"]):
        return "控制流语句", explain_control(stripped, block)

    if stripped.startswith("print("):
        return "打印输出", "这行向终端输出信息，用于提示运行状态、调试结果、错误信息或阶段完成情况。"

    if stripped.startswith("torch.save("):
        return "模型保存调用", "这行调用 `torch.save` 把模型权重或 checkpoint 写入磁盘，是训练与导出的关键持久化步骤。"

    if stripped.startswith("torch.load("):
        return "模型加载调用", "这行调用 `torch.load` 从磁盘读取权重或 checkpoint，以便恢复训练状态或加载模型参数。"

    if stripped.startswith("os.makedirs("):
        return "目录创建调用", "这行确保目标目录存在；如果目录不存在就自动创建，从而避免后续写文件时报路径错误。"

    if stripped.startswith("np.save("):
        return "NumPy 持久化调用", "这行把数组保存为 `.npy` 文件，用于缓存深度图等中间结果。"

    if stripped.startswith("cv2."):
        return "OpenCV 调用", "这行调用 OpenCV 的图像或视频处理接口，用于读写、绘制、显示或颜色空间转换。"

    if stripped.startswith("torch."):
        return "PyTorch 调用", "这行直接调用 PyTorch 的接口来完成张量运算、上下文控制、模型执行或训练辅助操作。"

    if stripped in {")", "]", "}", "),", "],", "},"}:
        return "结构收尾", "这行用于结束上一条多行表达式、容器定义或函数调用，使语法结构闭合。"

    if re.match(r'^[\"\\\"][^\"\\\n]+[\"\\\"]\s*:', stripped) or re.match(r"^'[^'\n]+'\s*:", stripped):
        key = stripped.split(":", 1)[0].strip()
        return "字典项", f"这行向前面的字典字面量中补充键 `{key}` 对应的值，用来继续定义配置、颜色或统计信息。"

    if stripped.endswith("("):
        return "调用起始行", "这行通常是一条多行函数调用或对象构造的起始行，下面若干缩进的参数行会共同组成完整调用。"

    if stripped.endswith("):"):
        return "语法头部", "这行是一个以冒号结尾的语法头部，下面的缩进块会作为其主体逻辑执行。"

    if stripped.endswith(","):
        return "续行", "这行通常是多行参数列表、列表项、字典项或元组项中的一部分，逗号表示后面还有内容继续。"

    context = block_label(block) if block else "模块级逻辑"
    return "普通语句", f"这行属于{context}内部的一条普通执行语句，用来推进当前步骤的计算、调用或状态更新。"


def generate_combined_document(metadata):
    """
    生成总版逐行说明文档。

    Args:
        metadata: 所有目标文件的结构化元数据集合。
    """
    with COMBINED_PATH.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write("# 项目 Python 源码逐行详解文档\n\n")
        handle.write("## 说明\n\n")
        handle.write("本文档面向 **项目自写 Python 源码** 进行逐文件、逐行解释。为避免把项目内置运行时中的标准库与第三方包也纳入文档，本文档 **不覆盖** [python/](./python) 目录，仅覆盖根目录和 [src/](./src) 目录下与本毕业设计直接相关的 15 个 Python 文件。\n\n")
        total_lines = sum(len(lines) for _, lines, _, _ in metadata)
        handle.write(f"- 覆盖文件数：`{len(metadata)}`\n")
        handle.write(f"- 覆盖代码总行数：`{total_lines}`\n")
        handle.write("- 文档组织方式：先给出文件职责、关键定义，再对每一行给出“原代码 + 行类型 + 详细解释”。\n")
        handle.write("- 行号约定：与当前仓库中的源码真实行号一一对应。\n\n")

        handle.write("## 覆盖文件清单\n\n")
        for path, lines, _, _ in metadata:
            handle.write(f"- [{rel_key(path)}](./{rel_key(path)})：`{len(lines)}` 行\n")
        handle.write("\n")

        for path, lines, blocks, doc_ranges in metadata:
            path_key = rel_key(path)
            handle.write(f"## 文件：`{path_key}`\n\n")
            handle.write(f"**文件职责**：{MODULE_DESCRIPTIONS.get(path_key, '该文件承担项目中的一部分业务逻辑。')}\n\n")
            handle.write(f"**总行数**：`{len(lines)}`\n\n")

            if blocks:
                handle.write("**关键定义清单**\n\n")
                for block in blocks:
                    desc = describe_symbol(path_key, block)
                    handle.write(f"- `{block['qualname']}`（L{block['lineno']}-L{block['end_lineno']}）：{desc}\n")
                handle.write("\n")

            handle.write("### 逐行说明\n\n")
            for index, line in enumerate(lines, 1):
                kind, explanation = explain_line(path_key, index, line, blocks, doc_ranges)
                handle.write(f"- **L{index:04d}**：`{escape_md(line)}`\n")
                handle.write(f"  - 行类型：{kind}\n")
                handle.write(f"  - 详细解释：{explanation}\n")
            handle.write("\n")


def split_combined_document():
    """
    将总版逐行说明文档拆分为按源码路径组织的独立 Markdown 文件。

    拆分后的目录结构与源码目录保持一致，便于按文件逐个查阅。
    """
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)

    lines = COMBINED_PATH.read_text(encoding="utf-8-sig").splitlines()
    sections = []
    current_path = None
    current_lines = []
    for line in lines:
        if line.startswith("## 文件：`") and line.endswith("`"):
            if current_path is not None:
                sections.append((current_path, "\n".join(current_lines).strip() + "\n"))
            current_path = line[len("## 文件：`") : -1]
            current_lines = [line]
        elif current_path is not None:
            current_lines.append(line)

    if current_path is not None:
        sections.append((current_path, "\n".join(current_lines).strip() + "\n"))

    readme_lines = [
        "# Python 逐文件逐行说明索引",
        "",
        "## 说明",
        "",
        "本目录由总文档 [PYTHON_LINE_BY_LINE_EXPLANATION.md](../../PYTHON_LINE_BY_LINE_EXPLANATION.md) 自动拆分而来。",
        "每个 Python 源文件对应一份单独的 Markdown，目录结构与源码路径保持一致，便于查阅。",
        "",
        f"- 拆分文件数：`{len(sections)}`",
        "- 覆盖范围：项目自写 Python 源码，不包含 `python/` 目录中的标准库和第三方包。",
        "",
        "## 文件列表",
        "",
    ]

    for rel_path, body in sections:
        target = SPLIT_ROOT / f"{rel_path}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        source_path = ROOT / rel_path
        source_rel = os.path.relpath(source_path, start=target.parent).replace("\\", "/")
        title = f"# 文件逐行说明：`{rel_path}`\n\n"
        intro = f"源文件：[{rel_path}]({source_rel})\n\n"
        target.write_text(title + intro + body, encoding="utf-8-sig", newline="\n")
        readme_lines.append(f"- [{rel_path}.md](./{rel_path}.md)")

    (SPLIT_ROOT / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8-sig", newline="\n")


def main():
    """
    脚本主入口。

    主流程包括：
    1. 收集所有目标源码文件的结构化信息；
    2. 生成总版逐行说明文档；
    3. 生成拆分后的独立说明文档目录；
    4. 输出覆盖文件数与行数统计。
    """
    metadata = []
    for path in TARGET_FILES:
        lines, blocks, doc_ranges = collect_metadata(path)
        metadata.append((path, lines, blocks, doc_ranges))

    generate_combined_document(metadata)
    split_combined_document()

    print(f"Generated combined document: {COMBINED_PATH}")
    print(f"Generated split docs directory: {SPLIT_ROOT}")
    print(f"Covered files: {len(metadata)}")
    print(f"Covered lines: {sum(len(lines) for _, lines, _, _ in metadata)}")


if __name__ == "__main__":
    main()
