#!/usr/bin/env python3
"""
模型导出脚本
Model Export Script

本模块负责把训练好的统一多任务模型导出为 ONNX，并给出 TensorRT INT8 构建
示例和 Jetson 部署建议，方便后续在边缘设备上完成部署验证。

本文件的定位并非完整部署框架，而是面向部署准备阶段的辅助工具，主要负责：
- 将当前 PyTorch 模型稳定导出为 ONNX；
- 提供 TensorRT INT8 引擎构建的参考代码；
- 给出 Jetson 平台部署验证时的若干注意事项。
"""

import os
import sys

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import onnx  # noqa: F401
except ImportError:
    onnx = None

from src.config import Config
from src.model import UnifiedMultiTaskModel
from src.utils import load_model_weights, resolve_model_weights


def export_qat_onnx(weights_path, onnx_path, device="cpu"):
    """
    将当前统一模型导出为 ONNX 文件。

    Args:
        weights_path: 待加载的权重文件路径；若为空或不存在，则使用随机初始化权重导出。
        onnx_path: 导出的 ONNX 文件保存路径。
        device: 导出时使用的设备，默认是 CPU。

    当前导出的是统一模型的三路输出：
    - 检测输出；
    - 雾分类输出；
    - beta 回归输出。

    这意味着后续部署侧不仅要做检测后处理，也要接住分类和回归两个分支。
    """
    if onnx is None:
        raise ImportError("ONNX export requires the 'onnx' package. Install it with `pip install onnx`.")

    cfg = Config()
    model = UnifiedMultiTaskModel(
        cfg.YOLO_BASE_MODEL,
        cfg.NUM_FOG_CLASSES,
        num_det_classes=cfg.NUM_DET_CLASSES,
    )

    # 如有可用权重，则优先加载训练好的模型参数；否则允许导出随机初始化版本。
    # 后者主要用于图结构验证或部署链路联调，不适合作为真实效果模型。
    if weights_path and os.path.exists(weights_path):
        report = load_model_weights(model, weights_path, map_location=device)
        skipped_yolo_keys = [
            key for key in report.get("skipped_mismatched_keys", []) if key.startswith("yolo.")
        ]
        print(f"Loaded weights from: {weights_path} ({report['source_type']})")
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
                "The exported ONNX graph will use the current single-class model definition, "
                "but box quality requires a checkpoint re-trained with `NUM_DET_CLASSES = 1`."
            )
    else:
        print(f"Weights were not found at {weights_path}, exporting with random initialization.")

    model.eval()
    model.to(device)

    # 构造一份与训练输入尺寸一致的虚拟样本，用于追踪计算图。
    dummy_input = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(device)
    print("Exporting ONNX...")
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
        input_names=["images"],
        output_names=["output0", "fog_cls", "fog_reg"],
        # 仅 batch 维保持动态，空间尺寸当前仍固定为训练输入尺寸。
        dynamic_axes={
            "images": {0: "batch_size"},
            "output0": {0: "batch_size"},
            "fog_cls": {0: "batch_size"},
            "fog_reg": {0: "batch_size"},
        },
    )
    print(f"ONNX export succeeded: {onnx_path}")


def get_trt_int8_config_example():
    """
    返回一个 TensorRT INT8 引擎构建示例代码片段。

    Returns:
        str: 可直接参考的 TensorRT Python API 示例。

    这里返回字符串而不是直接执行，是因为 TensorRT 环境通常与当前开发机环境不同，
    更适合把示例打印出来，交给目标部署环境参考执行。
    """
    cfg = Config()
    code_snippet = f'''
import tensorrt as trt

def build_int8_engine(onnx_file_path, engine_file_path):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    if builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("INT8 is enabled.")
    else:
        print("Platform does not support fast INT8, falling back to FP16.")
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, {cfg.IMG_SIZE}, {cfg.IMG_SIZE}), (4, 3, {cfg.IMG_SIZE}, {cfg.IMG_SIZE}), (8, 3, {cfg.IMG_SIZE}, {cfg.IMG_SIZE}))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
'''
    return code_snippet


def print_jetson_deployment_tips(onnx_path):
    """
    打印 Jetson 部署建议。

    Args:
        onnx_path: 已导出的 ONNX 文件路径，用于拼装演示命令。
    """
    print("\n" + "=" * 80)
    print("NVIDIA Jetson deployment tips")
    print("=" * 80)
    print("1. Quick validation with trtexec:")
    print(f"   trtexec --onnx={onnx_path} --saveEngine=model_int8.engine --int8 --fp16 --verbose")
    print("\n2. Environment:")
    print("   - Make sure JetPack and TensorRT versions match the target device.")
    print("   - Install pycuda or the official TensorRT Python API if you use Python inference.")
    print("\n3. Runtime checks:")
    print("   - Use 'jetson_clocks' for max-performance mode.")
    print("   - Validate detection, classification and regression outputs separately.")
    print("=" * 80 + "\n")


def main():
    """执行 ONNX 导出与部署提示输出流程。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = Config()

    # 自动解析最合适的权重文件，优先使用正式导出的模型文件。
    weights_path = resolve_model_weights(
        cfg.OUTPUT_DIR,
        cfg.CHECKPOINT_DIR,
        preferred_files=["unified_model.pt", "unified_model_best.pt"],
    )
    output_onnx = os.path.join(base_dir, "..", "outputs", "unified_multitask.onnx")
    os.makedirs(os.path.dirname(output_onnx), exist_ok=True)

    export_qat_onnx(weights_path, output_onnx, device="cpu")

    print("\nTensorRT INT8 build example (Python API):")
    print(get_trt_int8_config_example())
    print_jetson_deployment_tips(output_onnx)


if __name__ == "__main__":
    main()


