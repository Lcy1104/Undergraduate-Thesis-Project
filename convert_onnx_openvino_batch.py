import subprocess
import yaml
import shutil
from pathlib import Path
import random
import json
import sys
import time
import numpy as np
from typing import Tuple, List, Callable, Generator

# OpenVINO 2025.4.1 导入
from openvino import Core
import openvino as ov
import nncf

ONNX_ROOT = r"E:\final_exam_test\handcraft\handcrafted\runs\onnx_batch_test_fp32"
OV_ROOT   = r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models"
MODEL_NAMES = [
    "yolov8m", "yolo26n", "yolo26s", "yolo26m_laptop", "yolo26m_com"
]
DATASET_YAML = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml"
NUM_CALIB = 1652                     # 校准图片数量
CALIB_DIR = "calib_images_temp"     # 临时校准目录（自动创建删除）
INPUT_SIZE = 640                    # 模型输入尺寸

def check_dependencies():
    """检查依赖（参考您的脚本）"""
    print("=" * 70)
    print("🔍 检查依赖...")
    print("=" * 70)
    try:
        import nncf
        print(f"NNCF: {nncf.__version__}")
    except:
        print("安装NNCF...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nncf>=2.14.0"])
        import nncf
        print(f"NNCF安装完成")
    try:
        from PIL import Image
        print("Pillow: 已安装")
    except:
        print("安装Pillow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        print("Pillow安装完成")
    print(f"OpenVINO: {ov.__version__}")
    print("=" * 70)

def extract_calib_images() -> Tuple[str, int]:
    """从数据集中提取校准图片（只复制图片，不需要标注）"""
    print("\n提取校准图片...")
    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    all_images = []
    for split in ["train", "val", "test"]:
        if split in data:
            split_path = Path(data[split]).resolve()
            if split_path.exists():
                img_dir = split_path / "images" if (split_path / "images").exists() else split_path
                images = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png")) + list(img_dir.rglob("*.jpeg"))
                all_images.extend(images)
                print(f"{split}: {len(images)} 张图片")

    if not all_images:
        raise ValueError("未找到任何图片！")

    num_samples = min(NUM_CALIB, len(all_images))
    selected = random.sample(all_images, num_samples)

    calib_path = Path(CALIB_DIR)
    calib_path.mkdir(exist_ok=True)

    for idx, img_path in enumerate(selected):
        dest = calib_path / f"calib_{idx:04d}{img_path.suffix}"
        shutil.copy2(img_path, dest)

    print(f"✅ 提取完成: {num_samples} 张图片")
    return str(calib_path), num_samples

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """YOLO预处理：letterbox + normalize（完全复用您的成功代码）"""
    from PIL import Image
    if image_path.lower().endswith('.txt'):
        return None
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    ratio = min(target_size[0] / img_h, target_size[1] / img_w)
    new_w, new_h = int(img_w * ratio), int(img_h * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    pad_x = (target_size[1] - new_w) // 2
    pad_y = (target_size[0] - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(img)
    chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)

def get_calibration_data_loader(calib_dir: str, input_name: str, input_size: int) -> Generator:
    """返回一个生成器，每次 yield 一个 {input_name: img} 字典，用于 NNCF 校准"""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(calib_dir).glob(ext))
    if not image_files:
        raise ValueError(f"校准目录 {calib_dir} 中没有图片")
    print(f"  校准图片数量: {len(image_files)}")
    def generator():
        for img_path in image_files:
            img = preprocess_image(str(img_path), (input_size, input_size))
            if img is None:
                continue
            yield {input_name: img}
    return generator()

def convert_onnx_to_openvino(onnx_path: Path, output_dir: Path) -> Path:
    """将 ONNX 转换为 FP32 和 FP16 两个版本的 OpenVINO IR，返回 FP32 XML 路径"""
    print(f"  转换 ONNX -> OpenVINO FP32/FP16: {onnx_path.name}")
    # 转换为 FP32 模型
    ov_model = ov.convert_model(str(onnx_path))
    # 保存 FP32
    fp32_dir = output_dir / "FP32"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    fp32_xml = fp32_dir / f"{onnx_path.stem}.xml"
    ov.save_model(ov_model, str(fp32_xml))
    print(f"FP32 保存至: {fp32_xml}")
    # 保存 FP16（通过 compress_to_fp16 参数）
    fp16_dir = output_dir / "FP16"
    fp16_dir.mkdir(parents=True, exist_ok=True)
    fp16_xml = fp16_dir / f"{onnx_path.stem}.xml"
    ov.save_model(ov_model, str(fp16_xml), compress_to_fp16=True)
    print(f"FP16 保存至: {fp16_xml}")
    return fp32_xml

def get_gpu_device(core: Core) -> str:
    """获取可用的 GPU 设备（参考您的脚本）"""
    devices = core.available_devices
    gpu_devices = [d for d in devices if "GPU" in d]
    return gpu_devices[0] if gpu_devices else "CPU"

def quantize_to_int8(fp32_xml: Path, calib_loader: Generator, output_dir: Path, model_name: str):
    """使用 NNCF 对 FP32 IR 进行 INT8 量化"""
    print(f"  INT8 量化: {model_name}")
    core = Core()
    model = core.read_model(str(fp32_xml))
    # 获取输入名称
    try:
        input_name = model.inputs[0].get_any_name()
    except:
        input_name = "images"
    # 创建校准数据集（注意：calib_loader 是一个生成器，直接传给 Dataset）
    calib_dataset = nncf.Dataset(calib_loader)
    # 执行量化
    quantized_model = nncf.quantize(
        model,
        calib_dataset,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        subset_size=NUM_CALIB,
        fast_bias_correction=True,
    )
    # 保存 INT8 模型
    int8_dir = output_dir / "INT8"
    int8_dir.mkdir(parents=True, exist_ok=True)
    out_xml = int8_dir / f"{model_name}.xml"
    ov.save_model(quantized_model, str(out_xml))
    print(f"    INT8 保存至: {out_xml}")

def process_model(model_name: str, calib_loader: Generator = None):
    """处理单个模型：生成 FP32、FP16、INT8"""
    print(f"\n===== 处理模型: {model_name} =====")
    onnx_file = Path(ONNX_ROOT) / model_name / f"{model_name}.onnx"
    if not onnx_file.exists():
        print(f"  错误：ONNX 文件不存在 {onnx_file}")
        return
    output_dir = Path(OV_ROOT) / model_name
    # 1. 转换 FP32 和 FP16
    fp32_xml = convert_onnx_to_openvino(onnx_file, output_dir)
    # 2. 量化 INT8（需要校准数据生成器）
    if calib_loader is None:
        print("  跳过 INT8：未提供校准数据加载器")
    else:
        try:
            # 注意：calib_loader 是生成器，但每次调用后会被耗尽，所以需要为每个模型重新创建
            # 因此我们外部缓存生成器，而是在这里重新创建（通过重新读取校准目录）
            # 简单起见：让调用方传入生成器工厂函数，或者直接重新构建
            # 重新构建生成器（需要知道 input_name）
            # 为简化，要求 calib_loader 是一个可调用的工厂函数，每次返回新的生成器
            # 修改：process_model 接收一个工厂函数
            raise NotImplementedError("需要调整：每个模型应使用独立的生成器")
        except Exception as e:
            print(f"  INT8 量化失败: {e}")

def main():
    print("=" * 70)
    print("批量转换 ONNX → OpenVINO (FP32/FP16/INT8)")
    print("=" * 70)
    check_dependencies()

    # 1. 准备校准数据（如果数据集存在）
    calib_dir = None
    if not Path(DATASET_YAML).exists():
        raise FileNotFoundError(f"错误：需要 INT8 量化但找不到数据集 YAML: {DATASET_YAML}")
    try:
        calib_dir, num_images = extract_calib_images()
        print(f"校准数据准备完成，共 {num_images} 张图片，临时目录: {calib_dir}")
    except Exception as e:
        raise RuntimeError(f"准备校准数据失败: {e}")

    # 2. 获取模型的输入名称（从第一个 ONNX 推断）
    first_model = MODEL_NAMES[0]
    sample_onnx = Path(ONNX_ROOT) / first_model / f"{first_model}.onnx"
    if not sample_onnx.exists():
        raise FileNotFoundError(f"示例模型不存在: {sample_onnx}")
    ov_model = ov.convert_model(str(sample_onnx))
    input_name = ov_model.inputs[0].get_any_name()
    print(f"模型输入名称: {input_name}")

    # 3. 定义生成器工厂函数（每个模型独立创建，避免生成器被耗尽）
    def make_loader():
        return get_calibration_data_loader(calib_dir, input_name, INPUT_SIZE)

    # 4. 逐个处理模型
    for name in MODEL_NAMES:
        process_model_with_loader(name, make_loader)

    # 5. 清理临时校准目录
    if calib_dir and Path(calib_dir).exists():
        shutil.rmtree(calib_dir)
        print(f"\n已清理临时校准目录: {calib_dir}")

    print("\n全部处理完成！")
    print(f"输出根目录: {OV_ROOT}")
    for name in MODEL_NAMES:
        print(f"  └─ {name}/")
        print(f"      ├─ FP32/")
        print(f"      ├─ FP16/")
        print(f"      └─ INT8/ (如果量化成功)")

def process_model_with_loader(model_name: str, loader_factory):
    """处理模型，每次使用新的校准生成器"""
    print(f"\n===== 处理模型: {model_name} =====")
    onnx_file = Path(ONNX_ROOT) / model_name / f"{model_name}.onnx"
    if not onnx_file.exists():
        print(f"  错误：ONNX 文件不存在 {onnx_file}")
        return
    output_dir = Path(OV_ROOT) / model_name
    # 转换 FP32/FP16
    fp32_xml = convert_onnx_to_openvino(onnx_file, output_dir)
    # INT8 量化
    try:
        calib_loader = loader_factory()  # 获取新的生成器
        quantize_to_int8(fp32_xml, calib_loader, output_dir, model_name)
    except Exception as e:
        print(f"  INT8 量化失败: {e}")

if __name__ == "__main__":
    main()