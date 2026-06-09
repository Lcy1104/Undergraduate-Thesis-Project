import subprocess
import yaml
import shutil
from pathlib import Path
import random
import sys
import time
import numpy as np
from typing import Tuple

import openvino as ov
import nncf
from PIL import Image

# ==================== 用户配置 ====================
ONNX_ROOT = r"E:\final_exam_test\handcraft\handcrafted\runs\onnx_batch_test_fp32_pk"
OV_ROOT   = r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models_withoutnms"
MODEL_NAMES = [
    "yolo26s", "yolo26m_laptop", "yolo26n", "yolo26m_com"
    # 如果 yolov8m 也重新导出了，加进来；如果没动，注释掉
]
DATASET_YAML = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml"
NUM_CALIB = 300                     # 校准图数量，300张足够
CALIB_DIR = "calib_images_temp"     # 临时目录
INPUT_SIZE = 640
# =================================================

def check_dependencies():
    print("=" * 70)
    print("🔍 检查依赖...")
    print(f"✅ OpenVINO: {ov.__version__}")
    try:
        print(f"✅ NNCF: {nncf.__version__}")
    except Exception as e:
        print(f"⚠️ NNCF: {e}")
    print("=" * 70)

def extract_calib_images() -> Tuple[str, int]:
    print("\n📂 提取校准图片...")
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
                print(f"✅ {split}: {len(images)} 张")

    num_samples = min(NUM_CALIB, len(all_images))
    selected = random.sample(all_images, num_samples)

    calib_path = Path(CALIB_DIR)
    if calib_path.exists():
        shutil.rmtree(calib_path)
    calib_path.mkdir(exist_ok=True)

    for idx, img_path in enumerate(selected):
        dest = calib_path / f"calib_{idx:04d}{img_path.suffix}"
        shutil.copy2(img_path, dest)

    print(f"✅ 校准数据: {num_samples} 张")
    return str(calib_path), num_samples

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
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

def make_calib_loader(calib_dir: str, input_name: str, input_size: int):
    """返回 NNCF 可用的校准数据迭代器"""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(calib_dir).glob(ext))
    print(f"  校准图片: {len(image_files)} 张")

    def loader():
        for img_path in image_files:
            img = preprocess_image(str(img_path), (input_size, input_size))
            if img is not None:
                yield {input_name: img}
    return loader()

def convert_onnx_to_ir(onnx_path: Path, output_dir: Path, model_name: str):
    """ONNX -> FP32 + FP16 IR"""
    print(f"\n  ONNX -> IR: {onnx_path.name}")
    ov_model = ov.convert_model(str(onnx_path))
    print(f"    输入: {ov_model.input(0).shape}, 输出: {ov_model.output(0).shape}")

    # FP32
    fp32_dir = output_dir / "FP32"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    fp32_xml = fp32_dir / f"{model_name}.xml"
    ov.save_model(ov_model, str(fp32_xml), compress_to_fp16=False)
    print(f"    ✅ FP32: {fp32_xml}")

    # FP16
    fp16_dir = output_dir / "FP16"
    fp16_dir.mkdir(parents=True, exist_ok=True)
    fp16_xml = fp16_dir / f"{model_name}.xml"
    ov.save_model(ov_model, str(fp16_xml), compress_to_fp16=True)
    print(f"    ✅ FP16: {fp16_xml}")

    return fp32_xml

def quantize_ir_to_int8(fp32_xml: Path, calib_loader, output_dir: Path, model_name: str):
    """FP32 IR -> INT8"""
    print(f"\n  INT8 量化: {model_name}")
    core = ov.Core()
    model = core.read_model(str(fp32_xml))
    input_name = model.input(0).get_any_name()
    print(f"    输入节点: {input_name}")

    calibration_dataset = nncf.Dataset(calib_loader)

    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        subset_size=min(300, NUM_CALIB),
        fast_bias_correction=False,  # 对 GPU 数值稳定性更好
    )

    int8_dir = output_dir / "INT8"
    int8_dir.mkdir(parents=True, exist_ok=True)
    int8_xml = int8_dir / f"{model_name}.xml"
    ov.save_model(quantized_model, str(int8_xml))
    print(f"    ✅ INT8: {int8_xml}")

def main():
    print("=" * 70)
    print("🚀 重新转换 ONNX → OpenVINO IR (FP32/FP16/INT8)")
    print("=" * 70)
    check_dependencies()

    # 1. 准备校准数据
    calib_dir, _ = extract_calib_images()

    # 2. 逐个模型处理
    for model_name in MODEL_NAMES:
        print(f"\n{'='*70}")
        print(f"📦 {model_name}")
        print(f"{'='*70}")

        onnx_path = Path(ONNX_ROOT) / model_name / f"{model_name}.onnx"
        if not onnx_path.exists():
            print(f"  ❌ ONNX 不存在: {onnx_path}")
            continue

        output_dir = Path(OV_ROOT) / model_name

        # 清理旧 IR，避免混淆
        if output_dir.exists():
            print(f"  🧹 清理旧 IR...")
            shutil.rmtree(output_dir)

        try:
            # 转换 FP32/FP16
            fp32_xml = convert_onnx_to_ir(onnx_path, output_dir, model_name)

            # 获取输入名，创建校准加载器
            core = ov.Core()
            tmp_model = core.read_model(str(fp32_xml))
            input_name = tmp_model.input(0).get_any_name()
            calib_loader = make_calib_loader(calib_dir, input_name, INPUT_SIZE)

            # INT8 量化
            quantize_ir_to_int8(fp32_xml, calib_loader, output_dir, model_name)

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()

    # 清理临时目录
    if Path(calib_dir).exists():
        shutil.rmtree(calib_dir)
        print(f"\n🧹 已清理临时校准目录")

    print("\n🎉 全部完成！")

if __name__ == "__main__":
    main()