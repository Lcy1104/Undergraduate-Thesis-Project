# convert_int8_2024_3.py
import subprocess
import yaml
import shutil
from pathlib import Path
import random
import json
import sys
import time
import numpy as np
from typing import Tuple, List

# ==================== OpenVINO 2025.4.1 导入 ====================
from openvino import Core
import openvino as ov
import nncf

# ==============================================================

# ==================== 用户配置区域（修改这里） ====================
MODEL_DIR = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights_openvion\best_openvino_model"
DATASET_YAML = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml"
NUM_CALIB = 150
OUTPUT_DIR = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights_openvion\best_int8_model"
OUTPUT_NAME = "best_int8"
CALIB_DIR = "calib_images_2024_3"


# ==============================================================

def check_dependencies():
    """检查依赖"""
    print("=" * 70)
    print("🔍 检查依赖...")
    print("=" * 70)

    try:
        import nncf
        print(f"✅ NNCF: {nncf.__version__}")
    except:
        print("⏳ 安装NNCF...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nncf>=2.14.0"])
        import nncf
        print(f"✅ NNCF安装完成")

    try:
        from PIL import Image
        print("✅ Pillow: 已安装")
    except:
        print("⏳ 安装Pillow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        print("✅ Pillow安装完成")

    print(f"✅ OpenVINO: 2025.4.1")
    print("=" * 70)


def extract_calib_images() -> tuple[str, int, int]:
    """提取校准图片和标注，返回：(目录路径, 图片数量, 标注数量)"""
    print("\n📂 提取校准图片和标注...")

    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    all_images = []
    for split in ["train", "val", "test"]:
        if split in data:
            split_path = Path(data[split]).resolve()
            if split_path.exists():
                # 查找images子目录或直接使用
                img_dir = split_path / "images" if (split_path / "images").exists() else split_path
                txt_dir = split_path / "labels" if (split_path / "labels").exists() else split_path

                images = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png")) + \
                         list(img_dir.rglob("*.jpeg"))

                for img in images:
                    txt = txt_dir / f"{img.stem}.txt"
                    if txt.exists():
                        all_images.append((img, txt, split))

                valid = len([i for i in images if (txt_dir / f'{Path(i).stem}.txt').exists()])
                print(f"✅ {split}: {len(images)} 张图片，{valid} 个有效标注")

    if not all_images:
        raise ValueError("❌ 未找到任何有效图片+标注对！")

    # 随机采样
    num_samples = min(NUM_CALIB, len(all_images))
    selected = random.sample(all_images, num_samples)

    # 复制到校准目录
    calib_path = Path(CALIB_DIR)
    calib_path.mkdir(exist_ok=True)

    for idx, (img_path, txt_path, split) in enumerate(selected):
        base_name = f"{split}_{idx:04d}"

        # 复制图片
        img_dest = calib_path / f"{base_name}{img_path.suffix}"
        shutil.copy2(img_path, img_dest)

        # 复制标注（保留txt用于后续验证）
        txt_dest = calib_path / f"{base_name}.txt"
        shutil.copy2(txt_path, txt_dest)

    print(f"✅ 提取完成: {num_samples} 张图片 + {num_samples} 个标注")
    return str(calib_path), num_samples, num_samples


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """YOLO预处理：letterbox + normalize"""
    from PIL import Image

    # 关键：跳过txt文件
    if image_path.lower().endswith('.txt'):
        print(f"📄 跳过标注文件: {Path(image_path).name}")
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


def get_gpu_device(core: Core) -> str:
    """获取GPU设备"""
    devices = core.available_devices
    gpu_devices = [d for d in devices if "GPU" in d]
    return gpu_devices[0] if gpu_devices else "CPU"


def quantize_model():
    """主量化流程"""
    print("=" * 70)
    print("🚀 OpenVINO 2025.4.1 INT8 量化工具")
    print("=" * 70)

    # 1. 检查依赖
    check_dependencies()

    # 2. 提取校准数据（保留txt文件）
    calib_dir, num_images, num_txts = extract_calib_images()

    # 3. 加载模型
    core = Core()
    model_path = Path(MODEL_DIR)
    xml_path = model_path / "best.xml"
    bin_path = model_path / "best.bin"

    if not xml_path.exists():
        raise FileNotFoundError(f"❌ 模型不存在: {xml_path}")

    model = core.read_model(str(xml_path), str(bin_path))

    # 4. 固定输入形状
    input_shape = tuple(model.inputs[0].get_shape())
    if -1 in input_shape:
        input_shape = (1, 3, 640, 640)
        model.reshape({model.inputs[0]: input_shape})
        print(f"\n⚠️  固定输入形状: {input_shape}")

    print(f"\n✅ 模型加载: {xml_path.name}")

    # 5. 获取输入信息
    try:
        input_name = model.inputs[0].get_any_name()
    except:
        input_name = "images"

    print(f"模型输入名称: {input_name}")

    # 6. 获取设备
    device = get_gpu_device(core)
    print(f"✅ 使用设备: {device}")

    # 7. 创建图片路径列表（关键：只包含图片文件）
    print(f"\n⏳ 收集图片文件...")
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(Path(calib_dir).glob(ext))

    if not image_files:
        raise ValueError(f"❌ 在校准目录未找到任何图片: {calib_dir}")

    num_images = len(image_files)
    print(f"✅ 找到 {num_images} 张图片")
    print(f"📄 标注文件: {num_txts} 个（保留用于后续验证）")

    # 8. 执行量化
    print(f"\n{'=' * 70}")
    print(f"⏳ 开始量化...")
    print(f"设备: {device} | 图片: {num_images} | 预计: 5-15分钟")
    print("=" * 70)

    start_time = time.time()
    try:
        quantized_model = nncf.quantize(
            model,
            calibration_dataset=nncf.Dataset(
                image_files,  # 只传入图片文件
                lambda img_path: {input_name: preprocess_image(str(img_path), (input_shape[2], input_shape[3]))}
            ),
            preset=nncf.QuantizationPreset.PERFORMANCE,
            subset_size=min(NUM_CALIB, num_images),
            fast_bias_correction=True,
        )
        quant_time = time.time() - start_time

        print(f"\n✅ 量化完成！耗时: {quant_time:.1f} 秒")
        print(f"   设备: {device}")
        print(f"   校准图片: {num_images} 张")
        print(f"   标注保留: {calib_dir}（用于后续验证）")

    except Exception as e:
        print(f"\n❌ 量化失败: {e}")
        raise

    # 9. 保存模型
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    quantized_xml = output_path / f"{OUTPUT_NAME}.xml"
    quantized_bin = output_path / f"{OUTPUT_NAME}.bin"

    if quantized_xml.exists():
        print(f"\n⚠️  文件已存在: {quantized_xml}")
        if input("覆盖? (y/N): ").lower() != 'y':
            print("❌ 操作取消")
            return

    print(f"\n⏳ 保存模型...")
    # 修复：使用 ov.save_model()
    ov.save_model(quantized_model, str(quantized_xml))
    print(f"✅ 模型已保存: {quantized_xml.name}")

    # 10. 验证
    print(f"\n⏳ 验证模型...")
    compiled = core.compile_model(quantized_model, device)
    test_input = np.random.randn(*input_shape).astype(np.float32)
    result = compiled(test_input)
    print(f"✅ 验证成功！输出: {list(result.values())[0].shape}")

    # 11. 保存元数据
    meta = {
        "模型": str(xml_path),
        "输出": str(quantized_xml),
        "设备": device,
        "校准图片": num_images,
        "校准目录": calib_dir,
        "耗时(秒)": round(quant_time, 1),
        "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "OpenVINO版本": "2025.4.1",
        "NNCF版本": nncf.__version__
    }

    with open(output_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("🎉 量化全部完成！")
    print(f"模型: {quantized_xml}")
    print(f"设备: {device} | 耗时: {quant_time:.1f}秒")
    print(f"标注目录: {calib_dir}（保留用于验证）")
    print("=" * 70)


if __name__ == "__main__":
    quantize_model()