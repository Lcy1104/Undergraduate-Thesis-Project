# convert_int8_robust.py
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

# ==============================================================

# ==================== 用户配置区域（修改这里） ====================
MODEL_DIR = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\openvino_new\FP32"
DATASET_YAML = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml"
NUM_CALIB = 500
OUTPUT_DIR = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\openvino_new\int8_final"
OUTPUT_NAME = "model_int8"
CALIB_DIR = "calib_final"


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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nncf==2.14.0"])
        import nncf
        print(f"✅ NNCF安装完成")

    try:
        from PIL import Image
        print("✅ Pillow: 已安装")
    except:
        print("⏳ 安装Pillow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        print("✅ Pillow安装完成")

    print("=" * 70)
    return nncf


def extract_calib_images():
    """提取校准图片"""
    print("\n📂 提取校准图片...")

    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    all_images = []
    # 只使用验证集，通常质量更好
    for split in ["val", "test"]:
        if split in data:
            split_path = Path(data[split]).resolve()
            if split_path.exists():
                img_dir = split_path / "images" if (split_path / "images").exists() else split_path

                images = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png")) + \
                         list(img_dir.rglob("*.jpeg"))

                for img in images:
                    all_images.append((img, split))

    if not all_images:
        # 如果验证集没有，使用训练集
        split = "train"
        split_path = Path(data[split]).resolve()
        img_dir = split_path / "images" if (split_path / "images").exists() else split_path
        images = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png")) + \
                 list(img_dir.rglob("*.jpeg"))
        all_images = [(img, split) for img in images]

    if not all_images:
        raise ValueError("❌ 未找到任何图片！")

    # 随机采样
    num_samples = min(NUM_CALIB, len(all_images))
    selected = random.sample(all_images, num_samples)

    # 复制到校准目录
    calib_path = Path(CALIB_DIR)
    if calib_path.exists():
        shutil.rmtree(calib_path)
    calib_path.mkdir(exist_ok=True)

    for idx, (img_path, split) in enumerate(selected):
        base_name = f"{split}_{idx:04d}"
        img_dest = calib_path / f"{base_name}{img_path.suffix}"
        shutil.copy2(img_path, img_dest)

    print(f"✅ 提取完成: {num_samples} 张图片")
    return str(calib_path), num_samples


def preprocess_image_simple(image_path: str, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """简单的预处理函数"""
    from PIL import Image

    if not image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize(target_size[::-1], Image.Resampling.BILINEAR)  # (width, height)

        # 转换为numpy并归一化
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

        # HWC to CHW
        img_chw = img_array.transpose(2, 0, 1)

        # 添加batch维度
        img_batch = np.expand_dims(img_chw, axis=0)

        return img_batch

    except Exception as e:
        print(f"⚠️  预处理失败 {Path(image_path).name}: {e}")
        return None


def simple_quantization(model_path, calib_dir, output_dir, nncf):
    """
    重写：针对YOLO优化的稳健量化方法。
    核心改进：正确的ignored_scope、指定model_type、使用全部校准数据。
    """
    print("\n🔄 启动优化的量化流程...")

    # 1. 收集所有校准图片
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(list(Path(calib_dir).glob(ext)))
    image_files = image_files[:NUM_CALIB]  # 使用配置的数量，但不超过实际文件数

    if not image_files:
        raise ValueError(f"❌ 未找到校准图片: {calib_dir}")
    print(f"📊 将使用 {len(image_files)} 张图片进行校准")

    # 2. 加载模型
    core = Core()
    xml_path = Path(model_path) / "best.xml"
    bin_path = Path(model_path) / "best.bin"

    print(f"📦 加载FP32模型: {xml_path.name}")
    model = core.read_model(str(xml_path), str(bin_path))

    # 3. 固定输入形状 (YOLO通常是动态的)
    input_shape = tuple(model.inputs[0].get_shape())
    if -1 in input_shape:
        input_shape = (1, 3, 640, 640)  # 假设是YOLOv5/8的默认尺寸
        model.reshape({model.inputs[0]: input_shape})
        print(f"📏 已固定输入形状为: {input_shape}")

    input_name = model.inputs[0].get_any_name()
    print(f"🔧 模型输入名称: '{input_name}'")

    # 4. 创建高效的数据集生成器（避免一次性加载所有图片到内存）
    class CalibrationDataGenerator:
        def __init__(self, image_files, input_name, input_shape):
            self.image_files = image_files
            self.input_name = input_name
            self.input_shape = input_shape  # (B, C, H, W)

        def __iter__(self):
            for img_path in self.image_files:
                data = preprocess_image_simple(str(img_path), (self.input_shape[2], self.input_shape[3]))
                if data is not None:
                    yield {self.input_name: data}

        def __len__(self):
            return len(self.image_files)

    data_generator = CalibrationDataGenerator(image_files, input_name, input_shape)
    nncf_dataset = nncf.Dataset(data_generator)

    # 5. 【核心】配置并执行量化
    print(f"\n🎯 配置量化参数...")
    # 关键：针对YOLO模型的ignored_scope配置。目标是尽可能多地量化，只排除极少数对量化敏感的算子。
    ignored_scope = nncf.IgnoredScope(
        # 使用 patterns 来排除特定类型的层（保守策略开始，后续可减少）
        patterns=[
            # 通常对量化敏感或需要高精度计算的算子
            ".*aten::*",          # ATen算子（来自PyTorch）
            ".*roi_align.*",
            ".*interpolate.*",    # 上采样/插值层
            ".*sigmoid.*",        # 激活函数（有时需要保留精度）
            ".*softmax.*",        # Softmax层
            # ".*Reshape.*",      # Reshape (通常安全，可尝试量化)
            # ".*Concat.*",       # Concat (通常安全，可尝试量化)
        ],
        # 可选：也可以按层名称排除（如果你知道特定层名）
        # names=["/model.0/Reshape", "/model.1/Concat"]
    )

    # 量化预设：MIXED 比 PERFORMANCE 更激进，可能量化更多层
    quantization_preset = nncf.QuantizationPreset.MIXED
    # 明确指定模型类型有助于NNCF应用更合适的量化策略
    model_type = nncf.ModelType.TRANSFORMER  # 对于YOLO，TRANSFORMER或CNN可尝试

    print(f"   预设: {quantization_preset.name}")
    print(f"   模型类型: {model_type.value if hasattr(model_type, 'value') else model_type}")
    print(f"   忽略范围: 已配置 {len(ignored_scope.patterns)} 个模式")

    start_time = time.time()
    try:
        print("⚡ 正在执行量化（这可能需要几分钟）...")
        quantized_model = nncf.quantize(
            model=model,
            calibration_dataset=nncf_dataset,
            preset=quantization_preset,
            model_type=model_type,
            ignored_scope=ignored_scope,  # 应用我们的配置
            subset_size=len(image_files), # 使用全部校准数据
            fast_bias_correction=True,
            # 如果NNCF版本支持，可启用SmoothQuant
            # advanced_parameters=nncf.AdvancedQuantizationParameters(
            #     smooth_quant_alpha=0.5
            # )
        )
        quant_time = time.time() - start_time

        # 6. 立即检查量化效果
        int8_ratio = check_int8_ratio(quantized_model)
        print(f"\n✅ 量化完成！耗时: {quant_time:.1f} 秒")
        print(f"🎯 初步INT8量化比例: {int8_ratio:.1%}")

        # 7. 量化效果不理想的应急方案：尝试更激进的配置（减少忽略）
        if int8_ratio < 0.4:  # 如果比例仍然很低
            print(f"\n⚠️  INT8比例 ({int8_ratio:.1%}) 未达预期，尝试更激进的量化...")
            try:
                # 尝试一个几乎不忽略任何层的配置（风险较高，可能影响精度）
                aggressive_ignored_scope = nncf.IgnoredScope(
                    patterns=[".*softmax.*"],  # 只排除最敏感的
                )
                print("   尝试更激进的忽略范围...")
                quantized_model = nncf.quantize(
                    model=model,
                    calibration_dataset=nncf_dataset,
                    preset=nncf.QuantizationPreset.MIXED,
                    subset_size=min(100, len(image_files)),  # 用部分数据快速尝试
                    ignored_scope=aggressive_ignored_scope,
                )
                new_ratio = check_int8_ratio(quantized_model)
                print(f"   激进方案INT8比例: {new_ratio:.1%}")
                if new_ratio > int8_ratio:
                    int8_ratio = new_ratio
                    print("   ✅ 激进方案有效，已采用新模型。")
            except Exception as e_aggressive:
                print(f"   ⚠️  激进方案失败，保留原方案: {e_aggressive}")

        return quantized_model, int8_ratio

    except Exception as e:
        print(f"\n❌ 量化过程失败: {e}")
        # 提供具体的调试建议
        import traceback
        error_details = traceback.format_exc()
        if "ignored_scope" in str(e):
            print("\n💡 提示: `ignored_scope` 配置可能存在问题。尝试将其设为 `None` 或减少排除的模式。")
        print(f"\n🔍 详细错误:\n{error_details[:500]}...")  # 打印前500字符
        raise RuntimeError(f"量化失败: {e}")


def check_int8_ratio(model):
    """检查INT8比例"""
    int8_count = 0
    total_count = 0

    for node in model.get_ops():
        if node.get_type_name() == 'Constant':
            const_data = node.get_data()
            dtype = const_data.dtype

            total_count += 1
            if dtype == np.int8:
                int8_count += 1

    return int8_count / total_count if total_count > 0 else 0


def verify_and_save_model(model, output_dir, output_name, int8_ratio):
    """验证并保存模型"""
    print(f"\n💾 保存模型...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xml_path = output_path / f"{output_name}.xml"
    bin_path = output_path / f"{output_name}.bin"

    # 保存模型
    ov.save_model(model, str(xml_path))
    print(f"✅ 模型已保存: {xml_path.name}")

    # 验证模型
    print(f"🔍 验证模型...")
    core = Core()
    compiled = core.compile_model(model, "CPU")

    # 获取输入形状
    input_shape = tuple(model.inputs[0].get_shape())
    input_name = model.inputs[0].get_any_name()

    # 测试推理
    test_input = np.random.randn(*input_shape).astype(np.float32)
    result = compiled(test_input)
    print(f"✅ 推理测试成功")
    print(f"   输入: {test_input.shape}")
    print(f"   输出: {result[0].shape if isinstance(result, tuple) else result.shape}")

    # 计算模型大小
    original_size = Path(MODEL_DIR).joinpath("best.xml").stat().st_size / (1024 * 1024)
    quantized_size = xml_path.stat().st_size / (1024 * 1024)

    print(f"\n📊 模型大小对比:")
    print(f"   原始FP32模型: {original_size:.2f} MB")
    print(f"   INT8量化模型: {quantized_size:.2f} MB")
    print(f"   压缩比例: {(1 - quantized_size / original_size) * 100:.1f}%")
    print(f"   INT8量化比例: {int8_ratio:.1%}")

    return str(xml_path)


def manual_int8_conversion():
    """备选方案：手动转换到INT8"""
    print("\n🔧 尝试手动INT8转换...")

    # 加载模型
    core = Core()
    xml_path = Path(MODEL_DIR) / "best.xml"
    bin_path = Path(MODEL_DIR) / "best.bin"

    model = core.read_model(str(xml_path), str(bin_path))

    # 创建临时量化模型
    # 这里使用一个简单的转换策略：将所有浮点权重转换为INT8
    print("  注意：这是一个简化的INT8转换，可能影响精度")
    print("  仅用于测试目的")

    # 实际上，OpenVINO没有直接的API手动转换权重到INT8
    # 这里我们使用NNCF的最小配置量化

    return None


def quantize_model():
    """主量化流程"""
    print("=" * 70)
    print("🚀 稳健INT8量化工具")
    print("=" * 70)

    try:
        # 1. 检查依赖
        nncf = check_dependencies()

        # 2. 提取校准数据
        calib_dir, num_images = extract_calib_images()

        print(f"\n📈 量化配置:")
        print(f"   模型目录: {MODEL_DIR}")
        print(f"   校准图片: {num_images} 张")
        print(f"   输出目录: {OUTPUT_DIR}")

        # 3. 执行量化
        start_time = time.time()

        try:
            # 方法A: 简单量化
            print("\n" + "=" * 70)
            print("⚡ 开始量化...")
            print("=" * 70)

            quantized_model, int8_ratio = simple_quantization(
                MODEL_DIR, calib_dir, OUTPUT_DIR, nncf
            )

            quant_time = time.time() - start_time
            print(f"\n✅ 量化完成！耗时: {quant_time:.1f} 秒")
            print(f"   INT8比例: {int8_ratio:.1%}")

            # 评估量化效果
            if int8_ratio >= 0.7:
                print("  状态: 🎉 优秀 - 模型已深度量化")
            elif int8_ratio >= 0.4:
                print("  状态: ✅ 良好 - 模型已有效量化")
                print("  💡 提示: 可尝试增加校准图片或调整 `ignored_scope` 以追求更高比例。")
            else:
                print("  状态: ⚠️  一般 - 量化比例偏低")
                print(f"  💡 建议: 请检查模型结构是否特殊，或尝试使用更激进的 `ignored_scope = None`。")

            # 4. 验证并保存
            model_path = verify_and_save_model(
                quantized_model, OUTPUT_DIR, OUTPUT_NAME, int8_ratio
            )

            # 5. 保存元数据
            meta = {
                "原始模型": str(Path(MODEL_DIR) / "best.xml"),
                "INT8模型": model_path,
                "校准图片数量": num_images,
                "INT8量化比例": f"{int8_ratio:.1%}",
                "量化耗时(秒)": round(quant_time, 1),
                "量化时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                "OpenVINO版本": "2025.4.1",
                "NNCF版本": nncf.__version__,
                "备注": "使用稳健量化方法"
            }

            with open(Path(OUTPUT_DIR) / "quantization_info.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            print(f"\n📝 量化信息已保存: quantization_info.json")

        except Exception as e:
            print(f"\n❌ 量化失败: {e}")
            print("\n🔄 尝试备选方案...")

            # 备选方案：使用OpenVINO自带的Post-training Optimization Tool (POT)
            print("💡 建议使用OpenVINO的POT工具进行量化:")
            print("   1. 安装: pip install openvino-dev")
            print("   2. 使用POT命令行工具")
            print("   3. 或使用OpenVINO Model Optimizer的量化功能")

            raise

        print("\n" + "=" * 70)
        print("🎉 量化流程完成")
        print("=" * 70)

        if int8_ratio >= 0.7:
            print("✅ 成功！获得了高质量的INT8模型")
        elif int8_ratio >= 0.4:
            print("⚠️  部分成功！模型包含部分INT8权重")
            print("   性能可能有所提升")
        else:
            print("❌ INT8量化比例较低")
            print("   建议尝试其他量化工具或方法")

        print(f"\n📁 输出目录: {OUTPUT_DIR}")
        print(f"📄 模型文件: {OUTPUT_NAME}.xml")

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


def create_check_script():
    """创建检查脚本"""
    script_content = '''
#!/usr/bin/env python3
"""
INT8模型检查脚本
使用方法: python check_int8.py <模型路径>
"""

import sys
from pathlib import Path
import numpy as np
from openvino import Core

def analyze_model(model_path):
    """分析模型精度"""
    print("=" * 70)
    print("🔍 模型精度分析")
    print("=" * 70)

    xml_path = Path(model_path)
    if not xml_path.exists():
        print(f"❌ 模型文件不存在: {xml_path}")
        return

    # 加载模型
    core = Core()
    try:
        model = core.read_model(str(xml_path))
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    print(f"📄 模型文件: {xml_path.name}")
    print(f"📏 文件大小: {xml_path.stat().st_size / (1024*1024):.2f} MB")

    # 统计精度
    precision_stats = {}
    layer_details = []

    for node in model.get_ops():
        if node.get_type_name() == 'Constant':
            const_data = node.get_data()
            dtype = const_data.dtype
            dtype_str = str(dtype)

            precision_stats[dtype_str] = precision_stats.get(dtype_str, 0) + 1
            layer_details.append({
                'type': dtype_str,
                'shape': const_data.shape,
                'size': const_data.size
            })

    # 打印统计
    total_layers = sum(precision_stats.values())
    print(f"\n📊 总层数: {total_layers}")

    if total_layers == 0:
        print("❌ 未找到权重层")
        return

    # 按精度类型统计
    print("\n📈 精度分布:")
    for dtype, count in sorted(precision_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_layers) * 100
        print(f"  {dtype}: {count}层 ({percentage:.1f}%)")

    # 计算INT8比例
    int8_count = precision_stats.get('int8', 0)
    int8_ratio = int8_count / total_layers if total_layers > 0 else 0

    # 计算模型大小分布
    int8_size = sum(layer['size'] for layer in layer_details if layer['type'] == 'int8')
    fp32_size = sum(layer['size'] for layer in layer_details if layer['type'] == 'float32')
    total_size = int8_size + fp32_size

    print(f"\n🎯 INT8量化分析:")
    print(f"  INT8层数: {int8_count}")
    print(f"  INT8比例: {int8_ratio:.1%}")

    if total_size > 0:
        print(f"  INT8数据大小: {int8_size / (1024*1024):.2f} MB")
        print(f"  FP32数据大小: {fp32_size / (1024*1024):.2f} MB")
        print(f"  INT8数据比例: {(int8_size / total_size):.1%}")

    # 评估
    print(f"\n📋 量化效果评估:")
    if int8_ratio >= 0.8:
        print("  ✅ 优秀: 深度INT8量化")
    elif int8_ratio >= 0.6:
        print("  ✅ 良好: 较好的INT8量化")
    elif int8_ratio >= 0.4:
        print("  ⚠️  中等: 部分INT8量化")
    elif int8_ratio >= 0.2:
        print("  ⚠️  不足: 轻度INT8量化")
    else:
        print("  ❌ 较差: INT8量化不足")

    # 显示最大的层
    print(f"\n🔍 最大权重层:")
    layer_details.sort(key=lambda x: x['size'], reverse=True)
    for i, layer in enumerate(layer_details[:5]):
        size_mb = layer['size'] * 4 / (1024*1024) if layer['type'] == 'float32' else layer['size'] / (1024*1024)
        print(f"  {i+1}. {layer['type']}: {layer['shape']} ({size_mb:.2f} MB)")

    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)

    return int8_ratio

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认检查当前目录的模型
        model_paths = [
            "model_int8.xml",
            "best_int8.xml",
            "true_int8.xml"
        ]

        for model_path in model_paths:
            if Path(model_path).exists():
                analyze_model(model_path)
                break
        else:
            print("❌ 未找到模型文件")
            print("使用方法: python check_int8.py <模型路径>")
    else:
        analyze_model(sys.argv[1])
'''

    # 保存检查脚本
    script_path = Path(__file__).parent / "check_int8.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    print(f"\n📝 检查脚本已创建: {script_path}")


if __name__ == "__main__":
    # 创建检查脚本
    create_check_script()

    # 执行量化
    quantize_model()

    print(f"\n📋 使用说明:")
    print("   1. 检查量化结果: python check_int8.py")
    print("   2. 如果INT8比例不足，可以尝试:")
    print("      - 增加校准图片数量")
    print("      - 使用OpenVINO POT工具")
    print("      - 检查模型是否适合INT8量化")
    print("   3. 即使部分量化，也可能提升推理速度")