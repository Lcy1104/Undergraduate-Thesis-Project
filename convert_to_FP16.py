# convert_to_fp16.py
from openvino import Core
import openvino as ov
from pathlib import Path

# ==================== 配置 ====================
MODEL_DIR = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights_openvion\best_openvino_model"
OUTPUT_DIR = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights_openvion\best_fp16_model"
OUTPUT_NAME = "best_fp16"


# ==============================================

def convert_to_fp16():
    """将FP32模型转换为FP16"""
    print("=" * 70)
    print("🚀 OpenVINO 2025.4.1 FP16 转换工具")
    print("=" * 70)

    # 1. 加载FP32模型（需要xml + bin）
    core = Core()
    xml_path = Path(MODEL_DIR) / "best.xml"
    bin_path = Path(MODEL_DIR) / "best.bin"

    if not xml_path.exists() or not bin_path.exists():
        raise FileNotFoundError(f"❌ 需要FP32模型文件: {xml_path} 和 {bin_path}")

    model = core.read_model(str(xml_path), str(bin_path))
    print(f"✅ 加载FP32模型: {xml_path.name}")
    print(f"   权重大小: {bin_path.stat().st_size / (1024 ** 2):.1f} MB")

    # 2. 转换为FP16
    print(f"\n⏳ 转换为FP16...")
    # OpenVINO 2025.4.1会自动转换权重精度

    # 3. 保存FP16模型
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    fp16_xml = output_path / f"{OUTPUT_NAME}.xml"
    fp16_bin = output_path / f"{OUTPUT_NAME}.bin"

    # 关键：使用ov.save_model()保存
    ov.save_model(model, str(fp16_xml))

    # 验证文件大小减少
    print(f"✅ FP16模型保存: {fp16_xml.name}")
    print(f"   FP32权重: {bin_path.stat().st_size / (1024 ** 2):.1f} MB")
    print(f"   FP16权重: {fp16_bin.stat().st_size / (1024 ** 2):.1f} MB")
    print(f"   大小减少: {(1 - fp16_bin.stat().st_size / bin_path.stat().st_size) * 100:.1f}%")

    # 4. 验证推理
    print(f"\n⏳ 验证FP16模型...")
    compiled = core.compile_model(model, "GPU")
    test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    result = compiled(test_input)
    print(f"✅ 推理成功！输出: {list(result.values())[0].shape}")

    print("\n🎉 FP16转换完成！")
    print(f"模型: {fp16_xml}")
    print("=" * 70)


if __name__ == "__main__":
    import numpy as np  # 需要在这里导入numpy

    convert_to_fp16()