from openvino import Core
import numpy as np
import sys
from pathlib import Path


def check_model_precision(model_path, model_name="模型"):
    """检查指定模型的精度"""
    print(f"\n{'=' * 50}")
    print(f"🔍 正在检查: {model_name}")
    print(f"📂 文件路径: {model_path}")
    print('=' * 50)

    # 检查文件是否存在
    xml_path = Path(model_path)
    bin_path = xml_path.with_suffix('.bin')
    if not xml_path.exists():
        print(f"❌ 错误：找不到XML文件: {xml_path}")
        return
    if not bin_path.exists():
        print(f"⚠️  警告：找不到对应的BIN文件: {bin_path}")

    # 加载模型
    core = Core()
    try:
        model = core.read_model(str(xml_path))
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 检查所有权重的数据类型
    fp16_layers = 0
    fp32_layers = 0
    int8_layers = 0  # 新增：统计int8权重层
    uint8_layers = 0  # 新增：统计uint8权重层（有时用于输入/激活值）
    other_layers = 0
    other_types = {}  # 用于记录其他未知类型

    print("📊 权重层精度统计:")
    for i, node in enumerate(model.get_ops()):
        if node.get_type_name() == 'Constant':
            const_data = node.get_data()
            dtype = const_data.dtype

            # 统计精度类型（新增INT8/UINT8判断）
            if dtype == np.float16 or 'float16' in str(dtype):
                fp16_layers += 1
                layer_type = "FP16"
            elif dtype == np.float32 or 'float32' in str(dtype):
                fp32_layers += 1
                layer_type = "FP32"
            elif dtype == np.int8:  # 新增：INT8权重
                int8_layers += 1
                layer_type = "INT8"
            elif dtype == np.uint8:  # 新增：UINT8（可能用于量化激活值）
                uint8_layers += 1
                layer_type = "UINT8"
            else:
                other_layers += 1
                type_str = str(dtype)
                other_types[type_str] = other_types.get(type_str, 0) + 1
                layer_type = f"其他({type_str})"

            # 显示前3层的详细信息
            if i < 3:
                shape_str = 'x'.join(map(str, const_data.shape))
                print(f"  层{i}: {layer_type} [形状: {shape_str}]")
                if const_data.size > 0:
                    # 显示几个实际值（确保是真实数字，不是NaN/Inf）
                    sample = const_data.flatten()[:3]
                    print(f"      值样例: {sample}")

    # 打印统计摘要
    print(f"\n📈 精度摘要:")
    print(f"  FP32 权重层: {fp32_layers}")
    print(f"  FP16 权重层: {fp16_layers}")
    if int8_layers > 0:
        print(f"  INT8 权重层: {int8_layers}")
    if uint8_layers > 0:
        print(f"  UINT8 权重层: {uint8_layers}")
    if other_layers > 0:
        print(f"  其他精度层: {other_layers}")
        for t, count in other_types.items():
            print(f"    - {t}: {count}")

    # 核心判断（增强逻辑）
    total_layers = fp32_layers + fp16_layers + int8_layers + uint8_layers

    # 判断纯精度格式
    if fp32_layers == total_layers and total_layers > 0:
        print("✅ 结论: 模型权重为纯 FP32 格式")
    elif fp16_layers == total_layers and total_layers > 0:
        print("✅ 结论: 模型权重为纯 FP16 格式")
    elif int8_layers > 0 and (fp32_layers + fp16_layers + uint8_layers) == 0:
        print("✅ 结论: 模型权重为纯 INT8 格式（典型量化权重）")
    elif uint8_layers > 0 and (fp32_layers + fp16_layers + int8_layers) == 0:
        print("✅ 结论: 模型权重为纯 UINT8 格式")
    # 判断混合精度
    elif int8_layers > 0 and (fp16_layers > 0 or fp32_layers > 0):
        print("⚠️  结论: 模型包含混合量化权重 (INT8 + FP16/FP32)")
        if fp16_layers > 0:
            print("     说明: 部分层可能因敏感度较高，量化时保留了FP16精度")
    elif fp16_layers > 0 and fp32_layers > 0:
        print("⚠️  结论: 模型包含混合精度权重 (FP32 + FP16)")
    else:
        print("❓ 结论: 未检测到常见浮点/定点权重层")

    # 检查输入输出精度（通常保持FP32以兼容）
    print(f"\n🔌 模型接口:")
    for inp in model.inputs:
        print(f"  输入 '{inp.get_any_name()}': {inp.get_element_type()}")
    for out in model.outputs:
        print(f"  输出 '{out.get_any_name()}': {out.get_element_type()}")


def main():
    print("=" * 60)
    print("🛠️  OpenVINO 模型精度权威验证工具 (支持INT8检测)")
    print("=" * 60)

    # ==================== 在这里修改你的文件路径 ====================
    # 示例路径，请替换为你的实际路径
    model_a_path = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\openvino_new\FP32\best.xml"
    model_b_path = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\openvino_new\FP16\best.xml"
    model_c_path = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\openvino_new\best_int8_model_optimized\best_int8_optimized.xml"
    # ==============================================================

    # 检查FP32模型
    check_model_precision(model_a_path, "FP32模型")
    # 检查FP16模型
    check_model_precision(model_b_path, "FP16模型")

    # 检查INT8模型（需要你先量化生成）
    check_model_precision(model_c_path, "INT8量化模型")

    print("\n" + "=" * 60)
    print("🎯 验证完成！请根据上方的'结论'判断模型实际精度。")
    print("=" * 60)


if __name__ == "__main__":
    main()