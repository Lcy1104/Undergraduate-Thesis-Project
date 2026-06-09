
import sys
from pathlib import Path

def check_model_layers(model_path):
    """检查模型各层的数据类型"""
    from openvino import Core
    import numpy as np

    core = Core()
    model = core.read_model(str(model_path))

    layer_stats = {}
    for node in model.get_ops():
        if node.get_type_name() == 'Constant':
            const_data = node.get_data()
            dtype = const_data.dtype

            dtype_str = str(dtype)
            layer_stats[dtype_str] = layer_stats.get(dtype_str, 0) + 1

    print("📊 各层数据类型统计:")
    total_layers = sum(layer_stats.values())
    for dtype, count in layer_stats.items():
        percentage = (count / total_layers) * 100
        print(f"  {dtype}: {count} 层 ({percentage:.1f}%)")

    # 计算INT8比例
    int8_count = layer_stats.get('int8', 0)
    int8_percentage = (int8_count / total_layers) * 100 if total_layers > 0 else 0
    print(f"\n🎯 INT8量化比例: {int8_percentage:.1f}%")

    if int8_percentage > 70:
        print("✅ INT8量化成功！")
    elif int8_percentage > 30:
        print("⚠️  INT8量化部分成功，可以尝试进一步优化")
    else:
        print("❌ INT8量化不足，需要重新量化")

if __name__ == "__main__":
    model_path = Path(__file__).parent / "best_int8_optimized.xml"
    if model_path.exists():
        check_model_layers(model_path)
    else:
        print(f"❌ 模型文件不存在: {model_path}")
