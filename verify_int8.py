
#!/usr/bin/env python3
"""
INT8量化验证脚本
使用方法: python verify_int8.py <模型路径>
"""

import sys
from pathlib import Path
import numpy as np
from openvino import Core

def check_int8_quantization(model_path):
    """详细检查INT8量化情况"""
    print("=" * 70)
    print("🔍 INT8量化验证报告")
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

    # 详细统计
    precision_stats = {}
    layer_details = []

    for node in model.get_ops():
        if node.get_type_name() == 'Constant':
            const_data = node.get_data()
            dtype = const_data.dtype
            dtype_str = str(dtype)

            precision_stats[dtype_str] = precision_stats.get(dtype_str, 0) + 1
            layer_details.append({
                'name': node.get_friendly_name(),
                'dtype': dtype_str,
                'shape': const_data.shape,
                'size': const_data.size
            })

    # 打印统计
    total_layers = sum(precision_stats.values())
    print(f"
📊 总层数: {total_layers}")

    if total_layers == 0:
        print("❌ 未找到可量化的权重层")
        return

    # 按数量排序
    sorted_stats = sorted(precision_stats.items(), key=lambda x: x[1], reverse=True)

    print("
📈 精度分布:")
    for dtype, count in sorted_stats:
        percentage = (count / total_layers) * 100
        print(f"  {dtype}: {count}层 ({percentage:.1f}%)")

    # 计算INT8比例
    int8_count = precision_stats.get('int8', 0)
    int8_ratio = int8_count / total_layers

    # 浮点层数量
    fp_layers = sum(count for dtype, count in precision_stats.items() 
                   if 'float' in dtype or dtype in ['float32', 'float16'])

    print(f"
🎯 INT8量化效果:")
    print(f"  INT8层数: {int8_count}")
    print(f"  浮点层数: {fp_layers}")
    print(f"  INT8比例: {int8_ratio:.1%}")

    # 评估
    print(f"
📋 量化效果评估:")
    if int8_ratio >= 0.85:
        print("  ✅ 优秀: 深度INT8量化 (≥85%)")
        print("     模型已充分量化，性能最佳")
    elif int8_ratio >= 0.70:
        print("  ✅ 良好: 高度INT8量化 (70-85%)")
        print("     模型量化效果良好")
    elif int8_ratio >= 0.50:
        print("  ⚠️  中等: 部分INT8量化 (50-70%)")
        print("     可考虑进一步优化")
    elif int8_ratio >= 0.30:
        print("  ⚠️  不足: 轻度INT8量化 (30-50%)")
        print("     量化效果有限")
    else:
        print("  ❌ 较差: INT8量化严重不足 (<30%)")
        print("     需要重新量化")

    # 显示最大的INT8层
    int8_layers = [layer for layer in layer_details if layer['dtype'] == 'int8']
    if int8_layers:
        # 按大小排序
        int8_layers.sort(key=lambda x: x['size'], reverse=True)
        print(f"
🔍 最大的INT8权重层:")
        for i, layer in enumerate(int8_layers[:5]):
            size_mb = layer['size'] / (1024*1024)
            print(f"  {i+1}. {layer['name']}: {layer['shape']} ({size_mb:.2f} MB)")

    # 显示最大的浮点层
    fp_layers = [layer for layer in layer_details 
                if 'float' in layer['dtype'] or layer['dtype'] in ['float32', 'float16']]
    if fp_layers:
        fp_layers.sort(key=lambda x: x['size'], reverse=True)
        print(f"
🔍 最大的浮点权重层 (需要关注):")
        for i, layer in enumerate(fp_layers[:5]):
            size_mb = layer['size'] / (1024*1024)
            print(f"  {i+1}. {layer['name']}: {layer['shape']} ({layer['dtype']}, {size_mb:.2f} MB)")

    # 输入输出信息
    print(f"
🔌 模型接口:")
    for inp in model.inputs:
        print(f"  输入 '{inp.get_any_name()}': {inp.get_element_type()}")
    for out in model.outputs:
        print(f"  输出 '{out.get_any_name()}': {out.get_element_type()}")

    print("
" + "=" * 70)
    print("✅ 验证完成")
    print("=" * 70)

    return int8_ratio

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python verify_int8.py <模型路径>")
        print("示例: python verify_int8.py true_int8.xml")
        sys.exit(1)

    model_path = sys.argv[1]
    check_int8_quantization(model_path)
