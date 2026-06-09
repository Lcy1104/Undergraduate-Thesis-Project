
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
    print(f"
📊 总层数: {total_layers}")

    if total_layers == 0:
        print("❌ 未找到权重层")
        return

    # 按精度类型统计
    print("
📈 精度分布:")
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

    print(f"
🎯 INT8量化分析:")
    print(f"  INT8层数: {int8_count}")
    print(f"  INT8比例: {int8_ratio:.1%}")

    if total_size > 0:
        print(f"  INT8数据大小: {int8_size / (1024*1024):.2f} MB")
        print(f"  FP32数据大小: {fp32_size / (1024*1024):.2f} MB")
        print(f"  INT8数据比例: {(int8_size / total_size):.1%}")

    # 评估
    print(f"
📋 量化效果评估:")
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
    print(f"
🔍 最大权重层:")
    layer_details.sort(key=lambda x: x['size'], reverse=True)
    for i, layer in enumerate(layer_details[:5]):
        size_mb = layer['size'] * 4 / (1024*1024) if layer['type'] == 'float32' else layer['size'] / (1024*1024)
        print(f"  {i+1}. {layer['type']}: {layer['shape']} ({size_mb:.2f} MB)")

    print("
" + "=" * 70)
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
