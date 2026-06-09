import os
from pathlib import Path
from ultralytics import YOLO
import torch
models_map = {
    'yolov8m': r'E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights\best.pt',
    'yolo26n': r'E:\final_exam_test\handcraft\handcrafted\runs\detect\yolo26n_tomato5\weights\best.pt',
    'yolo26s': r'E:\final_exam_test\handcraft\handcrafted\runs\detect\yolo26s_tomato6\weights\best.pt',
    'yolo26m_laptop': r'E:\final_exam_test\handcraft\handcrafted\runs\detect\yolo26m_tomato5\weights\best.pt',
    'yolo26m_com': r'E:\final_exam_test\handcraft\handcrafted\runs\detect\yolo26m_tomato3\yolo26m_tomato3\weights\best.pt',
}

output_root = r'E:\final_exam_test\handcraft\handcrafted\runs\onnx_batch_test_fp32'  # 新目录，避免覆盖

for name, pt_path in models_map.items():
    out_dir = Path(output_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = out_dir / f"{name}.onnx"

    print(f"\n导出 {name} (FP32)...")
    model = YOLO(pt_path)
    print(f"模型类型: {model.model.args}")
    print(f"模型精度: {next(model.model.parameters()).dtype}")  # 应该是 float32

    # 【关键修改】去掉 half=True，使用 FP32
    model.export(format='onnx', imgsz=640, half=False, device='cpu')

    # 移动生成的文件
    src = Path(pt_path).parent / "best.onnx"
    if src.exists():
        src.rename(out_onnx)
        print(f"完成: {out_onnx}")
    else:
        print(f"错误：未找到生成的文件")

    import onnx

    onnx_model = onnx.load(out_onnx)
    print(f"ONNX 第一个卷积层权重范围:")
    for init in onnx_model.graph.initializer[:1]:
        import numpy as np

        arr = onnx.numpy_helper.to_array(init)
        print(f"  {init.name}: {arr.min():.3f} - {arr.max():.3f}")
        if arr.max() < 0.01 and arr.min() > -0.01:
            print("警告：权重几乎全为0，导出失败！")

print("\n全部导出完成！现在运行你的转换脚本生成真正的 FP32/FP16/INT8 IR 模型")