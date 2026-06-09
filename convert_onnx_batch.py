import os
from pathlib import Path
from ultralytics import YOLO

models_map = {
    'yolov8m': r'E:\final_exam_test\handcraft\handcrafted\runs\detect _convert\train8\weights\best.pt',
    'yolo26n': r'E:\final_exam_test\handcraft\handcrafted\runs\detect _convert\yolo26n_tomato5\weights\best.pt',
    'yolo26s': r'E:\final_exam_test\handcraft\handcrafted\runs\detect _convert\yolo26s_tomato6\weights\best.pt',
    'yolo26m_laptop': r'E:\final_exam_test\handcraft\handcrafted\runs\detect _convert\yolo26m_tomato5\weights\best.pt',
    'yolo26m_com': r'E:\final_exam_test\handcraft\handcrafted\runs\detect _convert\yolo26m_tomato3\yolo26m_tomato3\weights\best.pt',
}

output_root = r'E:\final_exam_test\handcraft\handcrafted\runs\onnx_batch_test'

for name, pt_path in models_map.items():
    out_dir = Path(output_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = out_dir / f"{name}.onnx"
    if out_onnx.exists():
        print(f"跳过 {name}，文件已存在")
        continue
    print(f"转换 {name} ...")
    model = YOLO(pt_path)
    # 官方导出，半精度，GPU
    model.export(format='onnx', imgsz=640, half=False, device=0)#half表示是否为半精度导出
    # 移动生成的文件（默认生成在 pt 文件同目录的 best.onnx）
    src = Path(pt_path).parent / "best.onnx"
    if src.exists():
        src.rename(out_onnx)
        print(f"成功: {out_onnx}")
    else:
        print(f"错误：未找到生成的文件")