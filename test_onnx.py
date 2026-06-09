import openvino as ov
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

# ========== 自动找第一张测试图 ==========
dataset_dir = Path(
    r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1")
img_files = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png"))
if not img_files:
    raise FileNotFoundError("找不到图片")
img_path = str(img_files[0])
print(f"使用测试图: {img_path}")

# ========== 预处理（与 yolo predict 一致） ==========
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img = img.astype(np.float32) / 255.0
blob = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # [1,3,640,640]
print(f"Blob范围: [{blob.min():.3f}, {blob.max():.3f}]")

# ========== 1. ONNX Runtime 推理 ==========
onnx_path = r"E:\final_exam_test\handcraft\handcrafted\runs\onnx_batch_test_fp32\yolov8m\yolov8m.onnx"
ort_sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
ort_out = ort_sess.run(None, {"images": blob})[0]
print(f"\nONNX输出形状: {ort_out.shape}")

# 解析 conf (sigmoid)
pred = ort_out[0].T  # [8400, 37]
conf = 1 / (1 + np.exp(-pred[:, 4]))
print(f"ONNX conf: {conf.min():.4f} - {conf.max():.4f}, 均值: {conf.mean():.4f}")
print(f"ONNX >0.25: {(conf > 0.25).sum()} 个")

# ========== 2. OpenVINO 直接读 ONNX ==========
core = ov.Core()
ov_model = core.read_model(onnx_path)
ov_compiled = core.compile_model(ov_model, "CPU")
ov_out = ov_compiled([blob])[ov_compiled.output(0)]
print(f"\nOpenVINO(直接读ONNX)输出形状: {ov_out.shape}")

pred2 = ov_out[0].T
conf2 = 1 / (1 + np.exp(-pred2[:, 4]))
print(f"OpenVINO(ONNX) conf: {conf2.min():.4f} - {conf2.max():.4f}, 均值: {conf2.mean():.4f}")
print(f"OpenVINO(ONNX) >0.25: {(conf2 > 0.25).sum()} 个")

# ========== 3. OpenVINO 读 IR ==========
xml_path = r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolov8m\FP32\yolov8m.xml"
if Path(xml_path).exists():
    ir_model = core.read_model(xml_path)
    ir_compiled = core.compile_model(ir_model, "CPU")
    ir_out = ir_compiled([blob])[ir_compiled.output(0)]
    print(f"\nOpenVINO(IR)输出形状: {ir_out.shape}")

    pred3 = ir_out[0].T
    conf3 = 1 / (1 + np.exp(-pred3[:, 4]))
    print(f"OpenVINO(IR) conf: {conf3.min():.4f} - {conf3.max():.4f}, 均值: {conf3.mean():.4f}")
    print(f"OpenVINO(IR) >0.25: {(conf3 > 0.25).sum()} 个")
else:
    print(f"\n⚠️ IR 文件不存在: {xml_path}")