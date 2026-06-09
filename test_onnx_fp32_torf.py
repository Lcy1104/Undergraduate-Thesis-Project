import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

# 【确保路径是新的 FP32 目录】
onnx_path = r"E:\final_exam_test\handcraft\handcrafted\runs\onnx_batch_test_fp32\yolo26m_com\yolo26m_com.onnx"

session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# 用你的测试图
img_path = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\train\images\ds1_IMG_0212_JPG.rf.51e464fbdd411d448fd79bb32b807bef.jpg"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img = img.astype(np.float32) / 255.0
blob = np.transpose(img, (2,0,1))[np.newaxis, ...]

outputs = session.run(None, {input_name: blob})[0]
print(f"输出形状: {outputs.shape}")

# 解析 conf
pred = outputs[0].T  # [8400, 37]
conf_logits = pred[:, 4]
conf = 1 / (1 + np.exp(-conf_logits))
print(f"conf 范围: {conf.min():.4f} - {conf.max():.4f}")
print(f"conf 均值: {conf.mean():.4f}")

# 应该有部分值接近 0，部分接近 1，而不是全是 0.5
if conf.max() > 0.6 or conf.min() < 0.4:
    print("✅ ONNX 正常！可以开始转换 IR 了")
else:
    print("❌ 还是全是 0.5，检查路径是否指向新文件")