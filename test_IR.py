import openvino as ov
from pathlib import Path

# 验证 yolo26s 的 FP32 模型
model_path = Path(r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolo26s_pk\FP32")
xml_file = list(model_path.glob("*.xml"))[0]

core = ov.Core()
model = core.read_model(str(xml_file))
print("模型输出形状:", model.output(0).shape)
# 预期输出: [1,37,8400] 或 [1,84,8400]