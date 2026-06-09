import torch
import cv2
import numpy as np
from openvino import Core
from pathlib import Path
import yaml
import random
from ultralytics import YOLO
import csv
import pandas as pd

# ==================== 路径配置（修改为您的实际路径） ====================
DATA_YAML = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml"
OV_BASE = Path(r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models")
PT_YOLOV8M = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights_openvion\best.pt"
PT_YOLOV26M = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\yolo26m_tomato3\yolo26m_tomato3\weights\best.pt"

# ==================== 测试配置 ====================
NUM_IMAGES = 20  # 测试图片数量

# OpenVINO 精度与设备组合
OV_PRECISIONS = ['FP32', 'FP16', 'INT8']
OV_DEVICES = ['GPU.0', 'CPU']

# PyTorch 设备
PT_DEVICES = ['cuda:0', 'cpu']

# ==================== 类别名称 ====================
CLASS_NAMES = [
    "Bacterial Spot", "Early_Blight", "Healthy", "Late_blight", "Leaf Mold",
    "Target_Spot", "black spot", "Apple Scab Leaf", "Apple leaf", "Apple rust leaf",
    "Bell_pepper leaf spot", "Bell_pepper leaf", "Blueberry leaf", "Cherry leaf",
    "Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf", "Peach leaf",
    "Potato leaf early blight", "Potato leaf late blight", "Potato leaf", "Raspberry leaf",
    "Soyabean leaf", "Soybean leaf", "Squash Powdery mildew leaf", "Strawberry leaf",
    "Tomato Septoria leaf spot", "Tomato leaf mosaic virus", "Tomato leaf yellow virus",
    "Tomato leaf", "Tomato two spotted spider mites leaf", "grape leaf black rot", "grape leaf"
]
NUM_CLASSES = len(CLASS_NAMES)

# ==================== 工具箱函数 ====================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh

def scale_boxes(boxes, orig_shape, r, dw, dh):
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])
    return boxes

def nms(boxes, scores, iou_threshold=0.45, conf_threshold=0.25):
    keep_conf = scores > conf_threshold
    boxes = boxes[keep_conf]
    scores = scores[keep_conf]
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) == 0:
        return []
    if isinstance(indices, tuple):
        indices = indices[0]
    return indices.flatten().tolist()

def postprocess_openvino(output, orig_shape, r, dw, dh, conf_thresh=0.25, iou_thresh=0.45):
    pred = output.squeeze().T          # [8400, 37]
    boxes_cxcywh = pred[:, :4]
    scores = pred[:, 4:4+NUM_CLASSES]
    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)

    mask = max_scores > conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return [], [], []

    boxes_xyxy = np.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    boxes_xyxy = scale_boxes(boxes_xyxy, orig_shape, r, dw, dh)
    keep = nms(boxes_xyxy, max_scores, iou_thresh, conf_thresh)
    if len(keep) == 0:
        return [], [], []
    return boxes_xyxy[keep], max_scores[keep], class_ids[keep]

def compute_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0]); inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2]); inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (area1 + area2 - inter_area + 1e-6)

# ==================== 加载测试图片 ====================
with open(DATA_YAML, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
img_pairs = []
for split in ['val', 'test']:
    if split in data:
        split_path = Path(data[split]).resolve()
        img_dir = split_path / 'images'
        lbl_dir = split_path / 'labels'
        if img_dir.exists() and lbl_dir.exists():
            for img_path in sorted(img_dir.glob('*.jpg')):
                lbl = lbl_dir / f'{img_path.stem}.txt'
                if lbl.exists():
                    img_pairs.append((img_path, lbl))
random.seed(42)
random.shuffle(img_pairs)
selected = img_pairs[:NUM_IMAGES]
print(f"测试图片数量: {len(selected)}")

# ==================== 模型加载与推理 ====================
def load_pt_model(pt_path, device):
    model = YOLO(str(pt_path))
    try:
        model.to(device)
        print(f"PyTorch 加载成功: {pt_path} on {device}")
    except Exception as e:
        print(f"PyTorch 设备 {device} 不可用,回退CPU: {e}")
        model.to('cpu')
        device = 'cpu'
    return model, device

def load_ov_model(ov_dir, device):
    core = Core()
    xml_files = list(ov_dir.glob('*.xml'))
    if not xml_files:
        raise FileNotFoundError(f"找不到 .xml 文件: {ov_dir}")
    model = core.read_model(str(xml_files[0]))
    try:
        compiled = core.compile_model(model, device)
        actual_device = device
        print(f"OV 模型加载成功: {ov_dir} on {device}")
    except Exception as e:
        print(f"OV 设备 {device} 不可用,回退CPU: {e}")
        compiled = core.compile_model(model, "CPU")
        actual_device = "CPU"
    return compiled, actual_device

# 缓存已加载的模型，避免重复加载
pt_cache = {}
ov_cache = {}

# ==================== 主循环 ====================
results = []  # 存储每组的统计: Model, PT_Device, OV_Precision, OV_Device, Avg_IoU, Offset...

# 生成所有配对
models_info = {
    'yolov8m': (PT_YOLOV8M, OV_BASE / 'yolov8m'),
    'yolov26m': (PT_YOLOV26M, OV_BASE / 'yolo26m_com')
}

for model_name, (pt_path, ov_base) in models_info.items():
    print(f"\n===== 模型: {model_name} =====")
    for pt_device in PT_DEVICES:
        # 加载/获取PyTorch模型
        if (pt_path, pt_device) not in pt_cache:
            pt_model, actual_pt_device = load_pt_model(pt_path, pt_device)
            pt_cache[(pt_path, pt_device)] = (pt_model, actual_pt_device)
        else:
            pt_model, actual_pt_device = pt_cache[(pt_path, pt_device)]

        for ov_prec in OV_PRECISIONS:
            ov_dir = ov_base / ov_prec
            for ov_device in OV_DEVICES:
                # 加载/获取OpenVINO模型
                cache_key = (str(ov_dir), ov_device)
                if cache_key not in ov_cache:
                    compiled, actual_ov_device = load_ov_model(ov_dir, ov_device)
                    ov_cache[cache_key] = (compiled, actual_ov_device)
                else:
                    compiled, actual_ov_device = ov_cache[cache_key]

                output_blob = compiled.output(0)
                print(f"  配对: PT {actual_pt_device} vs OV {ov_prec} {actual_ov_device}")

                # 对每张图片进行匹配
                ious = []
                offsets = []  # (cx_offset, cy_offset, w_offset, h_offset)
                for img_path, _ in selected:
                    img0 = cv2.imread(str(img_path))
                    if img0 is None:
                        continue
                    orig_h, orig_w = img0.shape[:2]

                    # PyTorch推理
                    results_pt = pt_model(str(img_path), verbose=False)
                    boxes_pt = results_pt[0].boxes.data.cpu().numpy()

                    # OpenVINO推理
                    img, r, dw, dh = letterbox(img0)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                    input_blob = np.expand_dims(img, axis=0)
                    result_ov = compiled([input_blob])[output_blob]
                    boxes_ov, scores_ov, ids_ov = postprocess_openvino(result_ov, (orig_h, orig_w), r, dw, dh)

                    if len(boxes_pt) == 0 or len(boxes_ov) == 0:
                        continue

                    # 最佳匹配（同一类别中IoU最大）
                    best_iou = 0
                    best_offsets = None
                    for pt_box in boxes_pt:
                        for ov_box, ov_score, ov_cls in zip(boxes_ov, scores_ov, ids_ov):
                            if pt_box[5] != ov_cls:
                                continue
                            iou = compute_iou(pt_box[:4], ov_box)
                            if iou > best_iou:
                                best_iou = iou
                                cx_pt = (pt_box[0] + pt_box[2]) / 2
                                cy_pt = (pt_box[1] + pt_box[3]) / 2
                                w_pt = pt_box[2] - pt_box[0]
                                h_pt = pt_box[3] - pt_box[1]
                                cx_ov = (ov_box[0] + ov_box[2]) / 2
                                cy_ov = (ov_box[1] + ov_box[3]) / 2
                                w_ov = ov_box[2] - ov_box[0]
                                h_ov = ov_box[3] - ov_box[1]
                                best_offsets = (
                                    abs(cx_pt - cx_ov),
                                    abs(cy_pt - cy_ov),
                                    abs(w_pt - w_ov),
                                    abs(h_pt - h_ov)
                                )
                    if best_offsets is not None:
                        ious.append(best_iou)
                        offsets.append(best_offsets)

                if ious:
                    avg_iou = np.mean(ious)
                    avg_offsets = np.mean(offsets, axis=0)
                    results.append({
                        'Model': model_name,
                        'PT_Device': actual_pt_device,
                        'OV_Precision': ov_prec,
                        'OV_Device': actual_ov_device,
                        'Avg_IoU': round(avg_iou, 4),
                        'Avg_cx_offset': round(avg_offsets[0], 2),
                        'Avg_cy_offset': round(avg_offsets[1], 2),
                        'Avg_w_offset': round(avg_offsets[2], 2),
                        'Avg_h_offset': round(avg_offsets[3], 2),
                    })
                    print(f"    → Avg IoU: {avg_iou:.4f}, Offset: {avg_offsets}")
                else:
                    print("    → 无有效匹配")

# ==================== 保存与打印结果 ====================
df = pd.DataFrame(results)
csv_path = "iou_validation_all_configs.csv"
df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"\n详细结果已保存至 {csv_path}")

# 打印汇总表
print("\n========== 偏移量验证汇总表 ==========")
print(df.to_string(index=False))

# 全局统计
overall_iou = df['Avg_IoU'].mean()
overall_cx = df['Avg_cx_offset'].mean()
overall_cy = df['Avg_cy_offset'].mean()
print("\n全局均值：")
print(f"  IoU = {overall_iou:.4f}")
print(f"  中心点偏移 = ({overall_cx:.2f}, {overall_cy:.2f}) 像素")