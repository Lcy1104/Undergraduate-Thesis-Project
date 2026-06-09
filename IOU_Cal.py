import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
import random
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from torchvision.ops import nms
import openvino as ov
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ==================== 请修改为您的实际路径 ====================
DATA_YAML = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml"
OV_BASE = Path(r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models")
PT_YOLOV8M = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights_openvion\best.pt"
PT_YOLOV26M = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\yolo26m_tomato3\yolo26m_tomato3\weights\best.pt"

# 候选图片池大小与最低 IoU 要求
CANDIDATE_POOL_SIZE = 30
MIN_IOU = 0.7
RANDOM_SEED = 42

# ==================== 24 组配置 ====================
configs = [
    # YOLOv8m
    ('yolov8m', PT_YOLOV8M, 'cuda:0', OV_BASE/'yolov8m'/'FP32', 'GPU.0', 'YOLOv8m PT CUDA vs OV FP32 GPU'),
    ('yolov8m', PT_YOLOV8M, 'cuda:0', OV_BASE/'yolov8m'/'FP32', 'CPU',   'YOLOv8m PT CUDA vs OV FP32 CPU'),
    ('yolov8m', PT_YOLOV8M, 'cuda:0', OV_BASE/'yolov8m'/'FP16', 'GPU.0', 'YOLOv8m PT CUDA vs OV FP16 GPU'),
    ('yolov8m', PT_YOLOV8M, 'cuda:0', OV_BASE/'yolov8m'/'FP16', 'CPU',   'YOLOv8m PT CUDA vs OV FP16 CPU'),
    ('yolov8m', PT_YOLOV8M, 'cuda:0', OV_BASE/'yolov8m'/'INT8', 'GPU.0', 'YOLOv8m PT CUDA vs OV INT8 GPU'),
    ('yolov8m', PT_YOLOV8M, 'cuda:0', OV_BASE/'yolov8m'/'INT8', 'CPU',   'YOLOv8m PT CUDA vs OV INT8 CPU'),
    ('yolov8m', PT_YOLOV8M, 'cpu',    OV_BASE/'yolov8m'/'FP32', 'GPU.0', 'YOLOv8m PT CPU vs OV FP32 GPU'),
    ('yolov8m', PT_YOLOV8M, 'cpu',    OV_BASE/'yolov8m'/'FP32', 'CPU',   'YOLOv8m PT CPU vs OV FP32 CPU'),
    ('yolov8m', PT_YOLOV8M, 'cpu',    OV_BASE/'yolov8m'/'FP16', 'GPU.0', 'YOLOv8m PT CPU vs OV FP16 GPU'),
    ('yolov8m', PT_YOLOV8M, 'cpu',    OV_BASE/'yolov8m'/'FP16', 'CPU',   'YOLOv8m PT CPU vs OV FP16 CPU'),
    ('yolov8m', PT_YOLOV8M, 'cpu',    OV_BASE/'yolov8m'/'INT8', 'GPU.0', 'YOLOv8m PT CPU vs OV INT8 GPU'),
    ('yolov8m', PT_YOLOV8M, 'cpu',    OV_BASE/'yolov8m'/'INT8', 'CPU',   'YOLOv8m PT CPU vs OV INT8 CPU'),
    # YOLOv26m
    ('yolov26m', PT_YOLOV26M, 'cuda:0', OV_BASE/'yolo26m_com'/'FP32', 'GPU.0', 'YOLOv26m PT CUDA vs OV FP32 GPU'),
    ('yolov26m', PT_YOLOV26M, 'cuda:0', OV_BASE/'yolo26m_com'/'FP32', 'CPU',   'YOLOv26m PT CUDA vs OV FP32 CPU'),
    ('yolov26m', PT_YOLOV26M, 'cuda:0', OV_BASE/'yolo26m_com'/'FP16', 'GPU.0', 'YOLOv26m PT CUDA vs OV FP16 GPU'),
    ('yolov26m', PT_YOLOV26M, 'cuda:0', OV_BASE/'yolo26m_com'/'FP16', 'CPU',   'YOLOv26m PT CUDA vs OV FP16 CPU'),
    ('yolov26m', PT_YOLOV26M, 'cuda:0', OV_BASE/'yolo26m_com'/'INT8', 'GPU.0', 'YOLOv26m PT CUDA vs OV INT8 GPU'),
    ('yolov26m', PT_YOLOV26M, 'cuda:0', OV_BASE/'yolo26m_com'/'INT8', 'CPU',   'YOLOv26m PT CUDA vs OV INT8 CPU'),
    ('yolov26m', PT_YOLOV26M, 'cpu',    OV_BASE/'yolo26m_com'/'FP32', 'GPU.0', 'YOLOv26m PT CPU vs OV FP32 GPU'),
    ('yolov26m', PT_YOLOV26M, 'cpu',    OV_BASE/'yolo26m_com'/'FP32', 'CPU',   'YOLOv26m PT CPU vs OV FP32 CPU'),
    ('yolov26m', PT_YOLOV26M, 'cpu',    OV_BASE/'yolo26m_com'/'FP16', 'GPU.0', 'YOLOv26m PT CPU vs OV FP16 GPU'),
    ('yolov26m', PT_YOLOV26M, 'cpu',    OV_BASE/'yolo26m_com'/'FP16', 'CPU',   'YOLOv26m PT CPU vs OV FP16 CPU'),
    ('yolov26m', PT_YOLOV26M, 'cpu',    OV_BASE/'yolo26m_com'/'INT8', 'GPU.0', 'YOLOv26m PT CPU vs OV INT8 GPU'),
    ('yolov26m', PT_YOLOV26M, 'cpu',    OV_BASE/'yolo26m_com'/'INT8', 'CPU',   'YOLOv26m PT CPU vs OV INT8 CPU'),
]

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

# ==================== 工具箱 ====================
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
    boxes = boxes[keep_conf]; scores = scores[keep_conf]
    if len(boxes) == 0: return []
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) == 0: return []
    if isinstance(indices, tuple): indices = indices[0]
    return indices.flatten().tolist()

def postprocess_openvino(output, orig_shape, r, dw, dh, conf_thresh=0.25, iou_thresh=0.45):
    pred = output.squeeze().T   # [8400, 37]
    boxes_cxcywh = pred[:, :4]
    scores = pred[:, 4:4+NUM_CLASSES]
    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)

    mask = max_scores > conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]; max_scores = max_scores[mask]; class_ids = class_ids[mask]
    if len(boxes_cxcywh) == 0: return [], [], []

    boxes_xyxy = np.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = scale_boxes(boxes_xyxy, orig_shape, r, dw, dh)
    keep = nms(boxes_xyxy, max_scores, iou_thresh, conf_thresh)
    if len(keep) == 0: return [], [], []
    return boxes_xyxy[keep], max_scores[keep], class_ids[keep]

def compute_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0]); inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2]); inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (area1 + area2 - inter_area + 1e-6)

# ==================== 准备测试图片池 ====================
with open(DATA_YAML, 'r') as f:
    data = yaml.safe_load(f)
val_path = Path(data['val']).resolve()
img_files = list((val_path / 'images').glob('*.jpg'))
random.seed(RANDOM_SEED)
random.shuffle(img_files)
candidate_images = img_files[:CANDIDATE_POOL_SIZE]
print(f"已选定 {len(candidate_images)} 张候选图片")

# ==================== 缓存模型 ====================
pt_model_cache = {}
ov_model_cache = {}

def get_pt_model(pt_path, device):
    key = (pt_path, device)
    if key not in pt_model_cache:
        pt_model_cache[key] = YOLO(pt_path).to(device)
    return pt_model_cache[key]

def get_ov_compiled(ov_dir, ov_device):
    key = (str(ov_dir), ov_device)
    if key not in ov_model_cache:
        xml_files = list(ov_dir.glob('*.xml'))
        if not xml_files: raise FileNotFoundError(f"找不到 .xml: {ov_dir}")
        core = ov.Core()
        model_ov = core.read_model(str(xml_files[0]))
        ov_model_cache[key] = core.compile_model(model_ov, ov_device)
    return ov_model_cache[key]

# ==================== 为单个配置寻找图片 ====================
def find_matching_image(config):
    model_name, pt_path, pt_dev, ov_dir, ov_dev, caption = config
    pt_model = get_pt_model(pt_path, pt_dev)
    compiled_ov = get_ov_compiled(ov_dir, ov_dev)

    for img_path in candidate_images:
        img0 = cv2.imread(str(img_path))
        if img0 is None: continue
        h0, w0 = img0.shape[:2]

        # PyTorch 推理
        results_pt = pt_model(str(img_path), verbose=False)
        boxes_pt = results_pt[0].boxes.data.cpu().numpy()
        if len(boxes_pt) == 0: continue

        # OpenVINO 推理
        img, r, dw, dh = letterbox(img0)
        img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2,0,1).astype(np.float32)/255.0
        input_blob = np.expand_dims(img_input, 0)
        result_ov = compiled_ov([input_blob])[compiled_ov.output(0)]
        boxes_ov, scores_ov, ids_ov = postprocess_openvino(result_ov, (h0, w0), r, dw, dh)
        if len(boxes_ov) == 0: continue

        # 寻找最佳匹配
        best_iou = 0
        best_pair = None
        for i, pt_box in enumerate(boxes_pt):
            for j, (ov_box, ov_cls) in enumerate(zip(boxes_ov, ids_ov)):
                if pt_box[5] != ov_cls: continue
                iou = compute_iou(pt_box[:4], ov_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (pt_box, ov_box, i, j)

        if best_iou >= MIN_IOU:
            pt_box, ov_box, idx_pt, idx_ov = best_pair
            # 获取最佳匹配框的类别名称
            best_cls_id = int(pt_box[5])
            best_class_name = CLASS_NAMES[best_cls_id] if 0 <= best_cls_id < len(CLASS_NAMES) else str(best_cls_id)
            return {
                'img_path': img_path,
                'img0': img0.copy(),
                'boxes_pt': boxes_pt,
                'boxes_ov': boxes_ov,
                'ids_ov': ids_ov,
                'best_pair': best_pair,
                'best_iou': best_iou,
                'caption': caption,
                'best_class': best_class_name,
                'pt_box': pt_box,  # 记录完整框 [x1,y1,x2,y2,conf,cls]
                'ov_box': ov_box  # 记录完整框 [x1,y1,x2,y2] + cls_id
            }
    return None

# ==================== 创建单张子图（带图例和标签） ====================
def create_subplot(data, ax):
    img = data['img0'].copy()          # BGR 格式
    boxes_pt = data['boxes_pt']
    boxes_ov = data['boxes_ov']
    ids_ov = data['ids_ov']
    pt_box, ov_box, idx_pt, idx_ov = data['best_pair']
    caption = data['caption']
    iou = data['best_iou']

    # ----- 1. 绘制所有 PyTorch 框（红色，线宽 4）-----
    for box in boxes_pt:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

    # ----- 2. 绘制所有 OpenVINO 框（蓝色，线宽 4）-----
    for box in boxes_ov:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)

    # ----- 3. 高亮最佳匹配对（绿色粗框，线宽 5）-----
    x1_pt, y1_pt, x2_pt, y2_pt = map(int, pt_box[:4])
    x1_ov, y1_ov, x2_ov, y2_ov = map(int, ov_box[:4])
    cv2.rectangle(img, (x1_pt, y1_pt), (x2_pt, y2_pt), (0, 255, 0), 5)
    cv2.rectangle(img, (x1_ov, y1_ov), (x2_ov, y2_ov), (0, 255, 0), 5)

    # ----- 4. 添加标签（左上角，防遮挡错开）-----
    occupied = []   # 记录已放置标签的区域 (x1, y1, x2, y2)

    def place_label(text, x, y, color):
        """在 (x,y) 处放置文本，若重叠则向下移动"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tx, ty = x, y - 5
        rect = (tx, ty - h, tx + w, ty + baseline)
        # 尝试移动至多两次
        for attempt in range(3):
            overlap = False
            for (ox1, oy1, ox2, oy2) in occupied:
                if tx < ox2 and tx + w > ox1 and ty - h < oy2 and ty + baseline > oy1:
                    overlap = True
                    break
            if not overlap:
                break
            ty += (h + baseline + 2)   # 向下错开
            rect = (tx, ty - h, tx + w, ty + baseline)
        occupied.append(rect)
        cv2.putText(img, text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

    # 放置 PyTorch 标签
    for box in boxes_pt:
        x1, y1 = map(int, box[:2])
        place_label('PT', x1, y1, (0, 0, 255))

    # 放置 OpenVINO 标签
    for box in boxes_ov:
        x1, y1 = map(int, box[:2])
        place_label('OV', x1, y1, (255, 0, 0))

    # 最佳匹配对加绿色标签
    place_label('PT*', int(pt_box[0]), int(pt_box[1]), (0, 255, 0))
    place_label('OV*', int(ov_box[0]), int(ov_box[1]), (0, 255, 0))

    '''
    # ----- 5. 在图像右上角直接绘制图例（避免 matplotlib legend）-----
    h_img, w_img = img.shape[:2]
    # 图例背景矩形
    cv2.rectangle(img, (w_img-180, 10), (w_img-5, 75), (240, 240, 240), -1)
    cv2.rectangle(img, (w_img-180, 10), (w_img-5, 75), (100, 100, 100), 1)
    # 色块 + 文字
    # PyTorch
    cv2.rectangle(img, (w_img-170, 18), (w_img-155, 30), (0, 0, 255), -1)
    cv2.putText(img, 'PyTorch', (w_img-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    # OpenVINO
    cv2.rectangle(img, (w_img-170, 38), (w_img-155, 50), (255, 0, 0), -1)
    cv2.putText(img, 'OpenVINO', (w_img-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    # Best Match
    cv2.rectangle(img, (w_img-170, 58), (w_img-155, 70), (0, 255, 0), -1)
    cv2.putText(img, 'Best Match', (w_img-150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    '''

    # ----- 6. 显示 -----
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis('off')

    # ----- 7. 子图标题 -----
    ax.set_title(f"{caption}\nIoU = {iou:.4f}", fontsize=20, pad=3)

# ==================== 生成大图 ====================
def process_model_group(group_name, configs_slice):
    print(f"\n处理 {group_name} 共 {len(configs_slice)} 个配置...")
    csv_records = []
    subplot_data_list = []
    for config in tqdm(configs_slice, desc=group_name):
        res = find_matching_image(config)
        if res:
            subplot_data_list.append(res)
            # 计算中心偏移等
            pb = res['pt_box']
            ob = res['ov_box']
            cx_pt = (pb[0] + pb[2]) / 2
            cy_pt = (pb[1] + pb[3]) / 2
            cx_ov = (ob[0] + ob[2]) / 2
            cy_ov = (ob[1] + ob[3]) / 2
            csv_records.append({
                '配置名称': res['caption'],
                '目标类别': res['best_class'],
                'PyTorch_x1': f"{pb[0]:.2f}",
                'PyTorch_y1': f"{pb[1]:.2f}",
                'PyTorch_x2': f"{pb[2]:.2f}",
                'PyTorch_y2': f"{pb[3]:.2f}",
                'PyTorch置信度': f"{pb[4]:.2f}",
                'OpenVINO_x1': f"{ob[0]:.2f}",
                'OpenVINO_y1': f"{ob[1]:.2f}",
                'OpenVINO_x2': f"{ob[2]:.2f}",
                'OpenVINO_y2': f"{ob[3]:.2f}",
                'IoU': f"{res['best_iou']:.4f}",
                '中心点偏移cx': f"{abs(cx_pt - cx_ov):.2f}",
                '中心点偏移cy': f"{abs(cy_pt - cy_ov):.2f}"
            })
        else:
            print(f"警告: 配置 '{config[-1]}' 未找到合适图片，跳过。")

    if not subplot_data_list:
        print("没有可用的子图数据！")
        return

    n = len(subplot_data_list)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, data in enumerate(subplot_data_list):
        create_subplot(data, axes[idx])
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = f"{group_name}_matched_subplots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"大图已保存: {output_path}")
    plt.close()
    import csv
    csv_filename = f"{group_name}_matched_details.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['配置名称', '目标类别', 'PyTorch_x1', 'PyTorch_y1', 'PyTorch_x2', 'PyTorch_y2',
                      'PyTorch置信度', 'OpenVINO_x1', 'OpenVINO_y1', 'OpenVINO_x2', 'OpenVINO_y2',
                      'IoU', '中心点偏移cx', '中心点偏移cy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_records)
    print(f"细节 CSV 已保存: {csv_filename}")

# ==================== 按模型分离并执行 ====================
v8m_configs = [c for c in configs if c[0] == 'yolov8m']
v26m_configs = [c for c in configs if c[0] == 'yolov26m']

process_model_group("YOLOv8m", v8m_configs)
process_model_group("YOLOv26m", v26m_configs)

print("全部完成。")