import cv2
import numpy as np
from openvino import Core
from pathlib import Path

# ==================== 你的 33 个类别名称（严格按训练顺序） ====================
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
NUM_CLASSES = len(CLASS_NAMES)  # = 33

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """调整图片大小并填充，保持宽高比，返回 (处理后的图片, 缩放比例, 填充的宽高)"""
    shape = img.shape[:2]  # 原图高、宽
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh

def scale_boxes(boxes, orig_shape, r, dw, dh):
    """将模型输出坐标（相对于填充后的640x640）映射回原图坐标"""
    # boxes: [N, 4] (x1, y1, x2, y2)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])
    return boxes

def nms(boxes, scores, iou_threshold=0.45, conf_threshold=0.25):
    """执行NMS，返回保留的索引"""
    # 先按置信度过滤
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

def postprocess_yolov8(output, input_size, orig_shape, r, dw, dh, conf_thresh=0.25, iou_thresh=0.45):
    """
    处理 YOLOv8 输出 [1, 37, 8400] -> 返回 (boxes_xyxy, scores, class_ids) 在原图坐标系
    """
    # 去掉 batch 维度，并转置为 [8400, 37]
    pred = output.squeeze().T  # 现在 shape = [8400, 37]
    # 分离边界框 (cx, cy, w, h) 和 33个类别分数
    boxes_cxcywh = pred[:, :4]
    scores = pred[:, 4:4+NUM_CLASSES]
    # 获取最大置信度和对应类别
    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)

    # 置信度过滤
    mask = max_scores > conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return [], [], []

    # 转换 cxcywh -> xyxy (相对于模型输入尺寸 640x640, 含填充)
    boxes_xyxy = np.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2

    # 映射到原图坐标
    boxes_xyxy = scale_boxes(boxes_xyxy, orig_shape, r, dw, dh)

    # NMS (输入已经是 xyxy)
    keep = nms(boxes_xyxy, max_scores, iou_thresh, conf_thresh)
    if len(keep) == 0:
        return [], [], []
    boxes_xyxy = boxes_xyxy[keep]
    max_scores = max_scores[keep]
    class_ids = class_ids[keep]

    return boxes_xyxy, max_scores, class_ids

def draw_detections(image, boxes, scores, class_ids, model_name):
    """在图片上绘制框和标签"""
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        if 0 <= cls_id < NUM_CLASSES:
            label = f"{model_name} {CLASS_NAMES[cls_id]}: {score:.2f}"
        else:
            label = f"{model_name} class_{cls_id}: {score:.2f}"
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def main():
    # ==================== 请修改为你的实际路径 ====================
    MODEL_V26M = r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolo26m_com\INT8\yolo26m_com.xml"
    MODEL_V8M = r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolov8m\INT8\yolov8m.xml"
    OUTPUT_DIR = Path("detection_results")  # 输出目录
    # 输入图片路径
    INPUT_IMAGE = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\val\images\ds2_blueberry_bushes_clipart_16_jpg.rf.6c9d98d01b97c2fa4bbf82a8c551f1ba.jpg"
    # ===========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ie = Core()

    # ---------- 加载 YOLOv8m ----------
    print("加载 YOLOv8m 模型...")
    model_v8 = ie.read_model(MODEL_V8M)
    compiled_v8 = ie.compile_model(model_v8, "CPU")
    output_v8 = compiled_v8.output(0)

    # ---------- 加载 YOLOv26m (如果存在) ----------
    print("加载 YOLOv26m 模型...")
    model_v26 = ie.read_model(MODEL_V26M)
    compiled_v26 = ie.compile_model(model_v26, "CPU")
    output_v26 = compiled_v26.output(0)

    # ---------- 读取图片并预处理 ----------
    img0 = cv2.imread(INPUT_IMAGE)
    if img0 is None:
        raise FileNotFoundError(f"图片不存在: {INPUT_IMAGE}")
    orig_h, orig_w = img0.shape[:2]

    # 使用 letterbox 预处理 (保持宽高比，填充到 640x640)
    img, r, dw, dh = letterbox(img0, (640, 640))
    # 转换为模型输入格式: BGR -> RGB, HWC -> CHW, 归一化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_blob = np.expand_dims(img, axis=0)  # [1, 3, 640, 640]

    # ---------- YOLOv8m 推理 ----------
    print("YOLOv8m 检测中...")
    result_v8 = compiled_v8([input_blob])[output_v8]  # shape: [1, 37, 8400]
    boxes_v8, scores_v8, ids_v8 = postprocess_yolov8(
        result_v8, 640, (orig_h, orig_w), r, dw, dh,
        conf_thresh=0.25, iou_thresh=0.45
    )
    print(f"YOLOv8m 检测到 {len(boxes_v8)} 个目标")

    # ---------- YOLOv26m 推理 (自动适配常见形状) ----------
    print("YOLOv26m 检测中...")
    result_v26 = compiled_v26([input_blob])[output_v26]
    print(f"YOLOv26m 原始输出形状: {result_v26.shape}")

    # 尝试通用解析 (支持 [1,37,N] 或 [1,N,37] 或 [1,84,N] 等)
    try:
        # 如果形状是 [1,37,8400] 就转置为 [8400,37]
        if len(result_v26.shape) == 3 and result_v26.shape[1] == 37 and result_v26.shape[2] > 37:
            pred = result_v26.squeeze().T  # [N, 37]
        elif len(result_v26.shape) == 3 and result_v26.shape[2] == 37:
            pred = result_v26.squeeze()     # [N, 37]
        else:
            # 其他形状直接尝试 squeeze
            pred = result_v26.squeeze()
            if pred.shape[1] == 37:
                pass
            elif pred.shape[0] == 37:
                pred = pred.T
            else:
                raise ValueError(f"不支持的形状: {result_v26.shape}")

        # 分离框和类别 (同样假设 37 维度)
        boxes_cxcywh = pred[:, :4]
        scores = pred[:, 4:4+NUM_CLASSES]
        max_scores = scores.max(axis=1)
        class_ids = scores.argmax(axis=1)
        mask = max_scores > 0.25
        boxes_cxcywh = boxes_cxcywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_cxcywh) > 0:
            boxes_xyxy = np.zeros_like(boxes_cxcywh)
            boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
            boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
            boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
            boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
            boxes_xyxy = scale_boxes(boxes_xyxy, (orig_h, orig_w), r, dw, dh)
            keep = nms(boxes_xyxy, max_scores, 0.45, 0.25)
            boxes_v26 = boxes_xyxy[keep]
            scores_v26 = max_scores[keep]
            ids_v26 = class_ids[keep]
        else:
            boxes_v26, scores_v26, ids_v26 = [], [], []
    except Exception as e:
        print(f"YOLOv26m 解析失败: {e}")
        boxes_v26, scores_v26, ids_v26 = [], [], []

    print(f"YOLOv26m 检测到 {len(boxes_v26)} 个目标")

    # ---------- 绘制并保存 ----------
    img_v8 = draw_detections(img0.copy(), boxes_v8, scores_v8, ids_v8, "YOLOv8m")
    img_v26 = draw_detections(img0.copy(), boxes_v26, scores_v26, ids_v26, "YOLOv26m")

    out_v8 = OUTPUT_DIR / "result_v8m.jpg"
    out_v26 = OUTPUT_DIR / "result_v26m.jpg"
    cv2.imwrite(str(out_v8), img_v8)
    cv2.imwrite(str(out_v26), img_v26)

    print(f"\n结果已保存到: {OUTPUT_DIR.absolute()}")
    print(f"  YOLOv8m: {out_v8}")
    print(f"  YOLOv26m: {out_v26}")

if __name__ == "__main__":
    main()

'''
MODEL_V26M = r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolo26m_com\INT8\yolo26m_com.xml"
    MODEL_V8M = r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolov8m\INT8\yolov8m.xml"

    # 输入图片路径
   SAMPLE_IMAGE_PATH =r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\val\images\ds2_blueberry_bushes_clipart_16_jpg.rf.6c9d98d01b97c2fa4bbf82a8c551f1ba.jpg"
'''