import cv2
import torch
from ultralytics import YOLO
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

def print_detections(model_name, results):
    """打印检测结果详情"""
    detections = results[0].boxes
    if detections is None or len(detections) == 0:
        print(f"\n{model_name} 检测到 0 个目标")
        return
    boxes = detections.xyxy.cpu().numpy()
    scores = detections.conf.cpu().numpy()
    class_ids = detections.cls.cpu().numpy().astype(int)
    print(f"\n{model_name} 检测到 {len(boxes)} 个目标:")
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids), 1):
        class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"未知类别({cls_id})"
        print(f"  {i}. {class_name} - 置信度: {score:.4f} - 位置: [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")

def draw_detections(image, results, model_name):
    """在图片上绘制检测框和标签（使用 results.plot() 更方便，但为了自定义标签我们手动绘制）"""
    detections = results[0].boxes
    if detections is None:
        return image
    boxes = detections.xyxy.cpu().numpy()
    scores = detections.conf.cpu().numpy()
    class_ids = detections.cls.cpu().numpy().astype(int)
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        if 0 <= cls_id < len(CLASS_NAMES):
            label = f"{model_name} {CLASS_NAMES[cls_id]}: {score:.2f}"
        else:
            label = f"{model_name} class_{cls_id}: {score:.2f}"
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def main():
    # ==================== 请修改为你的实际路径 ====================
    MODEL_V8M = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights\best.pt"      # YOLOv8m 的 .pt 文件
    MODEL_V26M = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\yolo26m_tomato3\yolo26m_tomato3\weights\best.pt" # YOLOv26m 的 .pt 文件
    INPUT_IMAGE = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\val\images\ds2_blueberry_bushes_clipart_16_jpg.rf.6c9d98d01b97c2fa4bbf82a8c551f1ba.jpg"
    OUTPUT_DIR = Path("detection_results_pt")
    CONF_THRESH = 0.25
    IOU_THRESH = 0.45
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # ===========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"使用设备: {DEVICE.upper()}")

    # 加载模型（自动使用 GPU）
    print("加载 YOLOv8m 模型...")
    model_v8 = YOLO(MODEL_V8M)
    model_v8.to(DEVICE)

    print("加载 YOLOv26m 模型...")
    model_v26 = YOLO(MODEL_V26M)
    model_v26.to(DEVICE)

    # 读取原始图片（用于绘图保存）
    img0 = cv2.imread(INPUT_IMAGE)
    if img0 is None:
        raise FileNotFoundError(f"图片不存在: {INPUT_IMAGE}")

    # ---------- YOLOv8m 推理 ----------
    print("\nYOLOv8m 检测中...")
    results_v8 = model_v8.predict(
        source=INPUT_IMAGE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        device=DEVICE,
        verbose=False
    )
    print_detections("YOLOv8m", results_v8)

    # ---------- YOLOv26m 推理 ----------
    print("\nYOLOv26m 检测中...")
    results_v26 = model_v26.predict(
        source=INPUT_IMAGE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        device=DEVICE,
        verbose=False
    )
    print_detections("YOLOv26m", results_v26)

    # 绘制并保存结果
    img_v8 = draw_detections(img0.copy(), results_v8, "YOLOv8m")
    img_v26 = draw_detections(img0.copy(), results_v26, "YOLOv26m")

    out_v8 = OUTPUT_DIR / "result_v8m.jpg"
    out_v26 = OUTPUT_DIR / "result_v26m.jpg"
    cv2.imwrite(str(out_v8), img_v8)
    cv2.imwrite(str(out_v26), img_v26)

    print(f"\n结果图片已保存到: {OUTPUT_DIR.absolute()}")
    print(f"  YOLOv8m: {out_v8}")
    print(f"  YOLOv26m: {out_v26}")

if __name__ == "__main__":
    main()