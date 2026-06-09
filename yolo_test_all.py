import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO


def main():
    # 加载训练好的最佳权重
    model = YOLO(r'E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights\best.pt')

    # 验证模型
    results = model.val(
        data=r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml",
        split='val',  # 验证集
        batch=16,
        imgsz=640,
        device=0
    )

    # 查看关键指标
    print(f"mAP50: {results.box.map50:.4f}")  # IoU=0.5 时的 mAP
    print(f"mAP50-95: {results.box.map:.4f}")  # IoU=0.5:0.95 时的 mAP
    print(f"Precision: {results.box.mp:.4f}")  # 精确率
    print(f"Recall: {results.box.mr:.4f}")  # 召回率


if __name__ == '__main__':
    main()