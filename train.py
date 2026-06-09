import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

def main():
    # 将所有训练代码放入函数中
    model = YOLO('yolov8m.pt')
    model.train(
        data=r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml",
        epochs=180,
        imgsz=640,
        batch=16,
        device=0,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=5,
        augment=True,
        cache=True,
        workers=4
    )

if __name__ == '__main__':
    main()