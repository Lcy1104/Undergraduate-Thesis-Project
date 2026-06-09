import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO


def main():
    # 加载模型
    model = YOLO(r'E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights\best.pt')

    # 对单张图像预测
    results = model.predict(
        source=r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\7\test\images\000_jpg.rf.79b07e9657b862eca6972264b5e2fe0c.jpg",  # 或文件夹路径、视频
        conf=0.25,  # 置信度阈值
        iou=0.45,  # NMS IoU阈值
        save=True,  # 保存结果图像
        save_txt=True,  # 保存标注文本
        show=True  # 弹出窗口显示结果
    )

    # 对文件夹批量预测
    # results = model.predict(
    #     source=r"E:\path\to\test\images_folder",
    #     conf=0.25,
    #     save=True,
    #     save_txt=True
    # )


if __name__ == '__main__':
    main()