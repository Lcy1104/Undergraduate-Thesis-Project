import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

def main():
    # 1. 直接加载 YOLO26-m，结构已内置新 head（去 DFL、去 NMS）
    model = YOLO('yolo26m.pt')          # 也可选 yolo26n/s/l/x

    # 2. 训练配置——只改“新增/必要”项，其余保持与原 yolov8 一致
    model.train(
        data=r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml",
        epochs=180,
        imgsz=640,
        batch=16,
        device=0,

        # ---- YOLO26 官方推荐优化器 ----
        optimizer='MuSGD',      # 替代 AdamW；lr0 仍可用 0.001
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=5,

        # ---- 打开 YOLO26 训练期新特性 ----
        progloss=True,          # 渐进式损失平衡
        stal=True,              # 小目标感知标签分配
        nmsfree=True,           # 训练期即采用 NMS-free 分支，推理直接受益

        # ---- 常规增强 ----
        augment=True,
        cache=True,
        workers=4,

        # 可选：把工程名改一下，方便 tensorboard 区分
        name='yolo26m_tomato'
        '''
        具体作用：
训练生成的所有中间文件（权重、日志、可视化图片等）都会放到
runs/detect/yolo26m_tomato/
而不是默认的 runs/detect/train/。
同时启动多次实验时，各自目录独立，不会相互覆盖。
TensorBoard 里也会以这个名字作为标签，方便对比曲线。
        '''
    )

    '''# 3. 验证 & 导出（验证 NMS-free 推理速度）
    metrics = model.val()
    model.export(format='onnx', nmsfree=True)  # 导出同样无 NMS 的 ONNX
    print('mAP50-95:', metrics.box.map)'''
    '''
    metrics = model.val()
用刚训练好的权重，在验证集上跑一次完整评估
返回的 metrics 里就包含 mAP@0.5、<EMAIL_ADDRESS>、推理速度（ms/img）等指标
因为训练时已经打开 nmsfree=True，这里默认也会走“无 NMS”分支，所以测到的速度就是真正的端到端延迟
model.export(format='onnx', nmsfree=True)
把 PyTorch 权重导出成 ONNX 文件
加 nmsfree=True 后，ONNX 图里不会再插 NMS 算子，推理侧直接拿到最终框+置信度，部署更简单、延迟更低
生成的文件在 runs/detect/yolo26m_tomato/weights/yolo26m_nmsfree.onnx（或类似路径）
print('mAP50-95:', metrics.box.map)
把最关键的 <EMAIL_ADDRESS> 打印到终端，一眼判断训练是否达标
    '''

if __name__ == '__main__':
    main()