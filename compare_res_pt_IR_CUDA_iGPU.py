import time
import cv2
import numpy as np
import torch
import openvino as ov
import yaml
import torchvision  # 为NMS提供torch.ops.torchvision.nms支持
from pathlib import Path
from tqdm import tqdm
import gc
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import warnings
import random

# ========== 全局设置 ==========
torch.serialization.DEFAULT_LOAD_WEIGHTS_ONLY = False
warnings.filterwarnings('ignore')

# ========== 内置 NMS ==========
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    if prediction.shape[-1] == 84:
        prediction = prediction[..., :6]
    output = []
    for xi, x in enumerate(prediction):
        if x.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=x.device))
            continue
        x = x[x[:, 4].argsort(descending=True)]
        keep = torch.ops.torchvision.nms(x[:, :4], x[:, 4], iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]
        output.append(x[keep])
    return output

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

# ========== mAP 计算 ==========
def compute_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou

def compute_map(predictions, targets, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    if not targets or not any(t for t in targets):
        return {'mAP_50': 0.0, 'mAP_50_95': 0.0}
    num_classes = max([t[4] for tlist in targets for t in tlist] + [0]) + 1
    ap_per_class = []
    for c in range(num_classes):
        preds_c = [p for img_preds in predictions for p in img_preds if int(p[5]) == c]
        targets_c = [t for img_targets in targets for t in img_targets if int(t[4]) == c]
        if len(targets_c) == 0:
            continue
        ap_list = []
        for iou_thr in iou_thresholds:
            matched_target = set()
            tp = np.zeros(len(preds_c))
            fp = np.zeros(len(preds_c))
            preds_sorted = sorted(preds_c, key=lambda x: x[4], reverse=True)
            for i, p in enumerate(preds_sorted):
                best_iou = 0
                best_j = -1
                for j, t in enumerate(targets_c):
                    if j in matched_target:
                        continue
                    iou = compute_iou(p[:4], t[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= iou_thr:
                    tp[i] = 1
                    matched_target.add(best_j)
                else:
                    fp[i] = 1
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            prec = tp_cum / (tp_cum + fp_cum + 1e-6)
            rec = tp_cum / len(targets_c)
            ap = np.trapz(prec, rec) if len(rec) > 1 else 0.0
            ap_list.append(ap)
        ap_per_class.append(np.mean(ap_list))
    mAP = np.mean(ap_per_class) if ap_per_class else 0.0
    # mAP@0.5
    ap50_per_class = []
    for c in range(num_classes):
        preds_c = [p for img_preds in predictions for p in img_preds if int(p[5]) == c]
        targets_c = [t for img_targets in targets for t in img_targets if int(t[4]) == c]
        if len(targets_c) == 0:
            continue
        matched_target = set()
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))
        preds_sorted = sorted(preds_c, key=lambda x: x[4], reverse=True)
        for i, p in enumerate(preds_sorted):
            best_iou = 0
            best_j = -1
            for j, t in enumerate(targets_c):
                if j in matched_target:
                    continue
                iou = compute_iou(p[:4], t[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= 0.5:
                tp[i] = 1
                matched_target.add(best_j)
            else:
                fp[i] = 1
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)
        rec = tp_cum / len(targets_c)
        ap50 = np.trapz(prec, rec) if len(rec) > 1 else 0.0
        ap50_per_class.append(ap50)
    mAP50 = np.mean(ap50_per_class) if ap50_per_class else 0.0
    return {'mAP_50': mAP50, 'mAP_50_95': mAP}

# ========== 单模型测试器 ==========
class SingleModelTester:
    def __init__(self, model_name, pt_path, ov_fp32_dir, ov_fp16_dir, ov_int8_dir, data_yaml,
                 num_samples=150, random_seed=42, fixed_image_paths=None, fixed_label_paths=None):
        self.model_name = model_name
        self.pt_path = Path(pt_path)
        self.ov_dirs = {'FP32': Path(ov_fp32_dir), 'FP16': Path(ov_fp16_dir), 'INT8': Path(ov_int8_dir)}
        self.data_yaml = Path(data_yaml)
        self.num_samples = num_samples
        self.random_seed = random_seed
        # 如果传入了固定的图片列表和标签字典，则直接使用；否则自己采样
        if fixed_image_paths is not None and fixed_label_paths is not None:
            self.image_paths = fixed_image_paths
            self.label_paths = fixed_label_paths
            # 仍需加载类别名称（从data.yaml读取）
            with open(self.data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            self.class_names = data['names']
            self.num_classes = len(self.class_names)
            print(f"\n📂 使用固定测试集: {len(self.image_paths)} 张图片")
        else:
            self._load_dataset()
        self.results = {}

    def _load_dataset(self):
        print(f"\n📂 加载数据集: {self.data_yaml}")
        with open(self.data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        self.class_names = data['names']
        self.num_classes = len(self.class_names)

        all_pairs = []
        for split in ['train', 'val', 'test']:
            if split not in data:
                continue
            split_path = Path(data[split]).resolve()
            img_dir = split_path / "images"
            lbl_dir = split_path / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                print(f"   ⚠️ 跳过 {split}：images或labels目录不存在")
                continue
            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
            for img in images:
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    all_pairs.append((img, lbl))
            print(f"   ✅ {split}: {len(images)} 张图片, {len([p for p in all_pairs if p[0].parent == img_dir])} 个有效标签")
        if not all_pairs:
            raise ValueError("未找到任何有效的图片-标签对！")
        random.seed(self.random_seed)
        if self.num_samples and self.num_samples < len(all_pairs):
            sampled = random.sample(all_pairs, self.num_samples)
        else:
            sampled = all_pairs
        self.image_paths = [p[0] for p in sampled]
        self.label_paths = {p[0].stem: p[1] for p in sampled}
        print(f"   📸 最终采样 {len(self.image_paths)} 张图片 (随机种子 {self.random_seed})")

    def _load_ground_truth(self, img_stem, img_path=None):
        """GT 坐标转换为原始图像绝对像素坐标"""
        label_file = self.label_paths.get(img_stem)
        if not label_file or not label_file.exists():
            return []

        # 获取原始图像尺寸
        if img_path is None:
            img_path = next((p for p in self.image_paths if p.stem == img_stem), None)
        if img_path is None:
            return []

        img = cv2.imread(str(img_path))
        if img is None:
            return []
        h, w = img.shape[:2]

        boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                xc, yc, w_norm, h_norm = map(float, parts[1:])

                # 转换为原始图像绝对坐标（不归一化到640）
                x1 = (xc - w_norm / 2) * w
                y1 = (yc - h_norm / 2) * h
                x2 = (xc + w_norm / 2) * w
                y2 = (yc + h_norm / 2) * h

                boxes.append([x1, y1, x2, y2, class_id])
        return boxes

    def _preprocess_image(self, img_path, backend):
        """使用 Ultralytics 官方预处理（与 yolo predict 完全一致）"""
        from ultralytics.data.augment import LetterBox
        import cv2
        import numpy as np

        # 读取原始图像 (BGR)
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            raise ValueError(f"无法读取图片: {img_path}")

        h0, w0 = im0.shape[:2]

        # 【关键】使用 Ultralytics 官方的 LetterBox
        # auto=True 会自动对齐 stride=32，这是与手动 resize 的关键区别
        letterbox = LetterBox(new_shape=640, auto=True, stride=32)
        im = letterbox(image=im0)  # 返回 640x640 BGR，灰色 114 填充

        # BGR -> RGB, HWC -> CHW, /255
        im = im[..., ::-1].transpose(2, 0, 1)  # RGB, CHW
        im = np.ascontiguousarray(im, dtype=np.float32) / 255.0

        # 计算后处理参数（与 LetterBox 内部一致）
        r = min(640 / h0, 640 / w0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        dw, dh = 640 - new_unpad[0], 640 - new_unpad[1]
        # auto=True 时，dw/dh 已经是 %32 对齐后的值
        dw /= 2  # 左右 padding
        dh /= 2  # 上下 padding

        pad_info = (r, dw, dh, w0, h0)

        if backend == "pytorch":
            return torch.from_numpy(im).unsqueeze(0), pad_info
        else:
            return im[np.newaxis, ...], pad_info

    def _postprocess_openvino(self, output, pad_info, conf_thres=0.25, iou_thres=0.45):
        r, dw, dh, orig_w, orig_h = pad_info

        # 情况1：已经过NMS的格式 [1, N, 6]
        if output.ndim == 3 and output.shape[2] == 6:
            boxes = torch.from_numpy(output[0]).float()  # [N, 6]
            conf = boxes[:, 4]
            print(f"   conf范围(输入): {conf.min():.4f}-{conf.max():.4f}, 均值: {conf.mean():.4f}")
            keep = boxes[:, 4] > conf_thres
            boxes = boxes[keep]
            if len(boxes) == 0:
                return np.empty((0, 6))
            boxes_np = boxes.cpu().numpy()
            boxes_np[:, [0, 2]] -= dw
            boxes_np[:, [1, 3]] -= dh
            boxes_np[:, :4] /= r
            boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, orig_w)
            boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, orig_h)
            return boxes_np

        # 情况2：原始检测头输出 [1, 37, 8400]
        if output.ndim == 3 and output.shape[1] == 37:
            pred = torch.from_numpy(output[0].T).float()  # [8400, 37]
            cx = pred[:, 0]
            if cx.max() <= 1.0:
                scale = 640.0
                print("   [自动检测] 坐标是归一化值 (0~1)，将乘以640")
            else:
                scale = 1.0
                print("   [自动检测] 坐标已是绝对像素值 (0~640)，无需缩放")
            print("前3个锚点的原始值:")
            for i in range(min(3, pred.shape[0])):
                print(
                    f"  anchor {i}: cx={pred[i, 0]:.4f}, cy={pred[i, 1]:.4f}, w={pred[i, 2]:.4f}, h={pred[i, 3]:.4f}, obj={pred[i, 4]:.4f}, cls_max={pred[i, 5:].max():.4f}")
            # 前4通道是绝对像素坐标 (cx, cy, w, h)
            cx, cy, w, h = pred[:, 0] * scale, pred[:, 1] * scale, pred[:, 2] * scale, pred[:, 3] * scale
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # 置信度：通道5~36是类别分数，取最大值作为置信度（原始值已在0~1之间）
            cls_scores = pred[:, 5:]  # [8400, num_classes]
            conf, cls_ids = torch.max(cls_scores, dim=1)

            # 打印 conf 统计（保留您原有的输出）
            print(f"   conf范围: {conf.min():.4f}-{conf.max():.4f}, 均值: {conf.mean():.4f}")

            boxes = torch.stack([x1, y1, x2, y2, conf, cls_ids.float()], dim=1)

            # 置信度过滤
            keep = conf > conf_thres
            boxes = boxes[keep]
            if len(boxes) == 0:
                return np.empty((0, 6))

            # NMS
            boxes_nms = non_max_suppression(boxes.unsqueeze(0),
                                            conf_thres=0.0,
                                            iou_thres=iou_thres)[0]
            if boxes_nms is None or len(boxes_nms) == 0:
                return np.empty((0, 6))

            boxes = boxes_nms.cpu().numpy()

            # 映射回原始图像尺寸
            boxes[:, [0, 2]] -= dw
            boxes[:, [1, 3]] -= dh
            boxes[:, :4] /= r
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

            return boxes

        raise ValueError(f"不支持的输出格式: {output.shape}")

    def test_pytorch_cuda(self):
        print(f"\n🔵 {self.model_name} | PyTorch CUDA (原始精度)")
        device = torch.device('cuda:0')
        model = YOLO(str(self.pt_path)).to(device)
        model.eval()

        times, all_preds, all_targets = [], [], []

        for img_path in tqdm(self.image_paths, desc="CUDA推理"):
            # 读取原始图像
            orig_img = cv2.imread(str(img_path))
            if orig_img is None:
                continue
            h, w = orig_img.shape[:2]

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                # 传入原始图像，YOLO内部做Letterbox，输出自动映射回原始尺寸
                results = model(orig_img, verbose=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

            # 提取结果（YOLO已经映射到原始图像坐标）
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.data.cpu().numpy()  # [N, 6] (x1,y1,x2,y2,conf,cls)
            else:
                boxes = np.empty((0, 6))

            all_preds.append(boxes)
            all_targets.append(self._load_ground_truth(img_path.stem, img_path))

        metrics = compute_map(all_preds, all_targets)
        stats = {'framework':'PyTorch','precision':'原始(FP32)','device':'CUDA',
                 'avg_time_ms':np.mean(times)*1000, 'fps':1/np.mean(times),
                 'mAP_50':metrics['mAP_50'], 'mAP_50_95':metrics['mAP_50_95']}
        self.results['pytorch_cuda'] = stats
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)
        return stats

    def test_pytorch_cpu(self):
        print(f"\n🔵 {self.model_name} | PyTorch CPU (原始精度)")
        device = torch.device('cpu')
        model = YOLO(str(self.pt_path)).to(device)
        model.eval()

        times, all_preds, all_targets = [], [], []

        for img_path in tqdm(self.image_paths, desc="CPU推理"):
            # 【修复】使用原始图像，让YOLO内部处理预处理
            orig_img = cv2.imread(str(img_path))
            if orig_img is None:
                continue

            start = time.perf_counter()
            with torch.no_grad():
                results = model(orig_img, verbose=False)  # YOLO内部做Letterbox
            times.append(time.perf_counter() - start)

            # 提取结果（YOLO已映射到原始坐标）
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.data.cpu().numpy()
            else:
                boxes = np.empty((0, 6))

            all_preds.append(boxes)
            all_targets.append(self._load_ground_truth(img_path.stem, img_path))

        metrics = compute_map(all_preds, all_targets)
        stats = {'framework': 'PyTorch', 'precision': '原始(FP32)', 'device': 'CPU',
                 'avg_time_ms': np.mean(times) * 1000, 'fps': 1 / np.mean(times),
                 'mAP_50': metrics['mAP_50'], 'mAP_50_95': metrics['mAP_50_95']}
        self.results['pytorch_cpu'] = stats
        del model
        gc.collect()
        time.sleep(5)
        return stats

    def test_openvino(self, precision, device_name):
        print(f"\n🔵 {self.model_name} | OpenVINO {precision} on {device_name}")

        model_dir = self.ov_dirs[precision]
        xml_file = next(model_dir.glob("*.xml"))

        import openvino as ov
        core = ov.Core()
        model = core.read_model(str(xml_file))
        print("模型输入形状:", model.input(0).shape)
        print("模型输出形状:", model.output(0).shape)

        # ===== 关键修改 1：直接使用 'GPU.0'，不转换为 'GPU' =====
        ov_device = device_name  # 原来是 'GPU' if 'GPU' in device_name else 'CPU'
        # ========================================================

        # 编译模型
        compiled_model = core.compile_model(model, ov_device)
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)

        times, all_preds, all_targets = [], [], []

        for idx, img_path in enumerate(tqdm(self.image_paths, desc=f"{precision}推理")):
            input_tensor, pad_info = self._preprocess_image(img_path, backend='openvino')

            start = time.perf_counter()

            # ===== 关键修改 2：显式创建 Tensor，声明类型和布局 =====
            ov_tensor = ov.Tensor(input_tensor)
            infer_request = compiled_model.create_infer_request()
            infer_request.set_input_tensor(0, ov_tensor)
            infer_request.infer()
            output = infer_request.get_output_tensor(0).data
            # =======================================================

            times.append(time.perf_counter() - start)

            # 诊断打印（仅 GPU 第一张图）
            if 'GPU' in ov_device and idx == 0:
                print(f"\n=== GPU 原始输出诊断 (模型: {self.model_name}, 精度: {precision}) ===")
                print(f"output shape: {output.shape}")
                print(f"output dtype: {output.dtype}")
                print(f"output min/max/mean: {output.min():.6f} / {output.max():.6f} / {output.mean():.6f}")
                print(f"output 是否包含 nan: {np.isnan(output).any()}")
                if output.shape[2] == 6:
                    for i in range(min(5, output.shape[1])):
                        print(f"  box[{i}]: {output[0, i]}")
                print("====================================\n")

            boxes = self._postprocess_openvino(output, pad_info, conf_thres=0.25, iou_thres=0.45)
            all_preds.append(boxes)
            all_targets.append(self._load_ground_truth(img_path.stem, img_path))

        metrics = compute_map(all_preds, all_targets)
        stats = {
            'framework': 'OpenVINO',
            'precision': precision,
            'device': ov_device,
            'avg_time_ms': np.mean(times) * 1000,
            'fps': 1 / np.mean(times),
            'mAP_50': metrics['mAP_50'],
            'mAP_50_95': metrics['mAP_50_95']
        }
        key = f"openvino_{precision.lower()}_{ov_device.lower().replace('.', '_')}"
        self.results[key] = stats
        gc.collect()
        time.sleep(2)
        return stats

    def _visualize_detections(self, img_path, pred_boxes, gt_boxes, save_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return
        for box in gt_boxes:
            x1, y1, x2, y2, cls_id = box[:5]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(img, f"GT:{self.class_names[int(cls_id)]}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        for box in pred_boxes:
            x1, y1, x2, y2, conf, cls_id = box[:6]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            cv2.putText(img, f"{self.class_names[int(cls_id)]}:{conf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imwrite(str(save_path), img)
        print(f"可视化保存至: {save_path}")

    def run_all_tests(self):
        self.test_pytorch_cuda()
        self.test_pytorch_cpu()

        # 自动检测OpenVINO设备
        core = ov.Core()
        available = core.available_devices
        print(f"可用OpenVINO设备: {available}")

        for prec in ['FP32', 'FP16', 'INT8']:
            # CPU 必须测试
            self.test_openvino(prec, 'CPU')
            # GPU仅当存在时测试
            gpu_device = 'GPU.0'
            if gpu_device:
                self.test_openvino(prec, gpu_device)
            else:
                print(f"⚠️ 跳过 {prec} GPU测试：设备不可用")

    def _create_visualizations(self, df, out_dir):
        """生成增强版可视化图表"""
        import matplotlib.pyplot as plt
        import numpy as np

        print("\n📊 生成可视化图表...")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']

        # 1. FPS对比图
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(df['测试名称'], df['FPS'], color=colors[:len(df)], alpha=0.8)
        ax.set_title('FPS性能对比', fontsize=16, fontweight='bold')
        ax.set_ylabel('FPS (越高越好)', fontsize=12)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_dir / "fps_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 2. 延迟对比图
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(df['测试名称'], df['平均延迟(ms)'], color=colors[:len(df)], alpha=0.8)
        ax.set_title('推理延迟对比', fontsize=16, fontweight='bold')
        ax.set_ylabel('平均延迟(ms) (越低越好)', fontsize=12)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}ms', ha='center', va='bottom', fontsize=9)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_dir / "latency_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 3. 精度对比图（双指标）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        bars1 = ax1.bar(df['测试名称'], df['mAP@0.5'], color=colors[:len(df)], alpha=0.8)
        ax1.set_title('mAP@0.5对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('mAP@0.5', fontsize=12)
        ax1.set_ylim(0, 1)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        bars2 = ax2.bar(df['测试名称'], df['mAP@0.5:0.95'], color=colors[:len(df)], alpha=0.8)
        ax2.set_title('mAP@0.5:0.95对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('mAP@0.5:0.95', fontsize=12)
        ax2.set_ylim(0, 1)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_dir / "accuracy_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 生成3张图表: fps_comparison.png, latency_comparison.png, accuracy_comparison.png")

    def save_report(self, output_dir):
        out_dir = Path(output_dir) / self.model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{
            '测试名称': k,
            '框架': v['framework'],
            '精度': v['precision'],
            '设备': v['device'],
            'FPS': v['fps'],
            '平均延迟(ms)': v['avg_time_ms'],
            'mAP@0.5': v['mAP_50'],
            'mAP@0.5:0.95': v['mAP_50_95']
        } for k, v in self.results.items()])
        df.to_csv(out_dir / f"{self.model_name}_report.csv", index=False, encoding='utf-8-sig')
        self._create_visualizations(df, out_dir)
        return df

    def _plot_charts(self, df, out_dir):
        # 修复：确保图表标签对齐，保留原有功能
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        x_pos = range(len(df))

        # FPS 对比图
        ax1.bar(x_pos, df['FPS'], color='skyblue')
        ax1.set_title('FPS Comparison (higher is better)')
        ax1.set_ylabel('FPS')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df['测试名称'], rotation=45, ha='right')

        # mAP 对比图
        ax2.bar(x_pos, df['mAP@0.5:0.95'], color='lightgreen')
        ax2.set_title('mAP@0.5:0.95 (higher is better)')
        ax2.set_ylabel('mAP')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(df['测试名称'], rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(out_dir / f"{self.model_name}_charts.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 保留：散点图（Performance vs Accuracy）
        plt.figure(figsize=(10, 8))
        for _, row in df.iterrows():
            plt.scatter(row['FPS'], row['mAP@0.5:0.95'], label=row['测试名称'], s=100)
        plt.xlabel('FPS (higher better)')
        plt.ylabel('mAP@0.5:0.95 (higher better)')
        plt.title('Performance vs Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{self.model_name}_scatter.png", dpi=150, bbox_inches='tight')
        plt.close()

# ========== 批量运行所有模型 ==========
def main():
    DATA_YAML = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml"
    NUM_SAMPLES =1638
    #NUM_SAMPLES = 100
    RANDOM_SEED = 42
    OUTPUT_BASE = r"E:\final_exam_test\comparison_results"
    all_pairs = []
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
    for split in ['train', 'val', 'test']:
        if split in data:
            split_path = Path(data[split]).resolve()
            img_dir = split_path / "images"
            lbl_dir = split_path / "labels"
            if img_dir.exists() and lbl_dir.exists():
                for img in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")):
                    lbl = lbl_dir / f"{img.stem}.txt"
                    if lbl.exists():
                        all_pairs.append((img, lbl))
    random.seed(RANDOM_SEED)
    if NUM_SAMPLES < len(all_pairs):
        sampled_pairs = random.sample(all_pairs, NUM_SAMPLES)
    else:
        sampled_pairs = all_pairs
    fixed_image_paths = [p[0] for p in sampled_pairs]
    fixed_label_paths = {p[0].stem: p[1] for p in sampled_pairs}
    print(f"统一测试集大小: {len(fixed_image_paths)} 张图片")
    models = {
        'yolov8m': {
            'pt': r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights_openvion\best.pt",
            'ov_fp32': r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolov8m\FP32",
            'ov_fp16': r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolov8m\FP16",
            'ov_int8': r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolov8m\INT8"
        },
        'yolo26m_com': {
            'pt': r"E:\final_exam_test\handcraft\handcrafted\runs\detect\yolo26m_tomato3\yolo26m_tomato3\weights\best.pt",
            'ov_fp32': r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolo26m_com\FP32",
            'ov_fp16': r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolo26m_com\FP16",
            'ov_int8': r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolo26m_com\INT8"
        }
    }

    all_summaries = []
    for name, paths in models.items():
        print("\n" + "="*80)
        print(f"🚀 开始测试模型: {name}")
        print("="*80)
        # 检查路径
        if not Path(paths['pt']).exists():
            print(f"❌ PyTorch模型不存在，跳过 {name}")
            continue
        missing = [p for p in ['ov_fp32','ov_fp16','ov_int8'] if not Path(paths[p]).exists()]
        if missing:
            print(f"❌ 缺少OpenVINO目录: {missing}，跳过 {name}")
            continue
        try:
            tester = SingleModelTester(
                model_name=name,
                pt_path=paths['pt'],
                ov_fp32_dir=paths['ov_fp32'],
                ov_fp16_dir=paths['ov_fp16'],
                ov_int8_dir=paths['ov_int8'],
                data_yaml=DATA_YAML,
                num_samples=NUM_SAMPLES,
                random_seed=RANDOM_SEED,
                fixed_image_paths=fixed_image_paths,
                fixed_label_paths=fixed_label_paths
            )
            tester.run_all_tests()
            tester.save_report(OUTPUT_BASE)
            for k, v in tester.results.items():
                all_summaries.append({
                    '模型': name,
                    '测试配置': k,
                    '框架': v['framework'],
                    '精度': v['precision'],
                    '设备': v['device'],
                    'FPS': v['fps'],
                    '延迟(ms)': v['avg_time_ms'],
                    'mAP@0.5': v['mAP_50'],
                    'mAP@0.5:0.95': v['mAP_50_95']
                })
        except Exception as e:
            print(f"❌ 模型 {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    # ========== 以下为新增：生成三个独立图表和雷达图（修正 GPU 键名匹配）==========
    if all_summaries:
        import matplotlib.pyplot as plt
        import numpy as np
        from math import pi
        plt.rcParams.update({
            'font.size': 24,
            'axes.titlesize': 30,
            'axes.labelsize': 26,
            'xtick.labelsize': 23,
            'ytick.labelsize': 23,
            'legend.fontsize': 23
        })
        df = pd.DataFrame(all_summaries)

        # 关键修正：GPU 的 key 中是下划线（因为原代码 replace('.', '_')）
        config_map = {
            'pytorch_cuda': 'Pytorch CUDA',
            'pytorch_cpu': 'Pytorch CPU',
            'openvino_fp32_cpu': 'OV FP32 CPU',
            'openvino_fp32_gpu_0': 'OV FP32 IGPU',  # 注意下划线
            'openvino_fp16_cpu': 'OV FP16 CPU',
            'openvino_fp16_gpu_0': 'OV FP16 IGPU',  # 注意下划线
            'openvino_int8_cpu': 'OV INT8 CPU',
            'openvino_int8_gpu_0': 'OV INT8 IGPU'  # 注意下划线
        }
        df['配置简称'] = df['测试配置'].map(config_map)
        df['模型简称'] = df['模型'].map({'yolov8m': 'YOLOv8m', 'yolo26m_com': 'YOLOv26m'})

        # 期望的横坐标顺序（全部8个，显示名称）
        DESIRED_CONFIGS = [
            'Pytorch CUDA',
            'Pytorch CPU',
            'OV FP32 CPU',
            'OV FP32 IGPU',
            'OV FP16 CPU',
            'OV FP16 IGPU',
            'OV INT8 CPU',
            'OV INT8 IGPU'
        ]

        # 强制使用完整列表，缺失自动补0
        configs = DESIRED_CONFIGS
        models = df['模型简称'].unique()

        # 构建数据透视表（直接使用 df 中的数值）
        pivot_fps = df.pivot_table(index='模型简称', columns='配置简称', values='FPS', aggfunc='first')
        pivot_latency = df.pivot_table(index='模型简称', columns='配置简称', values='延迟(ms)', aggfunc='first')
        pivot_map50 = df.pivot_table(index='模型简称', columns='配置简称', values='mAP@0.5', aggfunc='first')

        # 补充缺失的列并排序
        for cfg in configs:
            if cfg not in pivot_fps.columns:
                pivot_fps[cfg] = 0
            if cfg not in pivot_latency.columns:
                pivot_latency[cfg] = 0
            if cfg not in pivot_map50.columns:
                pivot_map50[cfg] = 0
        pivot_fps = pivot_fps[configs]
        pivot_latency = pivot_latency[configs]
        pivot_map50 = pivot_map50[configs]

        x = np.arange(len(configs))
        width = 0.35

        # 1. FPS 对比图
        plt.figure(figsize=(12, 6))
        for i, model in enumerate(models):
            if model in pivot_fps.index:
                vals = pivot_fps.loc[model].values
            else:
                vals = [0] * len(configs)
            plt.bar(x + i * width, vals, width, label=model)
        plt.xticks(x + width / 2, configs, rotation=45, ha='right')
        plt.ylabel('FPS')
        plt.title('FPS 对比')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(OUTPUT_BASE) / "fps_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ FPS对比图已保存: fps_comparison.png")

        # 2. 延迟对比图
        plt.figure(figsize=(12, 6))
        for i, model in enumerate(models):
            if model in pivot_latency.index:
                vals = pivot_latency.loc[model].values
            else:
                vals = [0] * len(configs)
            plt.bar(x + i * width, vals, width, label=model)
        plt.xticks(x + width / 2, configs, rotation=45, ha='right')
        plt.ylabel('延迟 (ms)')
        plt.title('推理延迟对比')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(OUTPUT_BASE) / "latency_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ 延迟对比图已保存: latency_comparison.png")

        # 3. mAP@0.5 对比图
        plt.figure(figsize=(12, 6))
        for i, model in enumerate(models):
            if model in pivot_map50.index:
                vals = pivot_map50.loc[model].values
            else:
                vals = [0] * len(configs)
            plt.bar(x + i * width, vals, width, label=model)
        plt.xticks(x + width / 2, configs, rotation=45, ha='right')
        plt.ylabel('mAP@0.5')
        plt.title('mAP@0.5 对比')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(OUTPUT_BASE) / "accuracy_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ mAP对比图已保存: accuracy_comparison.png")

        # ========== 雷达图部分（与独立图表数据一致）==========
        df['延迟倒数'] = 1.0 / (df['延迟(ms)'] + 1e-6)

        # 模型大小 (MB) 计算（路径复用，略，保持原样）
        model_paths = {
            'yolov8m': {
                'pt': Path(r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights_openvion\best.pt"),
                'ov_fp32': Path(r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolov8m\FP32"),
                'ov_fp16': Path(r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolov8m\FP16"),
                'ov_int8': Path(r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolov8m\INT8")
            },
            'yolo26m_com': {
                'pt': Path(
                    r"E:\final_exam_test\handcraft\handcrafted\runs\detect\yolo26m_tomato3\yolo26m_tomato3\weights\best.pt"),
                'ov_fp32': Path(r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolo26m_com\FP32"),
                'ov_fp16': Path(r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolo26m_com\FP16"),
                'ov_int8': Path(r"E:\final_exam_test\handcraft\handcrafted\runs\openvino_models\yolo26m_com\INT8")
            }
        }

        def get_model_size_mb(row):
            model_key = row['模型']
            framework = row['框架']
            precision = row['精度']
            paths = model_paths.get(model_key)
            if not paths:
                return 0.0
            if framework == 'PyTorch':
                p = paths['pt']
                return p.stat().st_size / (1024 * 1024) if p.exists() else 0.0
            elif framework == 'OpenVINO':
                if precision == 'FP32':
                    dir_path = paths['ov_fp32']
                elif precision == 'FP16':
                    dir_path = paths['ov_fp16']
                elif precision == 'INT8':
                    dir_path = paths['ov_int8']
                else:
                    return 0.0
                if dir_path.exists():
                    total = sum(f.stat().st_size for f in dir_path.glob('*') if f.is_file())
                    return total / (1024 * 1024)
                else:
                    return 0.0
            return 0.0

        df['模型大小(MB)'] = df.apply(get_model_size_mb, axis=1)

        # 归一化
        def normalize(metric):
            vals = df[metric].values
            minv, maxv = vals.min(), vals.max()
            if maxv - minv < 1e-6:
                return np.ones_like(vals) * 0.5
            return (vals - minv) / (maxv - minv)

        df['FPS_norm'] = normalize('FPS')
        df['延迟倒数_norm'] = normalize('延迟倒数')
        df['mAP50_norm'] = normalize('mAP@0.5')
        df['mAP50_95_norm'] = normalize('mAP@0.5:0.95')
        df['模型大小_norm_inv'] = 1 - normalize('模型大小(MB)')

        # 雷达图数据（使用全部配置）
        df_radar = df[df['配置简称'].isin(configs)]

        # 雷达图1：考虑 mAP
        metrics_with_map = ['FPS_norm', '延迟倒数_norm', 'mAP50_norm', 'mAP50_95_norm']
        labels_with_map = ['FPS', '1/Latency', 'mAP@0.5', 'mAP@0.5:0.95']
        angles = np.linspace(0, 2 * pi, len(metrics_with_map), endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
        colors = {'YOLOv8m': 'blue', 'YOLOv26m': 'red'}
        for model in models:
            sub = df_radar[df_radar['模型简称'] == model]
            for _, row in sub.iterrows():
                values = [row[m] for m in metrics_with_map]
                values += values[:1]
                label = row['配置简称']
                ax.plot(angles, values, 'o-', linewidth=4.5, markersize=10, alpha=0.9, label=label)
                ax.fill(angles, values, alpha=0.15, color=colors[model])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_with_map)
        ax.set_ylim(0, 1)
        ax.set_title('综合性能雷达图', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=21, ncol=4, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(Path(OUTPUT_BASE) / "radar_chart_with_map.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ 雷达图 (考虑 mAP) 已保存: radar_chart_with_map.png")

        # 雷达图2：不考虑 mAP
        metrics_without_map = ['FPS_norm', '延迟倒数_norm', '模型大小_norm_inv']
        labels_without_map = ['FPS', '1/Latency', 'Inverse Model Size']
        angles2 = np.linspace(0, 2 * pi, len(metrics_without_map), endpoint=False).tolist()
        angles2 += angles2[:1]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
        for model in models:
            sub = df_radar[df_radar['模型简称'] == model]
            for _, row in sub.iterrows():
                values = [row[m] for m in metrics_without_map]
                values += values[:1]
                label = row['配置简称']
                ax.plot(angles2, values, 'o-', linewidth=1.5, color=colors[model], alpha=0.7, label=label)
                ax.fill(angles2, values, alpha=0.15, color=colors[model])
        ax.set_xticks(angles2[:-1])
        ax.set_xticklabels(labels_without_map)
        ax.set_ylim(0, 1)
        ax.set_title('综合性能雷达图', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(Path(OUTPUT_BASE) / "radar_chart_without_map.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ 雷达图 (不考虑 mAP) 已保存: radar_chart_without_map.png")
        # ========== 新增代码结束 ==========
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(Path(OUTPUT_BASE) / "all_models_summary.csv", index=False, encoding='utf-8-sig')
        print("\n🎉 所有模型测试完成！汇总结果已保存。")
    else:
        print("\n❌ 没有成功测试任何模型，请检查路径配置。")

if __name__ == "__main__":
    main()