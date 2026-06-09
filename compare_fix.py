import time
import cv2
import numpy as np
import torch
import openvino as ov
import yaml
import torchvision
from pathlib import Path
from tqdm import tqdm
import gc
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import warnings
import random
from ultralytics.data.augment import LetterBox

warnings.filterwarnings('ignore')
torch.serialization.DEFAULT_LOAD_WEIGHTS_ONLY = False


# ========== 自己实现的工具函数（不依赖 ultralytics 内部 API）==========

def _xywh2xyxy(x):
    """YOLO格式 [cx,cy,w,h] -> [x1,y1,x2,y2]"""
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    将框从 img1_shape 缩放到 img0_shape
    ratio_pad: (ratio, dw, dh) 来自 LetterBox
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1], ratio_pad[2]

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img0_shape[1]) if isinstance(boxes, torch.Tensor) else np.clip(
        boxes[..., [0, 2]], 0, img0_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img0_shape[0]) if isinstance(boxes, torch.Tensor) else np.clip(
        boxes[..., [1, 3]], 0, img0_shape[0])
    return boxes


def _nms(boxes, scores, iou_thres):
    """torchvision.nms 包装"""
    return torch.ops.torchvision.nms(boxes, scores, iou_thres)


# ========== mAP 计算（不变）==========
def compute_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (area1 + area2 - inter_area + 1e-6)


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
                best_iou, best_j = 0, -1
                for j, t in enumerate(targets_c):
                    if j in matched_target:
                        continue
                    iou = compute_iou(p[:4], t[:4])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thr:
                    tp[i] = 1
                    matched_target.add(best_j)
                else:
                    fp[i] = 1
            tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
            prec = tp_cum / (tp_cum + fp_cum + 1e-6)
            rec = tp_cum / len(targets_c)
            ap = np.trapz(prec, rec) if len(rec) > 1 else 0.0
            ap_list.append(ap)
        ap_per_class.append(np.mean(ap_list))
    mAP = np.mean(ap_per_class) if ap_per_class else 0.0

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
            best_iou, best_j = 0, -1
            for j, t in enumerate(targets_c):
                if j in matched_target:
                    continue
                iou = compute_iou(p[:4], t[:4])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5:
                tp[i] = 1
                matched_target.add(best_j)
            else:
                fp[i] = 1
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)
        rec = tp_cum / len(targets_c)
        ap50 = np.trapz(prec, rec) if len(rec) > 1 else 0.0
        ap50_per_class.append(ap50)
    mAP50 = np.mean(ap50_per_class) if ap50_per_class else 0.0
    return {'mAP_50': mAP50, 'mAP_50_95': mAP}


class SingleModelTester:
    def __init__(self, model_name, pt_path, ov_fp32_dir, ov_fp16_dir, ov_int8_dir, data_yaml,
                 num_samples=150, random_seed=42, fixed_image_paths=None, fixed_label_paths=None):
        self.model_name = model_name
        self.pt_path = Path(pt_path)
        self.ov_dirs = {'FP32': Path(ov_fp32_dir), 'FP16': Path(ov_fp16_dir), 'INT8': Path(ov_int8_dir)}
        self.data_yaml = Path(data_yaml)
        self.num_samples = num_samples
        self.random_seed = random_seed

        if fixed_image_paths is not None and fixed_label_paths is not None:
            self.image_paths = fixed_image_paths
            self.label_paths = fixed_label_paths
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
            img_dir, lbl_dir = split_path / "images", split_path / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                print(f"   ⚠️ 跳过 {split}")
                continue
            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
            for img in images:
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    all_pairs.append((img, lbl))
        if not all_pairs:
            raise ValueError("未找到任何有效的图片-标签对！")
        random.seed(self.random_seed)
        sampled = random.sample(all_pairs,
                                min(self.num_samples, len(all_pairs))) if self.num_samples and self.num_samples < len(
            all_pairs) else all_pairs
        self.image_paths = [p[0] for p in sampled]
        self.label_paths = {p[0].stem: p[1] for p in sampled}
        print(f"   📸 最终采样 {len(self.image_paths)} 张图片")

    def _load_ground_truth(self, img_stem, img_path=None):
        label_file = self.label_paths.get(img_stem)
        if not label_file or not label_file.exists():
            return []
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
                x1 = (xc - w_norm / 2) * w
                y1 = (yc - h_norm / 2) * h
                x2 = (xc + w_norm / 2) * w
                y2 = (yc + h_norm / 2) * h
                boxes.append([x1, y1, x2, y2, class_id])
        return boxes

    # ========== 统一预处理 ==========
    def _preprocess_image(self, img_path):
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        h0, w0 = orig_img.shape[:2]

        # 使用 LetterBox
        letterbox = LetterBox(new_shape=640, auto=True, stride=32)
        im = letterbox(image=orig_img)

        # 计算 ratio_pad（与 LetterBox 内部一致）
        r = min(640 / h0, 640 / w0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        dw, dh = (640 - new_unpad[0]) / 2, (640 - new_unpad[1]) / 2

        # BGR -> RGB, HWC -> CHW, /255
        im = im[..., ::-1].transpose(2, 0, 1)
        im = np.ascontiguousarray(im, dtype=np.float32) / 255.0

        return im[np.newaxis, ...], orig_img, (r, dw, dh)

    # ========== 统一后处理 ==========
    def _postprocess(self, pred, orig_img, ratio_pad, conf_thres=0.25, iou_thres=0.45, max_det=300):
        device = pred.device if isinstance(pred, torch.Tensor) else torch.device('cpu')

        # 已做 NMS 的格式 [N, 6]
        if pred.ndim == 2 and pred.shape[1] == 6:
            boxes = pred
        else:
            # 原始输出 [8400, 84] 或 [1, 84, 8400]
            if pred.ndim == 3:
                pred = pred[0].T if pred.shape[0] == 1 else pred.T

            # 转换为 xyxy
            box = _xywh2xyxy(pred[:, :4])
            conf = pred[:, 4]
            cls_scores = pred[:, 5:]

            mask = conf > conf_thres
            if mask.sum() == 0:
                return torch.empty((0, 6), device=device)

            box = box[mask]
            conf = conf[mask]
            cls_ids = cls_scores[mask].argmax(dim=1).float()

            # NMS
            keep = _nms(box, conf, iou_thres)
            if len(keep) > max_det:
                keep = keep[:max_det]

            boxes = torch.cat([box[keep], conf[keep].unsqueeze(1), cls_ids[keep].unsqueeze(1)], dim=1)

        # 映射回原始图像
        img1_shape = (640, 640)
        img0_shape = orig_img.shape[:2]
        boxes[:, :4] = _scale_boxes(img1_shape, boxes[:, :4], img0_shape, ratio_pad=ratio_pad).round()

        return boxes

    def test_pytorch_cuda(self):
        print(f"\n🔵 {self.model_name} | PyTorch CUDA (原始精度)")
        device = torch.device('cuda:0')
        model = YOLO(str(self.pt_path)).to(device)
        model.eval()

        times, all_preds, all_targets = [], [], []

        for img_path in tqdm(self.image_paths, desc="CUDA推理"):
            orig_img = cv2.imread(str(img_path))
            if orig_img is None:
                continue

            im_tensor, _, ratio_pad = self._preprocess_image(img_path)
            im_tensor = torch.from_numpy(im_tensor).to(device)

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                pred = model.model(im_tensor)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

            boxes = self._postprocess(pred[0], orig_img, ratio_pad)
            all_preds.append(boxes.cpu().numpy())
            all_targets.append(self._load_ground_truth(img_path.stem, img_path))

        metrics = compute_map(all_preds, all_targets)
        stats = {'framework': 'PyTorch', 'precision': '原始(FP32)', 'device': 'CUDA',
                 'avg_time_ms': np.mean(times) * 1000, 'fps': 1 / np.mean(times),
                 'mAP_50': metrics['mAP_50'], 'mAP_50_95': metrics['mAP_50_95']}
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
            orig_img = cv2.imread(str(img_path))
            if orig_img is None:
                continue

            im_tensor, _, ratio_pad = self._preprocess_image(img_path)
            im_tensor = torch.from_numpy(im_tensor)

            start = time.perf_counter()
            with torch.no_grad():
                pred = model.model(im_tensor)
            times.append(time.perf_counter() - start)

            boxes = self._postprocess(pred[0], orig_img, ratio_pad)
            all_preds.append(boxes.cpu().numpy())
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

        core = ov.Core()
        model = core.read_model(str(xml_file))
        ov_device = device_name
        compiled_model = core.compile_model(model, ov_device)

        times, all_preds, all_targets = [], [], []

        for idx, img_path in enumerate(tqdm(self.image_paths, desc=f"{precision}推理")):
            im_np, orig_img, ratio_pad = self._preprocess_image(img_path)

            start = time.perf_counter()
            ov_tensor = ov.Tensor(im_np)
            infer_request = compiled_model.create_infer_request()
            infer_request.set_input_tensor(0, ov_tensor)
            infer_request.infer()
            output = infer_request.get_output_tensor(0).data
            times.append(time.perf_counter() - start)

            pred = torch.from_numpy(output)
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred[0]

            boxes = self._postprocess(pred, orig_img, ratio_pad)
            all_preds.append(boxes.cpu().numpy())
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

    def run_all_tests(self):
        self.test_pytorch_cuda()
        self.test_pytorch_cpu()

        core = ov.Core()
        available = core.available_devices
        print(f"可用OpenVINO设备: {available}")

        for prec in ['FP32', 'FP16', 'INT8']:
            self.test_openvino(prec, 'CPU')
            gpu_device = 'GPU.0'
            if gpu_device:
                self.test_openvino(prec, gpu_device)

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
        return df


def main():
    DATA_YAML = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1\data.yaml"
    NUM_SAMPLES = 100
    RANDOM_SEED = 42
    OUTPUT_BASE = r"E:\final_exam_test\comparison_results_pk"

    all_pairs = []
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
    for split in ['train', 'val', 'test']:
        if split in data:
            split_path = Path(data[split]).resolve()
            img_dir, lbl_dir = split_path / "images", split_path / "labels"
            if img_dir.exists() and lbl_dir.exists():
                for img in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")):
                    lbl = lbl_dir / f"{img.stem}.txt"
                    if lbl.exists():
                        all_pairs.append((img, lbl))
    random.seed(RANDOM_SEED)
    sampled_pairs = random.sample(all_pairs, min(NUM_SAMPLES, len(all_pairs)))
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
        print("\n" + "=" * 80)
        print(f"🚀 开始测试模型: {name}")
        print("=" * 80)
        if not Path(paths['pt']).exists():
            print(f"❌ PyTorch模型不存在，跳过 {name}")
            continue
        missing = [p for p in ['ov_fp32', 'ov_fp16', 'ov_int8'] if not Path(paths[p]).exists()]
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
                    '模型': name, '测试配置': k, '框架': v['framework'],
                    '精度': v['precision'], '设备': v['device'],
                    'FPS': v['fps'], '延迟(ms)': v['avg_time_ms'],
                    'mAP@0.5': v['mAP_50'], 'mAP@0.5:0.95': v['mAP_50_95']
                })
        except Exception as e:
            print(f"❌ 模型 {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(Path(OUTPUT_BASE) / "all_models_summary.csv", index=False, encoding='utf-8-sig')
        print("\n🎉 所有模型测试完成！汇总结果已保存。")


if __name__ == "__main__":
    main()