# fast_val_escape_ErrorForValTime.py
import sys, os, torch

# 把 ultralytics 目录加入 Python 路径（确保能找到）
ultra_root = os.path.dirname(os.path.abspath(__file__)) + r"\ultralytics\ultralytics"
sys.path.insert(0, ultra_root)

# 现在可以裸 import
from nn.autobackend import AutoBackend   # 一定存在
from models.yolo import YOLO             # 8.4.5 真实文件路径

# 用 AutoBackend 加载模型（FP32，GPU）
model = AutoBackend(weights="yolo26m.pt", device=torch.device("cuda"), fp16=False)

streams = [torch.cuda.Stream() for _ in range(4)]

def fast_val(loader):
    preds = []
    for i, x in enumerate(loader):
        idx = i % 4
        with torch.cuda.stream(streams[idx]):
            x = x.cuda(non_blocking=True)
            pred = model(x, augment=False)   # 每条流跑 4 张
            preds.append(pred.cpu())         # 异步回 CPU
    torch.cuda.synchronize()
    return torch.cat(preds, 0)

# 实例化 YOLO 并替换验证入口
yolo = YOLO("yolo26m.pt")
yolo.val = lambda **kw: fast_val(kw.pop("dataloader"))