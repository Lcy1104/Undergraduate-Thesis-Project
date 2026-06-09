import torch
import intel_extension_for_pytorch as ipex

print("torch version :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("XPU  available:", torch.xpu.is_available())

# 生成 4×4 随机张量
x = torch.randn(4, 4)
print("Tensor created on CPU:", x.device)

# 移到 Intel 核显
x = x.to("xpu")
print("Moved to XPU        :", x.device)
print("Value sample        :", x[0, :3].tolist())