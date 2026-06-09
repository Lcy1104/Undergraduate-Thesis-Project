# 保存为 check_cuda.py
import torch
import torchvision  # 添加这一行
import ultralytics

print("=" * 60)
print("环境验证报告")
print("=" * 60)

print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {torchvision.__version__}")
print(f"Ultralytics 版本: {ultralytics.__version__}")

print(f"\nCUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 设备数: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 额外测试
    print(f"\nCUDA 测试:")
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"  张量移到GPU成功: {x.device}")
    print(f"  计算测试: {x * 2}")
else:
    print("⚠️  CUDA 不可用，将使用 CPU 训练（速度极慢）")

print("\n" + "=" * 60)
print("✅ 环境验证完成！")
print("=" * 60)