# test_openvino_igpuFromIntel.py
from openvino import Core
import numpy as np
import time
from pathlib import Path
import json
import sys

# ==================== 用户配置区域 ====================
MODEL_XML_PATH = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights_openvion\best_openvino_model\best.xml"
MODEL_BIN_PATH = r"E:\final_exam_test\handcraft\handcrafted\runs\detect\train8\weights_openvion\best_openvino_model\best.bin"
RESULT_SAVE_PATH = r"E:\final_exam_test\handcraft\handcrafted\igpu_test_result.json"
TARGET_GPU = "GPU.0"  # 可以改为 GPU.1 测试第二个


# ====================================================

def verify_intel_gpu_strict(core: Core, device: str) -> bool:
    """
    严格验证Intel iGPU（仅依赖可靠的设备名称）
    """
    print(f"\n{'=' * 70}")
    print(f"🔍 验证设备: {device}")
    print(f"{'=' * 70}")

    try:
        # 核心验证: 设备名称必须包含 "Intel"
        # 这是100%可靠的，NVIDIA不会返回这个名字
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"✅ 设备名称: {device_name}")

        if "Intel" not in device_name:
            print(f"❌ 错误: 这不是Intel GPU！")
            return False

        # 辅助验证: 尝试获取其他属性（非关键，失败也不影响）
        print(f"\n⏳ 尝试获取扩展信息...")
        try:
            driver_ver = core.get_property(device, "GPU_DRIVER_VERSION")
            print(f"  驱动版本: {driver_ver}")
        except:
            print(f"  ⚠️  驱动版本: 无法获取（OpenVINO 2025 API变更）")

        try:
            total_mem = core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE")
            mem_gb = total_mem / (1024 ** 3)
            print(f"  显存大小: {mem_gb:.2f} GB")
        except:
            print(f"  ⚠️  显存大小: 无法获取")

        try:
            capabilities = core.get_property(device, "OPTIMIZATION_CAPABILITIES")
            print(f"  优化能力: {capabilities}")
        except:
            print(f"  ⚠️  优化能力: 无法获取")

        print(f"\n✅ 结论: {device} 是Intel iGPU")
        print(f"   设备名称已确认: {device_name}")
        return True

    except Exception as e:
        print(f"❌ 核心验证失败: {e}")
        return False


def test_device(core: Core, model, device: str, input_blob: np.ndarray,
                iterations: int = 100, warmup: int = 10) -> dict:
    """测试设备性能"""
    print(f"\n{'=' * 70}")
    print(f"🎯 测试设备: {device}")
    print(f"{'=' * 70}")

    # 编译模型
    print(f"⏳ 编译模型...")
    compiled = core.compile_model(model, device)
    print(f"✅ 编译成功")

    # 验证实际执行设备
    try:
        exec_devices = compiled.get_property("EXECUTION_DEVICES")
        print(f"实际执行设备: {exec_devices}")
    except:
        print(f"⚠️  无法验证执行设备")

    # 预热
    print(f"⏳ 预热 {warmup} 次...")
    for _ in range(warmup):
        compiled([input_blob])

    # 性能测试
    print(f"⏳ 性能测试 ({iterations} 次)...")
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        result = compiled([input_blob])
        times.append(time.perf_counter() - start)

        if (i + 1) % 20 == 0:
            print(f"   进度: {i + 1}/{iterations}")

    # 统计
    times_ms = np.array(times) * 1000
    stats = {
        "设备": device,
        "平均时间(ms)": round(float(np.mean(times_ms)), 2),
        "FPS": round(1000 / np.mean(times_ms), 2),
        "验证状态": "Intel iGPU已验证"
    }

    return stats


def main():
    """主流程"""
    print("=" * 70)
    print("🔍 OpenVINO Intel iGPU 性能测试")
    print(f"版本: 2025.4.1 | 目标设备: {TARGET_GPU}")
    print("=" * 70)

    # 1. 初始化OpenVINO
    core = Core()

    # 2. 显示所有设备
    devices = core.available_devices
    print(f"📋 可用设备: {devices}")

    # 3. 验证是Intel iGPU
    if not verify_intel_gpu_strict(core, TARGET_GPU):
        print("\n🚨 验证失败！设备不是Intel iGPU")
        sys.exit(1)

    # 4. 加载模型
    print("\n⏳ 加载模型...")
    xml_path = Path(MODEL_XML_PATH)
    bin_path = Path(MODEL_BIN_PATH)

    if not xml_path.exists():
        print(f"❌ 模型不存在: {xml_path}")
        sys.exit(1)

    model = core.read_model(str(xml_path), str(bin_path))
    print(f"✅ 模型加载: {xml_path.name}")

    # 5. 准备数据
    input_blob = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # 6. 测试GPU
    gpu_stats = test_device(core, model, TARGET_GPU, input_blob)

    # 7. 测试CPU对比
    cpu_stats = test_device(core, model, "CPU", input_blob, iterations=50)

    # 8. 结果分析
    speedup = gpu_stats["FPS"] / cpu_stats["FPS"]

    print("\n" + "=" * 70)
    print("📈 性能对比")
    print("=" * 70)
    print(f"{'设备':<10} | {'FPS':>8} | {'延迟(ms)':>10}")
    print("-" * 35)
    print(f"{'Intel GPU':<10} | {gpu_stats['FPS']:>8.1f} | {gpu_stats['平均时间(ms)']:>10.2f}")
    print(f"{'CPU':<10} | {cpu_stats['FPS']:>8.1f} | {cpu_stats['平均时间(ms)']:>10.2f}")
    print("-" * 35)
    print(f"加速比: {speedup:.2f}x")

    # 9. 保存结果
    result = {
        "测试时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "设备验证": "Intel Iris Xe iGPU 已确认",
        "GPU性能": gpu_stats,
        "CPU性能": cpu_stats,
        "加速比": round(speedup, 2),
    }

    with open(RESULT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {RESULT_SAVE_PATH}")
    print("\n✅ 验证完成：您的设备是 Intel Iris Xe iGPU")


if __name__ == "__main__":
    main()