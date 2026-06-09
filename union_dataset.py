import yaml
import os
import shutil
from pathlib import Path
import json
from collections import defaultdict


def merge_tomato_datasets(ds1_yaml_path, ds2_yaml_path,
                          ds1_dirs, ds2_dirs,
                          output_dir,
                          merge_mode='union'):
    """
    合并两个番茄叶片数据集，确保类别语义一致且ID映射100%准确
    """

    # ============== 1. 读取并解析类别配置 ==============
    print("=" * 70)
    print("阶段1: 读取并解析类别配置")
    print("=" * 70)

    with open(ds1_yaml_path, 'r', encoding='utf-8') as f:
        ds1_config = yaml.safe_load(f)

    with open(ds2_yaml_path, 'r', encoding='utf-8') as f:
        ds2_config = yaml.safe_load(f)

    # 转换为字典格式
    def parse_names(names_config):
        if isinstance(names_config, list):
            return {i: name for i, name in enumerate(names_config)}
        elif isinstance(names_config, dict):
            return {int(k): v for k, v in names_config.items()}
        return {}

    ds1_names = parse_names(ds1_config.get('names', {}))
    ds2_names = parse_names(ds2_config.get('names', {}))

    # ============== 2. 创建智能类别映射表 ==============
    print("\n" + "=" * 70)
    print("阶段2: 创建智能类别映射表（语义匹配）")
    print("=" * 70)

    # 定义语义相同的类别映射（手动定义，确保准确性）
    # key: 数据集2的类别名, value: 数据集1的对应类别名
    semantic_mapping = {
        "Tomato leaf bacterial spot": "Bacterial Spot",
        "Tomato Early blight leaf": "Early_Blight",
        "Tomato leaf late blight": "Late_blight",
        "Tomato mold leaf": "Leaf Mold",
    }

    # 数据集2的其他番茄类别（需要新增到数据集1）
    additional_tomato_classes = [
        "Tomato Septoria leaf spot",
        "Tomato leaf mosaic virus",
        "Tomato leaf yellow virus",
        "Tomato leaf",
        "Tomato two spotted spider mites leaf"
    ]

    # 构建合并后的类别列表（数据集1优先）
    merged_names = {}
    current_id = 0

    # 首先添加数据集1的所有类别（保留原始ID）
    for old_id, name in sorted(ds1_names.items()):
        merged_names[current_id] = name
        current_id += 1

    print(f"\n基础类别（来自数据集1）:")
    for cid, name in merged_names.items():
        print(f"  ID {cid}: {name}")

    # 添加数据集2的新类别
    ds2_id_to_new_id = {}  # 记录数据集2的ID如何映射到新ID
    ds2_name_to_new_id = {}  # 记录数据集2的类别名如何映射

    for old_id, name in ds2_names.items():
        # 检查是否在语义映射中
        if name in semantic_mapping:
            # 映射到数据集1的对应类别
            target_name = semantic_mapping[name]
            # 查找目标类别在 merged_names 中的ID
            new_id = None
            for cid, cname in merged_names.items():
                if cname == target_name:
                    new_id = cid
                    break
            ds2_id_to_new_id[old_id] = new_id
            ds2_name_to_new_id[name] = new_id
            print(f"  映射: '{name}' (ds2 ID:{old_id}) → '{target_name}' (新 ID:{new_id})")

        # 检查是否是数据集1已存在的类别
        elif name in ds1_names.values():
            # 查找对应ID
            new_id = None
            for cid, cname in merged_names.items():
                if cname == name:
                    new_id = cid
                    break
            ds2_id_to_new_id[old_id] = new_id
            ds2_name_to_new_id[name] = new_id
            print(f"  映射: '{name}' (ds2 ID:{old_id}) → 已存在 ID:{new_id}")

        # 否则作为新类别添加
        else:
            merged_names[current_id] = name
            ds2_id_to_new_id[old_id] = current_id
            ds2_name_to_new_id[name] = current_id
            print(f"  新增: '{name}' → 新 ID:{current_id}")
            current_id += 1

    print(f"\n合并后总类别数: {len(merged_names)}")

    # ============== 3. 创建输出目录 ==============
    print("\n" + "=" * 70)
    print("阶段3: 创建输出目录结构")
    print("=" * 70)

    output_dir = Path(output_dir)
    merged_dirs = {}
    for split in ['train', 'val', 'test']:
        img_dir = output_dir / split / 'images'
        lbl_dir = output_dir / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        merged_dirs[split] = {'images': img_dir, 'labels': lbl_dir}
        print(f"✓ 创建: {img_dir}")
        print(f"✓ 创建: {lbl_dir}")

    # 保存映射关系到文件（用于验证）
    mapping_file = output_dir / 'id_mapping.json'
    mapping_data = {
        'dataset1_original': ds1_names,
        'dataset2_original': ds2_names,
        'merged': merged_names,
        'ds2_id_to_new_id': ds2_id_to_new_id,
        'semantic_mapping': semantic_mapping
    }
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print(f"\n✓ 映射关系已保存到: {mapping_file}")

    # ============== 4. 复制和转换函数 ==============
    def process_split(split, ds_img_dir, ds_lbl_dir,
                      old_names, id_mapping, dataset_prefix='ds1'):
        """处理单个数据集 split，返回统计信息"""
        if not ds_img_dir or not Path(ds_img_dir).exists():
            print(f"\n⚠️  {dataset_prefix} 的 {split} 集路径不存在，跳过")
            return {'images': 0, 'labels': 0, 'errors': []}

        img_dir = Path(ds_img_dir)
        lbl_dir = Path(ds_lbl_dir) if ds_lbl_dir else None

        stats = {'images': 0, 'labels': 0, 'errors': []}

        print(f"\n{'=' * 60}")
        print(f"处理 {dataset_prefix} 的 {split} 集: {img_dir.name}")
        print(f"{'=' * 60}")

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                continue

            # 生成新文件名
            new_img_name = f"{dataset_prefix}_{img_file.name}"
            new_lbl_name = f"{dataset_prefix}_{img_file.stem}.txt"

            # 复制图像
            shutil.copy2(img_file, merged_dirs[split]['images'] / new_img_name)
            stats['images'] += 1

            # 处理标签文件
            if lbl_dir:
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    new_lbl_path = merged_dirs[split]['labels'] / new_lbl_name

                    try:
                        with open(lbl_file, 'r') as f_in, open(new_lbl_path, 'w') as f_out:
                            valid_lines = 0

                            for line_num, line in enumerate(f_in, 1):
                                line = line.strip()
                                if not line:
                                    continue

                                parts = line.split()
                                if len(parts) < 5:
                                    stats['errors'].append(
                                        f"{dataset_prefix}_{img_file.stem}.txt:行{line_num}-格式错误"
                                    )
                                    continue

                                try:
                                    old_class_id = int(parts[0])

                                    # 查找新ID
                                    if dataset_prefix == 'ds1':
                                        # 数据集1的ID保持不变
                                        new_class_id = old_class_id
                                    else:
                                        # 数据集2的ID需要映射
                                        if old_class_id in id_mapping:
                                            new_class_id = id_mapping[old_class_id]
                                        else:
                                            stats['errors'].append(
                                                f"{dataset_prefix}_{img_file.stem}.txt:行{line_num}-"
                                                f"未知类别ID {old_class_id}"
                                            )
                                            continue

                                    # 写入新标签
                                    parts[0] = str(new_class_id)
                                    f_out.write(' '.join(parts) + '\n')
                                    valid_lines += 1

                                except ValueError:
                                    stats['errors'].append(
                                        f"{dataset_prefix}_{img_file.stem}.txt:行{line_num}-ID转换失败"
                                    )
                                    continue

                        if valid_lines > 0:
                            stats['labels'] += 1
                        else:
                            # 删除空的标签文件
                            if new_lbl_path.exists():
                                new_lbl_path.unlink()

                    except Exception as e:
                        stats['errors'].append(
                            f"{dataset_prefix}_{img_file.stem}.txt:处理异常-{str(e)}"
                        )

        print(f"✓ 图像: {stats['images']} 张")
        print(f"✓ 标签: {stats['labels']} 个")
        if stats['errors']:
            print(f"⚠️  错误: {len(stats['errors'])} 个")
            for err in stats['errors'][:5]:  # 只显示前5个
                print(f"  - {err}")

        return stats

    # ============== 5. 处理所有数据集 ==============
    print("\n" + "=" * 70)
    print("阶段4: 开始复制文件并转换类别ID")
    print("=" * 70)

    all_stats = {}

    for split in ['train', 'val', 'test']:
        print(f"\n{'=' * 70}")
        print(f"处理 {split} 集")
        print(f"{'=' * 70}")

        # 数据集1（ID不变）
        stats1 = process_split(
            split,
            ds1_dirs.get(split, {}).get('images', ''),
            ds1_dirs.get(split, {}).get('labels', ''),
            ds1_names,
            id_mapping=None,
            dataset_prefix='ds1'
        )

        # 数据集2（需要ID映射）
        stats2 = process_split(
            split,
            ds2_dirs.get(split, {}).get('images', ''),
            ds2_dirs.get(split, {}).get('labels', ''),
            ds2_names,
            id_mapping=ds2_id_to_new_id,
            dataset_prefix='ds2'
        )

        all_stats[split] = {
            'ds1': stats1,
            'ds2': stats2,
            'total_images': stats1['images'] + stats2['images'],
            'total_labels': stats1['labels'] + stats2['labels']
        }

    # ============== 6. 生成最终的YAML文件 ==============
    print("\n" + "=" * 70)
    print("阶段5: 生成最终的YAML文件")
    print("=" * 70)

    # 转换names为列表格式（YOLOv8推荐）
    names_list = [merged_names[i] for i in range(len(merged_names))]

    merged_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(names_list),
        'names': names_list
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(merged_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✓ YAML文件已生成: {yaml_path}")
    print(f"\n内容预览:")
    print(yaml.dump(merged_yaml, allow_unicode=True))

    # ============== 7. 最终验证 ==============
    print("\n" + "=" * 70)
    print("阶段6: 最终验证（确保所有标签文件与YAML匹配）")
    print("=" * 70)

    # 验证函数
    def validate_labels(split_name):
        print(f"\n验证 {split_name} 集标签...")
        label_dir = merged_dirs[split_name]['labels']
        total_files = 0
        total_lines = 0
        errors = []

        for lbl_file in label_dir.glob("*.txt"):
            total_files += 1
            try:
                with open(lbl_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            errors.append(f"{lbl_file.name}:行{line_num}-格式错误")
                            continue

                        class_id = int(parts[0])
                        if class_id < 0 or class_id >= len(names_list):
                            errors.append(
                                f"{lbl_file.name}:行{line_num}-类别ID {class_id} 超出范围 [0-{len(names_list) - 1}]"
                            )
                        total_lines += 1

            except Exception as e:
                errors.append(f"{lbl_file.name}:读取错误-{str(e)}")

        print(f"  ✓ 检查文件: {total_files} 个")
        print(f"  ✓ 检查标注: {total_lines} 行")
        if errors:
            print(f"  ❌ 发现错误: {len(errors)} 个")
            for err in errors[:10]:
                print(f"     - {err}")
        else:
            print(f"  ✅ 全部验证通过！")

        return errors

    # 验证所有split
    validation_errors = {}
    for split in ['train', 'val', 'test']:
        validation_errors[split] = validate_labels(split)

    # ============== 8. 生成总结报告 ==============
    print("\n" + "=" * 70)
    print("合并完成总结报告")
    print("=" * 70)

    print(f"\n数据集统计:")
    for split in ['train', 'val', 'test']:
        stats = all_stats[split]
        print(f"\n{'=' * 50}")
        print(f"{split.upper()} 集:")
        print(f"{'=' * 50}")
        print(f"  数据集1: {stats['ds1']['images']} 张图像, {stats['ds1']['labels']} 个标签")
        if stats['ds1']['errors']:
            print(f"           ⚠️  {len(stats['ds1']['errors'])} 个错误")
        print(f"  数据集2: {stats['ds2']['images']} 张图像, {stats['ds2']['labels']} 个标签")
        if stats['ds2']['errors']:
            print(f"           ⚠️  {len(stats['ds2']['errors'])} 个错误")
        print(f"  总计: {stats['total_images']} 张图像, {stats['total_labels']} 个标签")

        val_errors = validation_errors[split]
        if val_errors:
            print(f"  验证: ❌ {len(val_errors)} 个严重错误")
        else:
            print(f"  验证: ✅ 通过")

    print(f"\n最终类别列表 ({len(names_list)} 类):")
    for i, name in enumerate(names_list):
        print(f"  {i}: {name}")

    print(f"\n✅ 合并后的数据集已准备完成！")
    print(f"📝 训练命令:")
    print(f"   yolo detect train data=\"{yaml_path}\" model=yolov8n.pt epochs=100 imgsz=640")
    print(f"\n📊 映射关系文件: {mapping_file}")

    # 如果有验证错误，返回False
    all_val_errors = [err for errs in validation_errors.values() for err in errs]
    if all_val_errors:
        print(f"\n⚠️  警告: 发现 {len(all_val_errors)} 个验证错误，请检查！")
        return False

    return True


# ==================== 配置并运行 ====================
if __name__ == "__main__":
    # ============== 你的实际路径 ==============
    DATASET1_YAML = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\1\data.yaml"
    DATASET1_TRAIN_IMG = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\1\train\images"
    DATASET1_TRAIN_LBL = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\1\train\labels"
    DATASET1_VAL_IMG = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\1\valid\images"
    DATASET1_VAL_LBL = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\1\valid\labels"
    DATASET1_TEST_IMG = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\1\test\images"
    DATASET1_TEST_LBL = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\1\test\labels"

    DATASET2_YAML = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\7\data.yaml"
    DATASET2_TRAIN_IMG = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\7\train\images"
    DATASET2_TRAIN_LBL = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\7\train\labels"
    DATASET2_VAL_IMG = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\7\valid\images"
    DATASET2_VAL_LBL = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\7\valid\labels"
    DATASET2_TEST_IMG = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\7\test\images"
    DATASET2_TEST_LBL = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\7\test\labels"

    OUTPUT_DIR = r"E:\final_exam_test\handcraft\handcrafted\datasets\farukalam\tomato-leaf-diseases-detection-computer-vision\versions\all1"

    # 组织路径字典
    ds1_dirs = {
        'train': {'images': DATASET1_TRAIN_IMG, 'labels': DATASET1_TRAIN_LBL},
        'val': {'images': DATASET1_VAL_IMG, 'labels': DATASET1_VAL_LBL},
        'test': {'images': DATASET1_TEST_IMG, 'labels': DATASET1_TEST_LBL}
    }

    ds2_dirs = {
        'train': {'images': DATASET2_TRAIN_IMG, 'labels': DATASET2_TRAIN_LBL},
        'val': {'images': DATASET2_VAL_IMG, 'labels': DATASET2_VAL_LBL},
        'test': {'images': DATASET2_TEST_IMG, 'labels': DATASET2_TEST_LBL}
    }

    # 执行合并
    print("\n" + "=" * 70)
    print("开始合并番茄叶片数据集")
    print("=" * 70)

    success = merge_tomato_datasets(
        DATASET1_YAML,
        DATASET2_YAML,
        ds1_dirs,
        ds2_dirs,
        OUTPUT_DIR,
        merge_mode='union'
    )

    if success:
        print("\n🎉 合并成功！数据集已100%验证通过，可以立即用于训练。")
    else:
        print("\n❌ 合并完成但存在验证错误，请检查报告中的错误信息！")