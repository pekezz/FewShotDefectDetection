"""
MVTec AD 数据集预处理脚本 (修复路径版)
功能:
  1. 遍历 MVTec AD 原始目录，识别所有物体类别
  2. 为每个类别分配独立 ID
  3. 将 PNG 掩码转换为 YOLO 格式 txt 标注
  4. 按比例划分为 train/val/test
  5. 生成 dataset.yaml (使用绝对路径，修复 FileNotFoundError)

用法:
  python scripts/prepare_mvtec.py --data_root data/MVTec_AD --output_dir data/processed
"""

import sys
import argparse
import random
import shutil
import yaml
import logging
import cv2
from pathlib import Path

# 确保从项目根目录运行时 src 包可以被找到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mask_to_bbox import MaskToBBoxConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_files(annotation_dir: Path, image_dir: Path,
                output_root: Path,
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                seed: int = 42):
    """
    划分数据集并移动文件
    """
    random.seed(seed)

    # 收集所有已生成的标注文件
    txt_files = sorted(list(annotation_dir.glob("*.txt")))
    if not txt_files:
        logger.error("临时目录里没有 .txt 文件，生成失败")
        return

    # 打乱顺序
    random.shuffle(txt_files)

    n_total = len(txt_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": txt_files[:n_train],
        "val": txt_files[n_train:n_train + n_val],
        "test": txt_files[n_train + n_val:]
    }

    for split_name, files in splits.items():
        label_out = output_root / "labels" / split_name
        image_out = output_root / "images" / split_name
        label_out.mkdir(parents=True, exist_ok=True)
        image_out.mkdir(parents=True, exist_ok=True)

        for txt_path in files:
            # 1. 复制标注文件
            shutil.copy2(txt_path, label_out / txt_path.name)

            # 2. 复制对应的图像文件
            img_path = image_dir / f"{txt_path.stem}.png"
            if img_path.exists():
                shutil.copy2(img_path, image_out / img_path.name)
            else:
                logger.warning(f"找不到图像: {img_path}")

        logger.info(f"  {split_name}: {len(files)} 个样本")


def main():
    parser = argparse.ArgumentParser(description="MVTec AD 数据集预处理 (绝对路径修复版)")
    parser.add_argument("--data_root", type=str, default="data/MVTec_AD", help="原始数据集路径")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="输出路径")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--min_area", type=int, default=50, help="最小缺陷面积")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    if not data_root.exists():
        logger.error(f"数据集目录不存在: {data_root}")
        return

    # 清理旧数据 (防止文件残留)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # 1. 扫描所有物体类别
    object_classes = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    logger.info(f"扫描到 {len(object_classes)} 个类别: {object_classes}")

    # 2. 初始化转换器
    converter = MaskToBBoxConverter(min_area=args.min_area)

    # 创建临时目录
    raw_labels_dir = output_dir / "_raw_labels"
    raw_images_dir = output_dir / "_raw_images"
    raw_labels_dir.mkdir(exist_ok=True)
    raw_images_dir.mkdir(exist_ok=True)

    # 3. 遍历每个类别并转换
    logger.info("Step 1: 转换标注与图像...")

    total_samples = 0
    for class_id, obj_name in enumerate(object_classes):
        obj_dir = data_root / obj_name
        test_dir = obj_dir / "test"

        if not test_dir.exists():
            continue

        logger.info(f"  处理类别 ID {class_id}: {obj_name}")

        # 遍历该物体下的所有缺陷类型
        for defect_dir in test_dir.iterdir():
            if not defect_dir.is_dir() or defect_dir.name == "good":
                continue

            defect_type = defect_dir.name
            gt_dir = obj_dir / "ground_truth" / defect_type

            if not gt_dir.exists():
                continue

            # 处理每张掩码
            for mask_file in gt_dir.glob("*.png"):
                stem = mask_file.stem.replace("_mask", "")
                img_file = defect_dir / f"{stem}.png"

                if not img_file.exists():
                    continue

                img = cv2.imread(str(img_file))
                if img is None: continue
                h, w = img.shape[:2]

                bboxes = converter.mask_to_bboxes(str(mask_file), w, h, class_id=class_id)

                if not bboxes:
                    continue

                # 生成唯一文件名
                unique_name = f"{obj_name}_{defect_type}_{stem}"

                # 保存 .txt
                txt_out = raw_labels_dir / f"{unique_name}.txt"
                with open(txt_out, "w") as f:
                    for bbox in bboxes:
                        f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")

                # 复制图片
                shutil.copy2(img_file, raw_images_dir / f"{unique_name}.png")
                total_samples += 1

    logger.info(f"共生成 {total_samples} 个样本。")

    # 4. 划分数据集
    logger.info("Step 2: 划分数据集 (Train/Val/Test)...")
    split_files(raw_labels_dir, raw_images_dir, output_dir,
                args.train_ratio, args.val_ratio, args.seed)

    # 5. 生成 dataset.yaml (关键修改：使用绝对路径)
    logger.info("Step 3: 生成 dataset.yaml...")

    # 获取 output_dir 的绝对路径
    abs_output_dir = output_dir.resolve()

    yaml_content = {
        # 使用绝对路径，确保 YOLO 在任何目录下运行都能找到图片
        "train": str(abs_output_dir / "images" / "train"),
        "val": str(abs_output_dir / "images" / "val"),
        "test": str(abs_output_dir / "images" / "test"),
        "nc": len(object_classes),
        "names": object_classes
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)

    # 清理临时文件
    shutil.rmtree(raw_labels_dir)
    shutil.rmtree(raw_images_dir)

    logger.info(f"预处理完成! dataset.yaml 已使用绝对路径生成。")


if __name__ == "__main__":
    main()