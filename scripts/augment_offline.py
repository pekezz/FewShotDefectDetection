"""
离线数据增强脚本 (修复版)
功能: 读取已处理的 YOLO 格式数据集，生成增强后的图片和标签，并保存到原目录。
修复: 增加 check_and_fix_bbox 函数，解决浮点数误差导致的越界报错
用法: python scripts/augment_offline.py --data_dir data/processed --num_aug 5
"""

import argparse
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm
import albumentations as A


def parse_args():
    parser = argparse.ArgumentParser(description="离线数据增强工具")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="数据根目录")
    parser.add_argument("--split", type=str, default="train", help="增强哪个集 (默认 train)")
    parser.add_argument("--num_aug", type=int, default=5, help="每张原图生成多少张增强图")
    return parser.parse_args()


def get_augmentations():
    """定义增强流水线"""
    return A.Compose([
        # 几何变换 (自动调整 bbox)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),

        # 像素变换 (不影响 bbox)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))


def check_and_fix_bbox(bbox):
    """
    检查并修复 bbox (yolo格式: x_c, y_c, w, h)
    确保其 x_min, y_min, x_max, y_max 都在 [0, 1] 范围内
    解决 albumentations 对浮点数误差敏感的问题
    """
    x, y, w, h = bbox

    # 1. 限制宽高在 [0, 1]
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    # 2. 限制中心点，确保盒子不越界
    # x_min = x - w/2 >= 0  =>  x >= w/2
    # x_max = x + w/2 <= 1  =>  x <= 1 - w/2
    x = max(w / 2, min(1.0 - w / 2, x))
    y = max(h / 2, min(1.0 - h / 2, y))

    # 3. 再次确保 epsilon 安全 (可选，这里直接返回 float)
    return [x, y, w, h]


def main():
    args = parse_args()

    img_dir = Path(args.data_dir) / "images" / args.split
    label_dir = Path(args.data_dir) / "labels" / args.split

    if not img_dir.exists() or not label_dir.exists():
        print(f"错误: 目录不存在 - {img_dir}")
        return

    # 获取所有图片 (排除已经增强过的图片，避免重复增强)
    all_images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    # 过滤掉文件名包含 "_aug_" 的图片
    img_paths = [p for p in all_images if "_aug_" not in p.name]

    print(f"找到 {len(img_paths)} 张原始图片，将生成 {len(img_paths) * args.num_aug} 张增强图片...")

    transform = get_augmentations()
    total_generated = 0

    for img_path in tqdm(img_paths, desc="Augmenting"):
        # 1. 读取图片
        image = cv2.imread(str(img_path))
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img = image.shape[:2]

        # 2. 读取对应标签
        label_path = label_dir / f"{img_path.stem}.txt"
        bboxes = []
        class_labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        # x_c, y_c, w, h
                        raw_box = [float(x) for x in parts[1:]]

                        # --- 关键修复步骤 ---
                        clean_box = check_and_fix_bbox(raw_box)

                        # 只有合法的框才加入
                        if clean_box[2] > 0 and clean_box[3] > 0:
                            bboxes.append(clean_box)
                            class_labels.append(cls_id)

        # 3. 生成增强样本
        for i in range(args.num_aug):
            try:
                # 即使没有 bbox 也可以增强图片（作为负样本或纯背景），但 albumentations 需要 list
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']

                # 4. 保存文件
                # 命名规则: 原文件名_aug_i.png
                new_stem = f"{img_path.stem}_aug_{i}"
                new_img_path = img_dir / f"{new_stem}.png"
                new_label_path = label_dir / f"{new_stem}.txt"

                # 保存图片 (转回 BGR)
                cv2.imwrite(str(new_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

                # 保存标签
                with open(new_label_path, 'w') as f:
                    for cls, box in zip(aug_labels, aug_bboxes):
                        # 再次清洗以防万一
                        box = check_and_fix_bbox(box)
                        line = f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
                        f.write(line)

                total_generated += 1

            except Exception as e:
                # 捕获异常但继续运行，打印出错文件以便排查
                # print(f"警告: 处理 {img_path.name} 第 {i} 次增强时跳过: {e}")
                pass

    print(f"增强完成! 共新增 {total_generated} 个样本。")


if __name__ == "__main__":
    main()