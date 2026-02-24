"""
少样本适配核心模块 (GUI 兼容版 - 类别聚合版)
功能: 将同一类别下的所有缺陷数据聚合，训练一个多分类专属模型。
"""

import os
import shutil
import random
import logging
from pathlib import Path
import yaml
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.proto_yolo import ProtoYOLO
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def run_few_shot_adaptation(
        meta_weights: str,
        yolo_weights: str,
        source_dir: str,
        category_name: str,
        defect_name: str,
        all_defects: list,  # 新增：该类别下所有的缺陷列表
        k_shot: int = 5,
        epochs: int = 50,
        work_dir: str = "experiments/few_shot_task",
        progress_callback=None
) -> str:
    # 1. 持久化存储少样本数据 (Support Set)
    support_base = Path("data/support_sets") / category_name
    current_defect_dir = support_base / defect_name

    if source_dir:
        # 如果用户上传了新数据，覆盖保存该缺陷的样本
        if current_defect_dir.exists():
            shutil.rmtree(current_defect_dir)
        current_defect_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(source_dir)
        valid_imgs = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            valid_imgs.extend(list(source_path.glob(ext)))
        valid_imgs = [img for img in valid_imgs if img.with_suffix(".txt").exists()]

        if len(valid_imgs) == 0:
            raise ValueError(f"未在 {source_dir} 中找到成对的图片和 .txt 标签！")

        actual_k = min(k_shot, len(valid_imgs))
        random.seed(42)
        random.shuffle(valid_imgs)
        support_set = valid_imgs[:actual_k]

        for img in support_set:
            shutil.copy(img, current_defect_dir / img.name)
            shutil.copy(img.with_suffix(".txt"), current_defect_dir / img.with_suffix(".txt").name)

    # 2. 构建聚合训练集 (将该类别下所有缺陷的数据合并)
    task_dir = Path(work_dir) / category_name
    if task_dir.exists():
        shutil.rmtree(task_dir)
    train_img_dir = task_dir / "images/train"
    train_lbl_dir = task_dir / "labels/train"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    # 按照 all_defects 列表的顺序，重新分配 class_id (0, 1, 2...)
    for class_id, d_name in enumerate(all_defects):
        d_dir = support_base / d_name
        if not d_dir.exists():
            continue

        for img_path in d_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
                continue
            lbl_path = img_path.with_suffix(".txt")
            if not lbl_path.exists():
                continue

            # 复制图片（加前缀防止重名）
            new_img_name = f"{d_name}_{img_path.name}"
            shutil.copy(img_path, train_img_dir / new_img_name)

            # 自动修正 txt 标签，强制把类别 ID 改为当前的 class_id
            new_lbl_path = train_lbl_dir / f"{d_name}_{lbl_path.name}"
            valid_lines = []
            with open(lbl_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = str(class_id)
                        valid_lines.append(" ".join(parts))

            with open(new_lbl_path, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(valid_lines))
            total_samples += 1

    if total_samples == 0:
        raise ValueError(f"类别 {category_name} 没有任何可用的训练数据！")

    # 3. 生成多分类 dataset.yaml
    yaml_path = task_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({
            "path": task_dir.resolve().as_posix(),
            "train": "images/train",
            "val": "images/train",
            "nc": len(all_defects),
            "names": all_defects
        }, f)

    # 4. 加载元学习骨架
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if progress_callback: progress_callback(5)

    # NumPy 兼容补丁
    import numpy as np
    if not hasattr(np, '_core') and 'numpy.core' in sys.modules:
        sys.modules['numpy._core'] = sys.modules['numpy.core']
        sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']

    proto_model = ProtoYOLO(yolo_weights=yolo_weights, num_classes=15)

    try:
        ckpt = torch.load(meta_weights, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(meta_weights, map_location=device)

    proto_model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    yolo_model = proto_model._yolo_wrapper[0]
    temp_pt = task_dir / "temp_meta.pt"
    yolo_model.save(str(temp_pt))

    if progress_callback: progress_callback(10)

    # 5. 开始多类别重组训练
    new_model = YOLO(str(temp_pt))

    if progress_callback:
        def on_train_epoch_end(trainer):
            progress = 10 + int(90 * (trainer.epoch + 1) / trainer.epochs)
            progress_callback(progress)

        new_model.add_callback("on_train_epoch_end", on_train_epoch_end)

    abs_work_dir = Path(work_dir).resolve()

    new_model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=min(4, total_samples),  # 根据总样本数自适应 batch
        freeze=10,
        project=str(abs_work_dir),
        name=category_name,  # 训练文件夹以类别名命名
        exist_ok=True,
        verbose=False,
        plots=False,
        amp=False
    )

    if hasattr(new_model, 'trainer') and new_model.trainer is not None:
        actual_save_dir = Path(new_model.trainer.save_dir)
    else:
        actual_save_dir = abs_work_dir / category_name

    best_model_path = actual_save_dir / "weights/best.pt"

    if not best_model_path.exists():
        raise RuntimeError(f"微调失败。YOLO预期保存路径为: {best_model_path}")

    return str(best_model_path)