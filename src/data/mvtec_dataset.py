"""
MVTec AD数据集加载器
支持少样本学习的数据集类 (修复了数据泄露和采样重叠问题)
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MVTecDataset(Dataset):
    """MVTec AD数据集加载器"""

    def __init__(
        self,
        data_root: str,
        annotation_dir: str,
        image_size: int = 640,
        split: str = 'train',
        transforms: Optional[A.Compose] = None,
        use_masks: bool = True
    ):
        self.data_root = Path(data_root)
        self.annotation_dir = Path(annotation_dir)
        self.image_size = image_size
        self.split = split
        self.use_masks = use_masks

        if transforms is None:
            self.transforms = self._get_default_transforms()
        else:
            self.transforms = transforms

        self.samples = self._load_samples()
        logger.info(f"加载 {split} 集: {len(self.samples)} 个样本")

    def _get_default_transforms(self) -> A.Compose:
        if self.split == 'train':
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def _load_samples(self) -> List[Dict]:
        samples = []
        for txt_file in self.annotation_dir.glob("*.txt"):
            parts = txt_file.stem.split('_')
            if len(parts) < 3: continue

            obj_class = parts[0]
            defect_type = '_'.join(parts[1:-1])
            images_dir = Path(str(self.annotation_dir).replace("labels", "images"))
            image_path = images_dir / f"{txt_file.stem}.png"

            if not image_path.exists(): continue

            samples.append({
                'image_path': str(image_path),
                'annotation_path': str(txt_file),
                'mask_path': None,
                'object_class': obj_class,
                'defect_type': defect_type
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise ValueError(f"无法读取图像: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []

        if Path(sample['annotation_path']).exists():
            with open(sample['annotation_path'], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        eps = 1e-6
                        x_c, y_c, w, h = [float(x) for x in parts[1:]]

                        # 严格的坐标限制
                        if x_c - w/2 < 0: x_c = w/2 + eps
                        if x_c + w/2 > 1: x_c = 1 - w/2 - eps
                        if y_c - h/2 < 0: y_c = h/2 + eps
                        if y_c + h/2 > 1: y_c = 1 - h/2 - eps
                        w = min(w, 1.0 - eps)
                        h = min(h, 1.0 - eps)

                        bboxes.append([x_c, y_c, w, h])
                        class_labels.append(class_id)

        if len(bboxes) > 0:
            transformed = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed['image']
            bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            class_labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
        else:
            transformed = self.transforms(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.long)

        return {
            'image': image, 'bboxes': bboxes, 'labels': class_labels,
            'image_path': sample['image_path'],
            'object_class': sample['object_class'],
            'defect_type': sample['defect_type']
        }


class FewShotMVTecDataset(Dataset):
    """少样本学习专用的MVTec数据集 (严格防止数据泄露版)"""

    def __init__(self, base_dataset: MVTecDataset, n_way: int = 5, k_shot: int = 5, query_num: int = 10):
        self.base_dataset = base_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_num = query_num
        self.class_to_samples = self._build_class_mapping()
        self.classes = list(self.class_to_samples.keys())

        logger.info(f"少样本数据集: {len(self.classes)} 个有效类别 (样本数 >= {k_shot+1})")
        logger.info(f"Episode配置: {n_way}-way {k_shot}-shot, {query_num} queries")

    def _build_class_mapping(self) -> Dict[int, List[int]]:
        class_to_samples = {}
        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset.samples[idx]
            # 这里简单地使用 defect_type 字符串哈希或 object_class + defect_type 作为区分
            # 为了简单起见，我们假设每个txt里的 class_id 是统一的，或者我们按照 defect_type 来分组
            # 实际上 MVTec 的 class_id 在 convert 时可能默认为 0。
            # 为了区分不同缺陷，我们应该用 (object_class, defect_type) 组合作为 key
            # 但为了配合 ProtoNet 的 int label，我们需要维护一个映射

            # 临时方案：读取标注文件里的 class_id。如果全是 0，需要修改 mask_to_bbox 让它生成不同 ID
            # 或者我们在这里自己重新分配 ID

            # 使用 object_class + defect_type 作为唯一标识
            unique_label = f"{sample['object_class']}_{sample['defect_type']}"
            if unique_label not in class_to_samples:
                class_to_samples[unique_label] = []
            class_to_samples[unique_label].append(idx)

        # 过滤样本不足的类别
        valid_class_to_samples = {}
        # 至少需要 k_shot + 1 张图 (k_shot 给 support, 至少 1 张给 query)
        min_required = self.k_shot + 1

        # 将字符串 key 映射为 int ID
        final_mapping = {}
        new_id_counter = 0

        for key, indices in class_to_samples.items():
            if len(indices) >= min_required:
                final_mapping[new_id_counter] = indices
                new_id_counter += 1
            else:
                logger.debug(f"忽略类别 {key}: 样本数 {len(indices)} < {min_required}")

        return final_mapping

    def __len__(self) -> int:
        return 200 # 限制每个 epoch 的 episode 数量

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. 随机选择 N 个类别
        available_classes = self.classes
        if len(available_classes) == 0:
            raise ValueError("没有足够样本的类别用于训练！请检查数据预处理。")

        selected_classes = np.random.choice(
            available_classes,
            size=min(self.n_way, len(available_classes)),
            replace=False
        )

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for class_idx, class_id in enumerate(selected_classes):
            # 获取该类别的所有图片索引
            all_indices = np.array(self.class_to_samples[class_id])

            # 关键步骤：打乱索引
            np.random.shuffle(all_indices)

            # 关键步骤：严格切分
            # 前 k_shot 个给 support
            support_indices = all_indices[:self.k_shot]

            # 剩下的给 query
            remaining_indices = all_indices[self.k_shot:]

            # 如果剩余的不够 query_num，则在剩余的中重复采样 (replace=True)
            # 但绝不从 support_indices 里采
            if len(remaining_indices) >= self.query_num:
                query_indices = remaining_indices[:self.query_num]
            else:
                query_indices = np.random.choice(remaining_indices, self.query_num, replace=True)

            # 再次检查是否有重叠 (Sanity Check)
            assert set(support_indices).isdisjoint(set(query_indices)), \
                f"Data Leakage Detected! Class {class_id}, Support: {support_indices}, Query: {query_indices}"

            # 加载 Support
            for idx in support_indices:
                img = self.base_dataset[idx]['image']
                support_images.append(img)
                support_labels.append(class_idx)

            # 加载 Query
            for idx in query_indices:
                img = self.base_dataset[idx]['image']
                query_images.append(img)
                query_labels.append(class_idx)

        # 堆叠
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels, dtype=torch.long)

        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels
        }