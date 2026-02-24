"""
Proto-YOLO: 融合YOLOv8和原型网络的少样本检测模型
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import sys

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class FeatureExtractor(nn.Module):
    """从YOLOv8提取特征 - 修复了数据流逻辑"""

    def __init__(self, yolo_model, extract_layer: int = -3):
        super().__init__()
        if YOLO is None: raise ImportError("需要安装ultralytics")

        if isinstance(yolo_model, YOLO):
            self.model = yolo_model.model
        else:
            self.model = yolo_model

        self.extract_layer = extract_layer
        self.freeze_backbone = False

    def set_freeze(self, freeze: bool = True):
        self.freeze_backbone = freeze
        for param in self.model.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = []
        # 处理负数索引
        target_layer = self.extract_layer
        if target_layer < 0: target_layer = len(self.model.model) + target_layer

        for i, m in enumerate(self.model.model):
            # 处理 YOLOv8 的路由 (from list)
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    # m.f 是列表时，表示 Concat
                    x = [x if j == -1 else y[j] for j in m.f]

            x = m(x)
            # 只有在 save 列表里的层才需要缓存输出
            y.append(x if m.i in self.model.save else None)

            if i == target_layer:
                return x
        return x


class ProtoYOLO(nn.Module):
    """
    Proto-YOLO模型
    """

    def __init__(
            self,
            yolo_weights: str = 'yolov8n.pt',
            num_classes: int = 5,
            proto_feature_dim: int = 256,
            freeze_backbone: bool = False,
            use_pretrained: bool = True
    ):
        super().__init__()

        if YOLO is None:
            raise ImportError("需要安装ultralytics: pip install ultralytics")

        self.num_classes = num_classes
        self.proto_feature_dim = proto_feature_dim

        # 加载YOLOv8模型
        if use_pretrained:
            _yolo = YOLO(yolo_weights)
        else:
            _yolo = YOLO('yolov8n.yaml')

        # --- 关键修改 1: 避免 self.yolo 被注册为 PyTorch 子模块 ---
        # 我们不把 _yolo 直接赋值给 self.yolo (如果是 nn.Module 的话)
        # 而是只取我们需要的部分：model (DetectionModel)
        self.backbone_model = _yolo.model

        # 如果你仍然需要 _yolo 对象来进行 detect 推理，可以把它藏在列表里
        # PyTorch 不会递归列表中的非 Module 对象，或者我们可以忽略它
        self._yolo_wrapper = [_yolo]

        # 特征提取器 (传入 backbone)
        self.feature_extractor = FeatureExtractor(
            self.backbone_model,  # 传入内部模型，而不是 wrapper
            extract_layer=-3
        )

        if freeze_backbone:
            self.feature_extractor.set_freeze(True)

        # 动态获取通道数
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 640, 640)
            dummy_output = self.feature_extractor(dummy_input)
            in_channels = dummy_output.shape[1]

        # 特征适配层
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(in_channels, proto_feature_dim, 1),
            nn.BatchNorm2d(proto_feature_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # 原型网络
        from src.models.prototypical_network import PrototypicalNetwork
        self.proto_net = PrototypicalNetwork(
            feature_dim=proto_feature_dim,
            num_classes=num_classes,
            distance_metric='euclidean',
            temperature=1.0
        )

        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    # --- 关键修改 2: 重写 train 方法，防止递归调用 ultralytics 的 train ---
    def train(self, mode=True):
        """
        重写 train 方法，确保只切换 PyTorch 模块的模式，
        而不触发 Ultralytics YOLO wrapper 的 train() 逻辑。
        """
        self.training = mode
        # 只对真正的 nn.Module 子模块调用 train
        for module in self.children():
            # 过滤掉可能的 YOLO wrapper 对象 (如果它被误认为是 Module)
            if isinstance(module, YOLO):
                continue
            module.train(mode)
        return self

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.feature_extractor(images)
        features = self.feature_adapter(feature_map)
        features = features.flatten(1)
        return features

    def forward_detection(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 从 wrapper 列表中取出 YOLO 对象进行推理
        return self._yolo_wrapper[0](images, verbose=False)

    def forward_prototype(
            self,
            support_images: torch.Tensor,
            support_labels: torch.Tensor,
            query_images: torch.Tensor,
            query_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        support_features = self.extract_features(support_images)
        query_features = self.extract_features(query_images)

        logits, prototypes, loss = self.proto_net(
            support_features,
            support_labels,
            query_features,
            query_labels
        )

        return logits, prototypes, loss

    def forward(
            self,
            images: torch.Tensor,
            mode: str = 'detection',
            support_images: Optional[torch.Tensor] = None,
            support_labels: Optional[torch.Tensor] = None,
            query_labels: Optional[torch.Tensor] = None
    ):
        if mode == 'detection':
            return self.forward_detection(images)
        elif mode == 'prototype':
            if support_images is None or support_labels is None:
                raise ValueError("prototype模式需要提供support_images和support_labels")
            return self.forward_prototype(
                support_images,
                support_labels,
                images,
                query_labels
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

class SimpleProtoYOLO(nn.Module):
    """Simple dummy model for testing without ultralytics"""
    def __init__(self, num_classes=5, proto_feature_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, proto_feature_dim)
        from src.models.prototypical_network import PrototypicalNetwork
        self.proto_net = PrototypicalNetwork(feature_dim=proto_feature_dim)

    def forward(self, support_images, support_labels, images, query_labels=None, mode='prototype'):
        s_f = self.fc(self.backbone(support_images).flatten(1))
        q_f = self.fc(self.backbone(images).flatten(1))
        return self.proto_net(s_f, support_labels, q_f, query_labels)