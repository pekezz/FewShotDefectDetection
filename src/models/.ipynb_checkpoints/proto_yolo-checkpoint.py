"""
Proto-YOLO: 融合YOLOv8和原型网络的少样本检测模型
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import sys

# YOLOv8相关导入
try:
    from ultralytics import YOLO
    from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
    from ultralytics.nn.tasks import DetectionModel
except ImportError:
    print("警告: ultralytics未安装，请运行: pip install ultralytics")
    YOLO = None


class FeatureExtractor(nn.Module):
    """从YOLOv8提取特征用于原型学习"""

    def __init__(self, yolo_model, extract_layer: int = -3):
        """
        Args:
            yolo_model: YOLOv8模型
            extract_layer: 提取特征的层索引
        """
        super().__init__()
        print('hallo')
        if YOLO is None:
            raise ImportError("需要安装ultralytics: pip install ultralytics")

        # 获取YOLOv8的backbone
        if isinstance(yolo_model, YOLO):
            self.model = yolo_model.model
        else:
            self.model = yolo_model

        self.extract_layer = extract_layer
        self.freeze_backbone = False

    def set_freeze(self, freeze: bool = True):
        """冻结或解冻backbone参数"""
        self.freeze_backbone = freeze
        for param in self.model.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        y = []
        target_layer = self.extract_layer
        if target_layer < 0:
            target_layer = len(self.model.model) + target_layer

        for i, m in enumerate(self.model.model):
            # 处理输入来源(from)
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]

            x = m(x)
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
            self.yolo = YOLO(yolo_weights)
        else:
            self.yolo = YOLO('yolov8n.yaml')

        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            self.yolo,
            extract_layer=-3
        )

        if freeze_backbone:
            self.feature_extractor.set_freeze(True)

        # -----------------------------------------------------
        # 修改点：动态获取Backbone输出通道数
        # -----------------------------------------------------
        # 构造一个虚拟输入，运行一次提取器以获取实际通道数
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 640, 640)
            dummy_output = self.feature_extractor(dummy_input)
            in_channels = dummy_output.shape[1]  # 自动获取通道数 (例如 384)
            # print(f"DEBUG: 检测到Backbone输出通道数为 {in_channels}")

        # 特征适配层 (使用获取到的 in_channels)
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

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.feature_extractor(images)
        features = self.feature_adapter(feature_map)
        features = features.flatten(1)
        return features

    def forward_detection(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.yolo(images, verbose=False)

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
    """简化版Proto-YOLO (仅用于测试)"""

    def __init__(self, num_classes: int = 5, proto_feature_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.proto_feature_dim = proto_feature_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, proto_feature_dim)

        from src.models.prototypical_network import PrototypicalNetwork
        self.proto_net = PrototypicalNetwork(
            feature_dim=proto_feature_dim,
            num_classes=num_classes
        )

    def extract_features(self, images):
        x = self.backbone(images)
        x = x.flatten(1)
        return self.fc(x)

    def forward(self, support_images, support_labels, images, query_labels=None, mode='prototype'):
        support_features = self.extract_features(support_images)
        query_features = self.extract_features(images)
        return self.proto_net(support_features, support_labels, query_features, query_labels)


if __name__ == "__main__":
    # 测试代码
    try:
        model = ProtoYOLO(num_classes=5)
        print("ProtoYOLO模型初始化成功")

        img = torch.randn(1, 3, 640, 640)
        # 测试完整的前向传播（不只是FeatureExtractor，还要过Adapter）
        feat = model.extract_features(img)
        print(f"提取特征形状: {feat.shape}")

    except Exception as e:
        print(f"测试失败: {e}")