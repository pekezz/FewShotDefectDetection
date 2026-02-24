"""
原型网络(Prototypical Networks)实现
用于少样本学习的原型计算和距离度量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PrototypicalNetwork(nn.Module):
    """原型网络"""
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = None,
        distance_metric: str = 'euclidean',
        temperature: float = 1.0
    ):
        """
        Args:
            feature_dim: 特征维度
            num_classes: 类别数(仅用于预训练，元学习时可为None)
            distance_metric: 距离度量方式 ('euclidean' or 'cosine')
            temperature: softmax温度参数
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.distance_metric = distance_metric
        self.temperature = temperature
        
        # 特征映射层(可选)
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # L2归一化
        self.normalize = nn.functional.normalize
    
    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算每个类别的原型向量
        
        Args:
            support_features: (N, feature_dim) 支持集特征
            support_labels: (N,) 支持集标签
            
        Returns:
            prototypes: (num_classes, feature_dim) 每个类别的原型
        """
        unique_labels = torch.unique(support_labels)
        num_classes = len(unique_labels)
        
        prototypes = torch.zeros(
            num_classes,
            self.feature_dim,
            device=support_features.device,
            dtype=support_features.dtype
        )
        
        for idx, label in enumerate(unique_labels):
            # 获取该类别的所有样本
            mask = support_labels == label
            class_features = support_features[mask]
            
            # 计算均值作为原型
            prototypes[idx] = class_features.mean(dim=0)
        
        return prototypes
    
    def compute_distances(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        计算查询样本与原型之间的距离
        
        Args:
            query_features: (M, feature_dim) 查询集特征
            prototypes: (num_classes, feature_dim) 原型向量
            
        Returns:
            distances: (M, num_classes) 距离矩阵
        """
        if self.distance_metric == 'euclidean':
            # 欧氏距离
            # (M, 1, D) - (1, C, D) -> (M, C, D) -> (M, C)
            distances = torch.cdist(query_features, prototypes, p=2)
            
        elif self.distance_metric == 'cosine':
            # 余弦距离 (1 - 余弦相似度)
            # L2归一化
            query_norm = F.normalize(query_features, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            
            # 计算余弦相似度
            similarity = torch.mm(query_norm, proto_norm.t())
            
            # 转换为距离
            distances = 1 - similarity
            
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            support_features: (N, feature_dim) 支持集特征
            support_labels: (N,) 支持集标签
            query_features: (M, feature_dim) 查询集特征
            query_labels: (M,) 查询集标签(可选，用于计算损失)
            
        Returns:
            logits: (M, num_classes) 分类logits
            prototypes: (num_classes, feature_dim) 原型向量
            loss: 交叉熵损失(如果提供了query_labels)
        """
        # 特征投影
        support_features = self.feature_projection(support_features)
        query_features = self.feature_projection(query_features)
        
        # 计算原型
        prototypes = self.compute_prototypes(support_features, support_labels)
        
        # 计算距离
        distances = self.compute_distances(query_features, prototypes)
        
        # 转换为logits (距离越小，logit越大)
        logits = -distances / self.temperature
        
        # 计算损失(如果提供了标签)
        loss = None
        if query_labels is not None:
            loss = F.cross_entropy(logits, query_labels)
        
        return logits, prototypes, loss


class PrototypicalLoss(nn.Module):
    """原型网络损失函数"""
    
    def __init__(
        self,
        distance_metric: str = 'euclidean',
        temperature: float = 1.0
    ):
        super().__init__()
        self.distance_metric = distance_metric
        self.temperature = temperature
    
    def forward(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算原型损失
        
        Args:
            support_features: (N, D) 支持集特征
            support_labels: (N,) 支持集标签
            query_features: (M, D) 查询集特征
            query_labels: (M,) 查询集标签
            
        Returns:
            loss: 标量损失值
        """
        # 计算原型
        unique_labels = torch.unique(support_labels)
        num_classes = len(unique_labels)
        feature_dim = support_features.size(1)
        
        prototypes = torch.zeros(
            num_classes,
            feature_dim,
            device=support_features.device,
            dtype=support_features.dtype
        )
        
        for idx, label in enumerate(unique_labels):
            mask = support_labels == label
            prototypes[idx] = support_features[mask].mean(dim=0)
        
        # 计算距离
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(query_features, prototypes, p=2)
        elif self.distance_metric == 'cosine':
            query_norm = F.normalize(query_features, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            similarity = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarity
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # 转换为logits
        logits = -distances / self.temperature
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, query_labels)
        
        return loss


class ProtoYOLOHead(nn.Module):
    """
    结合原型网络的YOLO检测头
    在标准YOLO检测的基础上增加原型分类分支
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        proto_feature_dim: int = 256,
        use_prototype: bool = True
    ):
        """
        Args:
            in_channels: 输入通道数
            num_classes: 类别数
            proto_feature_dim: 原型特征维度
            use_prototype: 是否使用原型分类
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_prototype = use_prototype
        
        # 标准YOLO分类分支
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        
        # 回归分支(边界框)
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 4, 1)  # x, y, w, h
        )
        
        # 原型特征提取分支
        if self.use_prototype:
            self.proto_conv = nn.Sequential(
                nn.Conv2d(in_channels, proto_feature_dim, 3, padding=1),
                nn.BatchNorm2d(proto_feature_dim),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)  # 全局平均池化
            )
            
            # 原型网络
            self.proto_net = PrototypicalNetwork(
                feature_dim=proto_feature_dim,
                num_classes=num_classes
            )
    
    def forward(
        self,
        x: torch.Tensor,
        support_features: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: (B, C, H, W) 输入特征
            support_features: 支持集特征(仅在少样本模式下)
            support_labels: 支持集标签
            
        Returns:
            cls_pred: (B, num_classes, H, W) 分类预测
            reg_pred: (B, 4, H, W) 回归预测
            proto_features: (B, proto_dim) 原型特征(如果使用)
        """
        # 标准YOLO预测
        cls_pred = self.cls_conv(x)
        reg_pred = self.reg_conv(x)
        
        # 原型特征
        proto_features = None
        if self.use_prototype:
            proto_features = self.proto_conv(x)
            proto_features = proto_features.flatten(1)  # (B, proto_dim)
            
            # 如果提供了支持集，使用原型分类
            if support_features is not None and support_labels is not None:
                proto_logits, _, _ = self.proto_net(
                    support_features,
                    support_labels,
                    proto_features
                )
                # 融合原型分类和标准分类
                # 这里简单地加权平均，实际可以设计更复杂的融合策略
                cls_pred = cls_pred + proto_logits.view(
                    cls_pred.size(0), -1, 1, 1
                ).expand_as(cls_pred)
        
        return cls_pred, reg_pred, proto_features


if __name__ == "__main__":
    # 测试原型网络
    print("测试原型网络...")
    
    # 创建模拟数据
    feature_dim = 256
    n_way = 5
    k_shot = 5
    query_num = 15
    
    # 支持集
    support_features = torch.randn(n_way * k_shot, feature_dim)
    support_labels = torch.tensor([i for i in range(n_way) for _ in range(k_shot)])
    
    # 查询集
    query_features = torch.randn(query_num, feature_dim)
    query_labels = torch.randint(0, n_way, (query_num,))
    
    # 创建原型网络
    proto_net = PrototypicalNetwork(feature_dim=feature_dim)
    
    # 前向传播
    logits, prototypes, loss = proto_net(
        support_features,
        support_labels,
        query_features,
        query_labels
    )
    
    print(f"Logits shape: {logits.shape}")
    print(f"Prototypes shape: {prototypes.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # 测试准确率
    pred = logits.argmax(dim=1)
    acc = (pred == query_labels).float().mean()
    print(f"Accuracy: {acc.item():.4f}")
    
    print("\n测试ProtoYOLO Head...")
    batch_size = 4
    in_channels = 256
    h, w = 20, 20
    
    x = torch.randn(batch_size, in_channels, h, w)
    head = ProtoYOLOHead(in_channels=in_channels, num_classes=n_way)
    
    cls_pred, reg_pred, proto_features = head(x)
    print(f"Cls pred shape: {cls_pred.shape}")
    print(f"Reg pred shape: {reg_pred.shape}")
    print(f"Proto features shape: {proto_features.shape if proto_features is not None else None}")
