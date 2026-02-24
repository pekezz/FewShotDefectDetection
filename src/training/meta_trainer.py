"""
元训练器 (Meta-Trainer)
实现episodic training for few-shot learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaTrainer:
    """元学习训练器"""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            checkpoint_dir: str = 'experiments/checkpoints',
            log_dir: str = 'experiments/logs'
    ):
        """
        Args:
            model: Proto-YOLO模型
            train_loader: 训练数据加载器(FewShotMVTecDataset)
            val_loader: 验证数据加载器
            lr: 学习率
            weight_decay: 权重衰减
            device: 设备
            checkpoint_dir: 检查点保存目录
            log_dir: 日志目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器(余弦退火)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader),
            eta_min=lr * 0.01
        )

        # 检查点和日志
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def _set_model_mode(self, training: bool):
        """
        安全地设置模型模式，处理Ultralytics YOLO .train() 方法冲突
        """
        try:
            if training:
                self.model.train()
            else:
                self.model.eval()
        except TypeError as e:
            # 捕获 Ultralytics YOLO 的 'bool' object is not callable 错误
            if "not callable" in str(e):
                self.model.training = training
                # 如果包含内部模型，也尝试设置它
                if hasattr(self.model, 'model') and isinstance(self.model.model, nn.Module):
                    if training:
                        self.model.model.train()
                    else:
                        self.model.model.eval()
            else:
                raise e

    def _process_batch_data(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        处理DataLoader增加的batch维度
        将 (1, N, C, H, W) -> (N, C, H, W)
        将 (1, N) -> (N,)
        """
        tensor = tensor.to(self.device)
        # 如果是5维张量 (B, N, C, H, W)，且B=1，则降维
        if tensor.dim() == 5 and tensor.size(0) == 1:
            return tensor.squeeze(0)
        # 如果是2维标签 (B, N)，且B=1，则降维
        if tensor.dim() == 2 and tensor.size(0) == 1:
            return tensor.squeeze(0)
        return tensor

    def train_episode(self, episode_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        训练单个episode
        """
        self._set_model_mode(True)

        # 获取并处理数据
        support_images = self._process_batch_data(episode_data['support_images'])
        support_labels = self._process_batch_data(episode_data['support_labels'])
        query_images = self._process_batch_data(episode_data['query_images'])
        query_labels = self._process_batch_data(episode_data['query_labels'])

        # 前向传播
        logits, prototypes, loss = self.model(
            support_images=support_images,
            support_labels=support_labels,
            images=query_images,
            query_labels=query_labels,
            mode='prototype'
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # 计算准确率
        pred = logits.argmax(dim=1)
        accuracy = (pred == query_labels).float().mean()

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }

    @torch.no_grad()
    def validate_episode(self, episode_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        验证单个episode
        """
        self._set_model_mode(False)

        # 获取并处理数据
        support_images = self._process_batch_data(episode_data['support_images'])
        support_labels = self._process_batch_data(episode_data['support_labels'])
        query_images = self._process_batch_data(episode_data['query_images'])
        query_labels = self._process_batch_data(episode_data['query_labels'])

        # 前向传播
        logits, prototypes, loss = self.model(
            support_images=support_images,
            support_labels=support_labels,
            images=query_images,
            query_labels=query_labels,
            mode='prototype'
        )

        # 计算准确率
        pred = logits.argmax(dim=1)
        accuracy = (pred == query_labels).float().mean()

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        epoch_metrics = {
            'loss': [],
            'accuracy': []
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for episode_data in pbar:
            # 训练单个episode
            metrics = self.train_episode(episode_data)

            # 记录指标
            epoch_metrics['loss'].append(metrics['loss'])
            epoch_metrics['accuracy'].append(metrics['accuracy'])

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            # 更新学习率
            self.scheduler.step()

        # 计算epoch平均指标
        avg_metrics = {
            'loss': np.mean(epoch_metrics['loss']),
            'accuracy': np.mean(epoch_metrics['accuracy'])
        }

        return avg_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {'loss': 0.0, 'accuracy': 0.0}

        val_metrics = {
            'loss': [],
            'accuracy': []
        }

        pbar = tqdm(self.val_loader, desc='Validation')

        for episode_data in pbar:
            metrics = self.validate_episode(episode_data)

            val_metrics['loss'].append(metrics['loss'])
            val_metrics['accuracy'].append(metrics['accuracy'])

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.4f}"
            })

        avg_metrics = {
            'loss': np.mean(val_metrics['loss']),
            'accuracy': np.mean(val_metrics['accuracy'])
        }

        return avg_metrics

    def train(self, num_epochs: int, save_freq: int = 10):
        """
        训练主循环
        """
        logger.info(f"开始元训练，共 {num_epochs} 个epoch")
        logger.info(f"设备: {self.device}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 记录历史
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_accuracy'].append(val_metrics['accuracy'])

            # 日志
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best.pt')
                logger.info(f"保存最佳模型 (Val Acc: {self.best_val_acc:.4f})")

            # 定期保存检查点
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')

        logger.info("训练完成!")
        logger.info(f"最佳验证准确率: {self.best_val_acc:.4f}")

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, filename: str):
        """加载检查点"""
        path = self.checkpoint_dir / filename

        if not path.exists():
            logger.warning(f"检查点不存在: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_history = checkpoint['train_history']

        logger.info(f"检查点已加载: {path}")
        logger.info(f"恢复训练从 epoch {self.current_epoch}, best val acc: {self.best_val_acc:.4f}")


if __name__ == "__main__":
    # 测试训练器
    from src_models_proto_yolo import SimpleProtoYOLO

    print("创建模型...")
    model = SimpleProtoYOLO(num_classes=5, proto_feature_dim=256)

    print("创建模拟数据加载器...")


    class DummyDataLoader:
        def __init__(self, num_episodes=100):
            self.num_episodes = num_episodes

        def __len__(self):
            return self.num_episodes

        def __iter__(self):
            for _ in range(self.num_episodes):
                n_way = 5
                k_shot = 5
                query_num = 10

                # 注意：这里模拟了DataLoader的Batch行为，增加了第0维
                yield {
                    'support_images': torch.randn(1, n_way * k_shot, 3, 224, 224),
                    'support_labels': torch.tensor([i for i in range(n_way) for _ in range(k_shot)]).unsqueeze(0),
                    'query_images': torch.randn(1, query_num, 3, 224, 224),
                    'query_labels': torch.randint(0, n_way, (1, query_num))
                }


    train_loader = DummyDataLoader(num_episodes=10)
    val_loader = DummyDataLoader(num_episodes=5)

    print("创建训练器...")
    trainer = MetaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        device='cpu'  # 使用CPU测试
    )

    print("开始训练...")
    trainer.train(num_epochs=3, save_freq=1)

    print("训练完成!")