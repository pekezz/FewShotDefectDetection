"""
元训练脚本 (episodic meta-training)
功能: 加载配置 → 构建数据加载器 → 创建 Proto-YOLO 模型 → 调用 MetaTrainer 训练

用法:
  python scripts/train_meta.py --config configs/train_config.yaml
  python scripts/train_meta.py --config configs/train_config.yaml --use_simple_model
  python scripts/train_meta.py --config configs/train_config.yaml --resume experiments/checkpoints/epoch_50.pt
"""

import sys
import argparse
import logging
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

# 确保从项目根目录运行时 src 包可以被找到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mvtec_dataset import MVTecDataset, FewShotMVTecDataset
from src.models.proto_yolo import ProtoYOLO, SimpleProtoYOLO
from src.training.meta_trainer import MetaTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Proto-YOLO 元训练")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    parser.add_argument("--use_simple_model", action="store_true",
                        help="使用简化模型(不依赖 ultralytics)")
    return parser.parse_args()


# ------------------------------------------------------------------
# 配置加载
# ------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# 数据加载器
# ------------------------------------------------------------------
def create_dataloaders(config: dict):
    data_cfg      = config["data"]
    few_shot_cfg  = config["few_shot"]
    device_cfg    = config["device"]
    processed     = Path(data_cfg["processed_root"])

    logger.info("创建数据集...")

    # 基础数据集
    train_dataset = MVTecDataset(
        data_root      = str(processed),  # processed_root 即可
        annotation_dir = str(processed / data_cfg["train_labels"]),
        image_size     = data_cfg["image_size"],
        split          = "train"
    )
    val_dataset = MVTecDataset(
        data_root      = str(processed),
        annotation_dir = str(processed / data_cfg["val_labels"]),
        image_size     = data_cfg["image_size"],
        split          = "val"
    )

    # 少样本包装
    train_few_shot = FewShotMVTecDataset(
        base_dataset=train_dataset,
        n_way      =few_shot_cfg["n_way"],
        k_shot     =few_shot_cfg["k_shot"],
        query_num  =few_shot_cfg["query_num"]
    )
    val_few_shot = FewShotMVTecDataset(
        base_dataset=val_dataset,
        n_way      =few_shot_cfg["n_way"],
        k_shot     = 1,
        query_num  =few_shot_cfg["query_num"]
    )

    train_loader = DataLoader(
        train_few_shot,
        batch_size =config["training"]["batch_size"],
        shuffle    =True,
        num_workers=device_cfg["num_workers"],
        pin_memory =device_cfg["pin_memory"]
    )
    val_loader = DataLoader(
        val_few_shot,
        batch_size =config["training"]["batch_size"],
        shuffle    =False,
        num_workers=device_cfg["num_workers"],
        pin_memory =device_cfg["pin_memory"]
    )

    logger.info(f"训练集: {len(train_dataset)} 个样本")
    logger.info(f"验证集: {len(val_dataset)} 个样本")
    return train_loader, val_loader


# ------------------------------------------------------------------
# 模型构建
# ------------------------------------------------------------------
def create_model(config: dict, use_simple: bool = False):
    model_cfg = config["model"]

    if use_simple:
        logger.info("使用简化模型 SimpleProtoYOLO (无 ultralytics 依赖)...")
        model = SimpleProtoYOLO(
            num_classes       =model_cfg["num_classes"],
            proto_feature_dim =model_cfg["proto_feature_dim"]
        )
    else:
        logger.info("使用完整模型 ProtoYOLO (依赖 ultralytics)...")
        try:
            model = ProtoYOLO(
                yolo_weights      =model_cfg["yolo_weights"],
                num_classes       =model_cfg["num_classes"],
                proto_feature_dim =model_cfg["proto_feature_dim"],
                freeze_backbone   =model_cfg["freeze_backbone"],
                use_pretrained    =model_cfg["use_pretrained"]
            )
        except Exception as e:
            logger.warning(f"无法创建 ProtoYOLO: {e}\n回退到 SimpleProtoYOLO...")
            model = SimpleProtoYOLO(
                num_classes       =model_cfg["num_classes"],
                proto_feature_dim =model_cfg["proto_feature_dim"]
            )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数: {total:,}  可训练参数: {trainable:,}")
    return model


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def main():
    args   = parse_args()
    config = load_config(args.config)

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(config["device"]["gpu_ids"][0])
        logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("使用 CPU")

    # 数据
    train_loader, val_loader = create_dataloaders(config)

    # 模型
    model = create_model(config, use_simple=args.use_simple_model)

    # 训练器
    trainer = MetaTrainer(
        model          =model,
        train_loader   =train_loader,
        val_loader     =val_loader,
        lr             =config["training"]["lr"],
        weight_decay   =config["training"]["weight_decay"],
        device         =device,
        checkpoint_dir =config["checkpoint"]["save_dir"],
        log_dir        =config["logging"]["log_dir"]
    )

    # 恢复训练
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 开始训练
    logger.info("开始元训练...")
    trainer.train(
        num_epochs=config["training"]["num_epochs"],
        save_freq =config["checkpoint"]["save_freq"]
    )
    logger.info("元训练完成!")


if __name__ == "__main__":
    main()
