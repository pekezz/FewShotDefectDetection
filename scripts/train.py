"""
标准训练脚本 (非 episode 模式的全量训练入口)
功能: 读取配置 → 构建数据加载器 → 创建模型 → 训练 → 保存最优权重

与 train_meta.py 的区别:
  - train_meta.py : episodic meta-learning, 使用 MetaTrainer
  - train.py      : 标准 epoch-level 训练, 适用于初步调试和数据量充足的场景

用法:
  python scripts/train.py --config configs/train_config.yaml
  python scripts/train.py --config configs/train_config.yaml --use_simple_model --epochs 50
  python scripts/train.py --config configs/train_config.yaml --resume experiments/checkpoints/epoch_30.pt
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保从项目根目录运行时 src 包可以被找到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mvtec_dataset import MVTecDataset
from src.models.proto_yolo import ProtoYOLO, SimpleProtoYOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="标准训练脚本")
    parser.add_argument("--config",           type=str,  default="configs/train_config.yaml")
    parser.add_argument("--data_path",        type=str,  default=None,
                        help="如果原始MVTec数据还没预处理，传入原始路径会自动触发预处理")
    parser.add_argument("--weights",          type=str,  default=None, help="预训练权重路径")
    parser.add_argument("--resume",           type=str,  default=None, help="恢复训练的检查点")
    parser.add_argument("--device",           type=str,  default=None)
    parser.add_argument("--batch_size",       type=int,  default=None)
    parser.add_argument("--epochs",           type=int,  default=None)
    parser.add_argument("--use_simple_model", action="store_true",
                        help="使用简化模型(不依赖 ultralytics)")
    return parser.parse_args()


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# 数据加载器
# ------------------------------------------------------------------
def create_dataloaders(config: dict):
    data_cfg  = config["data"]
    processed = Path(data_cfg["processed_root"])

    train_dataset = MVTecDataset(
        data_root      = str(processed),
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

    batch_size  = config["training"]["batch_size"]
    num_workers = config["device"]["num_workers"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info(f"训练集: {len(train_dataset)} 个样本  | 验证集: {len(val_dataset)} 个样本")
    return train_loader, val_loader


# ------------------------------------------------------------------
# 模型构建
# ------------------------------------------------------------------
def build_model(config: dict, args) -> nn.Module:
    model_cfg = config["model"]

    if args.use_simple_model:
        logger.info("使用 SimpleProtoYOLO...")
        model = SimpleProtoYOLO(
            num_classes       =model_cfg["num_classes"],
            proto_feature_dim =model_cfg["proto_feature_dim"]
        )
    else:
        logger.info("使用 ProtoYOLO...")
        weights = args.weights or model_cfg["yolo_weights"]
        try:
            model = ProtoYOLO(
                yolo_weights      =weights,
                num_classes       =model_cfg["num_classes"],
                proto_feature_dim =model_cfg["proto_feature_dim"],
                freeze_backbone   =model_cfg["freeze_backbone"],
                use_pretrained    =model_cfg["use_pretrained"]
            )
        except Exception as e:
            logger.warning(f"ProtoYOLO 创建失败: {e}\n回退到 SimpleProtoYOLO...")
            model = SimpleProtoYOLO(
                num_classes       =model_cfg["num_classes"],
                proto_feature_dim =model_cfg["proto_feature_dim"]
            )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total:,}  可训练: {trainable:,}")
    return model


# ------------------------------------------------------------------
# 训练循环
# ------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, use_amp=True):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device == "cuda")
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp and device == "cuda"):
            outputs = model(images)
            # outputs 是 forward() 返回的 dict;
            # 如果 loss 已在 forward 里计算就直接取；否则用占位损失
            loss = outputs.get("loss", torch.tensor(0.0, device=device, requires_grad=True))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, use_amp=True):
    model.eval()
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(loader, desc="Val ", leave=False):
        images = batch["images"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp and device == "cuda"):
            outputs = model(images)
            loss = outputs.get("loss", torch.tensor(0.0, device=device))
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def main():
    args   = parse_args()
    config = load_config(args.config)

    # 命令行覆盖
    if args.device:      config["training"]["device"]     = args.device
    if args.batch_size:  config["training"]["batch_size"] = args.batch_size
    if args.epochs:      config["training"]["num_epochs"] = args.epochs

    set_seed(42)

    # 设备
    device = config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    logger.info(f"设备: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 如果传了 --data_path 且数据没预处理过，自动调用预处理
    if args.data_path:
        processed = Path(config["data"]["processed_root"])
        if not (processed / "labels" / "train").exists():
            logger.info("自动触发数据预处理...")
            import subprocess
            subprocess.run([
                sys.executable, "scripts/prepare_mvtec.py",
                "--data_root", args.data_path,
                "--output_dir", str(processed)
            ], check=True)

    # 数据
    train_loader, val_loader = create_dataloaders(config)

    # 模型
    model = build_model(config, args).to(device)

    # 优化器 + 调度器
    optimizer = optim.AdamW(model.parameters(),
                            lr=config["training"]["lr"],
                            weight_decay=config["training"]["weight_decay"])
    num_epochs = config["training"]["num_epochs"]
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 检查点目录
    ckpt_dir = Path(config["checkpoint"]["save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 恢复训练
    start_epoch  = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"从 epoch {start_epoch} 恢复训练, best_val_loss={best_val_loss:.4f}")

    # ---------- 训练主循环 ----------
    logger.info(f"开始训练: epoch {start_epoch} → {num_epochs}")
    use_amp = config["training"].get("use_amp", True)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, use_amp)
        val_loss   = validate(model, val_loader, device, use_amp)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(f"Epoch [{epoch+1:>3}/{num_epochs}]  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}  time={elapsed:.1f}s")

        # 保存最优
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss":      best_val_loss
            }, ckpt_dir / "best.pt")
            logger.info(f"  → 保存最优权重 best.pt (val_loss={val_loss:.4f})")

        # 定期保存
        if (epoch + 1) % config["checkpoint"]["save_freq"] == 0:
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss":      best_val_loss
            }, ckpt_dir / f"epoch_{epoch+1}.pt")

        # 早停
        es_cfg = config["training"].get("early_stopping", {})
        if es_cfg.get("enabled", False):
            patience = es_cfg.get("patience", 20)
            if not hasattr(main, "_no_improve"):
                main._no_improve = 0
            if val_loss < best_val_loss + es_cfg.get("min_delta", 0.0):
                main._no_improve = 0
            else:
                main._no_improve += 1
            if main._no_improve >= patience:
                logger.info(f"早停触发 (patience={patience})")
                break

    # 保存最终模型
    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    logger.info(f"训练完成! 最优 val_loss={best_val_loss:.4f}  权重保存在 {ckpt_dir}")


if __name__ == "__main__":
    main()
