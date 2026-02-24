"""
YOLO 检测头微调脚本
功能: 使用 data/processed/dataset.yaml 微调 YOLOv8，使其适应缺陷检测任务
用法: python scripts/train_yolo.py --epochs 100
"""

import argparse
from ultralytics import YOLO
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 微调脚本")
    parser.add_argument("--data", type=str, default="data/processed/dataset.yaml", help="数据集配置文件")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="预训练模型")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="图像大小")
    parser.add_argument("--project", type=str, default="experiments", help="保存项目路径")
    parser.add_argument("--name", type=str, default="finetune_yolo", help="实验名称")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载模型 (加载预训练权重)
    logger.info(f"加载模型: {args.model}...")
    model = YOLO(args.model)

    # 2. 开始训练
    logger.info(f"开始在 {args.data} 上微调...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=20,  # 早停
        save=True,  # 保存检查点
        exist_ok=True,  # 覆盖同名实验
        pretrained=True,  # 使用预训练权重
        verbose=True,
        amp=False
    )

    logger.info("训练完成!")
    logger.info(f"最佳权重保存在: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()