"""
推理检测与评估脚本
功能: 加载训练好的权重 → 对图像/目录批量检测 → 绘制检测框 → (可选)计算mAP指标

用法:
  # 必须指定 --yolo_weights 指向微调过的 YOLO 权重 (以匹配类别数)
  python scripts/detect.py \
    --weights experiments/checkpoints/best.pt \
    --yolo_weights experiments/finetune_yolo/weights/best.pt \
    --source data/processed/images/test \
    --data data/processed/dataset.yaml \
    --num_classes 15 \
    --compute_map
"""

import sys
import argparse
import logging
from pathlib import Path
import warnings
import yaml

import torch
import cv2
import numpy as np

# 确保从项目根目录运行时 src 包可以被找到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.proto_yolo import ProtoYOLO, SimpleProtoYOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="缺陷检测推理与评估脚本")
    parser.add_argument("--weights", type=str, required=True, help="元学习模型权重路径 (experiments/checkpoints/best.pt)")
    # --- 新增关键参数 ---
    parser.add_argument("--yolo_weights", type=str, default="yolov8n.pt", help="基础 YOLO 权重路径 (用于初始化骨架，必须与训练时一致)")
    # -------------------
    parser.add_argument("--source", type=str, required=True, help="图像文件或目录路径")
    parser.add_argument("--output", type=str, default="outputs/predictions", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--img_size", type=int, default=640, help="输入图像大小")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_simple_model", action="store_true", help="使用 SimpleProtoYOLO")
    parser.add_argument("--num_classes", type=int, default=5, help="类别数量")
    parser.add_argument("--save_txt", action="store_true", help="保存 txt 标注")
    parser.add_argument("--save_conf", action="store_true", help="txt 中附加置信度")
    parser.add_argument("--compute_map", action="store_true", help="计算 mAP 指标 (需要对应的 labels)")
    parser.add_argument("--data", type=str, default="data/processed/dataset.yaml", help="数据集配置文件路径 (用于显示类别名)")
    return parser.parse_args()


# ------------------------------------------------------------------
# 指标计算工具函数
# ------------------------------------------------------------------
def box_iou(box1, box2):
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre, mrec

def process_batch(detections, labels, iou_threshold=0.5):
    if len(detections) == 0:
        return np.zeros((0, 1), dtype=bool)
    correct = np.zeros(len(detections), dtype=bool)
    if len(labels) == 0:
        return correct

    det_boxes = np.array([d[:4] for d in detections])
    det_classes = np.array([d[5] for d in detections])
    gt_classes = np.array([l[0] for l in labels])
    gt_boxes = np.array([l[1:] for l in labels])

    iou = box_iou(det_boxes, gt_boxes)
    detected_gt = []

    for i, d_cls in enumerate(det_classes):
        gt_indices = np.where(gt_classes == d_cls)[0]
        if len(gt_indices) == 0: continue
        cls_ious = iou[i, gt_indices]
        max_iou_idx = np.argmax(cls_ious)
        max_iou = cls_ious[max_iou_idx]
        original_gt_idx = gt_indices[max_iou_idx]

        if max_iou >= iou_threshold and original_gt_idx not in detected_gt:
            correct[i] = True
            detected_gt.append(original_gt_idx)
    return correct


# ------------------------------------------------------------------
# 模型加载
# ------------------------------------------------------------------
def load_model(args) -> torch.nn.Module:
    device = args.device
    weights_path = Path(args.weights)

    if args.use_simple_model:
        logger.info("加载 SimpleProtoYOLO...")
        model = SimpleProtoYOLO(num_classes=args.num_classes)
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    else:
        logger.info("加载 ProtoYOLO...")
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            is_meta_ckpt = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        except Exception:
            try:
                checkpoint = torch.load(weights_path, map_location=device)
                is_meta_ckpt = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            except Exception as e:
                logger.error(f"无法加载权重文件: {e}")
                sys.exit(1)

        if is_meta_ckpt:
            logger.info(f"检测到 Meta-Learning 检查点")
            logger.info(f"初始化骨架: num_classes={args.num_classes}, yolo_weights={args.yolo_weights}")

            # --- 关键修改: 传入 yolo_weights 以便初始化正确的 backbone 结构 ---
            try:
                model = ProtoYOLO(
                    yolo_weights=args.yolo_weights,
                    num_classes=args.num_classes
                )
            except Exception as e:
                logger.error(f"模型初始化失败: {e}")
                logger.error("请确保 --yolo_weights 指向了正确的微调权重 (类别数需匹配)")
                sys.exit(1)
            # ------------------------------------------------------------

            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.info("检测到标准 YOLO 权重，直接初始化...")
            model = ProtoYOLO(yolo_weights=str(weights_path), num_classes=args.num_classes)

    model = model.to(device)
    model.eval()
    logger.info("模型加载完成!")
    return model


def preprocess_image(img_bgr: np.ndarray, img_size: int, device: str):
    resized = cv2.resize(img_bgr, (img_size, img_size))
    tensor = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


# ------------------------------------------------------------------
# 绘图与推理
# ------------------------------------------------------------------
COLORS = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,255,0), (255,0,255)]

def draw_detections(img_bgr: np.ndarray, detections: list, img_size: int, class_names: list = None) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        x1, y1 = int(x1 * w / img_size), int(y1 * h / img_size)
        x2, y2 = int(x2 * w / img_size), int(y2 * h / img_size)
        cls, score = int(det["cls"]), det["score"]
        color = COLORS[cls % len(COLORS)]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_text = class_names[cls] if (class_names and 0 <= cls < len(class_names)) else f"cls{cls}"
        label = f"{label_text}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(vis, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis

def detect_single(model, img_bgr: np.ndarray, args) -> list:
    tensor = preprocess_image(img_bgr, args.img_size, args.device)
    with torch.no_grad():
        preds = model(tensor, mode='detection')
    detections = []
    if preds and len(preds) > 0:
        res = preds[0]
        if hasattr(res, 'boxes') and res.boxes is not None:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            scores_conf = res.boxes.conf.cpu().numpy()
            classes_cls = res.boxes.cls.cpu().numpy()
            for i in range(len(boxes_xyxy)):
                score = float(scores_conf[i])
                if score < args.conf: continue
                detections.append({
                    "box": boxes_xyxy[i].tolist(),
                    "score": score,
                    "cls": int(classes_cls[i])
                })
    return detections

def save_txt(detections: list, txt_path: Path, save_conf: bool = False):
    with open(txt_path, "w") as f:
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            line = f"{det['cls']} {x1} {y1} {x2} {y2}"
            if save_conf: line += f" {det['score']:.4f}"
            f.write(line + "\n")

# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def main_refined():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available(): args.device = "cpu"
    logger.info(f"设备: {args.device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载类别名称
    class_names = None
    if args.data:
        data_path = Path(args.data)
        if data_path.exists():
            try:
                with open(data_path, 'r') as f:
                    data_cfg = yaml.safe_load(f)
                    names = data_cfg.get('names')
                    if isinstance(names, dict): class_names = [names[i] for i in sorted(names.keys())]
                    elif isinstance(names, list): class_names = names
                    logger.info(f"已加载 {len(class_names) if class_names else 0} 个类别名称")
            except Exception as e: logger.warning(f"加载数据集配置失败: {e}")

    # 加载模型
    model = load_model(args)

    source = Path(args.source)
    if source.is_file(): image_paths = [source]
    else: image_paths = sorted(list(source.glob("*.png")) + list(source.glob("*.jpg")))
    logger.info(f"开始检测 {len(image_paths)} 张图像...")

    all_preds, target_counts = [], {}

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue

        detections = detect_single(model, img_bgr, args)
        vis = draw_detections(img_bgr, detections, args.img_size, class_names=class_names)
        cv2.imwrite(str(output_dir / img_path.name), vis)

        if args.compute_map:
            label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
            if not label_path.exists() and "images" not in str(img_path): # 备用路径逻辑
                 label_path = img_path.parent.parent / "labels" / img_path.parent.name / img_path.with_suffix(".txt").name

            gt_labels = []
            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            c = int(parts[0])
                            target_counts[c] = target_counts.get(c, 0) + 1
                            cx, cy, bw, bh = [float(x) for x in parts[1:]]
                            x1, y1 = (cx - bw/2)*args.img_size, (cy - bh/2)*args.img_size
                            x2, y2 = (cx + bw/2)*args.img_size, (cy + bh/2)*args.img_size
                            gt_labels.append([c, x1, y1, x2, y2])

            pred_data = [d['box'] + [d['score'], d['cls']] for d in detections]
            correct = process_batch(pred_data, gt_labels)
            for i, det in enumerate(detections):
                all_preds.append([correct[i], det['score'], det['cls']])

    if args.compute_map and len(all_preds) > 0:
        logger.info("-" * 80)
        logger.info(f"{'Class':<20} {'Images':<10} {'Targets':<10} {'P':<10} {'R':<10} {'mAP@.5':<10}")
        all_preds = np.array(all_preds)
        unique_classes = np.unique(all_preds[:, 2])
        aps = []
        for cls_id in unique_classes:
            cls_id = int(cls_id)
            n_gt = target_counts.get(cls_id, 0)
            cls_name = class_names[cls_id] if (class_names and cls_id < len(class_names)) else str(cls_id)
            cls_mask = all_preds[:, 2] == cls_id
            cls_preds = all_preds[cls_mask]
            if n_gt == 0 and len(cls_preds) == 0: continue
            if len(cls_preds) == 0: ap, p, r = 0, 0, 0
            else:
                sort_idx = np.argsort(-cls_preds[:, 1])
                cls_preds = cls_preds[sort_idx]
                tp = cls_preds[:, 0].astype(int)
                fp, tp_cum = 1 - tp, np.cumsum(tp)
                recall = tp_cum / (n_gt + 1e-6)
                precision = tp_cum / (tp_cum + np.cumsum(fp) + 1e-6)
                ap, _, _ = compute_ap(recall, precision)
                p, r = precision[-1], recall[-1]
            aps.append(ap)
            logger.info(f"{cls_name:<20} {len(image_paths):<10} {n_gt:<10} {p:.3f}      {r:.3f}      {ap:.3f}")
        logger.info("-" * 80)
        logger.info(f"{'All':<20} {len(image_paths):<10} {sum(target_counts.values()):<10} {'-':<10} {'-':<10} {np.mean(aps) if aps else 0:.3f}")
        logger.info("-" * 80)
    elif args.compute_map:
        logger.warning("未检测到任何目标或未找到标签文件，无法计算指标。")

if __name__ == "__main__":
    main_refined()