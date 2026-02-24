# 基于YOLOv8的少样本工业零件缺陷检测系统 — 项目结构

## 目录树（仅列实际交付的文件）

```
FewShotDefectDetection/
│
├── README.md                          # 项目说明
├── CODE_LIST.md                       # 文件清单和功能简介
├── PROJECT_STRUCTURE.md               # 本文件
├── DEPLOYMENT_GUIDE.md                # 部署和文件组织指南
├── requirements.txt                   # Python 依赖列表
│
├── configs/
│   └── train_config.yaml              # 统一训练/数据/模型配置
│
├── data/
│   ├── MVTec_AD/                      # ← 用户自行下载并解压到此处
│   └── processed/                     # ← prepare_mvtec.py 自动生成
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── dataset.yaml               # 自动生成的数据集描述
│
├── src/                               # 核心源代码包
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mask_to_bbox.py            # PNG掩码 → YOLO边界框转换
│   │   └── mvtec_dataset.py           # Dataset / FewShotDataset 加载器
│   ├── models/
│   │   ├── __init__.py
│   │   ├── prototypical_network.py    # 原型网络(距离度量 + 原型计算)
│   │   └── proto_yolo.py              # ProtoYOLO / SimpleProtoYOLO 主模型
│   ├── training/
│   │   ├── __init__.py
│   │   └── meta_trainer.py            # Episode级别的元训练器
│   └── gui/
│       ├── __init__.py
│       └── main_window.py             # PyQt5 图形界面
│
├── scripts/                           # 命令行入口脚本
│   ├── prepare_mvtec.py               # 数据预处理(掩码转换 + 划分 + 生成yaml)
│   ├── train_meta.py                  # 元训练脚本(episodic)
│   ├── train.py                       # 标准训练脚本(epoch-level)
│   └── detect.py                      # 推理检测脚本
│
├── experiments/                       # 训练产出(运行后自动创建)
│   ├── checkpoints/                   # 模型权重 best.pt / epoch_N.pt
│   └── logs/                          # 训练日志
│
└── outputs/                           # 推理产出(运行后自动创建)
    └── predictions/                   # 检测结果图像和txt
```

---

## 各模块说明

### configs/
唯一的配置入口 `train_config.yaml`。所有脚本均从此文件读取参数，**不需要**再维护多个 yaml。

### src/ 包结构
| 子包 | 文件 | 核心职责 |
|------|------|----------|
| data | mask_to_bbox.py | 读取MVTec的PNG二值掩码，用 `cv2.connectedComponentsWithStats` 提取连通域，转为YOLO格式 `(cls, cx, cy, w, h)` |
| data | mvtec_dataset.py | `MVTecDataset`：标准 PyTorch Dataset；`FewShotMVTecDataset`：按 N-way K-shot 采样 episode |
| models | prototypical_network.py | 计算类原型向量，支持欧氏/余弦距离度量 |
| models | proto_yolo.py | `ProtoYOLO`（依赖ultralytics）和 `SimpleProtoYOLO`（独立CNN backbone），双模式前向: detection / prototype |
| training | meta_trainer.py | Episode训练循环、AdamW优化器、余弦调度、梯度裁剪、检查点管理 |
| gui | main_window.py | PyQt5 主窗口: 训练配置面板、图像加载、检测结果展示 |

### scripts/ 入口说明
| 脚本 | 用途 | 对应的核心模块 |
|------|------|---------------|
| prepare_mvtec.py | 一次性数据预处理 | `src.data.mask_to_bbox` |
| train_meta.py | episodic 元训练 | `src.data.mvtec_dataset`, `src.models.proto_yolo`, `src.training.meta_trainer` |
| train.py | 标准 epoch 训练 | `src.data.mvtec_dataset`, `src.models.proto_yolo` |
| detect.py | 加权重推理 | `src.models.proto_yolo` |

---

## 完整使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据预处理(需要先把MVTec AD解压到 data/MVTec_AD/)
python scripts/prepare_mvtec.py --data_root data/MVTec_AD --output_dir data/processed

# 3a. 元训练(推荐)
python scripts/train_meta.py --config configs/train_config.yaml

# 3b. 标准训练(备选)
python scripts/train.py --config configs/train_config.yaml

# 3c. 简化模型(无需 ultralytics)
python scripts/train_meta.py --config configs/train_config.yaml --use_simple_model

# 4. 推理检测
python scripts/detect.py --weights experiments/checkpoints/best.pt --source data/processed/images/test

# 5. GUI
python -m src.gui.main_window
```
