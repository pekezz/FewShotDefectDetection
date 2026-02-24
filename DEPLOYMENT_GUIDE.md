# 代码部署和使用指南

---

## 第一步：创建项目目录

```bash
mkdir FewShotDefectDetection
cd FewShotDefectDetection
```

---

## 第二步：把下载的文件放到对应位置

下载的文件名带有路径前缀（用下划线拼接），按照以下对照表**重命名并放入正确目录**即可。

### 2.1 根目录文件（直接放进去）

```
FewShotDefectDetection/
├── requirements.txt
├── README.md
├── CODE_LIST.md
├── PROJECT_STRUCTURE.md
└── DEPLOYMENT_GUIDE.md
```

### 2.2 configs/

```bash
mkdir -p configs
```

| 下载文件名 | 放入路径 |
|-----------|----------|
| train_config.yaml | configs/train_config.yaml |

### 2.3 src/ (核心源代码包)

```bash
mkdir -p src/data src/models src/training src/gui
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/training/__init__.py src/gui/__init__.py
```

| 下载文件名 | 放入路径 |
|-----------|----------|
| src_data_mask_to_bbox.py | src/data/mask_to_bbox.py |
| src_data_mvtec_dataset.py | src/data/mvtec_dataset.py |
| src_models_prototypical_network.py | src/models/prototypical_network.py |
| src_models_proto_yolo.py | src/models/proto_yolo.py |
| src_training_meta_trainer.py | src/training/meta_trainer.py |
| src_gui_main_window.py | src/gui/main_window.py |

### 2.4 scripts/ (命令行入口)

```bash
mkdir -p scripts
```

| 下载文件名 | 放入路径 |
|-----------|----------|
| scripts_prepare_mvtec.py | scripts/prepare_mvtec.py |
| scripts_train_meta.py | scripts/train_meta.py |
| scripts_train.py | scripts/train.py |
| scripts_detect.py | scripts/detect.py |

### 2.5 其他目录（空文件夹，运行时自动创建也可）

```bash
mkdir -p data/MVTec_AD          # 用户需手动将MVTec AD数据集解压到此
mkdir -p data/processed         # prepare_mvtec.py 自动生成
mkdir -p experiments/checkpoints
mkdir -p experiments/logs
mkdir -p outputs/predictions
```

---

## 最终目录结构（和 PROJECT_STRUCTURE.md 完全一致）

```
FewShotDefectDetection/
│
├── README.md
├── CODE_LIST.md
├── PROJECT_STRUCTURE.md
├── DEPLOYMENT_GUIDE.md
├── requirements.txt
│
├── configs/
│   └── train_config.yaml
│
├── data/
│   ├── MVTec_AD/                  # ← 用户下载并解压到此
│   └── processed/                 # ← prepare_mvtec.py 生成
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/
│       └── dataset.yaml
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mask_to_bbox.py
│   │   └── mvtec_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── prototypical_network.py
│   │   └── proto_yolo.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── meta_trainer.py
│   └── gui/
│       ├── __init__.py
│       └── main_window.py
│
├── scripts/
│   ├── prepare_mvtec.py
│   ├── train_meta.py
│   ├── train.py
│   └── detect.py
│
├── experiments/
│   ├── checkpoints/
│   └── logs/
│
└── outputs/
    └── predictions/
```

---

## 快速组织脚本

如果嫌逐个移动麻烦，把以下内容保存为 `organize_files.sh`，放在下载文件所在目录运行即可：

```bash
#!/bin/bash
# organize_files.sh — 自动把下载的文件放到正确目录

set -e

# 创建目录
mkdir -p configs
mkdir -p src/{data,models,training,gui}
mkdir -p scripts
mkdir -p data/{MVTec_AD,processed}
mkdir -p experiments/{checkpoints,logs}
mkdir -p outputs/predictions

# 创建 __init__.py
touch src/__init__.py
touch src/{data,models,training,gui}/__init__.py

# --- configs ---
mv train_config.yaml            configs/                          2>/dev/null || true

# --- src/data ---
mv src_data_mask_to_bbox.py     src/data/mask_to_bbox.py          2>/dev/null || true
mv src_data_mvtec_dataset.py    src/data/mvtec_dataset.py         2>/dev/null || true

# --- src/models ---
mv src_models_prototypical_network.py  src/models/prototypical_network.py  2>/dev/null || true
mv src_models_proto_yolo.py            src/models/proto_yolo.py            2>/dev/null || true

# --- src/training ---
mv src_training_meta_trainer.py src/training/meta_trainer.py      2>/dev/null || true

# --- src/gui ---
mv src_gui_main_window.py       src/gui/main_window.py            2>/dev/null || true

# --- scripts ---
mv scripts_prepare_mvtec.py     scripts/prepare_mvtec.py          2>/dev/null || true
mv scripts_train_meta.py        scripts/train_meta.py             2>/dev/null || true
mv scripts_train.py             scripts/train.py                  2>/dev/null || true
mv scripts_detect.py            scripts/detect.py                 2>/dev/null || true

echo "✅ 文件组织完成！"
echo "   下一步: pip install -r requirements.txt"
```

```bash
chmod +x organize_files.sh
./organize_files.sh
```

---

## 环境安装

```bash
pip install -r requirements.txt
# ultralytics 已包含在 requirements.txt 中；
# 如果 pip install ultralytics 失败，可以用 --use_simple_model 标记绕过。
```

---

## 数据准备

1. 访问 https://www.mvtec.com/company/research/datasets/mvtec-ad 下载数据集
2. 解压到 `data/MVTec_AD/`
3. 运行预处理：

```bash
python scripts/prepare_mvtec.py --data_root data/MVTec_AD --output_dir data/processed
```

预处理完成后 `data/processed/` 内会自动生成 `images/`、`labels/`、`dataset.yaml`。

---

## 训练

```bash
# 元训练（推荐，少样本场景）
python scripts/train_meta.py --config configs/train_config.yaml

# 标准训练
python scripts/train.py --config configs/train_config.yaml

# 不依赖 ultralytics 的简化模型
python scripts/train_meta.py --config configs/train_config.yaml --use_simple_model

# 从检查点恢复
python scripts/train_meta.py --config configs/train_config.yaml --resume experiments/checkpoints/epoch_50.pt
```

---

## 推理检测

```bash
# 单图
python scripts/detect.py --weights experiments/checkpoints/best.pt --source path/to/image.png

# 批量检测目录
python scripts/detect.py --weights experiments/checkpoints/best.pt --source data/processed/images/test

# 同时保存 txt 标注
python scripts/detect.py --weights experiments/checkpoints/best.pt --source images/ --save_txt --save_conf
```

---

## GUI 运行

```bash
python -m src.gui.main_window
```

> **注意**：必须从项目根目录 `FewShotDefectDetection/` 下执行，否则 `src` 包找不到。

---

## 常见问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'src'` | 没有从项目根目录运行 | `cd FewShotDefectDetection` 后再执行 |
| `ModuleNotFoundError: No module named 'ultralytics'` | ultralytics 未装 | `pip install ultralytics` 或加 `--use_simple_model` |
| `FileNotFoundError: data/processed/...` | 没跑预处理 | 先执行 `prepare_mvtec.py` |
| CUDA OOM | 显存不够 | 在 `train_config.yaml` 里把 `image_size` 降为 512，或换用 `yolov8n` |
| `__init__.py` 缺失 | 没创建空文件 | 用 `organize_files.sh` 或手动 `touch` |

---

## 导入路径约定

所有代码内部的 import 都基于项目根目录：

```python
from src.data.mask_to_bbox         import MaskToBBoxConverter
from src.data.mvtec_dataset        import MVTecDataset, FewShotMVTecDataset
from src.models.prototypical_network import PrototypicalNetwork
from src.models.proto_yolo         import ProtoYOLO, SimpleProtoYOLO
from src.training.meta_trainer     import MetaTrainer
```

如果需要在 `src/` 内部某文件里引用兄弟包，也用完整路径（如 `from src.models.prototypical_network import ...`），保证无论从哪里调用都一致。
