# 代码文件清单

## 📦 文件列表及说明

### 📄 文档文件

| 文件名 | 说明 | 重要性 |
|--------|------|--------|
| README.md | 项目完整说明文档，包含安装、使用、原理等 | ⭐⭐⭐⭐⭐ |
| PROJECT_STRUCTURE.md | 详细的项目结构说明 | ⭐⭐⭐⭐ |
| DEPLOYMENT_GUIDE.md | 部署和使用指南 | ⭐⭐⭐⭐⭐ |
| CODE_LIST.md | 本文件，代码清单 | ⭐⭐⭐ |

### 🔧 配置文件

| 文件名 | 目标位置 | 说明 |
|--------|----------|------|
| requirements.txt | 根目录 | Python依赖包列表 |
| train_config.yaml | configs/ | 训练配置文件 |

### 💻 核心代码文件

#### 数据处理模块 (src/data/)

| 原文件名 | 目标文件名 | 功能说明 |
|----------|-----------|----------|
| src_data_mask_to_bbox.py | src/data/mask_to_bbox.py | **MVTec AD掩码转YOLO格式边界框** <br> - 读取PNG二值掩码 <br> - 连通区域检测 <br> - 转换为YOLO格式 <br> - 支持批量处理 |
| src_data_mvtec_dataset.py | src/data/mvtec_dataset.py | **数据集加载器** <br> - MVTecDataset类 <br> - FewShotMVTecDataset类 <br> - Episode采样 <br> - 数据增强集成 |

**关键特性**:
- ✅ 处理像素级掩码（MVTec AD特有）
- ✅ 自动边界框提取
- ✅ N-way K-shot采样
- ✅ Albumentations增强

#### 模型模块 (src/models/)

| 原文件名 | 目标文件名 | 功能说明 |
|----------|-----------|----------|
| src_models_prototypical_network.py | src/models/prototypical_network.py | **原型网络实现** <br> - 原型计算 <br> - 距离度量（欧氏/余弦） <br> - 原型分类 <br> - ProtoYOLO Head |
| src_models_proto_yolo.py | src/models/proto_yolo.py | **Proto-YOLO主模型** <br> - YOLOv8集成 <br> - 特征提取 <br> - 双模式架构 <br> - SimpleProtoYOLO（不依赖ultralytics） |

**关键特性**:
- ✅ 原型学习算法
- ✅ YOLOv8特征复用
- ✅ 双分支架构
- ✅ 简化版本（用于演示）

#### 训练模块 (src/training/)

| 原文件名 | 目标文件名 | 功能说明 |
|----------|-----------|----------|
| src_training_meta_trainer.py | src/training/meta_trainer.py | **元训练器** <br> - Episode训练 <br> - 学习率调度 <br> - 检查点管理 <br> - 训练日志 |

**关键特性**:
- ✅ Episodic training
- ✅ 余弦退火学习率
- ✅ 梯度裁剪
- ✅ 早停机制

#### GUI模块 (src/gui/)

| 原文件名 | 目标文件名 | 功能说明 |
|----------|-----------|----------|
| src_gui_main_window.py | src/gui/main_window.py | **PyQt5图形界面** <br> - 训练配置界面 <br> - 实时检测界面 <br> - 结果可视化 <br> - 多线程训练 |

**关键特性**:
- ✅ 友好的用户界面
- ✅ 参数配置
- ✅ 实时日志
- ✅ 图像可视化

### 🚀 脚本文件 (scripts/)

| 原文件名 | 目标文件名 | 功能说明 |
|----------|-----------|----------|
| scripts_prepare_mvtec.py | scripts/prepare_mvtec.py | **数据预处理** <br> - 批量掩码转换 <br> - 数据集划分 <br> - 生成配置文件 |
| scripts_train_meta.py | scripts/train_meta.py | **元训练主脚本** <br> - 加载配置 <br> - 创建数据加载器 <br> - 启动训练 |
| scripts_train.py | scripts/train.py | 标准训练脚本（备用） |
| scripts_detect.py | scripts/detect.py | 推理检测脚本（备用） |

## 🎯 核心功能实现说明

### 1. 掩码转边界框 (最重要！)

**为什么重要**: MVTec AD数据集提供的是PNG格式的像素级二值掩码，而YOLO需要边界框标注。这是使用该数据集的关键步骤。

**实现方法**:
```python
# src/data/mask_to_bbox.py
class MaskToBBoxConverter:
    def mask_to_bboxes(self, mask_path, width, height, class_id):
        # 1. 读取掩码图像
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. 二值化
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 3. 连通区域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        
        # 4. 提取每个区域的边界框
        for i in range(1, num_labels):
            x, y, w, h = stats[i, :4]
            # 转换为YOLO格式（归一化的中心坐标）
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            norm_w = w / width
            norm_h = h / height
            bboxes.append((class_id, x_center, y_center, norm_w, norm_h))
```

### 2. 少样本学习实现

**核心思想**: Episode训练 + 原型网络

**实现方法**:
```python
# 每个训练episode:
1. 随机选择N个类别
2. 每个类别采样K个支持样本和Q个查询样本
3. 计算每个类别的原型（支持集特征均值）
4. 查询样本与原型计算距离进行分类
5. 反向传播更新模型
```

**代码位置**:
- Episode采样: `src/data/mvtec_dataset.py` → `FewShotMVTecDataset`
- 原型计算: `src/models/prototypical_network.py` → `compute_prototypes`
- 训练循环: `src/training/meta_trainer.py` → `train_episode`

### 3. YOLOv8集成

**方法1**: 使用ultralytics包（推荐）
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
```

**方法2**: 简化版本（不依赖ultralytics）
```python
from src.models.proto_yolo import SimpleProtoYOLO
model = SimpleProtoYOLO(num_classes=5)
```

**代码位置**: `src/models/proto_yolo.py`

## 📊 数据流程图

```
MVTec AD数据集 (PNG掩码)
        ↓
[mask_to_bbox.py] 转换
        ↓
YOLO格式标注 (txt)
        ↓
[mvtec_dataset.py] 加载
        ↓
Episode采样 (N-way K-shot)
        ↓
[proto_yolo.py] 特征提取
        ↓
[prototypical_network.py] 原型学习
        ↓
[meta_trainer.py] 训练优化
        ↓
保存模型检查点
```

## 🔑 关键参数说明

### 训练配置 (train_config.yaml)

```yaml
# 少样本配置
few_shot:
  n_way: 5          # 类别数（建议5-10）
  k_shot: 5         # 每类样本数（建议5-10）
  query_num: 10     # 查询样本数（建议10-15）

# 训练参数
training:
  num_epochs: 200   # 训练轮数
  lr: 0.001         # 学习率
  batch_size: 1     # Episode级别，固定为1
  
# 模型参数
model:
  yolo_weights: "yolov8n.pt"  # 预训练权重
  num_classes: 5              # 类别总数
  proto_feature_dim: 256      # 原型特征维度
```

## ⚙️ 代码使用流程

### 完整流程

```bash
# 1. 组织文件结构
./organize_files.sh

# 2. 安装依赖
pip install -r requirements.txt
pip install ultralytics

# 3. 下载数据集
# 手动下载MVTec AD → data/MVTec_AD/

# 4. 预处理数据
python scripts/prepare_mvtec.py \
    --data_root data/MVTec_AD \
    --output_dir data/processed

# 5. 开始训练
python scripts/train_meta.py \
    --config configs/train_config.yaml

# 6. 运行GUI
python -m src.gui.main_window
```

### 快速测试（无需完整数据集）

```python
# 测试原型网络
python src/models/prototypical_network.py

# 测试简化模型
python src/models/proto_yolo.py

# 测试训练器
python src/training/meta_trainer.py
```

## 📝 代码修改建议

### 如需支持更多数据集

修改 `src/data/mvtec_dataset.py`:
```python
class CustomDataset(Dataset):
    def __init__(self, ...):
        # 实现自己的数据加载逻辑
        pass
```

### 如需添加新的增强方法

修改 `src/data/mvtec_dataset.py` 中的 `_get_default_transforms`:
```python
transforms = A.Compose([
    A.YourCustomAugmentation(),
    # ...
])
```

### 如需修改网络结构

修改 `src/models/proto_yolo.py`:
```python
class ProtoYOLO(nn.Module):
    def __init__(self, ...):
        # 修改backbone或添加新模块
        pass
```

## 🐛 常见问题速查

| 问题 | 解决方案 | 文件位置 |
|------|----------|----------|
| 导入错误 | 确保在根目录运行 | 所有Python文件 |
| 掩码转换失败 | 检查掩码路径和格式 | src/data/mask_to_bbox.py |
| 显存不足 | 减小batch_size和image_size | configs/train_config.yaml |
| YOLOv8安装失败 | 使用SimpleProtoYOLO | src/models/proto_yolo.py |

## 📈 性能优化建议

1. **数据加载**: 增加num_workers
2. **混合精度**: 启用AMP训练
3. **模型大小**: 使用yolov8n而非yolov8x
4. **推理加速**: 导出ONNX/TensorRT

## ✅ 代码验证清单

- [ ] 所有文件已按正确结构放置
- [ ] 创建了所有`__init__.py`
- [ ] 安装了所有依赖包
- [ ] YOLOv8可以正常导入
- [ ] 数据集下载并解压
- [ ] 运行预处理脚本成功
- [ ] 训练脚本可以启动
- [ ] GUI界面可以打开

---

**重要提醒**:
1. 文件名中的前缀（如`src_`、`scripts_`）需要去掉
2. 所有命令从项目根目录运行
3. 先阅读README.md了解整体架构
4. 遇到问题查看DEPLOYMENT_GUIDE.md

---

**文件总数**: 14个
**核心代码文件**: 8个
**配置文件**: 2个
**文档文件**: 4个

完整代码已提供，可直接运行！
