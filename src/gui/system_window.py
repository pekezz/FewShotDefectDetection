"""
工业缺陷检测系统 GUI (PyQt5)
功能：类别多缺陷聚合版 + 全局真正暗黑模式 + 支持删除 + 【可缩放平移视图】
"""

import sys
import time
import json
import cv2
import yaml
import torch
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QGroupBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox,
    QSplitter, QFrame, QDialog, QListWidget, QInputDialog, QProgressDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QPainter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.proto_yolo import ProtoYOLO
from ultralytics import YOLO


# -------------------------------------------------------------------------
# 新增自定义控件：支持鼠标滚轮缩放和平移的图片视图
# -------------------------------------------------------------------------
class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # 初始化场景
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None

        # 设置交互模式
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # 允许鼠标拖拽平移
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # 以鼠标中心缩放
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # 隐藏滚动条，保持界面简洁 (可选，根据喜好注释掉)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 设置背景色与主题融合
        self.setBackgroundBrush(QColor("#1e1e1e"))
        self.setStyleSheet("border: 1px solid #3c3f41; border-radius: 10px;")

    def set_image(self, qpixmap):
        """加载新图片并自动适应视图大小"""
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(qpixmap)
        self.setSceneRect(QRectF(qpixmap.rect()))  # 设置场景边界
        # 初始加载时，将图片完整适配到视图中
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        """重写滚轮事件以实现缩放"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # 保存当前的变换矩阵
        old_matrix = self.transform()

        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

        # 获取当前缩放级别
        current_scale = self.transform().m11()

        # 限制最大最小缩放比例 (例如：最小 0.1倍，最大 10倍)
        if current_scale < 0.1 or current_scale > 10.0:
            self.setTransform(old_matrix)


# -------------------------------------------------------------------------
# 配置管理
# -------------------------------------------------------------------------
CONFIG_FILE = "data/gui_config.json"
DATASET_YAML = "data/processed/dataset.yaml"
DEFAULT_YOLO = "experiments/finetune_yolo/weights/best.pt"
DEFAULT_META = "experiments/checkpoints/best.pt"


class ConfigManager:
    def __init__(self):
        self.base_categories = []
        self.config = self.load_or_init()

    def load_or_init(self):
        config = {"categories": {}, "adapted_models": {}}
        if Path(CONFIG_FILE).exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)

        base_names = []
        if Path(DATASET_YAML).exists():
            with open(DATASET_YAML, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                names = data.get('names', [])
                if isinstance(names, dict):
                    base_names = [names[i] for i in sorted(names.keys())]
                elif isinstance(names, list):
                    base_names = names

        self.base_categories = base_names.copy()

        for name in base_names:
            if name not in config["categories"]:
                config["categories"][name] = []

        train_img_dir = Path("data/processed/images/train")
        if train_img_dir.exists():
            for img_path in train_img_dir.glob("*.*"):
                filename = img_path.stem
                for cat in base_names:
                    if filename.startswith(cat + "_"):
                        remainder = filename[len(cat) + 1:]
                        if "_aug_" in remainder:
                            remainder = remainder.split("_aug_")[0]
                        parts = remainder.rsplit('_', 1)
                        if len(parts) >= 1:
                            defect_type = parts[0]
                            if defect_type and defect_type not in config["categories"][cat]:
                                config["categories"][cat].append(defect_type)
                        break
        self.save_internal(config)
        return config

    def save_internal(self, config):
        Path(CONFIG_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    def save(self):
        self.save_internal(self.config)


# -------------------------------------------------------------------------
# 线程 1: 少样本微调工作线程
# -------------------------------------------------------------------------
class AdaptationThread(QThread):
    progress_update = pyqtSignal(int)
    finished_success = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, category, defect, folder, all_defects):
        super().__init__()
        self.category = category
        self.defect = defect
        self.folder = folder
        self.all_defects = all_defects

    def run(self):
        try:
            from scripts.few_shot_adapt import run_few_shot_adaptation
            model_path = run_few_shot_adaptation(
                meta_weights=DEFAULT_META,
                yolo_weights=DEFAULT_YOLO,
                source_dir=self.folder,
                category_name=self.category,
                defect_name=self.defect,
                all_defects=self.all_defects,
                k_shot=5,
                epochs=50,
                progress_callback=self.progress_update.emit
            )
            self.finished_success.emit(model_path)
        except Exception as e:
            self.error_occurred.emit(str(e))


# -------------------------------------------------------------------------
# Layer 3: 缺陷类型管理窗口
# -------------------------------------------------------------------------
class DefectDialog(QDialog):
    def __init__(self, category_name, config_mgr, parent=None):
        super().__init__(parent)
        self.category_name = category_name
        self.config_mgr = config_mgr
        self.setWindowTitle(f"[{category_name}] 缺陷管理")
        self.resize(450, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.refresh_list()

        btn_add = QPushButton("➕ 添加新缺陷类型 (并更新类别模型)")
        btn_add.clicked.connect(self.add_defect)

        btn_tune = QPushButton("🔄 对选中的已有缺陷提供新样本并微调")
        btn_tune.clicked.connect(self.tune_existing_defect)

        btn_del = QPushButton("🗑️ 删除选中的缺陷标签 (不改动模型)")
        btn_del.setStyleSheet("background-color: #c0392b; color: white;")
        btn_del.clicked.connect(self.delete_defect)

        layout.addWidget(QLabel(f"类别: {self.category_name} (每次添加或微调都会更新专属模型)"))
        layout.addWidget(self.list_widget)
        layout.addWidget(btn_add)
        layout.addWidget(btn_tune)
        layout.addWidget(btn_del)
        self.setLayout(layout)

    def refresh_list(self):
        self.list_widget.clear()
        defects = self.config_mgr.config["categories"].get(self.category_name, [])
        for d in defects: self.list_widget.addItem(d)

    def add_defect(self):
        defect_name, ok = QInputDialog.getText(self, "新增", "输入缺陷名称:")
        if not ok or not defect_name.strip(): return
        defect_name = defect_name.strip()
        if defect_name in self.config_mgr.config["categories"][self.category_name]:
            QMessageBox.warning(self, "警告", "已存在！")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择数据集文件夹")
        if folder: self.start_adaptation(defect_name, folder, is_new=True)

    def tune_existing_defect(self):
        item = self.list_widget.currentItem()
        if not item: return
        defect_name = item.text()
        folder = QFileDialog.getExistingDirectory(self, f"选择 [{defect_name}] 数据集")
        if folder: self.start_adaptation(defect_name, folder, is_new=False)

    def delete_defect(self):
        item = self.list_widget.currentItem()
        if not item: return
        defect_name = item.text()

        reply = QMessageBox.question(self, '确认',
                                     f"确定要移除缺陷标签 [{defect_name}] 吗？\n(这仅会从界面列表中将其隐藏，不会修改已有的模型文件)",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if defect_name in self.config_mgr.config["categories"].get(self.category_name, []):
                self.config_mgr.config["categories"][self.category_name].remove(defect_name)
                self.config_mgr.save()
                self.refresh_list()
                if self.parent() and self.parent().parent():
                    self.parent().parent().refresh_model_list()

    def start_adaptation(self, defect_name, folder, is_new=True):
        self.is_new = is_new
        all_defects = self.config_mgr.config["categories"][self.category_name].copy()
        if is_new and defect_name not in all_defects:
            all_defects.append(defect_name)

        self.progress_dlg = QProgressDialog(f"正在聚合训练 [{self.category_name}] 的模型...", "取消", 0, 100, self)
        self.progress_dlg.setWindowModality(Qt.WindowModal)
        self.progress_dlg.show()

        self.thread = AdaptationThread(self.category_name, defect_name, folder, all_defects)
        self.thread.progress_update.connect(self.progress_dlg.setValue)
        self.thread.finished_success.connect(lambda path: self.on_success(defect_name, path))
        self.thread.error_occurred.connect(self.on_error)
        self.progress_dlg.canceled.connect(self.thread.terminate)
        self.thread.start()

    def on_success(self, defect_name, model_path):
        self.progress_dlg.close()
        if self.is_new:
            self.config_mgr.config["categories"][self.category_name].append(defect_name)

        self.config_mgr.config["adapted_models"][self.category_name] = model_path
        self.config_mgr.save()

        self.refresh_list()
        QMessageBox.information(self, "成功", f"[{self.category_name}] 类专属多分类模型已成功更新！")
        if self.parent() and self.parent().parent(): self.parent().parent().refresh_model_list()

    def on_error(self, err):
        self.progress_dlg.close()
        QMessageBox.critical(self, "失败", str(err))


# -------------------------------------------------------------------------
# Layer 2: 类别管理窗口
# -------------------------------------------------------------------------
class CategoryDialog(QDialog):
    def __init__(self, config_mgr, parent=None):
        super().__init__(parent)
        self.config_mgr = config_mgr
        self.setWindowTitle("类别管理")
        self.resize(350, 450)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.refresh_list()

        btn_add = QPushButton("1. ➕ 添加新类别 (Object)")
        btn_add.clicked.connect(self.add_category)

        btn_view = QPushButton("2. 🔍 查看/管理 该类别的缺陷 ->")
        btn_view.clicked.connect(self.view_defects)

        btn_del = QPushButton("3. 🗑️ 删除选中的自定义类别")
        btn_del.setStyleSheet("background-color: #c0392b; color: white;")
        btn_del.clicked.connect(self.delete_category)

        layout.addWidget(QLabel("物体类别 (双击查看下属缺陷):"))
        layout.addWidget(self.list_widget)
        layout.addWidget(btn_add)
        layout.addWidget(btn_view)
        layout.addWidget(btn_del)

        self.list_widget.itemDoubleClicked.connect(self.view_defects)
        self.setLayout(layout)

    def refresh_list(self):
        self.list_widget.clear()
        for cat in self.config_mgr.config["categories"].keys():
            if cat in self.config_mgr.base_categories:
                self.list_widget.addItem(f"{cat} (基础)")
            else:
                self.list_widget.addItem(cat)

    def _get_real_cat_name(self, text):
        return text.replace(" (基础)", "")

    def add_category(self):
        cat_name, ok = QInputDialog.getText(self, "新增", "类别名称:")
        if not ok or not cat_name.strip(): return
        cat_name = cat_name.strip()
        if cat_name in self.config_mgr.config["categories"]:
            QMessageBox.warning(self, "警告", "该类别已存在！")
            return

        self.config_mgr.config["categories"][cat_name] = []
        self.config_mgr.save()
        self.refresh_list()
        self.open_defect(cat_name)

    def view_defects(self):
        item = self.list_widget.currentItem()
        if item: self.open_defect(self._get_real_cat_name(item.text()))

    def delete_category(self):
        item = self.list_widget.currentItem()
        if not item: return
        cat_name = self._get_real_cat_name(item.text())

        if cat_name in self.config_mgr.base_categories:
            QMessageBox.warning(self, "操作被拒绝", f"[{cat_name}] 属于基础数据集类别，不允许删除！")
            return

        reply = QMessageBox.question(self, '确认删除',
                                     f"警告：确定要删除类别 [{cat_name}] 吗？\n\n这将会清除该类别下的所有缺陷记录，并且彻底删除硬盘上的专属模型文件夹和数据！",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            keys_to_delete = []
            for key, path_str in self.config_mgr.config.get("adapted_models", {}).items():
                if key == cat_name or key.startswith(cat_name + "_"):
                    keys_to_delete.append(key)
                    model_path = Path(path_str)
                    if model_path.exists():
                        task_folder = model_path.parent.parent
                        try:
                            shutil.rmtree(task_folder)
                            print(f"已彻底删除训练文件夹: {task_folder}")
                        except Exception as e:
                            print(f"删除训练文件夹失败: {e}")

            for key in keys_to_delete:
                del self.config_mgr.config["adapted_models"][key]

            support_dir = Path("data/support_sets") / cat_name
            if support_dir.exists():
                try:
                    shutil.rmtree(support_dir)
                except Exception as e:
                    pass

            if cat_name in self.config_mgr.config["categories"]:
                del self.config_mgr.config["categories"][cat_name]

            self.config_mgr.save()
            self.refresh_list()
            QMessageBox.information(self, "成功", f"类别 [{cat_name}] 及其所有底层模型、缓存文件夹已被彻底清除。")

            if self.parent():
                self.parent().refresh_model_list()

    def open_defect(self, cat_name):
        dlg = DefectDialog(cat_name, self.config_mgr, self)
        dlg.exec_()
        if self.parent(): self.parent().refresh_model_list()


# -------------------------------------------------------------------------
# 线程 2: 推理检测线程
# -------------------------------------------------------------------------
class InferenceThread(QThread):
    frame_processed = pyqtSignal(np.ndarray, np.ndarray, list, float, str, str)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_info, conf_thres, source_type, source_path):
        super().__init__()
        self.model_info = model_info
        self.conf_thres = conf_thres
        self.source_type = source_type
        self.source_path = source_path
        self.running = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        self.status_update.emit("正在加载模型...")
        try:
            import sys
            import numpy as np
            if not hasattr(np, '_core') and 'numpy.core' in sys.modules:
                sys.modules['numpy._core'] = sys.modules['numpy.core']
                sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']

            mode = self.model_info['mode']
            if mode == "adapted":
                self.model = YOLO(self.model_info['model_path'])
            else:
                self.model = ProtoYOLO(yolo_weights=self.model_info['yolo_w'], num_classes=15)
                if mode == "meta":
                    try:
                        ckpt = torch.load(self.model_info['meta_w'], map_location=self.device, weights_only=False)
                    except TypeError:
                        ckpt = torch.load(self.model_info['meta_w'], map_location=self.device)
                    self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
                self.model.to(self.device)
                self.model.eval()
            return True
        except Exception as e:
            self.status_update.emit(f"模型加载失败: {e}")
            return False

    def detect_single(self, img_bgr):
        t0 = time.time()
        detections = []
        mode = self.model_info['mode']

        if mode == "adapted":
            res = self.model.predict(img_bgr, verbose=False, conf=self.conf_thres)[0]
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                for i in range(len(boxes)):
                    detections.append({"box": boxes[i].tolist(), "score": float(confs[i]), "cls": int(clss[i])})
        else:
            img_size = 640
            resized = cv2.resize(img_bgr, (img_size, img_size))
            tensor = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.model(tensor, mode='detection')

            if preds and len(preds) > 0 and hasattr(preds[0], 'boxes') and preds[0].boxes is not None:
                res = preds[0]
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                h, w = img_bgr.shape[:2]
                for i in range(len(boxes)):
                    if confs[i] >= self.conf_thres:
                        bx = boxes[i].tolist()
                        bx = [bx[0] * w / img_size, bx[1] * h / img_size, bx[2] * w / img_size, bx[3] * h / img_size]
                        detections.append({"box": bx, "score": float(confs[i]), "cls": int(clss[i])})
        return detections, time.time() - t0

    def draw(self, img, dets):
        vis = img.copy()
        names = self.model_info['cls_names']
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]

        for d in dets:
            x1, y1, x2, y2 = map(int, d['box'])
            cls_id = d['cls']
            label = names[cls_id] if cls_id < len(names) else f"cls{cls_id}"
            text = f"{label} {d['score']:.2f}"
            c = colors[cls_id % len(colors)]

            # 1. 画检测框
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)

            # 2. 获取文字尺寸
            font_scale = 0.6
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # 计算背景框坐标
            txt_y1 = max(0, y1 - th - baseline - 5)
            txt_y2 = txt_y1 + th + baseline + 5

            # 3. 画文字底部的实心背景色块
            cv2.rectangle(vis, (x1, txt_y1), (x1 + tw, txt_y2), c, -1)

            # 4. 画文字 (开启抗锯齿)
            cv2.putText(vis, text, (x1, txt_y2 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

        return vis

    def run(self):
        self.running = True
        if not self.load_model(): return

        if self.source_type in ['image', 'video']:
            filename = Path(self.source_path).stem
            input_cat = filename.split('_')[0] if '_' in filename else filename
        else:
            input_cat = "未知"

        title_in = f"【输入】"

        if self.source_type == 'image':
            img = cv2.imread(self.source_path)
            if img is not None:
                dets, t = self.detect_single(img)
                vis = self.draw(img, dets)

                if not dets:
                    title_out = f"【结果】(状态: 正常)"
                else:
                    det_cats = []
                    for d in dets:
                        idx = d['cls']
                        names = self.model_info['cls_names']
                        c = names[idx] if idx < len(names) else "未知"
                        if c not in det_cats: det_cats.append(c)
                    title_out = f"【结果】 (发现缺陷: {','.join(det_cats)})"

                self.frame_processed.emit(img, vis, dets, t, title_in, title_out)
                self.status_update.emit(f"检测完成，耗时: {t * 1000:.1f} ms")

        elif self.source_type == 'video':
            cap = cv2.VideoCapture(self.source_path)
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                dets, t = self.detect_single(frame)
                vis = self.draw(frame, dets)

                if not dets:
                    title_out = f"【结果】 (状态: 正常)"
                else:
                    det_cats = []
                    for d in dets:
                        idx = d['cls']
                        names = self.model_info['cls_names']
                        c = names[idx] if idx < len(names) else "未知"
                        if c not in det_cats: det_cats.append(c)
                    title_out = f"【结果】 (发现缺陷: {','.join(det_cats)})"

                self.frame_processed.emit(frame, vis, dets, t, title_in, title_out)
                time.sleep(0.01)
            cap.release()

        self.running = False
        self.finished.emit()

    def stop(self):
        self.running = False


# -------------------------------------------------------------------------
# Layer 1: 主窗口 UI
# -------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("少样本工业缺陷检测系统")
        self.resize(1280, 800)
        self.config_mgr = ConfigManager()
        self.init_ui()
        self.inf_thread = None

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        ctrl_panel = QFrame()
        ctrl_panel.setFixedWidth(340)
        ctrl_panel.setStyleSheet(
            "QFrame { background-color: #2b2b2b; border-radius: 10px; border: 1px solid #3c3f41; }")
        ctrl_layout = QVBoxLayout()
        ctrl_panel.setLayout(ctrl_layout)

        mgr_group = QGroupBox("✨ 模型与任务管理")
        mgr_layout = QVBoxLayout()

        btn_manage = QPushButton("📂 类别与缺陷管理 (微调)")
        btn_manage.clicked.connect(self.open_category_manager)
        mgr_layout.addWidget(btn_manage)

        mgr_layout.addWidget(QLabel("当前使用模型:"))
        self.combo_model = QComboBox()
        self.refresh_model_list()
        mgr_layout.addWidget(self.combo_model)

        mgr_group.setLayout(mgr_layout)
        ctrl_layout.addWidget(mgr_group)

        param_group = QGroupBox("⚙️ 检测控制")
        param_layout = QVBoxLayout()

        param_layout.addWidget(QLabel("置信度阈值 (Conf):"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        param_layout.addWidget(self.spin_conf)

        btn_img = QPushButton("🖼️ 检测单张图片")
        btn_img.clicked.connect(lambda: self.start_detect('image'))

        btn_vid = QPushButton("🎞️ 检测视频流")
        btn_vid.clicked.connect(lambda: self.start_detect('video'))

        self.btn_stop = QPushButton("🛑 停止检测")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_detect)

        param_layout.addWidget(btn_img)
        param_layout.addWidget(btn_vid)
        param_layout.addWidget(self.btn_stop)
        param_group.setLayout(param_layout)
        ctrl_layout.addWidget(param_group)

        log_group = QGroupBox("📝 运行日志")
        log_layout = QVBoxLayout()
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setStyleSheet("border: none; background-color: #1e1e1e; color: #a9b7c6; border-radius: 5px;")
        log_layout.addWidget(self.log_viewer)
        log_group.setLayout(log_layout)
        ctrl_layout.addWidget(log_group)

        main_layout.addWidget(ctrl_panel)

        display_panel = QWidget()
        display_layout = QVBoxLayout()
        display_panel.setLayout(display_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)

        # 替换为自定义的可缩放视图控件
        self.view_src = ZoomableImageView()
        self.lbl_src_title = QLabel("【输入】 类别: -")
        self.lbl_src_title.setAlignment(Qt.AlignCenter)
        self.lbl_src_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #4facfe; margin-top: 10px;")

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.view_src, stretch=1)
        left_layout.addWidget(self.lbl_src_title)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # 替换为自定义的可缩放视图控件
        self.view_dst = ZoomableImageView()
        self.lbl_dst_title = QLabel("【结果】 类别: -")
        self.lbl_dst_title.setAlignment(Qt.AlignCenter)
        self.lbl_dst_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ff7675; margin-top: 10px;")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.view_dst, stretch=1)
        right_layout.addWidget(self.lbl_dst_title)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        display_layout.addWidget(splitter, stretch=1)

        self.lbl_stats = QLabel("🚀 系统就绪 | 等待检测任务")
        self.lbl_stats.setStyleSheet("font-weight: bold; font-size: 15px; color: #a9b7c6; margin-top: 10px;")
        display_layout.addWidget(self.lbl_stats)

        main_layout.addWidget(display_panel, stretch=1)

    def refresh_model_list(self):
        self.combo_model.clear()
        self.model_configs = []
        base_names = list(self.config_mgr.config["categories"].keys())
        self.combo_model.addItem("✨ [原装] YOLO 全监督模型 (15类)")
        self.model_configs.append({'mode': 'yolo', 'yolo_w': DEFAULT_YOLO, 'meta_w': None, 'cls_names': base_names})

        self.combo_model.addItem("🧠 [原装] 元学习 ProtoYOLO (15类)")
        self.model_configs.append(
            {'mode': 'meta', 'yolo_w': DEFAULT_YOLO, 'meta_w': DEFAULT_META, 'cls_names': base_names})

        for cat, model_path in self.config_mgr.config.get("adapted_models", {}).items():
            if Path(model_path).exists():
                self.combo_model.addItem(f"🎯 [专属类别模型] {cat}")
                cls_names = self.config_mgr.config["categories"].get(cat, [cat])
                self.model_configs.append({'mode': 'adapted', 'model_path': model_path, 'cls_names': cls_names})

    def open_category_manager(self):
        dlg = CategoryDialog(self.config_mgr, self)
        dlg.exec_()
        self.refresh_model_list()

    def start_detect(self, s_type):
        if s_type == 'image':
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mkv)")

        if not path: return

        if self.inf_thread and self.inf_thread.isRunning():
            self.inf_thread.stop()
            self.inf_thread.wait()

        idx = self.combo_model.currentIndex()
        if idx < 0: return

        self.inf_thread = InferenceThread(
            model_info=self.model_configs[idx],
            conf_thres=self.spin_conf.value(),
            source_type=s_type,
            source_path=path
        )
        self.inf_thread.frame_processed.connect(self.update_ui)
        self.inf_thread.status_update.connect(self.log)
        self.inf_thread.finished.connect(lambda: self.btn_stop.setEnabled(False))

        self.btn_stop.setEnabled(True)
        self.inf_thread.start()

    def stop_detect(self):
        if self.inf_thread: self.inf_thread.stop()

    def update_ui(self, src, dst, dets, t, title_in, title_out):
        # 将图片转换为 QPixmap 并传递给自定义视图
        self.view_src.set_image(self._convert_cv2_to_pixmap(src))
        self.view_dst.set_image(self._convert_cv2_to_pixmap(dst))

        self.lbl_src_title.setText(title_in)
        self.lbl_dst_title.setText(title_out)

        fps = 1.0 / (t + 1e-6)
        self.lbl_stats.setText(f"🚀 FPS: {fps:.1f} | 检出目标数: {len(dets)} | 耗时: {t * 1000:.1f} ms")

    def _convert_cv2_to_pixmap(self, img_bgr):
        """辅助函数：将 OpenCV 图像转换为 QPixmap"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, c * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def log(self, text):
        self.log_viewer.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")


def apply_dark_theme(app):
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(43, 43, 43))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.AlternateBase, QColor(43, 43, 43))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(60, 63, 65))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

    style_sheet = """
        QGroupBox {
            font-size: 15px;
            font-weight: bold;
            margin-top: 15px;
            border: 1px solid #555555;
            border-radius: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #4facfe;
        }
        QPushButton {
            background-color: #0e639c;
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #1177bb; }
        QPushButton:disabled { background-color: #555555; color: #888888; }
        QPushButton#btn_stop { background-color: #c0392b; }
    """
    app.setStyleSheet(style_sheet)


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 11))
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())