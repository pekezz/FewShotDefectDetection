"""
基于PyQt5的GUI主窗口
简化版示例
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QTabWidget,
    QProgressBar, QGroupBox, QFormLayout, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import torch
import cv2
import numpy as np


class TrainingThread(QThread):
    """训练线程"""
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def run(self):
        """运行训练"""
        self.log.emit("开始训练...")
        
        # 模拟训练过程
        for epoch in range(10):
            self.log.emit(f"Epoch {epoch+1}/10")
            self.progress.emit(int((epoch + 1) / 10 * 100))
            self.msleep(1000)  # 模拟训练时间
        
        self.log.emit("训练完成!")
        self.finished.emit()


class TrainWidget(QWidget):
    """训练界面"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 配置区域
        config_group = QGroupBox("训练配置")
        config_layout = QFormLayout()
        
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 1000)
        self.epoch_spin.setValue(200)
        config_layout.addRow("训练轮数:", self.epoch_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        config_layout.addRow("学习率:", self.lr_spin)
        
        self.nway_spin = QSpinBox()
        self.nway_spin.setRange(2, 20)
        self.nway_spin.setValue(5)
        config_layout.addRow("N-way:", self.nway_spin)
        
        self.kshot_spin = QSpinBox()
        self.kshot_spin.setRange(1, 50)
        self.kshot_spin.setValue(5)
        config_layout.addRow("K-shot:", self.kshot_spin)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 数据集选择
        data_group = QGroupBox("数据集")
        data_layout = QVBoxLayout()
        
        data_btn_layout = QHBoxLayout()
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setText("data/processed")
        self.data_select_btn = QPushButton("选择目录")
        self.data_select_btn.clicked.connect(self.select_data_dir)
        data_btn_layout.addWidget(self.data_path_edit)
        data_btn_layout.addWidget(self.data_select_btn)
        data_layout.addLayout(data_btn_layout)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.train_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 日志显示
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
    
    def select_data_dir(self):
        """选择数据目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if dir_path:
            self.data_path_edit.setText(dir_path)
    
    def start_training(self):
        """开始训练"""
        self.log_text.append("准备开始训练...")
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 创建训练配置
        config = {
            'epochs': self.epoch_spin.value(),
            'lr': self.lr_spin.value(),
            'n_way': self.nway_spin.value(),
            'k_shot': self.kshot_spin.value(),
            'data_path': self.data_path_edit.text()
        }
        
        # 启动训练线程
        self.training_thread = TrainingThread(config)
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.log.connect(self.append_log)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def append_log(self, text):
        """添加日志"""
        self.log_text.append(text)
    
    def training_finished(self):
        """训练完成"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


class DetectWidget(QWidget):
    """检测界面"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 模型加载
        model_group = QGroupBox("模型")
        model_layout = QHBoxLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setText("experiments/checkpoints/best.pt")
        self.model_select_btn = QPushButton("选择模型")
        self.model_select_btn.clicked.connect(self.select_model)
        self.model_load_btn = QPushButton("加载模型")
        self.model_load_btn.clicked.connect(self.load_model)
        
        model_layout.addWidget(QLabel("模型路径:"))
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.model_select_btn)
        model_layout.addWidget(self.model_load_btn)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 图像选择和检测
        detect_group = QGroupBox("检测")
        detect_layout = QVBoxLayout()
        
        img_btn_layout = QHBoxLayout()
        self.img_select_btn = QPushButton("选择图像")
        self.img_select_btn.clicked.connect(self.select_image)
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.clicked.connect(self.detect_image)
        self.detect_btn.setEnabled(False)
        img_btn_layout.addWidget(self.img_select_btn)
        img_btn_layout.addWidget(self.detect_btn)
        detect_layout.addLayout(img_btn_layout)
        
        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)
        
        # 图像显示
        img_layout = QHBoxLayout()
        
        # 原图
        orig_group = QGroupBox("原图")
        orig_layout = QVBoxLayout()
        self.orig_label = QLabel()
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setMinimumSize(400, 400)
        self.orig_label.setStyleSheet("border: 1px solid gray;")
        orig_layout.addWidget(self.orig_label)
        orig_group.setLayout(orig_layout)
        img_layout.addWidget(orig_group)
        
        # 检测结果
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 400)
        self.result_label.setStyleSheet("border: 1px solid gray;")
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)
        img_layout.addWidget(result_group)
        
        layout.addLayout(img_layout)
        
        # 检测结果文本
        result_text_group = QGroupBox("检测信息")
        result_text_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        result_text_layout.addWidget(self.result_text)
        result_text_group.setLayout(result_text_layout)
        layout.addWidget(result_text_group)
        
        self.setLayout(layout)
        
        self.current_image = None
    
    def select_model(self):
        """选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pt *.pth)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def load_model(self):
        """加载模型"""
        self.result_text.append("加载模型...")
        # TODO: 实际加载模型
        self.result_text.append("模型加载成功!")
        self.detect_btn.setEnabled(True)
    
    def select_image(self):
        """选择图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image, self.orig_label)
            self.result_text.append(f"已加载图像: {file_path}")
    
    def display_image(self, img, label):
        """显示图像"""
        if img is None:
            return
        
        # 转换为RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小以适应label
        h, w = img_rgb.shape[:2]
        label_w, label_h = label.width(), label.height()
        
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        # 转换为QImage
        height, width, channel = img_resized.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 显示
        label.setPixmap(QPixmap.fromImage(q_img))
    
    def detect_image(self):
        """检测图像"""
        if self.current_image is None:
            self.result_text.append("请先选择图像!")
            return
        
        self.result_text.append("开始检测...")
        
        # TODO: 实际检测逻辑
        # 这里仅做演示
        result_img = self.current_image.copy()
        
        # 画一个示例框
        h, w = result_img.shape[:2]
        cv2.rectangle(result_img, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        cv2.putText(result_img, "Defect: crack (0.95)", (w//4, h//4-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        self.display_image(result_img, self.result_label)
        
        self.result_text.append("检测完成!")
        self.result_text.append("检测到 1 个缺陷:")
        self.result_text.append("  - 类型: crack, 置信度: 0.95")


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("基于YOLOv8的少样本工业零件缺陷检测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("少样本工业零件缺陷检测系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # 选项卡
        self.tabs = QTabWidget()
        
        # 训练页面
        self.train_widget = TrainWidget()
        self.tabs.addTab(self.train_widget, "模型训练")
        
        # 检测页面
        self.detect_widget = DetectWidget()
        self.tabs.addTab(self.detect_widget, "缺陷检测")
        
        main_layout.addWidget(self.tabs)
        
        central_widget.setLayout(main_layout)
        
        # 状态栏
        self.statusBar().showMessage("就绪")


def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
