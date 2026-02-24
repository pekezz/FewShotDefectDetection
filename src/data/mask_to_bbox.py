"""
MVTec AD数据集掩码转边界框工具
将PNG格式的像素级二值掩码转换为YOLO格式的边界框标注
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaskToBBoxConverter:
    """将二值掩码转换为YOLO格式边界框"""
    
    def __init__(self, min_area: int = 50):
        """
        Args:
            min_area: 最小缺陷区域面积(像素数)，小于此值的区域将被忽略
        """
        self.min_area = min_area
    
    def mask_to_bboxes(self, mask_path: str, image_width: int, image_height: int,
                       class_id: int = 0) -> List[Tuple[int, float, float, float, float]]:
        """
        将掩码图像转换为YOLO格式的边界框
        
        Args:
            mask_path: 掩码图像路径(.png)
            image_width: 原图宽度
            image_height: 原图高度
            class_id: 类别ID(默认为0，表示缺陷)
            
        Returns:
            边界框列表，每个元素为 (class_id, x_center, y_center, width, height)
            所有坐标值都已归一化到[0, 1]
        """
        # 读取掩码图像(灰度)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            logger.warning(f"无法读取掩码: {mask_path}")
            return []
        
        # 二值化(阈值设为127)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 查找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        bboxes = []
        
        # 遍历每个连通区域(跳过背景，label=0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 过滤小面积区域
            if area < self.min_area:
                continue
            
            # 获取边界框坐标
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 转换为YOLO格式(归一化的中心坐标和宽高)
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            norm_width = w / image_width
            norm_height = h / image_height
            
            # 裁剪到 [0, 1] 范围，避免浮点误差导致的越界
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_width = max(0.0, min(1.0, norm_width))
            norm_height = max(0.0, min(1.0, norm_height))
            
            bboxes.append((class_id, x_center, y_center, norm_width, norm_height))
        
        return bboxes
    
    def convert_dataset(self, data_root: Path, output_dir: Path, 
                       category_map: Dict[str, int] = None):
        """
        批量转换MVTec AD数据集
        
        Args:
            data_root: MVTec AD数据集根目录
            output_dir: 输出目录
            category_map: 类别名到ID的映射，如 {'crack': 0, 'scratch': 1}
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # MVTec AD数据集结构: data_root/object_class/defect_type/ground_truth/*.png
        for obj_class_dir in data_root.iterdir():
            if not obj_class_dir.is_dir():
                continue
            
            logger.info(f"处理类别: {obj_class_dir.name}")
            
            # 遍历缺陷类型(test目录下)
            test_dir = obj_class_dir / "test"
            if not test_dir.exists():
                continue
            
            for defect_type_dir in test_dir.iterdir():
                if not defect_type_dir.is_dir() or defect_type_dir.name == "good":
                    continue
                
                # 获取类别ID
                if category_map:
                    class_id = category_map.get(defect_type_dir.name, 0)
                else:
                    class_id = 0  # 默认所有缺陷为类别0
                
                logger.info(f"  缺陷类型: {defect_type_dir.name} (class_id={class_id})")
                
                # ground_truth目录包含掩码
                gt_dir = obj_class_dir / "ground_truth" / defect_type_dir.name
                if not gt_dir.exists():
                    continue
                
                # 对应的原图目录
                image_dir = defect_type_dir
                
                # 处理每个掩码文件
                for mask_file in gt_dir.glob("*.png"):
                    # 对应的原图文件（去掉 _mask 后缀）
                    # 掩码文件名: 000_mask.png -> stem: 000_mask
                    # 原图文件名: 000.png
                    image_name = mask_file.stem.replace("_mask", "") + ".png"
                    image_file = image_dir / image_name
                    
                    if not image_file.exists():
                        logger.warning(f"找不到对应原图: {image_file}")
                        continue
                    
                    # 读取原图获取尺寸
                    img = cv2.imread(str(image_file))
                    if img is None:
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # 转换掩码为边界框
                    bboxes = self.mask_to_bboxes(str(mask_file), w, h, class_id)
                    
                    if not bboxes:
                        logger.warning(f"未检测到有效区域: {mask_file}")
                        continue
                    
                    # 保存YOLO格式标注
                    # 标注文件名应该和原图文件名对应（去掉 _mask）
                    base_name = mask_file.stem.replace("_mask", "")
                    output_txt = output_dir / f"{obj_class_dir.name}_{defect_type_dir.name}_{base_name}.txt"
                    
                    with open(output_txt, 'w') as f:
                        for bbox in bboxes:
                            # YOLO格式: class_id x_center y_center width height
                            f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
                    
                    logger.info(f"    保存标注: {output_txt.name} ({len(bboxes)} 个边界框)")
    
    def visualize_conversion(self, mask_path: str, image_path: str, output_path: str):
        """
        可视化掩码到边界框的转换结果
        
        Args:
            mask_path: 掩码图像路径
            image_path: 原图路径
            output_path: 输出可视化结果路径
        """
        # 读取图像
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            logger.error("无法读取图像或掩码")
            return
        
        h, w = img.shape[:2]
        
        # 获取边界框
        bboxes = self.mask_to_bboxes(mask_path, w, h)
        
        # 在图像上绘制边界框
        for bbox in bboxes:
            class_id, x_center, y_center, box_w, box_h = bbox
            
            # 反归一化
            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h
            
            # 计算左上角和右下角坐标
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)
            
            # 绘制矩形
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Class {class_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 创建可视化结果(原图+掩码叠加+边界框)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[mask > 127] = [0, 0, 255]  # 红色显示缺陷区域
        
        # 叠加掩码到原图
        overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
        
        # 保存结果
        cv2.imwrite(output_path, overlay)
        logger.info(f"可视化结果已保存: {output_path}")


if __name__ == "__main__":
    # 示例用法
    converter = MaskToBBoxConverter(min_area=50)
    
    # 单个文件转换示例
    # converter.visualize_conversion(
    #     mask_path="data/MVTec_AD/bottle/ground_truth/broken_large/000.png",
    #     image_path="data/MVTec_AD/bottle/test/broken_large/000.png",
    #     output_path="output_visualization.png"
    # )
    
    # 批量转换示例
    data_root = Path("data/MVTec_AD")
    output_dir = Path("data/annotations/train")
    
    # 定义缺陷类型到ID的映射(可选)
    category_map = {
        'broken_large': 0,
        'broken_small': 1,
        'contamination': 2,
        'crack': 3,
        'scratch': 4,
        # 添加更多缺陷类型...
    }
    
    # converter.convert_dataset(data_root, output_dir, category_map)
    
    print("掩码转边界框工具已就绪")
