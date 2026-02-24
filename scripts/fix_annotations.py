"""
修复现有 YOLO 标注文件中的坐标越界问题
将所有坐标裁剪到 [0, 1] 范围内

用法:
  python scripts/fix_annotations.py
  python scripts/fix_annotations.py --data_dir data/processed
"""

import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fix_annotation_file(txt_path: Path) -> int:
    """
    修复单个标注文件
    
    Returns:
        修复的坐标数量
    """
    fixed_count = 0
    lines = []
    
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = parts[0]
                    coords = [float(x) for x in parts[1:]]
                    
                    # 检查是否需要修复
                    needs_fix = any(c < 0.0 or c > 1.0 for c in coords)
                    
                    if needs_fix:
                        fixed_count += 1
                        # 裁剪到 [0, 1]
                        coords = [max(0.0, min(1.0, c)) for c in coords]
                        logger.debug(f"  修复 {txt_path.name}: {line.strip()} -> {class_id} {coords}")
                    
                    # 重新格式化
                    line = f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n"
                
                lines.append(line)
        
        # 写回文件
        if fixed_count > 0:
            with open(txt_path, 'w') as f:
                f.writelines(lines)
    
    except Exception as e:
        logger.error(f"处理文件 {txt_path} 时出错: {e}")
        return 0
    
    return fixed_count


def main():
    parser = argparse.ArgumentParser(description="修复 YOLO 标注文件中的越界坐标")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="预处理后的数据目录")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细信息")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        logger.info(f"请确保已运行数据预处理: python scripts/prepare_mvtec.py")
        return
    
    # 遍历所有标注文件
    labels_dir = data_dir / "labels"
    if not labels_dir.exists():
        logger.error(f"标注目录不存在: {labels_dir}")
        return
    
    total_files = 0
    total_fixed_coords = 0
    files_with_issues = 0
    
    logger.info("开始修复标注文件...")
    
    for split in ["train", "val", "test"]:
        split_dir = labels_dir / split
        if not split_dir.exists():
            logger.warning(f"跳过不存在的目录: {split}")
            continue
        
        logger.info(f"\n处理 {split} 集...")
        
        txt_files = list(split_dir.glob("*.txt"))
        if not txt_files:
            logger.warning(f"  {split} 集没有标注文件")
            continue
        
        for txt_file in txt_files:
            fixed = fix_annotation_file(txt_file)
            if fixed > 0:
                logger.info(f"  ✓ {txt_file.name}: 修复 {fixed} 个越界坐标")
                total_fixed_coords += fixed
                files_with_issues += 1
            total_files += 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"修复完成!")
    logger.info(f"  总文件数: {total_files}")
    logger.info(f"  有问题的文件数: {files_with_issues}")
    logger.info(f"  修复的坐标数: {total_fixed_coords}")
    logger.info(f"{'='*60}")
    
    if total_fixed_coords > 0:
        logger.info("\n现在可以重新运行训练:")
        logger.info("  python scripts/train_meta.py --config configs/train_config.yaml")
    else:
        logger.info("\n没有发现越界坐标，数据集正常")


if __name__ == "__main__":
    main()
