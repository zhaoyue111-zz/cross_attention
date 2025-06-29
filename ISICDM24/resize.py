import os
from PIL import Image

def resize_images_in_folders(source_dir, target_dir, size):
    """
    将文件夹中的所有图片调整为指定大小，仅对文件夹名称中包含 'images' 的目录操作，
    并保持原有的子文件夹结构和命名。

    参数:
    - source_dir (str): 源文件夹路径，包含若干子文件夹和图片。
    - target_dir (str): 调整后的图片保存路径。
    - size (tuple): 目标大小，例如 (128, 128)。
    """
    # 遍历源文件夹
    for root, dirs, files in os.walk(source_dir):
        # 判断当前文件夹是否包含 "images" 关键字
        # if "masks_oc" in os.path.basename(root).lower() or "masks_od" in os.path.basename(root).lower():
        if 'masks' in os.path.basename(root).lower():
            # 计算目标路径
            relative_path = os.path.relpath(root, source_dir)
            target_subdir = os.path.join(target_dir, relative_path)
            os.makedirs(target_subdir, exist_ok=True)  # 确保子目录存在

            for file in files:
                # 检查文件是否为图片
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    source_file_path = os.path.join(root, file)
                    target_file_path = os.path.join(target_subdir, file)

                    try:
                        # 打开图片并调整大小
                        with Image.open(source_file_path) as img:
                            img_resized = img.resize(size, Image.LANCZOS)  # 使用 LANCZOS 替代 ANTIALIAS
                            img_resized.save(target_file_path)

                        print(f"Resized: {source_file_path} -> {target_file_path}")

                    except Exception as e:
                        print(f"处理文件 {source_file_path} 时出错: {e}")

# 使用示例
source_directory = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent/val"  # 源文件夹路径
target_directory = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent/masks"  # 目标文件夹路径
resize_size = (512, 512)  # 调整后的图片大小

resize_images_in_folders(source_directory, target_directory, resize_size)
