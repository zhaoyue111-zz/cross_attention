import os
import numpy as np
from PIL import Image, UnidentifiedImageError

def convert_rgb_labels_to_binary_in_folders(source_dir):
    """
    将文件夹中包含 'masks_oc' 或 'masks_od' 的目录内的所有 RGB 标签图像转换为灰度图，并将像素值从 0,255 转换为 0,1。

    参数:
    - source_dir (str): 源文件夹路径，包含若干子文件夹和标签图像。
    """
    for root, dirs, files in os.walk(source_dir):
        # 只处理文件夹名称包含 'masks_oc' 或 'masks_od' 的目录
        # if "masks_oc" in os.path.basename(root).lower() or "masks_od" in os.path.basename(root).lower():
        # if "masks" in os.path.basename(root).lower():
        for file in files:
            # 检查文件是否为图片
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)

                try:
                    # 打开图片
                    with Image.open(file_path) as img:
                        # 转换为灰度图
                        grayscale_img = img.convert("L")  # 转为灰度图
                        label_array = np.array(grayscale_img)

                        # 将像素值从 255 转换为 1
                        binary_label_array = (label_array > 128).astype(np.uint8)

                        # 保存为新的图片，覆盖原始文件
                        binary_label_img = Image.fromarray(binary_label_array)  # 保存为 0,255 格式
                        binary_label_img.save(file_path)
                        print(f"Converted: {file_path}")

                except UnidentifiedImageError:
                    print(f"Unidentified image format: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# 示例使用
source_directory = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/unet/dataset/Crescent/labelsVal"
convert_rgb_labels_to_binary_in_folders(source_directory)
