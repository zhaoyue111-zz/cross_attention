import os
from PIL import Image, UnidentifiedImageError

def convert_png_to_jpg_in_folders(source_dir):
    """
    将文件夹中包含 'images' 的目录内的所有 .png 图片转换为 .jpg，并删除原来的 .png 文件。

    参数:
    - source_dir (str): 源文件夹路径，包含若干子文件夹和图片。
    """
    for root, dirs, files in os.walk(source_dir):
        # 只处理文件夹名称包含 'images' 的目录
        # if "images" in os.path.basename(root).lower():
        if "masks_oc" in os.path.basename(root).lower() or "masks_od" in os.path.basename(root).lower():
            for file in files:
                # 检查文件是否为 .png
                if file.lower().endswith('.jpg'):
                    source_file_path = os.path.join(root, file)
                    target_file_path = os.path.splitext(source_file_path)[0] + ".png"

                    try:
                        # 打开 .png 图片并保存为 .jpg 格式
                        with Image.open(source_file_path) as img:
                            # 检查图片模式，如果不是 RGB，转换为 RGB
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            img.save(target_file_path, format="PNG")
                            print(f"Converted: {source_file_path} -> {target_file_path}")

                        # 删除原始 .png 文件
                        os.remove(source_file_path)
                        print(f"Deleted: {source_file_path}")

                    except UnidentifiedImageError:
                        print(f"Unidentified image format: {source_file_path}")
                    except Exception as e:
                        print(f"Error converting {source_file_path}: {e}")

# 示例使用
source_directory = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OCD"
convert_png_to_jpg_in_folders(source_directory)
