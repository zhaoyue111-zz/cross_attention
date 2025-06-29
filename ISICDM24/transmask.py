import os
from PIL import Image
import numpy as np

def convert_mask_to_grayscale(source_dir, output_dir, target_rgb=(128, 0, 0), target_gray=255, background_gray=0):
    """
    将文件夹中的RGB mask图像转换为灰度图像，其中指定的RGB值转换为灰度255，其他像素为灰度0。

    参数:
    - source_dir (str): 源文件夹路径，包含RGB mask图像。
    - output_dir (str): 转换后的灰度图像存储路径。
    - target_rgb (tuple): 需要转换为灰度255的目标RGB值，默认是 (128, 0, 0)。
    - target_gray (int): 转换为灰度图像时目标RGB对应的灰度值，默认是 255。
    - background_gray (int): 转换为灰度图像时其他RGB像素对应的灰度值，默认是 0。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历源文件夹中的文件
    for filename in os.listdir(source_dir):
        # 检查是否为图像文件
        file_path = os.path.join(source_dir, filename)
        if os.path.isfile(file_path):
            try:
                # 打开图像并转换为numpy数组
                image = Image.open(file_path).convert('RGB')
                image_np = np.array(image)

                # 创建一个空白灰度图
                grayscale_mask = np.full(image_np.shape[:2], background_gray, dtype=np.uint8)

                # 找到匹配目标RGB的像素位置
                target_mask = (image_np[:, :, 0] == target_rgb[0]) & \
                              (image_np[:, :, 1] == target_rgb[1]) & \
                              (image_np[:, :, 2] == target_rgb[2])

                # 将目标像素位置设置为目标灰度值
                grayscale_mask[target_mask] = target_gray

                # 保存灰度图像
                output_path = os.path.join(output_dir, filename)
                Image.fromarray(grayscale_mask).save(output_path)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

# 使用示例
# 源文件夹路径
source_directory = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent_ori/val/label"
# 输出文件夹路径
output_directory = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent_ori/val/labelbi"

# 调用函数
convert_mask_to_grayscale(source_directory, output_directory)
