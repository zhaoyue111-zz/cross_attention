from PIL import Image
import os
import numpy as np

def process_masks(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # 根据文件后缀可自行调整
            # 读取图像
            mask_path = os.path.join(input_folder, filename)
            mask = Image.open(mask_path).convert('L')  # 转换为灰度图
            
            # 将像素值乘以 255
            mask_array = np.array(mask) * 255
            mask_array = mask_array.astype(np.uint8)  # 转换为 8 位图像

            # 保存结果图像
            output_image = Image.fromarray(mask_array)
            output_path = os.path.join(output_folder, filename)
            output_image.save(output_path)
            print(f"Processed and saved: {output_path}")

# 输入和输出文件夹路径
input_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OD/masks/04'  # 替换为你的输入文件夹
output_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OD/masks/04'  # 替换为你的输出文件夹

process_masks(input_folder, output_folder)
