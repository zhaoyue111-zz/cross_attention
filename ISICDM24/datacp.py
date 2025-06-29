import os
import shutil

def classify_and_copy_images_by_name(source_dir, dir1, dir2, dir3):
    """
    遍历 source_dir 中的图像文件，根据名称中是否包含关键词 "pasm" 或 "masson" 进行分类复制。
    
    参数:
    - source_dir (str): 源文件夹路径。
    - dir1 (str): 如果文件名中含有 "pasm"，复制到的目录路径。
    - dir2 (str): 如果文件名中含有 "masson"，复制到的目录路径。
    - dir3 (str): 如果文件名中不含 "pasm" 和 "masson"，复制到的目录路径。
    """
    # 确保目标文件夹存在
    os.makedirs(dir1, exist_ok=True)
    os.makedirs(dir2, exist_ok=True)
    os.makedirs(dir3, exist_ok=True)

    # 遍历源文件夹中的文件
    for filename in os.listdir(source_dir):
        # 构造文件完整路径
        file_path = os.path.join(source_dir, filename)
        # 检查是否为文件
        if os.path.isfile(file_path):
            if "pasm" in filename.lower():
                shutil.copy(file_path, os.path.join(dir1, filename))
            elif "masson" in filename.lower():
                shutil.copy(file_path, os.path.join(dir2, filename))
            else:
                shutil.copy(file_path, os.path.join(dir3, filename))

# 使用示例
# # 源文件夹路径
# source_directory = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent_ori/train/labelbi"
# # 分类存储的目标文件夹
# directory1 = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent/train/03/masks"    # pasm
# directory2 = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent/train/01/masks"    # masson
# directory3 = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent/train/02/masks"    # pas

# 源文件夹路径
source_directory = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent_ori/val/labelbi"
# 分类存储的目标文件夹
directory1 = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent/val/03/masks"    # pasm
directory2 = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent/val/01/masks"    # masson
directory3 = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Crescent/val/02/masks"    # pas

# 调用函数
classify_and_copy_images_by_name(source_directory, directory1, directory2, directory3)
