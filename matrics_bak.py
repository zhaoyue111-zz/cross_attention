import os
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import adapted_rand_error, variation_of_information
from skimage.transform import resize
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm  # 导入 tqdm 以显示进度条

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

def mean_iou(y_true, y_pred):
    intersection = np.sum((y_true * y_pred), axis=(1, 2))
    union = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2)) - intersection
    return np.mean(intersection / (union + 1e-6))

def hausdorff_distance(y_true, y_pred):
    # 将二值图像转换为坐标点集
    y_true_coords = np.argwhere(y_true > 0)
    y_pred_coords = np.argwhere(y_pred > 0)

    # 进行 Hausdorff 距离计算
    if len(y_true_coords) == 0 or len(y_pred_coords) == 0:
        return float('inf')  # 如果没有点，返回无穷大
    return max(directed_hausdorff(y_true_coords, y_pred_coords)[0],
               directed_hausdorff(y_pred_coords, y_true_coords)[0])

def process_images(gt_folder, pred_folder, output_excel):
    results = []

    gt_filenames = [f for f in os.listdir(gt_folder) if f.endswith('.png')]  # 获取 GT 图像文件

    # 在处理 GT 图像时添加进度条
    for gt_filename in tqdm(gt_filenames, desc="Processing GT Images", ncols=100):
        gt_path = os.path.join(gt_folder, gt_filename)

        # 读取 GT 图像
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        gt_image = (gt_image > 0).astype(np.uint8)  # 二值化 GT 图像

        base_name = gt_filename[:-4]  # 去掉 .png 后缀
        pred_filename_prefix = base_name + "_"  # 匹配方式
        pred_filenames = [f for f in os.listdir(pred_folder) if f.startswith(pred_filename_prefix)]

        # 在处理预测图像时添加进度条
        for pred_filename in tqdm(pred_filenames, desc=f"Processing: {base_name}", leave=False):
            pred_path = os.path.join(pred_folder, pred_filename)

            # 读取预测图像
            pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
            pred_image = (pred_image > 0).astype(np.uint8)  # 二值化预测图像

            # 计算各项指标
            dice = dice_coefficient(gt_image, pred_image)
            iou = mean_iou(gt_image[None, :, :], pred_image[None, :, :])  # 扩展维度以匹配
            hausdorff = hausdorff_distance(gt_image, pred_image)

            domain_info = pred_filename.split('_')[2]  # 提取域信息

            # 添加结果到列表
            results.append([gt_filename, domain_info, dice, iou, hausdorff])

    # 保存到 Excel 文件
    results_df = pd.DataFrame(results, columns=['GT Filename', 'Domain', 'Dice Coefficient', 'Mean IoU', 'Hausdorff Distance'])
    results_df.to_excel(output_excel, index=False)

if __name__ == "__main__":
    gt_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OD/masks/01'  # 替换为你的GT图像的文件夹路径
    pred_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_tiny16/output'  # 替换为你的预测图像的文件夹路径
    output_excel = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_tiny16/results.xlsx'  # 替换为目标Excel文件路径

    process_images(gt_folder, pred_folder, output_excel)
