import os
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import adapted_rand_error, variation_of_information
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)


def mean_iou(y_true, y_pred):
    intersection = np.sum((y_true * y_pred), axis=(1, 2))
    union = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2)) - intersection
    return np.mean(intersection / (union + 1e-6))


def hausdorff_distance(y_true, y_pred):
    y_true_coords = np.argwhere(y_true > 0)
    y_pred_coords = np.argwhere(y_pred > 0)
    if len(y_true_coords) == 0 or len(y_pred_coords) == 0:
        return float('inf')
    return max(directed_hausdorff(y_true_coords, y_pred_coords)[0],
               directed_hausdorff(y_pred_coords, y_true_coords)[0])


def get_domain_from_filename(filename):
    """根据文件名确定源域，OC01作为目标域"""
    parts = filename.split('_')
    if len(parts) >= 3:
        identifier = parts[2][0]  # 获取第二个下划线后的第一个字符

        # 域映射规则
        domain_mapping = {
            'G': 'OC02',
            'N': 'OC02',
            'S': 'OC02',
            'g': 'OC03',
            'n': 'OC03',
            'V': 'OC04',
            'T': 'OC05'
        }

        return domain_mapping.get(identifier, 'Unknown')
    return 'Unknown'


def get_domain_from_filename_n1(filename):
    """根据文件名确定源域，非OC01作为目标域"""
    parts = filename.split('_')
    if len(parts) >= 3:
        return 'OC01'

    identifier = parts[1][0]  # 获取第一个下划线后的第一个字符

    # 域映射规则
    domain_mapping = {
        'G': 'OC02',
        'N': 'OC02',
        'S': 'OC02',
        'g': 'OC03',
        'n': 'OC03',
        'V': 'OC04',
        'T': 'OC05'
    }

    return domain_mapping.get(identifier, 'Unknown')


def process_images(gt_folder, pred_folder, output_excel):
    results = []
    gt_filenames = [f for f in os.listdir(gt_folder) if f.endswith('.png')]

    for gt_filename in tqdm(gt_filenames, desc="Processing GT Images"):
        gt_path = os.path.join(gt_folder, gt_filename)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_image = ((gt_image > 100) & (gt_image < 200)).astype(np.uint8)  # OC
        # gt_image = (gt_image > 200).astype(np.uint8)  # OD

        base_name = gt_filename[:-4]
        pred_filename_prefix = base_name + "_"
        pred_filenames = [f for f in os.listdir(pred_folder) if f.startswith(pred_filename_prefix)]

        for pred_filename in tqdm(pred_filenames, desc=f"Processing: {base_name}", leave=False):
            pred_path = os.path.join(pred_folder, pred_filename)
            pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            pred_image = ((pred_image > 100) & (pred_image < 200)).astype(np.uint8)  # OC
            # pred_image = (pred_image < 100).astype(np.uint8)  # OD

            dice = dice_coefficient(gt_image, pred_image)
            iou = mean_iou(gt_image[None, :, :], pred_image[None, :, :])
            hausdorff = hausdorff_distance(gt_image, pred_image)

            # 设置01或非01的映射函数
            domain_info = get_domain_from_filename_n1(pred_filename)
            results.append([gt_filename, domain_info, dice, iou, hausdorff])

    results_df = pd.DataFrame(results,
                              columns=['GT Filename', 'Domain', 'Dice Coefficient', 'Mean IoU', 'Hausdorff Distance'])
    results_df.to_excel(output_excel, index=False)


def process_excel_data(input_file, output_file):
    data = pd.read_excel(input_file)

    # 计算每个域的平均值
    average_results = data.groupby('Domain').agg({
        'Dice Coefficient': 'mean',
        'Mean IoU': 'mean',
        'Hausdorff Distance': 'mean'
    }).reset_index()

    # 找出每个GT图像的最佳结果
    max_values = data.loc[data.groupby('GT Filename')['Dice Coefficient'].idxmax()]

    # 计算最佳结果的平均值
    max_averages = max_values[['Dice Coefficient', 'Mean IoU', 'Hausdorff Distance']].mean()
    max_averages['Domain'] = 'Max'
    max_averages_df = pd.DataFrame([max_averages])

    # 设置目标域
    average_results['TargetDomain'] = 'OC05'
    max_averages_df['TargetDomain'] = 'OC05'

    # 合并结果
    merged_results = pd.concat([average_results, max_averages_df], ignore_index=True)
    merged_results = merged_results[['TargetDomain', 'Domain', 'Dice Coefficient', 'Mean IoU', 'Hausdorff Distance']]

    # 提取域编号
    merged_results['DomainNum'] = merged_results['Domain'].str.extract(r'OC(\d+)')

    # 对相同域编号的结果进行分组并计算平均值
    merged_results = merged_results.groupby('DomainNum', as_index=False).agg({
        'TargetDomain': 'first',  # 保持OC
        'Domain': lambda x: f"OC{x.iloc[0][-2:]}",  # 保持原始域名格式
        'Dice Coefficient': 'mean',
        'Mean IoU': 'mean',
        'Hausdorff Distance': 'mean'
    })

    # 添加Max行
    max_row = merged_results[merged_results['Domain'] == 'Max'].copy()
    merged_results = pd.concat([merged_results, max_row], ignore_index=True)

    # 删除DomainNum列
    merged_results = merged_results.drop('DomainNum', axis=1)

    # 保存最终汇总结果
    merged_results.to_excel(output_file, index=False)


def main(gt_folder, pred_folder, output_excel, input_excel_file, output_excel_file):
    process_images(gt_folder, pred_folder, output_excel)
    process_excel_data(input_excel_file, output_excel_file)


if __name__ == "__main__":
    gt_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet_bak/datasets/Fundus512/05/test/mask'
    pred_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet_bak/Fundus256/result_1e-3_9e-1_small16/05/output'
    output_excel = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet_bak/Fundus256/result_1e-3_9e-1_small16/05/results_OC.xlsx'
    input_excel_file = output_excel  # Assuming you want to process the same file
    output_excel_file = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet_bak/Fundus256/result_1e-3_9e-1_small16/05/mean_OC.xlsx'
    main(gt_folder, pred_folder, output_excel, input_excel_file, output_excel_file)
