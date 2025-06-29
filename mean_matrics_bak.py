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

def process_images(gt_folder, pred_folder, output_excel):
    results = []
    gt_filenames = [f for f in os.listdir(gt_folder) if f.endswith('.png')]

    for gt_filename in tqdm(gt_filenames, desc="Processing GT Images"):
        gt_path = os.path.join(gt_folder, gt_filename)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_image = (gt_image > 0).astype(np.uint8)

        base_name = gt_filename[:-4]
        pred_filename_prefix = base_name + "_"
        pred_filenames = [f for f in os.listdir(pred_folder) if f.startswith(pred_filename_prefix)]

        for pred_filename in tqdm(pred_filenames, desc=f"Processing: {base_name}", leave=False):
            pred_path = os.path.join(pred_folder, pred_filename)
            pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            pred_image = (pred_image > 0).astype(np.uint8)

            dice = dice_coefficient(gt_image, pred_image)
            iou = mean_iou(gt_image[None, :, :], pred_image[None, :, :])
            hausdorff = hausdorff_distance(gt_image, pred_image)

            domain_info = pred_filename.split('_')[2]

            results.append([gt_filename, domain_info, dice, iou, hausdorff])

    results_df = pd.DataFrame(results, columns=['GT Filename', 'Domain', 'Dice Coefficient', 'Mean IoU', 'Hausdorff Distance'])
    results_df.to_excel(output_excel, index=False)

def process_excel_data(input_file, output_file):
    data = pd.read_excel(input_file)

    average_results = data.groupby('Domain').agg({
        'Dice Coefficient': 'mean',
        'Mean IoU': 'mean',
        'Hausdorff Distance': 'mean'
    }).reset_index()

    max_values = data.loc[data.groupby('GT Filename')['Dice Coefficient'].idxmax()]

    max_averages = max_values[['Dice Coefficient', 'Mean IoU', 'Hausdorff Distance']].mean()
    max_averages['Domain'] = 'Max'
    max_averages_df = pd.DataFrame([max_averages])

    target_domain = max_values['GT Filename'].str[:4].unique()[0]
    average_results['TargetDomain'] = target_domain
    max_averages_df['TargetDomain'] = target_domain

    sheet2_content = pd.concat([average_results, max_averages_df], ignore_index=True)
    sheet2_content = sheet2_content[['TargetDomain', 'Domain', 'Dice Coefficient', 'Mean IoU', 'Hausdorff Distance']]

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        data.to_excel(writer, sheet_name='Sheet1', index=False)
        sheet2_content.to_excel(writer, sheet_name='Sheet2', index=False)

def main(gt_folder, pred_folder, output_excel, input_excel_file, output_excel_file):
    process_images(gt_folder, pred_folder, output_excel)
    process_excel_data(input_excel_file, output_excel_file)

if __name__ == "__main__":
    # 指定路径
    # gt_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OD/masks/04'
    # pred_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_small16/04/output'
    # output_excel = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_small16/04/results.xlsx'
    # input_excel_file = output_excel  # Assuming you want to process the same file
    # output_excel_file = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_small16/04/mean.xlsx'

    # gt_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OD/masks/01'
    # pred_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_base16/01/output'
    # output_excel = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_base16/01/results.xlsx'
    # input_excel_file = output_excel  # Assuming you want to process the same file
    # output_excel_file = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_base16/01/mean.xlsx'

    gt_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OD/masks/01'
    pred_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_small16/01/output'
    output_excel = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_small16/01/results.xlsx'
    input_excel_file = output_excel  # Assuming you want to process the same file
    output_excel_file = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/results_1e-3_9e-1_small16/01/mean.xlsx'


    main(gt_folder, pred_folder, output_excel, input_excel_file, output_excel_file)
