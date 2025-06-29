import os
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import adapted_rand_error, variation_of_information
from skimage.transform import resize
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm  # 导入 tqdm 以显示进度条


def multi_class_dice(y_true, y_pred, class_value):
    """计算特定类别的Dice系数"""
    # 将特定类别的像素设为1，其他为0
    y_true_class = (y_true == class_value).astype(np.uint8)
    y_pred_class = (y_pred == class_value).astype(np.uint8)

    intersection = np.sum(y_true_class * y_pred_class)
    return (2. * intersection) / (np.sum(y_true_class) + np.sum(y_pred_class) + 1e-6)


def multi_class_iou(y_true, y_pred, class_value):
    """计算特定类别的IoU"""
    y_true_class = (y_true == class_value).astype(np.uint8)
    y_pred_class = (y_pred == class_value).astype(np.uint8)

    intersection = np.sum(y_true_class * y_pred_class)
    union = np.sum(y_true_class) + np.sum(y_pred_class) - intersection
    return intersection / (union + 1e-6)


def hausdorff_distance(y_true, y_pred, class_value):
    """计算特定类别的Hausdorff距离"""
    y_true_class = (y_true == class_value).astype(np.uint8)
    y_pred_class = (y_pred == class_value).astype(np.uint8)

    y_true_coords = np.argwhere(y_true_class > 0)
    y_pred_coords = np.argwhere(y_pred_class > 0)

    if len(y_true_coords) == 0 or len(y_pred_coords) == 0:
        return float('inf')
    return max(directed_hausdorff(y_true_coords, y_pred_coords)[0],
               directed_hausdorff(y_pred_coords, y_true_coords)[0])


def process_images(gt_folder, pred_folder, output_excel):
    results = []
    # 定义三个类别的颜色值
    classes = {
        'black': 0,
        'gray': 128,
        'white': 255
    }

    gt_filenames = [f for f in os.listdir(gt_folder) if f.endswith('.png')]

    for gt_filename in tqdm(gt_filenames, desc="处理图像中", ncols=100):
        gt_path = os.path.join(gt_folder, gt_filename)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        base_name = gt_filename[:-4]
        pred_filename_prefix = base_name + "_"
        pred_filenames = [f for f in os.listdir(pred_folder) if f.startswith(pred_filename_prefix)]

        if not pred_filenames:  # 如果没有找到对应的预测文件，跳过当前GT文件
            print(f"警告: 未找到与 {gt_filename} 对应的预测文件")
            continue

        for pred_filename in tqdm(pred_filenames, desc=f"处理: {base_name}", leave=False):
            try:
                pred_path = os.path.join(pred_folder, pred_filename)
                pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

                if pred_image is None:
                    print(f"警告: 无法读取预测图像 {pred_path}")
                    continue

                # 确保图像大小一致
                if gt_image.shape != pred_image.shape:
                    pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))

                # 计算每个类别的指标
                metrics = {}
                for class_name, class_value in classes.items():
                    dice = multi_class_dice(gt_image, pred_image, class_value)
                    iou = multi_class_iou(gt_image, pred_image, class_value)
                    hd = hausdorff_distance(gt_image, pred_image, class_value)

                    metrics[f'{class_name}_dice'] = dice
                    metrics[f'{class_name}_iou'] = iou
                    metrics[f'{class_name}_hd'] = hd

                # 计算平均指标
                avg_dice = np.mean([metrics[f'{c}_dice'] for c in classes.keys()])
                avg_iou = np.mean([metrics[f'{c}_iou'] for c in classes.keys()])
                avg_hd = np.mean([metrics[f'{c}_hd'] for c in classes.keys() if metrics[f'{c}_hd'] != float('inf')])

                # 提取域信息
                try:
                    domain_info = pred_filename.split('_')[2]
                except IndexError:
                    domain_info = "unknown"

                # 添加结果
                result_row = {
                    'GT_Filename': gt_filename,
                    'Pred_Filename': pred_filename,
                    'Domain': domain_info,
                    'Average_Dice': avg_dice,
                    'Average_IoU': avg_iou,
                    'Average_Hausdorff': avg_hd
                }
                # 添加每个类别的指标
                result_row.update(metrics)
                results.append(result_row)

            except Exception as e:
                print(f"处理文件 {pred_filename} 时出错: {str(e)}")
                continue

    if not results:
        print("警告: 没有成功处理任何图像对")
        return

    # 创建DataFrame并保存
    results_df = pd.DataFrame(results)

    # 确保输出路径包含.xlsx扩展名
    if not output_excel.endswith('.xlsx'):
        output_excel = output_excel + '/results.xlsx'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)

    # 保存结果
    results_df.to_excel(output_excel, index=False)

    # 打印汇总统计
    print("\n=== 评估指标汇总 ===")
    print(f"处理的图像对数量: {len(results_df)}")
    print(f"平均 Dice: {results_df['Average_Dice'].mean():.4f}")
    print(f"平均 IoU: {results_df['Average_IoU'].mean():.4f}")
    print(f"平均 Hausdorff距离: {results_df['Average_Hausdorff'].mean():.4f}")

    # 打印每个类别的平均指标
    for class_name in classes.keys():
        print(f"\n{class_name.capitalize()} 类别指标:")
        print(f"Dice: {results_df[f'{class_name}_dice'].mean():.4f}")
        print(f"IoU: {results_df[f'{class_name}_iou'].mean():.4f}")
        print(f"Hausdorff: {results_df[f'{class_name}_hd'].mean():.4f}")


if __name__ == "__main__":
    gt_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet_bak/datasets/Fundus512/05/test/mask'
    pred_folder = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet_bak/Fundus256/result_1e-3_9e-1_small16/05/output'
    output_excel = '/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet_bak/Fundus256/result_1e-3_9e-1_small16/05/results.xlsx'

    process_images(gt_folder, pred_folder, output_excel)
