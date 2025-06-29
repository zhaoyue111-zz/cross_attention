import os
import cv2
import glob
import numpy as np
from PIL import Image


def find_black_center(mask_gray):
    """
    计算掩膜中黑色区域(像素值=0)的质心。
    """
    mask_bin = (mask_gray == 0).astype(np.uint8)
    M = cv2.moments(mask_bin, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def crop_with_black_padding(img, cx, cy, crop_size=800, fill_color=(0, 0, 0)):
    """
    以(cx, cy)为中心裁剪图像，超出部分填充指定颜色。
    fill_color: 超出边界的填充颜色。
    """
    H, W = img.shape[:2]
    half = crop_size // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + crop_size, y1 + crop_size

    # 有效裁剪区域
    valid_x1 = max(0, x1)
    valid_y1 = max(0, y1)
    valid_x2 = min(W, x2)
    valid_y2 = min(H, y2)

    # 画布上的偏移
    canvas_x1 = max(0, -x1)
    canvas_y1 = max(0, -y1)

    # 创建填充颜色的画布
    if len(img.shape) == 3 and img.shape[2] == 3:
        canvas = np.full((crop_size, crop_size, 3), fill_color, dtype=img.dtype)
    else:
        canvas = np.full((crop_size, crop_size), fill_color[0], dtype=img.dtype)

    if valid_x1 >= valid_x2 or valid_y1 >= valid_y2:
        return canvas

    # 裁剪有效区域
    roi = img[valid_y1:valid_y2, valid_x1:valid_x2]
    h_roi, w_roi = roi.shape[:2]
    canvas[canvas_y1:canvas_y1 + h_roi, canvas_x1:canvas_x1 + w_roi] = roi

    return canvas


def get_mask_files(mask_dir):
    """
    获取 mask 目录下的所有 bmp 或 png 文件。
    """
    mask_paths = glob.glob(os.path.join(mask_dir, "*.[Bb][Mm][Pp]")) or \
                  glob.glob(os.path.join(mask_dir, "*.[Pp][Nn][Gg]"))
    
    if not mask_paths:
        print(f"[警告] 在 {mask_dir} 找不到任何 .bmp 或 .png 掩膜文件！")
    
    return mask_paths


def get_image_file(image_dir, base_name):
    """
    获取 image 目录下的图像文件，优先匹配 jpg，然后是 png，忽略大小写。
    """
    patterns = [
        os.path.join(image_dir, f"{base_name}.[Jj][Pp][Gg]"),
        os.path.join(image_dir, f"{base_name}.[Jj][Pp][Ee][Gg]"),
        os.path.join(image_dir, f"{base_name}.[Pp][Nn][Gg]")
    ]

    for pattern in patterns:
        image_files = glob.glob(pattern)
        if image_files:
            print(f"[匹配成功] 找到图像文件: {image_files[0]}")
            return image_files[0]

    print(f"[警告] 在 {image_dir} 中找不到 {base_name} 的有效图像文件！")
    return None


def safe_read_image(file_path, is_mask=False):
    """
    安全读取图像文件，尝试使用 OpenCV 和 Pillow。
    """
    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return None

    # 尝试使用 OpenCV 读取
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if image is not None:
        print(f"[成功] OpenCV 成功读取文件: {file_path}")
        return image

    # 如果 OpenCV 失败，尝试使用 Pillow
    try:
        with Image.open(file_path) as img:
            img = img.convert('L') if is_mask else img.convert('RGB')
            image = np.array(img)
            print(f"[成功] 使用 Pillow 成功读取文件: {file_path}")
            return image
    except Exception as e:
        print(f"[错误] 无法读取文件 {file_path}: {e}")
        return None


def process_directory_structure(root_dir, save_root_dir, crop_size=800):
    """
    遍历 domain1-5 目录结构，批量裁剪图像和掩膜并保持目录结构。
    """
    for domain in range(1, 6):  # domain1-5
        for phase in ['test', 'train']:
            image_dir = os.path.join(root_dir, f"Domain{domain}", phase, "image")
            mask_dir = os.path.join(root_dir, f"Domain{domain}", phase, "mask")
            save_image_dir = os.path.join(save_root_dir, f"Domain{domain}", phase, "image")
            save_mask_dir = os.path.join(save_root_dir, f"Domain{domain}", phase, "mask")

            os.makedirs(save_image_dir, exist_ok=True)
            os.makedirs(save_mask_dir, exist_ok=True)

            mask_paths = get_mask_files(mask_dir)
            if not mask_paths:
                continue

            for mask_path in mask_paths:
                base_name = os.path.splitext(os.path.basename(mask_path))[0]
                image_path = get_image_file(image_dir, base_name)
                if not image_path:
                    continue

                mask_gray = safe_read_image(mask_path, is_mask=True)
                image = safe_read_image(image_path, is_mask=False)

                if mask_gray is None or image is None:
                    print(f"[跳过] 无法读取文件: {mask_path} 或 {image_path}")
                    continue

                center = find_black_center(mask_gray)
                if center is None:
                    print(f"[跳过] 掩膜 {mask_path} 中不存在黑色像素(=0)")
                    continue
                cx, cy = center

                image_cropped = crop_with_black_padding(image, cx, cy, crop_size, fill_color=(0, 0, 0))
                mask_cropped = crop_with_black_padding(mask_gray, cx, cy, crop_size, fill_color=(255,))

                cv2.imwrite(os.path.join(save_image_dir, f"{base_name}.jpg"), image_cropped)
                cv2.imwrite(os.path.join(save_mask_dir, f"{base_name}.png"), mask_cropped)

                print(f"[完成] {base_name}: → {save_image_dir}, {save_mask_dir}")


if __name__ == "__main__":
    root_dir = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Fundus/Origin"       # 源数据根目录
    save_root_dir = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Fundus/Fundus600"     # 目标根目录
    crop_size = 600  

    process_directory_structure(root_dir, save_root_dir, crop_size)



# move mask image files based on mask file names
# # import os
# import shutil
# import glob


# def move_images_based_on_masks(image_path, mask_paths, target_paths):
#     """
#     根据 mask 的文件名，在 image 文件夹中查找对应的 jpg 文件，
#     并将其移动到对应的目标路径。

#     :param image_path: 存放 jpg 图像的路径
#     :param mask_paths: 掩膜路径列表 [mask_path_1, mask_path_2]
#     :param target_paths: 目标路径列表 [target_path_1, target_path_2]
#     """
#     # 确保 mask_paths 和 target_paths 数量一致
#     if len(mask_paths) != len(target_paths):
#         raise ValueError("mask_paths 和 target_paths 的数量必须一致！")

#     for mask_path, target_path in zip(mask_paths, target_paths):
#         # 确保目标路径存在
#         if not os.path.exists(target_path):
#             os.makedirs(target_path)

#         # 获取当前掩膜路径下所有 .bmp 文件
#         mask_files = glob.glob(os.path.join(mask_path, "*.bmp"))

#         if not mask_files:
#             print(f"[警告] 在 {mask_path} 找不到任何 .bmp 掩膜文件！")
#             continue

#         # 遍历当前掩膜路径下的所有文件
#         for mask_file in mask_files:
#             base_name = os.path.splitext(os.path.basename(mask_file))[0]  # 获取文件名（无扩展名）

#             # 在图像路径中寻找同名 jpg 文件
#             image_file = os.path.join(image_path, f"{base_name}.jpg")

#             if os.path.exists(image_file):
#                 # 移动文件到对应的目标路径
#                 target_file = os.path.join(target_path, f"{base_name}.jpg")
#                 shutil.move(image_file, target_file)
#                 print(f"[已移动] {image_file} → {target_file}")
#             else:
#                 print(f"[未找到] {image_file} 对应的 jpg 文件不存在，跳过。")


# if __name__ == "__main__":
#     # 📁 文件路径（请根据实际路径修改）
#     image_path = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Fundus/Origin/Domain5/Test400"           # 存放 jpg 图像的文件夹
#     mask_path_1 = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Fundus/Origin/Domain5/train/mask"          # 第一个存放 bmp 掩膜的路径
#     mask_path_2 = "/path/to/mask2"          # 第二个存放 bmp 掩膜的路径
#     target_path_1 = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Fundus/Origin/Domain5/test/image"      # 第一个目标路径
#     target_path_2 = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/Fundus/Origin/Domain5/train/image"      # 第二个目标路径

#     # 执行批量移动
#     move_images_based_on_masks(
#         image_path=image_path,
#         mask_paths=[mask_path_1, mask_path_2],
#         target_paths=[target_path_1, target_path_2]
#     )

# refuge2 test400
# import os
# import cv2
# import glob
# import numpy as np

# def find_black_center(mask_gray):
#     """
#     计算掩膜中黑色区域(像素值=0)的质心。
#     """
#     mask_bin = (mask_gray == 0).astype(np.uint8)
#     M = cv2.moments(mask_bin, binaryImage=True)
#     if M["m00"] == 0:
#         return None
#     cx = int(M["m10"] / M["m00"])
#     cy = int(M["m01"] / M["m00"])
#     return (cx, cy)


# def crop_with_black_padding(img, cx, cy, crop_size=800):
#     """
#     以(cx, cy)为中心裁剪图像，超出部分用黑色填充。
#     """
#     H, W = img.shape[:2]
#     half = crop_size // 2
#     x1, y1 = cx - half, cy - half
#     x2, y2 = x1 + crop_size, y1 + crop_size

#     # 有效裁剪区域
#     valid_x1 = max(0, x1)
#     valid_y1 = max(0, y1)
#     valid_x2 = min(W, x2)
#     valid_y2 = min(H, y2)

#     # 画布上的偏移
#     canvas_x1 = max(0, -x1)
#     canvas_y1 = max(0, -y1)

#     # 创建黑色画布
#     if len(img.shape) == 3 and img.shape[2] == 3:
#         canvas = np.zeros((crop_size, crop_size, 3), dtype=img.dtype)
#     else:
#         canvas = np.zeros((crop_size, crop_size), dtype=img.dtype)

#     if valid_x1 >= valid_x2 or valid_y1 >= valid_y2:
#         return canvas

#     # 裁剪有效区域
#     roi = img[valid_y1:valid_y2, valid_x1:valid_x2]
#     h_roi, w_roi = roi.shape[:2]
#     canvas[canvas_y1:canvas_y1 + h_roi, canvas_x1:canvas_x1 + w_roi] = roi

#     return canvas


# def batch_crop_images_and_masks(
#         image_dir, 
#         mask_dir, 
#         save_image_dir, 
#         save_mask_dir, 
#         crop_size=800
#     ):
#     """
#     批量处理 BMP 格式图像和掩膜，裁剪并保存为 JPG 和 PNG 格式。
#     """
#     if not os.path.exists(save_image_dir):
#         os.makedirs(save_image_dir)
#     if not os.path.exists(save_mask_dir):
#         os.makedirs(save_mask_dir)

#     # 遍历所有 mask 文件（.bmp）
#     mask_paths = glob.glob(os.path.join(mask_dir, "*.bmp"))
#     if not mask_paths:
#         print(f"在 {mask_dir} 找不到任何 .bmp 掩膜文件！")
#         return

#     for mask_path in mask_paths:
#         base_name = os.path.splitext(os.path.basename(mask_path))[0]
#         image_path = os.path.join(image_dir, base_name + ".jpg")

#         if not os.path.exists(image_path):
#             print(f"[跳过] 没有找到对应的原图: {image_path}")
#             continue

#         # 读取掩膜和原图
#         mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)

#         if mask_gray is None:
#             print(f"[跳过] 掩膜读取失败: {mask_path}")
#             continue
#         if image is None:
#             print(f"[跳过] 原图读取失败: {image_path}")
#             continue

#         # 找到黑色区域质心
#         center = find_black_center(mask_gray)
#         if center is None:
#             print(f"[跳过] 掩膜 {mask_path} 中不存在黑色像素(=0)")
#             continue
#         cx, cy = center

#         # 裁剪
#         image_cropped = crop_with_black_padding(image, cx, cy, crop_size)
#         mask_cropped = crop_with_black_padding(mask_gray, cx, cy, crop_size)

#         # 保存为 jpg 和 png
#         save_image_path = os.path.join(save_image_dir, f"{base_name}.jpg")
#         save_mask_path = os.path.join(save_mask_dir, f"{base_name}.png")
#         cv2.imwrite(save_image_path, image_cropped)
#         cv2.imwrite(save_mask_path, mask_cropped)

#         print(f"[完成] {base_name}: 中心({cx},{cy}) → {save_image_path}, {save_mask_path}")


# if __name__ == "__main__":
#     # 📁 文件路径（请根据实际路径修改）
#     image_dir = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OCD/05/Test400"              # 原图 BMP 文件夹
#     mask_dir = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OCD/05/N"               # 掩膜 BMP 文件夹
#     save_image_dir = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OCD/05/ROIs/n"  # 裁剪后原图 JPG 文件夹
#     save_mask_dir = "/IMBR_Data/Student-home/2023M_ShiGuangze/code/SwinTV2_UNet/SwinTV2UNet/datasets/data/OCD/05/ROIs/n_mask"    # 裁剪后掩膜 PNG 文件夹
#     crop_size = 800                   # 裁剪尺寸

#     batch_crop_images_and_masks(
#         image_dir,
#         mask_dir,
#         save_image_dir,
#         save_mask_dir,
#         crop_size=crop_size
#     )
