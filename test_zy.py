import glob
import os
import torch
import cv2
import numpy as np
from tqdm.auto import tqdm
import warnings
from utils import losses
from models import unet_swin
from datasets.mydataset import prepare_test_dataloader
from config import get_args
import albumentations as A
import matplotlib.pyplot as plt

def make_dirs(save_dir):
    """创建保存结果的目录"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# 提取某个域的风格（可以继续升级为提取多个域的混合风格）
def extract_domain_style(model, domain_path, device, batch_size=4):
    """提取某个域的整体风格特征"""
    model.eval()
    print(f"\nExtracting style from domain: {os.path.basename(domain_path)}")
    
    image_paths = glob.glob(os.path.join(domain_path, 'JPEGImages', '*.jpg'))
    image_paths.sort()
    print(f"Found {len(image_paths)} images in domain")
    
    transforms = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 初始化累积特征
    sum_features = None
    count = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_imgs = []
            
            for img_path in batch_paths:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                transformed = transforms(image=img)
                img = transformed['image']
                img = torch.from_numpy(img).float()
                img = img.permute(2, 0, 1)
                batch_imgs.append(img)
            
            batch_tensor = torch.stack(batch_imgs).to(device)
            
            # 获取kv特征 (现在是单个tensor而不是列表)
            kv_feature = model.encoder.extract_domain_kv(batch_tensor)
            
            # 累积特征
            if sum_features is None:
                sum_features = kv_feature
            else:
                sum_features = sum_features + kv_feature
            count += 1
            
            # 释放内存
            del batch_tensor, kv_feature
            torch.cuda.empty_cache()
    
    # 计算平均值
    avg_features = sum_features / count
    
    return avg_features

# 保持图片原来的风格
def test_epoch_self(model, test_loader, device, save_dir):
    """测试一个epoch"""
    model.eval()
    model.encoder.use_checkpoint = False
    
    # 尝试多个阈值
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    test_pbar = tqdm(test_loader,
                     desc='Testing',
                     total=len(test_loader),
                     ncols=100,
                     leave=True,
                     position=0)

    with torch.no_grad():
        for batch_idx, (batch_img, img_names) in enumerate(test_pbar):
            # 将图像移到设备上并处理维度
            batch_img = batch_img.to(device).float()
            
            # 对batch中的每张图片单独处理
            for i in range(batch_img.shape[0]):
                img = batch_img[i:i+1]  # [1, 3, 512, 512]
                img_name = img_names[i]

                # 创建组合输入 (与训练时一致)
                img2 = img.clone()
                combined = torch.cat([img, img2], dim=1)
                
                # 添加预测值的详细统计
                pred = model(combined)
                
                pred = torch.sigmoid(pred)
                
                # 对每个阈值都生成一个预测掩码
                for threshold in thresholds:
                    pred_mask = (pred > threshold).float()
                    pred_mask_np = pred_mask.cpu().numpy()[0, 0]
                    
                    original_img = img.cpu().numpy()[0]
                    original_img = np.transpose(original_img, (1, 2, 0))
                    
                    # 保存带阈值信息的结果
                    threshold_img_name = f"{img_name}_thresh{threshold:.1f}"
                    save_results(
                        original_image=original_img,
                        pred_mask=pred_mask_np,
                        pred_raw=pred.cpu().numpy()[0, 0],
                        save_dir=save_dir,
                        img_name=threshold_img_name
                    )

            test_pbar.set_postfix_str(f'Processing batch {batch_idx}')

    test_pbar.close()

# 风格替换测试
def test_epoch_style(model, test_loader, device, save_dir, args):
    """风格替换测试"""
    model.eval()
    model.encoder.use_checkpoint = False
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # 提取各个域的风格
    domain_styles = {}
    for domain in ['02', '03', '04']:
        try:
            domain_path = os.path.join(args.data_dir, domain)
            style = extract_domain_style(model, domain_path, device)  # 现在返回单个tensor而不是列表
            domain_styles[domain] = style
            print(f"Successfully extracted style for domain {domain}")
        except Exception as e:
            print(f"Error extracting style for domain {domain}: {str(e)}")
            continue
    
    test_pbar = tqdm(test_loader, desc='Testing')
    
    with torch.no_grad():
        for batch_idx, (batch_img, img_names) in enumerate(test_pbar):
            batch_img = batch_img.to(device)
            
            for i in range(batch_img.shape[0]):
                img = batch_img[i:i + 1]
                img_name = img_names[i]
                
                content_features = model.encoder.get_unet_feature(img)
                
                for domain, style in domain_styles.items():
                    try:
                        # 直接使用单层风格特征
                        decoder_output = model.decoder(content_features, style)
                        pred = model.segmentation_head(decoder_output)
                        
                        for threshold in thresholds:
                            pred = torch.sigmoid(pred)
                            pred_mask = (pred > threshold).float()
                            
                            print("pred_mask shape:", pred_mask.shape)
                            pred_mask = pred_mask.cpu().numpy()[0, 0]
                            original_img = img.cpu().numpy()[0]
                            original_img = np.transpose(original_img, (1, 2, 0))
                            
                            threshold_img_name = f"domain{domain}_thresh{threshold:.1f}_{img_name}"
                            save_results(
                                original_image=original_img,
                                pred_mask=pred_mask,
                                pred_raw=pred.cpu().numpy()[0, 0],
                                save_dir=save_dir,
                                img_name=threshold_img_name
                            )
                    except Exception as e:
                        print(f"Error processing domain {domain} for image {img_name}: {str(e)}")
                        continue
                        
            test_pbar.set_postfix_str(f'Processing batch {batch_idx}')
    
    test_pbar.close()

def save_results(original_image, pred_mask, pred_raw, save_dir, img_name):
    """保存测试结果"""
    # 保存统计信息
    stats_path = os.path.join(save_dir, 'prediction_stats.txt')
    with open(stats_path, 'a') as f:
        f.write(f"\n=== {img_name} ===\n")
        f.write(f"Raw prediction range: [{pred_raw.min():.3f}, {pred_raw.max():.3f}]\n")
        f.write(f"Raw prediction mean: {pred_raw.mean():.3f}\n")
        f.write(f"Raw prediction std: {pred_raw.std():.3f}\n")
        f.write(f"Positive pixels: {(pred_mask > 0).sum()}\n")
        f.write(f"Positive ratio: {(pred_mask > 0).sum() / pred_mask.size:.3f}\n")

    print(f"Saved results for save_dir:", save_dir)
    pred_mask = (pred_mask * 255).astype(np.uint8)  # 转换为0-255范围
    print(f"Pred mask shape:", pred_mask.shape)
    cv2.imwrite(os.path.join(save_dir, f'{img_name}_mask.png'), pred_mask)


    pred_raw_normalized = ((pred_raw - pred_raw.min()) / (pred_raw.max() - pred_raw.min()) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f'{img_name}_raw.png'), pred_raw_normalized)


def test(args):
    # 设置设备
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    # TODO 修改保存位置
    save_dir = os.path.join(args.save_result)
    # save_dir = os.path.join(args.result_dir, 'test_results')
    make_dirs(save_dir)
    
    # 加载模型
    model = unet_swin(size=args.size, img_size=512, config=args).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 准备数据
    test_loader = prepare_test_dataloader(cfg=args)
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # 开始测试
    print("\nStarting test...")
    # test_epoch_self(model, test_loader, device, save_dir)
    test_epoch_style(model, test_loader, device, save_dir, args)
    print(f"\nTest finished! Results saved to {save_dir}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    args = get_args()
    test(args)

