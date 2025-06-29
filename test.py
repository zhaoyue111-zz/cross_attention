import os
import torch
import numpy as np
import cv2
from tqdm.auto import tqdm
import warnings
import pandas as pd  # 导入pandas库
import torch.nn.functional as F

# Internal imports
from config import get_args
from models import unet_swin
from datasets.mydataset import prepare_test_loaders


def save_prediction(pred, filename, save_result):
    """处理三分类预测结果"""
    print(f"输入预测形状: {pred.shape}")

    # 确保预测结果是三通道概率图
    if len(pred.shape) == 3:
        pred = pred.transpose(2, 0, 1)  # 转换为 (C, H, W) 格式

    # 获取每个类别的最大概率
    pred_classes = np.argmax(pred, axis=0)

    # 创建输出图像
    output_image = np.zeros_like(pred_classes, dtype=np.uint8)

    # 映射类别到颜色值
    # 0: 黑色 (0)
    # 1: 灰色 (128)
    # 2: 白色 (255)
    output_image[pred_classes == 0] = 0
    output_image[pred_classes == 1] = 128
    output_image[pred_classes == 2] = 255

    print(f"最终输出形状: {output_image.shape}")

    # 保存
    os.makedirs(save_result, exist_ok=True)
    cv2.imwrite(os.path.join(save_result, filename), output_image)


def get_domain_from_path(img_path):
    """从图像路径中提取域信息
    Args:
        img_path: 图像路径，格式可能是：
            - 数字格式："01_OD_001..."
            - 数据集名称格式："gdrishtiGS_..."
    Returns:
        domain_id: 域ID (0-4)
    """
    try:
        # 从路径中提取域号
        parts = img_path.split('/')
        for part in parts:
            if part in ['01', '02', '03', '04', '05']:
                domain_id = int(part) - 1  # 将1-5映射到0-4
                print(f"[DEBUG] 从路径 {img_path} 提取到域ID: {domain_id}")
                return domain_id

    except Exception as e:
        print(f"[ERROR] 域ID提取失败: {str(e)}")
        print(f"[ERROR] 问题路径: {img_path}")
        return 0  # 默认返回域0

    print(f"[WARNING] 无法从路径提取域信息: {img_path}")
    return 0  # 默认返回域0


def test(args):
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 创建结果目录
    os.makedirs(args.save_result, exist_ok=True)

    # 加载模型
    model = unet_swin(size=args.size, img_size=256, config=args).to(device)

    # 加载最佳检查点
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model_best.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # 获取训练和测试域
    train_domains = args.train_domains.split(',') if args.train_domains else None
    test_domains = args.test_domains.split(',') if args.test_domains else None

    # 准备测试数据加载器
    test_loader = prepare_test_loaders(cfg=args, train_domains=train_domains, test_domains=test_domains)

    # 测试模式
    model.eval()

    # 测试进度条
    test_pbar = tqdm(test_loader, desc='Testing', total=len(test_loader), ncols=100)

    # 保存预测结果和指标
    metrics_list = []

    # 禁用梯度计算
    with torch.no_grad():
        for batch_idx, (img1, img2, img1_path, img2_path) in enumerate(test_pbar):
            img1 = img1.to(device).float().permute(0, 3, 1, 2).contiguous()
            img2 = img2.to(device).float().permute(0, 3, 1, 2).contiguous()

            # 获取源域和目标域ID
            source_domain = get_domain_from_path(img1_path[0])
            target_domain = get_domain_from_path(img2_path[0])

            print(f"[DEBUG] Processing: source_domain={source_domain}, target_domain={target_domain}")
            print(f"[DEBUG] Image paths: {img1_path[0]} -> {img2_path[0]}")

            # 从源域获取Q特征
            q_features = model.encoder.get_unet_feature(img1)

            # 使用目标域memory bank中已有的KV特征
            target_bank = model.domain_banks.banks[f'domain{target_domain}']
            if target_bank.memory:
                # 使用已存储的KV特征
                stored_kv = target_bank.memory[0].to(device)  # 获取存储的KV特征
                print(f"[DEBUG] 使用域{target_domain}的存储KV特征")
            else:
                # 如果目标域还没有存储的KV特征，则从当前图像提取
                kv_features = model.encoder.extract_domain_kv(img2)
                stored_kv = kv_features
                target_bank.update(kv_features, torch.tensor(1.0))
                print(f"[DEBUG] 域{target_domain}没有存储的KV特征，使用当前图像的KV特征")

            # 打印特征形状以进行调试
            print(f"[DEBUG] Q features shapes: {[f.shape for f in q_features]}")
            print(f"[DEBUG] KV features shape: {stored_kv.shape}")

            try:
                # 确保KV特征格式正确
                if isinstance(stored_kv, list):
                    decoder_output = model.decoder(q_features, stored_kv)
                else:
                    # 重新组织KV特征的维度
                    kv_features_list = []
                    # 假设stored_kv的形状是 [2, 1, 24, 64, 32]
                    stored_kv = stored_kv.squeeze(1)  # 移除冗余维度

                    # 为每个Q特征层级创建对应的KV特征
                    for idx, q_feat in enumerate(q_features):
                        target_h, target_w = q_feat.shape[2:]
                        target_c = q_feat.shape[1]

                        # 首先将stored_kv重组为标准的4D张量格式
                        current_kv = stored_kv[0]  # 只使用key特征
                        # 将特征展平并重组
                        current_kv = current_kv.reshape(-1, stored_kv.shape[-2] * stored_kv.shape[-1])
                        current_kv = current_kv.reshape(1, -1, stored_kv.shape[-2], stored_kv.shape[-1])

                        # 调整空间维度
                        resized_kv = F.interpolate(
                            current_kv,
                            size=(target_h, target_w),
                            mode='bilinear',
                            align_corners=True
                        )

                        # 调整通道数以匹配Q特征
                        if resized_kv.shape[1] != target_c:
                            # 使用1x1卷积调整通道数
                            channel_adjust = torch.nn.Conv2d(
                                resized_kv.shape[1],
                                target_c,
                                kernel_size=1
                            ).to(device)
                            resized_kv = channel_adjust(resized_kv)

                        print(f"[DEBUG] 层级 {idx}: Q形状 {q_feat.shape}, KV形状 {resized_kv.shape}")
                        kv_features_list.append(resized_kv)

                    print(f"[DEBUG] 调整后的KV特征形状: {[f.shape for f in kv_features_list]}")
                    decoder_output = model.decoder(q_features, kv_features_list)

            except Exception as e:
                print(f"[ERROR] 解码器处理失败: {str(e)}")
                print(f"[ERROR] Q特征形状: {[f.shape for f in q_features]}")
                print(f"[ERROR] KV特征形状: {stored_kv.shape}")
                print(f"[ERROR] 详细错误: {str(e)}")
                raise e

            outputs = model.segmentation_head(decoder_output)

            # 使用softmax处理输出
            outputs_softmax = F.softmax(outputs, dim=1)
            pred = outputs_softmax.cpu().numpy()
            pred = pred[0].transpose(1, 2, 0)

            # 保存预测结果
            img1_basename = os.path.basename(img1_path[0])
            img2_basename = os.path.basename(img2_path[0])
            save_filename = f"{img1_basename[:-4]}_{img2_basename[:-4]}.png"

            save_prediction(pred, save_filename, args.save_result)

            # 存储指标到列表
            metrics_list.append({
                'batch_idx': batch_idx,
                'sample_idx': img1_basename,
                'filename': save_filename,
                'source_domain': source_domain,
                'target_domain': target_domain,
                'used_stored_kv': bool(target_bank.memory)
            })

            # 更新进度条
            test_pbar.set_postfix_str(f'Processed {batch_idx + 1}/{len(test_loader)}')

    test_pbar.close()

    # 保存指标到 Excel 文件
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_excel(os.path.join(args.save_result, 'test_metrics.xlsx'), index=False)


def main():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    args = get_args()
    test(args)


if __name__ == "__main__":
    main()
