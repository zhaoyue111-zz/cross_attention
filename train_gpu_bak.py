# python imports
import os
import glob
import warnings
import random
# external imports
import torch
import numpy as np
import cv2
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from utils import losses
from config import get_args
from datasets.mydataset import prepare_train_loaders
from models import unet_swin
from natsort import natsorted
import time
from tqdm.auto import tqdm
import sys
from logger import nnUNetLogger  # 导入 nnUNetLogger
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 添加在文件开头


def make_dirs(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, name, args):
    cv2.imwrite(os.path.join(args.result_dir, name), img)


def compute_dice(gt, pred):
    """
        计算多类别的Dice系数
        Args:
            gt: [B, H, W] 值为0,1,2
            pred: [B, 2, H, W] logits
        """
    pred = F.softmax(pred, dim=1)  # 先应用softmax
    pred = pred.argmax(dim=1)  # [B, H, W]

    dice_scores = []
    for class_idx in range(1, 3):  # 计算两个前景类别的Dice
        pred_class = (pred == class_idx)
        gt_class = (gt == class_idx)

        intersection = (pred_class & gt_class).sum().float()
        union = pred_class.sum() + gt_class.sum()

        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_scores.append(dice.item())

    return np.mean(dice_scores)  # 返回平均Dice


def set_seed(seed=42):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, args):
    model.train()
    model.encoder.use_checkpoint = True
    print(f'\nEpoch {epoch}/{args.n_iter}')
    print('-' * 20)

    train_losses = []
    single_losses = []
    consistency_losses = []
    memory_losses = []
    scaler = GradScaler()
    accumulation_steps = 2
    optimizer.zero_grad()

    pbar = tqdm(train_loader,
                desc=f'Training Epoch {epoch}/{args.n_iter}',
                total=len(train_loader),
                ncols=100,
                leave=True,
                position=0)

    for i, batch_data in enumerate(pbar):
        torch.cuda.empty_cache()

        img1, img2, label1, domain_idx = batch_data
        img1, img2 = img1.to(device), img2.to(device)
        label1 = label1.to(device)
        domain_idx = domain_idx.to(device)

        if torch.isnan(img1).any() or torch.isnan(img2).any():
            print("Warning: NaN values detected in input images")
            continue

        img1 = img1.permute(0, 3, 1, 2).contiguous()
        img2 = img2.permute(0, 3, 1, 2).contiguous()

        # 使用混合精度训练
        with autocast():
            combined = torch.cat([img1, img2], dim=1)
            outputs, x1_outputs, interaction_score, losses_dict = model(combined, domain_idx)

            # 检查输出是否包含nan
            if torch.isnan(outputs).any() or torch.isnan(x1_outputs).any():
                print(f"[WARNING] NaN detected in model outputs at batch {i}")
                continue

            if label1.max() >= outputs.size(1):
                print(f"Label values: {torch.unique(label1)}")
                print(f"Number of output channels: {outputs.size(1)}")
                continue

            # 1.1. 分割损失：交互分割与真实标签
            main_loss = criterion(outputs, label1) / accumulation_steps

            # 1.2. 分割损失：单独分割与真实标签
            single_loss = criterion(x1_outputs, label1) / accumulation_steps

            # 检查损失值
            if torch.isnan(main_loss) or torch.isnan(single_loss):
                print(f"[WARNING] NaN detected in segmentation loss at batch {i}")
                continue

            # 2. 一致性损失：添加epsilon避免数值不稳定
            epsilon = 1e-6
            consistency_loss = F.mse_loss(
                F.softmax(outputs, dim=1) + epsilon,
                F.softmax(x1_outputs, dim=1) + epsilon
            ) / accumulation_steps

            # 3. 对比损失：memory bank loss
            memory_loss = losses_dict['memory_loss'] / accumulation_steps

            # 检查所有损失
            if torch.isnan(consistency_loss) or torch.isnan(memory_loss):
                print(f"[WARNING] NaN detected in consistency or memory loss at batch {i}")
                continue

            # 4. 总损失计算：添加梯度裁剪
            loss = main_loss + single_loss + consistency_loss + memory_loss

            # 检查总损失
            if torch.isnan(loss):
                print(f"[WARNING] NaN detected in total loss at batch {i}")
                continue

        # 使用梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 记录各种损失
        train_losses.append(main_loss.item() * accumulation_steps)
        single_losses.append(single_loss.item() * accumulation_steps)
        consistency_losses.append(consistency_loss.item() * accumulation_steps)
        memory_losses.append(memory_loss.item() * accumulation_steps)

        # 计算当前批次的Dice系数
        with torch.no_grad():
            dice = compute_dice(label1, outputs)

        pbar.set_postfix_str(f'Loss={loss.item():.4f}, Dice={dice:.4f}')
        pbar.refresh()

    pbar.close()

    return {
        'main_loss': np.mean(train_losses),
        'single_loss': np.mean(single_losses),
        'consistency_loss': np.mean(consistency_losses),
        'memory_loss': np.mean(memory_losses)
    }


def val_epoch(model, valid_loader, device):
    model.eval()
    model.encoder.use_checkpoint = False
    DSC = []
    memory_losses = []

    val_pbar = tqdm(valid_loader, desc='Validating', total=len(valid_loader))

    with torch.no_grad():
        for batch_data in val_pbar:
            img1, img2, label1, domain_idx = batch_data
            current_domain_idx = domain_idx[0].item()

            img1 = img1.to(device).float()
            img2 = img2.to(device).float()
            label1 = label1.to(device)
            domain_idx = domain_idx.to(device)

            img1 = img1.permute(0, 3, 1, 2).contiguous()
            img2 = img2.permute(0, 3, 1, 2).contiguous()
            combined = torch.cat([img1, img2], dim=1)

            with autocast():
                # 获取当前域的存储特征
                current_bank = model.domain_banks.banks[f'domain{current_domain_idx}']
                if current_bank.memory:
                    # 将所有存储的特征堆叠并计算平均值
                    stored_features = torch.stack([feat.to(device) for feat in current_bank.memory])
                    stored_features = stored_features.mean(dim=0, keepdim=True)  # [1, 1, C, H, W]
                else:
                    stored_features = None

                print("shape of stored_features: ", stored_features.shape)
                pred, x1_outputs, interaction_score, losses_dict = model(
                    combined, 
                    domain_idx,
                    stored_kv=stored_features
                )
                
                memory_losses.append(losses_dict['memory_loss'].item())
                dice = compute_dice(label1, pred)
                DSC.append(dice)

            val_pbar.set_postfix_str(
                f'Dice={dice:.4f}, MemoryLoss={losses_dict["memory_loss"].item():.4f}'
            )

    val_pbar.close()
    return np.mean(DSC), np.std(DSC), np.mean(memory_losses)


def train(args):
    set_seed(42)
    make_dirs(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = unet_swin(size=args.size, img_size=args.img_size, config=args).to(device)

    # 初始化优化器和调度器
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=args.alpha, patience=5, verbose=True
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # 初始化训练状态
    start_epoch = 1
    best_dice = -1
    best_epoch = 0
    current_lr = args.lr

    # 实例化日志记录器
    logger = nnUNetLogger()

    history = {
        'train_loss': [],  # main loss
        'single_loss': [],  # 单独分割损失
        'consistency_loss': [],  # 一致性损失
        'memory_loss': [],  # 对比损失
        'val_dice': [],
        'val_dice_std': [],
        'learning_rates': [],
        'val_memory_losses': []
    }

    # 如果有resume路径，加载checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)

            # 恢复训练状态
            start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch继续
            best_dice = checkpoint['best_dice']
            best_epoch = checkpoint['best_epoch']
            history = checkpoint['history']

            # 加载模型权重
            model.load_state_dict(checkpoint['state_dict'])

            # 恢复优化器状态
            opt.load_state_dict(checkpoint['optimizer'])

            # 恢复学习率调度器状态
            scheduler.load_state_dict(checkpoint['scheduler'])

            print(f"=> resumed from epoch {checkpoint['epoch']}")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)

    # 如果是续训，追加模式打开日志文件
    mode = 'a' if args.resume else 'w'
    f = open(os.path.join(args.log_dir, log_name + ".txt"), mode)

    # 获取训练和测试域
    train_domains = args.train_domains.split(',') if args.train_domains else None
    test_domains = args.test_domains.split(',') if args.test_domains else None
    # 准备数据
    train_loader, valid_loader = prepare_train_loaders(cfg=args, train_domains=train_domains, test_domains=test_domains)
    print("train loader size:", len(train_loader.dataset))
    print("valid loader size:", len(valid_loader.dataset))
    print("Batch size:", args.train_bs)

    # 训练循环
    for epoch in range(start_epoch, args.n_iter + 1):
        # 训练
        start_time = time.time()
        epoch_loss = train_epoch(model, train_loader, opt, loss_fn, device, epoch, args)
        epoch_time = time.time() - start_time

        # 验证
        mean_dice, std_dice, val_memory_loss = val_epoch(model, valid_loader, device)
        history['train_loss'].append(epoch_loss['main_loss'])
        history['single_loss'].append(epoch_loss['single_loss'])
        history['consistency_loss'].append(epoch_loss['consistency_loss'])
        history['memory_loss'].append(epoch_loss['memory_loss'])
        history['val_dice'].append(mean_dice)
        history['val_dice_std'].append(std_dice)
        history['learning_rates'].append(opt.param_groups[0]['lr'])
        history['val_memory_losses'].append(val_memory_loss)

        # 记录日志信息
        logger.log('train_losses', epoch_loss['main_loss'], epoch)
        logger.log('single_losses', epoch_loss['single_loss'], epoch)
        logger.log('consistency_losses', epoch_loss['consistency_loss'], epoch)
        logger.log('memory_losses', epoch_loss['memory_loss'], epoch)
        logger.log('val_dice', mean_dice, epoch)
        logger.log('val_memory_losses', val_memory_loss, epoch)
        logger.log('learning_rates', opt.param_groups[0]['lr'], epoch)

        # 更新学习率
        scheduler.step(mean_dice)

        # 保存检查点
        if best_dice > mean_dice:  # 修正条件判断
            best_dice = mean_dice
            state = {
                'model': model.state_dict(),
                'memory_banks': model.domain_banks.banks,
                'optimizer': opt.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice
            }
            save_checkpoint(state, args.model_dir)

        # 打印epoch总结
        print(f'\nEpoch {epoch} Summary:')
        print(f'train time: {epoch_time:.2f}s')
        print(f'main loss: {epoch_loss["main_loss"]:.4f}')
        print(f'single loss: {epoch_loss["single_loss"]:.4f}')
        print(f'consistency loss: {epoch_loss["consistency_loss"]:.4f}')
        print(f'memory loss: {epoch_loss["memory_loss"]:.4f}')
        print(f'valid Dice: {mean_dice:.4f} ± {std_dice:.4f}')
        print(f'valid memory loss: {val_memory_loss:.4f}')
        print(f'learning rate: {opt.param_groups[0]["lr"]:.6f}')
        print(f'best Dice: {best_dice:.4f} (Epoch {epoch})')

        # 写入日志文件
        f.write(f"Epoch {epoch}, "
                f"MainLoss: {epoch_loss['main_loss']:.4f}, "
                f"SingleLoss: {epoch_loss['single_loss']:.4f}, "
                f"ConsistencyLoss: {epoch_loss['consistency_loss']:.4f}, "
                f"MemoryLoss: {epoch_loss['memory_loss']:.4f}, "
                f"Dice: {mean_dice:.4f} ± {std_dice:.4f}, "
                f"ValMemoryLoss: {val_memory_loss:.4f}, "
                f"LR: {opt.param_groups[0]['lr']:.6f}\n")
        f.flush()

    # 训练结束，保存最终结果
    print('\nTraining finished!')
    print(f'Best validation Dice: {best_dice:.4f} (Epoch {epoch})')

    # 保存训练历史为TXT文件
    history_file_path = os.path.join(args.log_dir, f'training_history_{log_name}.txt')
    with open(history_file_path, 'w') as f_history:
        f_history.write(
            "Epoch, Main Loss, Single Loss, Consistency Loss, Memory Loss, "
            "Validation Dice, Validation Dice Std, Validation Memory Loss, Learning Rate\n")
        for i in range(len(history['train_loss'])):
            f_history.write(f"{i + 1}, "
                            f"{history['train_loss'][i]:.4f}, "
                            f"{history['single_loss'][i]:.4f}, "
                            f"{history['consistency_loss'][i]:.4f}, "
                            f"{history['memory_loss'][i]:.4f}, "
                            f"{history['val_dice'][i]:.4f}, "
                            f"{history['val_dice_std'][i]:.4f}, "
                            f"{history['val_memory_losses'][i]:.4f}, "
                            f"{history['learning_rates'][i]:.6f}\n")

    # 绘制并保存训练过程图
    logger.plot_progress_png(args.log_dir)

    f.close()


def save_checkpoint(state, save_dir):
    """保存检查点"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 分别保存模型和memory banks
    model_path = os.path.join(save_dir, 'best_model.pth')
    memory_path = os.path.join(save_dir, 'memory_banks.pth')
    
    # 保存模型状态
    model_state = {
        'model': state['model'],
        'optimizer': state['optimizer'],
        'epoch': state['epoch'],
        'best_dice': state['best_dice']
    }
    torch.save(model_state, model_path)
    
    # 保存memory banks前将特征移到CPU并转换为列表
    memory_banks = {}
    for domain_name, bank in state['memory_banks'].items():
        memory_banks[domain_name] = {
            'memory': [feat.cpu().detach() for feat in bank.memory],
            'scores': bank.scores,
            'max_size': bank.max_size
        }
    
    memory_state = {
        'memory_banks': memory_banks
    }
    torch.save(memory_state, memory_path)


def load_checkpoint(model, checkpoint_path, device):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        return None
        
    # 加载模型状态
    model_path = os.path.join(checkpoint_path, 'best_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        
        # 加载memory banks并移到正确的设备
        memory_path = os.path.join(checkpoint_path, 'memory_banks.pth')
        if os.path.exists(memory_path):
            memory_state = torch.load(memory_path)
            for domain_name, bank_data in memory_state['memory_banks'].items():
                model.domain_banks.banks[domain_name].memory = [feat.to(device) for feat in bank_data['memory']]
                model.domain_banks.banks[domain_name].scores = bank_data['scores']
                model.domain_banks.banks[domain_name].max_size = bank_data['max_size']
            
        return checkpoint
    return None


def resize_images(data_dir, size=256):
    """
    递归调整目录下所有图片的大小
    Args:
        data_dir: 数据目录路径
        size: 目标大小
    """
    # 支持的图片格式
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

    # 获取所有图片路径
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))

    print(f"Found {len(img_paths)} images")

    # 处理每张图片
    for img_path in tqdm(img_paths, desc="Resizing images"):
        try:
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            # 检查图片大小
            h, w = img.shape[:2]
            if h == size and w == size:
                continue  # 如果已经是目标大小，跳过

            # 根据是否是掩码图像选择不同的插值方法
            dsize = (int(size), int(size))
            if 'masks' or 'mask' in img_path:
                # 对于掩码图像使用最近邻插值以保持标签值
                resized = cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)
            else:
                # 对于普通图像使用区域插值
                resized = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

            # 保存回原路径
            cv2.imwrite(img_path, resized)

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            print(f"Image path: {img_path}")
            print(f"Image shape: {img.shape if img is not None else 'None'}")

    print("Done!")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    args = get_args()
    # resize_images(args.data_dir, size=args.img_size)
    train(args)
