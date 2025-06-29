import cv2
import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import Image
import albumentations as A


class PairDataset(Dataset):
    def __init__(self, cfg=None, transforms=None, mode='train', is_train_split=True):
        """
        Args:
            mode: 'train' 或 'test'
            is_train_split: True表示使用80%训练集，False表示使用20%验证集
        """
        self.mode = mode
        self.cfg = cfg
        self.transforms = transforms
        self.is_train_split = is_train_split

        random.seed(42)

        if self.mode == 'train':
            # 获取每个域的图像路径并按数字顺序排序
            self.domain_paths = {}
            self.train_indices = {}
            self.val_indices = {}

            for domain in ['02', '03', '04']:
                # 获取图像路径
                img_paths = glob.glob(os.path.join(cfg.data_dir, f'{domain}/JPEGImages', '*.jpg'))
                # 获取对应的标签路径
                label_paths = glob.glob(os.path.join(cfg.data_dir, f'{domain}/SegmentationClass', '*.png'))
                # 确保图像和标签一一对应
                img_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                label_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                
                self.domain_paths[domain] = {'images': img_paths, 'labels': label_paths}
                
                total_images = len(img_paths)
                indices = [int(os.path.basename(p).split('_')[1].split('.')[0]) for p in img_paths]
                train_size = int(0.8 * total_images)
                
                random.seed(42)
                random.shuffle(indices)
                
                self.train_indices[domain] = indices[:train_size]
                self.val_indices[domain] = indices[train_size:]

            # 生成图像对
            self.pairs = []
            for main_domain in ['02', '03', '04']:
                main_indices = self.train_indices[main_domain] if is_train_split else self.val_indices[main_domain]
                for main_idx in main_indices:
                    other_domains = [d for d in ['02', '03', '04'] if d != main_domain]
                    for other_domain in other_domains:
                        self.pairs.append((main_domain, main_idx, other_domain))
                        
        elif self.mode == 'test':
            self.image_paths = glob.glob(os.path.join(cfg.data_dir, '01/JPEGImages', '*.jpg'))
            self.image_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

        if self.mode == 'train':
            assert all(len(paths) > 0 for paths in self.domain_paths.values()), "训练域文件夹为空"
            assert len(self.pairs) > 0, "没有生成图像对"
        else:
            assert len(self.image_paths) > 0, "测试文件夹为空"

    def __len__(self):
        if self.mode == 'train':
            return len(self.pairs)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.mode == 'train':
            main_domain, main_idx, other_domain = self.pairs[idx]
            
            # 构建文件名
            img1_name = f'OD{main_domain}_{main_idx:03d}.jpg'
            img1_path = os.path.join(self.cfg.data_dir, main_domain, 'JPEGImages', img1_name)
            label1_name = f'OD{main_domain}_{main_idx:03d}.png'
            label1_path = os.path.join(self.cfg.data_dir, main_domain, 'SegmentationClass', label1_name)
            
            available_indices = self.train_indices[other_domain] if self.is_train_split else self.val_indices[other_domain]
            random_idx = random.choice(available_indices)
            img2_name = f'OD{other_domain}_{random_idx:03d}.jpg'
            img2_path = os.path.join(self.cfg.data_dir, other_domain, 'JPEGImages', img2_name)

            # 读取图像和标签  图像 3通道   label 1通道
            img1 = cv2.imread(img1_path)
            label1 = cv2.imread(label1_path,cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path)
            # 确保标签值是二值的

            # 应用transforms
            if self.transforms:
                transformed = self.transforms(image=img1, mask=label1)
                img1 = transformed['image']
                label1 = transformed['mask']
                transformed = self.transforms(image=img2)
                img2 = transformed['image']

            # 转换为tensor并归一化
            img1 = torch.from_numpy(img1).float() / 255.0
            img2 = torch.from_numpy(img2).float() / 255.0
            label1 = torch.from_numpy(label1).float()

            return img1, img2, label1
        else:
            img_path = self.image_paths[idx]
            img = cv2.imread(img_path)
            
            if self.transforms:
                transformed = self.transforms(image=img)
                img = transformed['image']
            
            img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
            return img


def get_transforms(train=True, cfg=None):
    if train:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
        ])
    else:
        return A.Compose([])  # 验证集不需要数据增强


def prepare_train_loaders(cfg, debug=False):
    # 创建训练集
    train_dataset = PairDataset(
        cfg=cfg,
        transforms=get_transforms(train=True, cfg=cfg),
        mode='train',
        is_train_split=True
    )

    # 创建验证集
    valid_dataset = PairDataset(
        cfg=cfg,
        transforms=get_transforms(train=False, cfg=cfg),
        mode='train',
        is_train_split=False
    )

    if debug:
        train_size = min(len(train_dataset), 20)
        valid_size = min(len(valid_dataset), 20)
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
        valid_dataset = torch.utils.data.Subset(valid_dataset, range(valid_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_bs if not cfg.debug else 20,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_bs if not cfg.debug else 20,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, valid_loader

def prepare_test_dataloader(cfg):
    """准备测试数据加载器"""
    test_dataset = TestDataset(cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    return test_loader


class TestDataset(Dataset):
    def __init__(self, cfg):
        self.image_paths = glob.glob(os.path.join(cfg.data_dir, '01/JPEGImages', '*.jpg'))
        self.image_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        # 使用与训练时一致的预处理
        self.transforms = A.Compose([
            A.Resize(512, 512),
            # 添加与训练时相同的归一化
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path).split('.')[0]
        
        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        # 转换为tensor并归一化 (与训练时一致)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        
        return image, img_name
