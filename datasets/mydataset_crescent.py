import cv2
import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import albumentations as A

class PairDataset(Dataset):
    def __init__(self, cfg=None, transforms=None, mode='train', is_train_split=True, train_domains=None, test_domains=None):
        """
        Args:
            cfg: 配置对象
            transforms: 数据增强
            mode: 'train' 或 'test'
            is_train_split: True表示使用80%训练集，False表示使用20%验证集
            train_domains: 自定义的训练域列表
            test_domains: 自定义的测试域列表
        """
        self.mode = mode
        self.cfg = cfg
        self.transforms = transforms
        self.is_train_split = is_train_split
        self.train_domains = train_domains if train_domains else ['02', '03', '04']  # 默认值
        self.test_domains = test_domains if test_domains else ['01']  # 默认值

        random.seed(42)

        if self.mode == 'train':
            self.domain_paths = {}
            self.train_indices = {}
            self.val_indices = {}

            for domain in self.train_domains:
                train_img_paths = glob.glob(os.path.join(self.cfg.data_dir, domain, 'train/images', '*.jpg'))    # 训练集图片路径
                train_mask_paths = glob.glob(os.path.join(self.cfg.data_dir, domain, 'train/masks', '*.png'))
                # print("train_img_paths", train_img_paths)
                # print(type)
                train_img_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x)))
                train_mask_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x)))

                val_img_paths = glob.glob(os.path.join(self.cfg.data_dir, domain, 'test/images', '*.jpg'))    # 验证集图片路径
                val_mask_paths = glob.glob(os.path.join(self.cfg.data_dir, domain, 'test/masks', '*.png'))
                val_img_paths.sort(key=lambda x: os.path.splitext(x))
                val_mask_paths.sort(key=lambda x: os.path.splitext(x))

                self.domain_paths[domain] = {'images': train_img_paths, 'labels': train_mask_paths}

                train_images = len(train_img_paths)
                val_images = len(val_img_paths)
                train_indices = [os.path.splitext(p)[0] for p in train_img_paths]
                val_indices = [os.path.splitext(p)[0] for p in val_img_paths]
                train_size = int(train_images)
                val_size = int(val_images)

                random.seed(42)
                random.shuffle(val_indices)

                self.train_indices[domain] = train_indices[:train_size]
                self.val_indices[domain] = val_indices[:val_size]

            self.pairs = []
            for main_domain in self.train_domains:
                main_indices = self.train_indices[main_domain] if is_train_split else self.val_indices[main_domain]
                for main_idx in main_indices:
                    # other_domains = [d for d in self.train_domains if d != main_domain]   # 除去主域
                    other_domains = [d for d in self.train_domains]         # 所有域都参与训练
                    for other_domain in other_domains:
                        self.pairs.append((main_domain, main_idx, other_domain))

        elif self.mode == 'test':
            self.domain_paths = {}
            self.image_paths = []

            for test_domain in self.test_domains:
                img_paths = glob.glob(os.path.join(self.cfg.data_dir, test_domain, 'test/images', '*.jpg'))    # 测试集图片路径
                img_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
                self.domain_paths[test_domain] = {'images': img_paths}
                if test_domain == self.test_domains[0]:
                    self.image_paths = img_paths

            self.pairs = []
            for other_domain in self.train_domains:
                print(f"Loading images for domain {self.train_domains}")
                img_paths = glob.glob(os.path.join(self.cfg.data_dir, other_domain, 'train/images', '*.jpg'))    # 训练集图片路径
                img_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

                if img_paths:
                    for img1_path in self.image_paths:
                        self.domain_paths[other_domain] = {'images': img_paths}
                        self.pairs.append((self.test_domains[0], img1_path, other_domain))
                else:
                    print(f"No images found for domain {other_domain}")

            assert len(self.image_paths) > 0, "测试文件夹为空"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.mode == 'train':
            if self.is_train_split:
                main_domain, main_idx, other_domain = self.pairs[idx]
                img1_name = os.path.join(os.path.basename(main_idx) + '.jpg')     # 图片名称
                # print("img1_name--------------------------", img1_name)
                img1_path = os.path.join(self.cfg.data_dir, main_domain, 'train/images', img1_name)
                label1_name = os.path.join(os.path.basename(main_idx) + '.png')
                label1_path = os.path.join(self.cfg.data_dir, main_domain, 'train/masks', label1_name)
                # available_indices = self.train_indices[other_domain] if self.is_train_split else self.val_indices[other_domain]
                available_indices = self.train_indices[other_domain]
                random_idx = random.choice(available_indices)
                img2_name = os.path.join(os.path.basename(random_idx) + '.jpg')   # 图片名称
                img2_path = os.path.join(self.cfg.data_dir, other_domain, 'train/images', img2_name)
            else:
                main_domain, main_idx, other_domain = self.pairs[idx]
                img1_name = os.path.join(os.path.basename(main_idx) + '.jpg')     # 图片名称
                img1_path = os.path.join(self.cfg.data_dir, main_domain, 'test/images', img1_name)
                label1_name = os.path.join(os.path.basename(main_idx) + '.png')
                label1_path = os.path.join(self.cfg.data_dir, main_domain, 'test/masks', label1_name)
                available_indices= self.val_indices[other_domain]
                random_idx = random.choice(available_indices)
                img2_name = os.path.join(os.path.basename(random_idx) + '.jpg')   # 图片名称
                img2_path = os.path.join(self.cfg.data_dir, other_domain, 'test/images', img2_name)

            img1 = cv2.imread(img1_path)
            label1 = cv2.imread(label1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path)

            if self.transforms:
                transformed = self.transforms(image=img1, mask=label1)
                img1 = transformed['image']
                label1 = transformed['mask']
                transformed = self.transforms(image=img2)
                img2 = transformed['image']

            img1 = torch.from_numpy(img1).float() / 255.0
            img2 = torch.from_numpy(img2).float() / 255.0
            label1 = torch.from_numpy(label1).float()

            return img1, img2, label1
        else:
            main_domain, img1_path, other_domain = self.pairs[idx]
            img1 = cv2.imread(img1_path)

            img2_paths = self.domain_paths[other_domain]['images']
            random_idx = random.choice(range(len(img2_paths)))
            img2_path = img2_paths[random_idx]

            img2 = cv2.imread(img2_path)
            if self.transforms:
                transformed = self.transforms(image=img2)
                img2 = transformed['image']

            img1 = torch.from_numpy(img1).float() / 255.0
            img2 = torch.from_numpy(img2).float() / 255.0

            return img1, img2, img1_path, img2_path



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

def prepare_train_loaders(cfg, train_domains=None, test_domains=None, debug=False):
    # 创建训练集
    train_dataset = PairDataset(
        cfg=cfg,
        transforms=get_transforms(train=True, cfg=cfg),
        mode='train',
        is_train_split=True,
        train_domains=train_domains,  # 新增参数
        test_domains=test_domains
    )

    # 创建验证集
    valid_dataset = PairDataset(
        cfg=cfg,
        transforms=get_transforms(train=False, cfg=cfg),
        mode='train',
        is_train_split=False,
        train_domains=train_domains,  # 新增参数
        test_domains=test_domains
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

def prepare_test_loaders(cfg, train_domains=None, test_domains=None, debug=False):
    # 创建测试集
    test_dataset = PairDataset(
        cfg=cfg,
        transforms=get_transforms(train=False, cfg=cfg),  # 测试模式不需要增强
        mode='test',  # 设置 mode 为 'test'
        train_domains=train_domains,
        test_domains=test_domains  # 新增参数
    )

    if debug:
        test_size = min(len(test_dataset), 20)
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_bs if not cfg.debug else 20,  # 可根据需要设置测试批大小
        num_workers=cfg.num_workers,
        shuffle=False,  # 测试时不打乱顺序
        pin_memory=True
    )

    return test_loader
