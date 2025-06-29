import os
import glob
import numpy as np
import torch.utils.data as Data

import torchvision.transforms as transforms
from PIL import Image

# Define a transformation to convert images to tensors
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel (grayscale)
    transforms.ToTensor(),
])

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''



class Dataset(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, idx):
        # # 索引数据集中的某个数据，还可以对数据进行预处理
        # # 下标index参数是必须有的，名字任意
        # img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        # index = self.files[index][59:61]
        # # 返回值自动转换为torch的tensor类型
        # return img_arr, index

        img_path = self.files[idx]
        img = Image.open(img_path)
        img = transform(img)  # Apply transformations
        return img.unsqueeze(0)
