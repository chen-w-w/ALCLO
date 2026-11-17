# ==================== 数据加载模块 ====================
"""
CRMark 数据集加载器。

功能：
- 从指定目录递归加载图像文件
- 支持多种图像格式（PNG、JPG、TIFF等）
- 自动调整图像尺寸并转换为张量
- 生成随机水印比特
"""

import os
import glob
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class HideImage(Dataset):
    """
    图像水印数据集类。

    功能：
    - 加载封面图像（载体图像）
    - 生成随机水印比特
    - 统一图像尺寸和格式

    参数:
        root_path: 图像文件夹路径
        im_size: 图像尺寸（正方形）
        bit_length: 水印比特长度
        channel_dim: 通道数（1=灰度，3=RGB）
    """

    def __init__(self, root_path, im_size, bit_length, channel_dim):
        self.root_path = root_path
        self.im_size = im_size
        self.bit_length = bit_length
        self.channel_dim = channel_dim
        self.cover = None  # 图像路径列表

        # 图像预处理：转为张量
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 加载图像路径
        self.load_images()

    def load_images(self):
        """
        递归加载所有支持的图像文件。

        支持的格式：PNG、TIFF、BMP、JPG、JPEG、GIF、WebP
        """
        file_modes = ('*.png', '*.tif', '*.tiff', '*.bmp', '*.jpg', '*.jpeg', '*.gif', '*.webp')
        img_paths = []

        # 递归搜索所有子目录
        for mode in file_modes:
            pattern = os.path.join(self.root_path, '**', mode)
            for img_path in glob.glob(pattern, recursive=True):
                img_paths.append(img_path)

        self.cover = img_paths

    def __len__(self):
        """返回数据集大小。"""
        return len(self.cover)

    def __getitem__(self, idx):
        """
        获取单个样本。

        返回:
            cover_tensor: 封面图像张量 [C, H, W]，值范围 [0, 1]
            secret_tensor: 随机水印比特 [bit_length]，值范围 [0, 1]
        """
        # 加载图像
        cover_path = self.cover[idx]

        # 根据通道数选择模式
        if self.channel_dim == 3:
            cover = Image.open(cover_path).convert('RGB')  # 彩色图
        else:
            cover = Image.open(cover_path).convert('L')    # 灰度图

        # 调整图像尺寸（保持比例裁剪）
        cover = ImageOps.fit(cover, (self.im_size, self.im_size))

        # 转为张量
        cover_tensor = self.transform(cover)

        # 生成随机水印比特
        secret_tensor = torch.randint(0, 2, size=(self.bit_length,)).float()

        return cover_tensor, secret_tensor
