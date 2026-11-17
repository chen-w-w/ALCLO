# ==================== Dataset loading ====================
"""Image dataset loader for watermark training and evaluation.

Recursively discovers supported image files, crops and resizes each image,
converts it to a tensor, and generates random watermark bits.
"""

import os
import glob
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class HideImage(Dataset):
    """Load cover images and generate random watermark bits.

    Args:
        root_path: Root directory containing images.
        im_size: Target square image size.
        bit_length: Number of watermark bits.
        channel_dim: Number of image channels: 1 for grayscale or 3 for RGB.
    """

    def __init__(self, root_path, im_size, bit_length, channel_dim):
        self.root_path = root_path
        self.im_size = im_size
        self.bit_length = bit_length
        self.channel_dim = channel_dim
        self.cover = None  # Image path list.

        # Convert images to tensors.
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Load image paths.
        self.load_images()

    def load_images(self):
        """Recursively load PNG, TIFF, BMP, JPEG, GIF, and WebP image paths."""
        file_modes = ('*.png', '*.tif', '*.tiff', '*.bmp', '*.jpg', '*.jpeg', '*.gif', '*.webp')
        img_paths = []

        # Recursively search all subdirectories.
        for mode in file_modes:
            pattern = os.path.join(self.root_path, '**', mode)
            for img_path in glob.glob(pattern, recursive=True):
                img_paths.append(img_path)

        self.cover = img_paths

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.cover)

    def __getitem__(self, idx):
        """Return a cover tensor and random watermark bits for one sample.

        Returns:
            cover_tensor: Image shaped [C, H, W], with values in [0, 1].
            secret_tensor: Binary float tensor shaped [bit_length].
        """
        # Load the cover image.
        cover_path = self.cover[idx]

        # Select the image mode from the channel count.
        if self.channel_dim == 3:
            cover = Image.open(cover_path).convert('RGB')  # RGB image.
        else:
            cover = Image.open(cover_path).convert('L')    # Grayscale image.

        # Resize with an aspect-preserving crop.
        cover = ImageOps.fit(cover, (self.im_size, self.im_size))

        # Convert to a tensor.
        cover_tensor = self.transform(cover)

        # Generate random watermark bits.
        secret_tensor = torch.randint(0, 2, size=(self.bit_length,)).float()

        return cover_tensor, secret_tensor
