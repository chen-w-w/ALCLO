"""Grayscale ALCLO evaluation with configurable Phase 2/3 checkpoints.

Default architecture from phase2_gray logs: L 512x512, 256 raw bits, k=5,
min_size=32, fc=False. Default pair: Phase 2 model_best.pth and Phase 3
score_model_53.pth (a local experiment default; pass your selected weights).
Phase 3 needs BOTH checkpoints; use --skip_phase3 to evaluate Phase 2 only.
See README.md for checkpoint paths and training/evaluation examples.
"""

import argparse
import ast
import contextlib
import glob
import hashlib
import importlib.util
import json
import os
import random
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps, features
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def absolute_path(path):
    return Path(os.path.abspath(os.path.expanduser(str(path))))


SCRIPT_DIR = absolute_path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
# DEFAULT_DATASET = absolute_path(REPO_ROOT / "../DIV2K/val")
DEFAULT_DATASET = absolute_path(REPO_ROOT / "../Kodak24")
# DEFAULT_DATASET = absolute_path(REPO_ROOT / "../AID")
# DEFAULT_DATASET = absolute_path(REPO_ROOT / "../DiffusionDB")
# DEFAULT_DATASET = absolute_path(REPO_ROOT / "../CelebA-HQ")
# DEFAULT_DATASET = absolute_path(REPO_ROOT / "../COCO/test")
DEFAULT_BASELINE = REPO_ROOT / "pretrain/crmark_gray_size_512_bit_256.pth"
GRAY_DEFAULTS = {"img_size": 512, "channel_dim": 1, "bit_length": 256,
                 "k": 5, "min_size": 32, "fc": False}
Z_OPT_DEFAULTS = {"z_opt_steps": 30, "z_opt_lr": 0.03, "z_opt_sigma": 0.01,
                  "z_opt_lambda_prior": 0.5, "z_opt_lambda_ent": 0.3}
# Preserve the local experiment default. Published examples use explicit paths.
LOG_SELECTED_SCORE_EPOCH = 53


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def default_checkpoint():
    # Logs were written under scripts/logs/phase2_gray with checkpoint_path=checkpoints/.
    candidates = [SCRIPT_DIR / "checkpoints/phase2_gray/model_best.pth",
                  REPO_ROOT / "checkpoints/phase2_gray/model_best.pth"]
    return next((path for path in candidates if path.is_file()), candidates[0])


def default_score_checkpoint():
    # Resolve only the selected epoch; do not substitute the last epoch or
    # silently disable Phase 3 when its remote weights are absent locally.
    filename = "score_model_{}.pth".format(LOG_SELECTED_SCORE_EPOCH)
    candidates = [SCRIPT_DIR / "checkpoints" / "phase3_gray_hybrid" / filename,
                  REPO_ROOT / "checkpoints" / "phase3_gray_hybrid" / filename]
    return next((path for path in candidates if path.is_file()), candidates[0])


# Attack definitions, quality metrics and tables are unchanged from evaluate.py.

class AttackLibrary:
    """
    攻击库: 包含已见攻击和未见攻击的统一接口。

    所有攻击函数签名: attack(image_tensor) -> attacked_tensor
    输入输出均为 (B, C, H, W) 范围 [0, 1] 的 float tensor。
    """

    # ---------- 已见攻击 (训练噪声池中的 6 种) ----------

    @staticmethod
    def jpeg(x, quality=50):
        """JPEG 压缩 (逐张处理，用 PIL 实现)"""
        import io
        B, C, H, W = x.shape
        result = torch.zeros_like(x)
        for i in range(B):
            img = x[i].detach().cpu()
            if C == 1:
                pil_img = transforms.ToPILImage()(img).convert('L')
            else:
                pil_img = transforms.ToPILImage()(img)
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            pil_img = Image.open(buf)
            result[i] = transforms.ToTensor()(pil_img).to(x.device)
        return result

    @staticmethod
    def gaussian_noise(x, sigma=0.15):
        """高斯噪声"""
        return torch.clamp(x + torch.randn_like(x) * sigma, 0, 1)

    @staticmethod
    def gaussian_blur(x, sigma=3.0):
        """高斯模糊"""
        kernel_size = int(6 * sigma + 1) | 1  # 确保奇数
        return transforms.GaussianBlur(kernel_size, sigma)(x)

    @staticmethod
    def salt_pepper(x, p=0.3):
        """椒盐噪声"""
        mask = torch.rand_like(x)
        result = x.clone()
        result[mask < p / 2] = 0.0
        result[mask > 1 - p / 2] = 1.0
        return result

    @staticmethod
    def median_filter(x, kernel_size=9):
        """中值滤波 (逐张处理)"""
        B, C, H, W = x.shape
        result = torch.zeros_like(x)
        for i in range(B):
            img = x[i].detach().cpu()
            pil_img = transforms.ToPILImage()(img)
            pil_img = pil_img.filter(ImageFilter.MedianFilter(kernel_size))
            result[i] = transforms.ToTensor()(pil_img).to(x.device)
        return result

    @staticmethod
    def dropout(x, p=0.5):
        """随机丢弃像素 (置零)"""
        mask = (torch.rand_like(x) > p).float()
        return x * mask

    # ---------- 未见攻击 (训练时从未出现) ----------

    @staticmethod
    def mean_filter(x, kernel_size=5):
        """均值滤波"""
        C = x.shape[1]
        kernel = torch.ones(C, 1, kernel_size, kernel_size, device=x.device) / (kernel_size ** 2)
        pad = kernel_size // 2
        return torch.clamp(F.conv2d(x, kernel, padding=pad, groups=C), 0, 1)

    @staticmethod
    def sharpen(x, strength=1.0):
        """锐化"""
        B, C, H, W = x.shape
        result = torch.zeros_like(x)
        for i in range(B):
            img = x[i].detach().cpu()
            pil_img = transforms.ToPILImage()(img)
            sharp = pil_img.filter(ImageFilter.SHARPEN)
            sharp_t = transforms.ToTensor()(sharp).to(x.device)
            # 按 strength 混合
            result[i] = torch.clamp(x[i] + strength * (sharp_t - x[i]), 0, 1)
        return result

    @staticmethod
    def brightness(x, delta=20):
        """亮度调整 (delta 为像素值偏移量，范围 0-255 尺度)"""
        return torch.clamp(x + delta / 255.0, 0, 1)

    @staticmethod
    def contrast(x, factor=0.8):
        """对比度调整"""
        mean = x.mean(dim=[2, 3], keepdim=True)
        return torch.clamp(mean + factor * (x - mean), 0, 1)

    @staticmethod
    def poisson_noise(x, lam=30.0):
        """泊松噪声"""
        # noisy = torch.poisson(x * lam) / lam
        noisy = torch.poisson(torch.clamp(x * lam,min=0.0)) / lam

        return torch.clamp(noisy, 0, 1)

    @staticmethod
    def webp(x, quality=50):
        """WebP 压缩"""
        import io
        B, C, H, W = x.shape
        result = torch.zeros_like(x)
        for i in range(B):
            img = x[i].detach().cpu()
            pil_img = transforms.ToPILImage()(img)
            buf = io.BytesIO()
            pil_img.save(buf, format='WEBP', quality=quality)
            buf.seek(0)
            pil_img = Image.open(buf).convert('RGB' if C == 3 else 'L')
            result[i] = transforms.ToTensor()(pil_img).to(x.device)
        return result

    @staticmethod
    def resize_attack(x, scale=0.5):
        """缩放攻击: 缩小再放大"""
        _, _, H, W = x.shape
        small = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        return F.interpolate(small, size=(H, W), mode='bilinear', align_corners=False)

    @staticmethod
    def rotation(x, angle=5.0):
        """小角度旋转 (旋转后裁剪回原尺寸,中心对齐)"""
        B, C, H, W = x.shape
        theta = torch.tensor([
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
            [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0],
        ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return torch.clamp(F.grid_sample(x, grid, mode='bilinear',
                                         padding_mode='reflection', align_corners=False), 0, 1)

    @staticmethod
    def center_crop_resize(x, crop_ratio=0.8):
        """中心裁剪再放大回原尺寸"""
        _, _, H, W = x.shape
        ch, cw = int(H * crop_ratio), int(W * crop_ratio)
        top, left = (H - ch) // 2, (W - cw) // 2
        cropped = x[:, :, top:top + ch, left:left + cw]
        return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)

    @staticmethod
    def jpeg2000(x, quality=30):
        """JPEG2000 压缩 (通过 PIL 的 JP2 编码)"""
        import io
        B, C, H, W = x.shape
        result = torch.zeros_like(x)
        for i in range(B):
            img = x[i].detach().cpu()
            pil_img = transforms.ToPILImage()(img)
            buf = io.BytesIO()
            try:
                pil_img.save(buf, format='JPEG2000', quality_mode='rates',
                             quality_layers=[100.0 / max(quality, 1)])
                buf.seek(0)
                pil_img = Image.open(buf).convert('RGB' if C == 3 else 'L')
                result[i] = transforms.ToTensor()(pil_img).to(x.device)
            except Exception:
                result[i] = x[i]
        return result

    @staticmethod
    def motion_blur(x, kernel_size=9):
        """运动模糊 (水平方向)"""
        C = x.shape[1]
        kernel = torch.zeros(C, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
        kernel[:, :, kernel_size // 2, :] = 1.0 / kernel_size
        pad = kernel_size // 2
        return torch.clamp(F.conv2d(x, kernel, padding=pad, groups=C), 0, 1)

    @staticmethod
    def hue_shift(x, shift=0.1):
        """色调偏移 (仅 3 通道生效)"""
        if x.shape[1] != 3:
            return x
        B = x.shape[0]
        result = torch.zeros_like(x)
        for i in range(B):
            img = x[i].detach().cpu()
            pil_img = transforms.ToPILImage()(img)
            jitter = transforms.ColorJitter(hue=shift)
            pil_img = jitter(pil_img)
            result[i] = transforms.ToTensor()(pil_img).to(x.device)
        return result

    @staticmethod
    def chain(x, *fns):
        """级联多个攻击"""
        for fn in fns:
            x = fn(x)
        return x


def get_seen_attacks():
    """返回已见攻击配置 (与训练噪声池对齐)"""
    atk = AttackLibrary
    return OrderedDict([
        # ("JPEG_Q10",          lambda x: atk.jpeg(x, quality=10)),
        # ("JPEG_Q20",          lambda x: atk.jpeg(x, quality=20)),
        ("JPEG_Q30",          lambda x: atk.jpeg(x, quality=30)),
        # ("JPEG_Q40",          lambda x: atk.jpeg(x, quality=40)),
        # ("JPEG_Q50",          lambda x: atk.jpeg(x, quality=50)),
        # ("JPEG_Q70",          lambda x: atk.jpeg(x, quality=70)),
        # ("GaussNoise_0.10",   lambda x: atk.gaussian_noise(x, sigma=0.10)),
        ("GaussNoise_0.20",   lambda x: atk.gaussian_noise(x, sigma=0.20)),
        # ("GaussNoise_0.30",   lambda x: atk.gaussian_noise(x, sigma=0.30)),
        # ("GaussNoise_0.40",   lambda x: atk.gaussian_noise(x, sigma=0.40)),
        ("GaussBlur_2.0",     lambda x: atk.gaussian_blur(x, sigma=2.0)),
        # ("GaussBlur_3.0",     lambda x: atk.gaussian_blur(x, sigma=3.0)),
        # ("GaussBlur_4.0",     lambda x: atk.gaussian_blur(x, sigma=4.0)),
        # ("GaussBlur_5.0",     lambda x: atk.gaussian_blur(x, sigma=5.0)),
        # ("SaltPepper_0.1",    lambda x: atk.salt_pepper(x, p=0.1)),
        ("SaltPepper_0.2",    lambda x: atk.salt_pepper(x, p=0.2)),
        # ("SaltPepper_0.3",    lambda x: atk.salt_pepper(x, p=0.3)),
        # ("SaltPepper_0.4",    lambda x: atk.salt_pepper(x, p=0.4)),
        # ("SaltPepper_0.5",    lambda x: atk.salt_pepper(x, p=0.5)),
        # ("SaltPepper_0.6",    lambda x: atk.salt_pepper(x, p=0.6)),
        # ("SaltPepper_0.7",    lambda x: atk.salt_pepper(x, p=0.7)),
        # ("SaltPepper_0.8",    lambda x: atk.salt_pepper(x, p=0.8)),
        # ("SaltPepper_0.9",    lambda x: atk.salt_pepper(x, p=0.9)),
        # ("MedianFilter_11",   lambda x: atk.median_filter(x, kernel_size=11)),
        # ("MedianFilter_13",   lambda x: atk.median_filter(x, kernel_size=13)),
        # ("MedianFilter_15",   lambda x: atk.median_filter(x, kernel_size=15)),
        ("MedianFilter_17",   lambda x: atk.median_filter(x, kernel_size=17)),
        # ("MedianFilter_19",   lambda x: atk.median_filter(x, kernel_size=19)),
        # ("Dropout_0.1",       lambda x: atk.dropout(x, p=0.1)),
        ("Dropout_0.3",       lambda x: atk.dropout(x, p=0.3)),
        # ("Dropout_0.5",       lambda x: atk.dropout(x, p=0.5)),
        # ("Dropout_0.7",       lambda x: atk.dropout(x, p=0.7)),
        # ("Dropout_0.9",       lambda x: atk.dropout(x, p=0.9)),
    ])


def get_unseen_attacks():
    """
    返回未见攻击配置 (训练时完全未出现) — 加强版

    目标: 避免天花板效应,让 Phase 1 vs Phase 2 的差异能显现出来。
    策略: (1) 现有攻击加大参数  (2) 新增更强的攻击类型  (3) 级联攻击
    """
    atk = AttackLibrary
    return OrderedDict([
        # ---- 线性滤波 (加大 kernel) ----
        # ("MeanFilter_7",      lambda x: atk.mean_filter(x, kernel_size=7)),
        # ("MeanFilter_9",      lambda x: atk.mean_filter(x, kernel_size=9)),
        # ("MotionBlur_9",      lambda x: atk.motion_blur(x, kernel_size=9)),
        ("MotionBlur_15",     lambda x: atk.motion_blur(x, kernel_size=15)),

        # ---- 锐化 (加大强度) ----
        # ("Sharpen_1.5",       lambda x: atk.sharpen(x, strength=1.5)),
        ("Sharpen_2.0",       lambda x: atk.sharpen(x, strength=2.0)),

        # ---- 亮度/对比度 (加大偏移) ----
        # ("Brightness_-40",    lambda x: atk.brightness(x, delta=-40)),
        ("Brightness_+40",    lambda x: atk.brightness(x, delta=40)),
        # ("Contrast_0.5",      lambda x: atk.contrast(x, factor=0.5)),
        ("Contrast_1.6",      lambda x: atk.contrast(x, factor=1.6)),

        # ---- 色调偏移 (新增) ----
        # ("HueShift_0.1",      lambda x: atk.hue_shift(x, shift=0.1)),
        # ("HueShift_0.2",      lambda x: atk.hue_shift(x, shift=0.2)),

        # ---- 泊松噪声 (加大,lam 越小噪声越强) ----
        # ("Poisson_15",        lambda x: atk.poisson_noise(x, lam=15)),
        ("Poisson_8",         lambda x: atk.poisson_noise(x, lam=8)),

        # ---- WebP 压缩 (加大压缩率) ----
        ("WebP_Q20",          lambda x: atk.webp(x, quality=20)),
        # ("WebP_Q10",          lambda x: atk.webp(x, quality=10)),

        # ---- JPEG2000 (新增压缩类型) ----
        # ("JPEG2000_Q30",      lambda x: atk.jpeg2000(x, quality=30)),
        # ("JPEG2000_Q15",      lambda x: atk.jpeg2000(x, quality=15)),

        # ---- 缩放攻击 (加大,比例越小损失越多) ----
        ("Resize_0.5x",       lambda x: atk.resize_attack(x, scale=0.5)),
        # ("Resize_0.35x",      lambda x: atk.resize_attack(x, scale=0.35)),
        # ("Resize_0.25x",      lambda x: atk.resize_attack(x, scale=0.25)),

        # ---- 几何变换 (新增) ----
        ("Rotation_5deg",     lambda x: atk.rotation(x, angle=5.0)),
        # ("Rotation_10deg",    lambda x: atk.rotation(x, angle=10.0)),
        ("CenterCrop_0.8",    lambda x: atk.center_crop_resize(x, crop_ratio=0.8)),
        # ("CenterCrop_0.7",    lambda x: atk.center_crop_resize(x, crop_ratio=0.7)),

        # ---- 级联攻击 (新增, 模拟真实 pipeline) ----
        ("Resize_0.5+WebP_30", lambda x: atk.chain(
            x,
            lambda y: atk.resize_attack(y, scale=0.5),
            lambda y: atk.webp(y, quality=30),
        )),
        ("Bright_30+Contrast_0.7", lambda x: atk.chain(
            x,
            lambda y: atk.brightness(y, delta=30),
            lambda y: atk.contrast(y, factor=0.7),
        )),
        # ("MeanFilter_5+Sharpen_1.5", lambda x: atk.chain(
        #     x,
        #     lambda y: atk.mean_filter(y, kernel_size=5),
        #     lambda y: atk.sharpen(y, strength=1.5),
        # )),
    ])


def compute_psnr_np(img1, img2):
    """计算 PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_ssim_torch(img1, img2, window_size=11, sigma=1.5):
    """计算 SSIM，输入为 NCHW、取值范围 [0, 1] 的 tensor。"""
    if img1.shape != img2.shape:
        raise ValueError(f"SSIM 输入尺寸不一致: {img1.shape} vs {img2.shape}")

    _, channel, _, _ = img1.shape
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device)
    coords = coords - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    window = torch.outer(gauss, gauss).view(1, 1, window_size, window_size)
    window = window.expand(channel, 1, window_size, window_size)
    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


def print_results_table(results, quality, model_name="Model"):
    """打印单模型结果表"""
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"  Stego PSNR: {quality['psnr']:.2f} dB  |  SSIM: {quality['ssim']:.4f}")
    print(f"{'=' * 60}")
    print(f"  {'Attack':<22} {'BER':>8} {'Accuracy':>10}")
    print(f"  {'-' * 42}")
    for name, r in results.items():
        print(f"  {name:<22} {r['ber']:>8.4f} {r['accuracy']:>9.2%}")

    avg_ber = np.mean([r['ber'] for r in results.values()])
    print(f"  {'-' * 42}")
    print(f"  {'AVERAGE':<22} {avg_ber:>8.4f} {1 - avg_ber:>9.2%}")


def print_comparison_table(results1, quality1, results2, quality2,
                           name1="Phase1", name2="Phase2"):
    """打印双模型对比表"""
    print(f"\n{'=' * 75}")
    print(f"  {name1} PSNR: {quality1['psnr']:.2f} dB, SSIM: {quality1['ssim']:.4f}  |  "
          f"{name2} PSNR: {quality2['psnr']:.2f} dB, SSIM: {quality2['ssim']:.4f}")
    print(f"{'=' * 75}")
    print(f"  {'Attack':<22} {name1 + ' Acc':>12} {name2 + ' Acc':>12} {'Delta':>10}")
    print(f"  {'-' * 58}")

    for name in results1.keys():
        acc1 = results1[name]['accuracy']
        acc2 = results2[name]['accuracy']
        delta = acc2 - acc1
        marker = "+" if delta > 0.005 else ("-" if delta < -0.005 else " ")
        print(f"  {name:<22} {acc1:>11.2%} {acc2:>11.2%} {marker}{abs(delta):>8.2%}")

    avg1 = np.mean([r['accuracy'] for r in results1.values()])
    avg2 = np.mean([r['accuracy'] for r in results2.values()])
    delta = avg2 - avg1
    marker = "+" if delta > 0.005 else "-" if delta < -0.005 else " "
    print(f"  {'-' * 58}")
    print(f"  {'AVERAGE':<22} {avg1:>11.2%} {avg2:>11.2%} {marker}{abs(delta):>8.2%}")


class GrayEvaluationDataset(Dataset):
    """Fixed grayscale covers/messages shared by every model and attack."""

    def __init__(self, root_path, num_images=100, seed=42, im_size=512, bit_length=256):
        root_path = absolute_path(root_path)
        if not root_path.is_dir():
            raise ValueError("Dataset directory does not exist: {}".format(root_path))
        paths = []
        for extension in ("png", "tif", "tiff", "bmp", "jpg", "jpeg", "gif", "webp"):
            paths.extend(glob.glob(str(root_path / "**" / ("*." + extension)), recursive=True))
        self.cover = sorted(paths)
        self.total_images = len(self.cover)
        if not self.cover:
            raise ValueError("No supported images found in: {}".format(root_path))
        if num_images > 0:
            indices = torch.randperm(self.total_images, generator=torch.Generator().manual_seed(seed))[:num_images]
            self.cover = [self.cover[index] for index in indices.tolist()]
        self.seed, self.im_size, self.bit_length = seed, im_size, bit_length
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.cover)

    def __getitem__(self, index):
        with Image.open(self.cover[index]) as image:
            cover = self.to_tensor(ImageOps.fit(image.convert("L"), (self.im_size, self.im_size)))
        message = torch.randint(0, 2, (self.bit_length,),
                                generator=torch.Generator().manual_seed(self.seed + index)).float()
        return cover, message


def read_checkpoint(path, state_key):
    path = absolute_path(path)
    if not path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(path))
    # Training checkpoints can contain optimizer state / NumPy values as well.
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or state_key not in checkpoint:
        raise ValueError("{} must contain {}".format(path, state_key))
    return checkpoint


def resolve_model_config(params, overrides=None):
    """Checkpoint metadata wins over defaults; explicit conflicting args fail."""
    overrides = overrides or {}
    config = {}
    for name, default in GRAY_DEFAULTS.items():
        explicit = overrides.get(name)
        saved = params.get(name)
        if explicit is not None and saved is not None and explicit != saved:
            raise ValueError("{} mismatch: checkpoint={!r}, argument={!r}".format(name, saved, explicit))
        config[name] = saved if saved is not None else explicit if explicit is not None else default
    if config["channel_dim"] != 1:
        raise ValueError("evaluate_gray.py requires channel_dim=1; use evaluate.py for color checkpoints")
    for name in ("img_size", "bit_length", "k", "min_size"):
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] < 1:
            raise ValueError("{} must be a positive integer".format(name))
    ratio, remainder = divmod(config["img_size"], config["min_size"])
    if remainder or ratio < 1 or ratio & (ratio - 1):
        raise ValueError("img_size/min_size must be an integer power of two")
    if not config["fc"]:
        side = int(config["bit_length"] ** 0.5)
        if side ** 2 != config["bit_length"] or side > config["min_size"]:
            raise ValueError("fc=False requires square bit_length with sqrt(bit_length) <= min_size")
    return config


@lru_cache(maxsize=1)
def inference_model_class():
    """Use native iIWN classes/forward without constructing training-only LPIPS.

    nets.py imports all compression/training utilities. Select only its native
    architecture classes and the native StochasticRound class by AST, unchanged.
    The subclass replaces ONLY initialization of unused loss modules. No copied
    block/rounding/forward implementation or external pretrained AlexNet is used.
    """
    scope = {"__name__": "_alclo_gray_inference", "torch": torch, "nn": nn, "np": np}
    sources = [
        (REPO_ROOT / "crmark/compressor/utils.py", {"StochasticRound"}),
        (REPO_ROOT / "crmark/nets/nets.py", {"ConvTP2d", "Conv2D", "SpatialAttention",
                                          "BaseFuncDown", "BaseFuncUp", "InvertibleBlock", "Model"}),
    ]
    for path, names in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        if {node.name for node in selected} != names:
            raise ImportError("Native iIWN source classes changed: {}".format(path))
        module = ast.Module(body=selected, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    native_model, native_block = scope["Model"], scope["InvertibleBlock"]

    class InferenceModel(native_model):
        def __init__(self, img_size, channel_dim, bit_length, k, min_size=32, fc=False):
            nn.Module.__init__(self)
            self.img_size, self.channel_dim, self.bit_length = img_size, channel_dim, bit_length
            self.k, self.min_size, self.fc = k, min_size, fc
            self.inn_blocks = nn.Sequential(*[
                native_block(img_size, channel_dim, bit_length, min_size=min_size, fc=fc) for _ in range(k)])

    return InferenceModel


def load_model(checkpoint_path, device, overrides=None):
    checkpoint = read_checkpoint(checkpoint_path, "model_state_dict")
    params = checkpoint.get("param_dict", {}) or {}
    config = resolve_model_config(params, overrides)
    model = inference_model_class()(**config)
    state = checkpoint["model_state_dict"]
    state = OrderedDict((name[7:] if name.startswith("module.") else name, value) for name, value in state.items())
    # Pretrained and Phase 2 saves include LLoss.lpips.*; it is not part of
    # embedding/extraction. All other unknown keys and all missing iIWN keys fail.
    ignored = [name for name in state if name.startswith("LLoss.")]
    state = OrderedDict((name, value) for name, value in state.items() if not name.startswith("LLoss."))
    model.load_state_dict(state, strict=True)
    metadata = {
        "checkpoint": str(absolute_path(checkpoint_path)), "config": config,
        "epoch": checkpoint.get("epoch", params.get("epoch")),
        "val_psnr": float(checkpoint["val_psnr"]) if "val_psnr" in checkpoint else None,
        "val_acc": float(checkpoint["val_acc"]) if "val_acc" in checkpoint else None,
        "training_only_loss_entries_ignored": len(ignored),
    }
    return model.to(device).eval().requires_grad_(False), metadata


@lru_cache(maxsize=2)
def load_local_module(name):
    path = REPO_ROOT / "alclo/score" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_gray_" + name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_score_network(checkpoint_path, device, model_config, args):
    checkpoint = read_checkpoint(checkpoint_path, "score_net_state_dict")
    params = checkpoint.get("param_dict", {}) or {}
    for key, expected in (("in_channels", 1), ("channel_dim", 1),
                          ("bit_length", model_config["bit_length"]), ("img_size", model_config["img_size"])):
        if key in params and params[key] != expected:
            raise ValueError("Score checkpoint {}={} does not match grayscale iIWN ({})".format(key, params[key], expected))
    architecture = {
        "in_channels": 1, "base_channels": params.get("base_channels", 32),
        "channel_mults": tuple(params.get("channel_mults", [1, 2, 4, 4])),
        "embed_dim": params.get("embed_dim", 256),
    }
    score_net = load_local_module("score_network").ScoreNetwork(**architecture)
    state = OrderedDict((name[7:] if name.startswith("module.") else name, value)
                        for name, value in checkpoint["score_net_state_dict"].items())
    score_net.load_state_dict(state, strict=True)
    options = {key: getattr(args, key) if getattr(args, key) is not None else params.get(key, default)
               for key, default in Z_OPT_DEFAULTS.items()}
    if not isinstance(options["z_opt_steps"], int) or options["z_opt_steps"] < 1:
        raise ValueError("z_opt_steps must be a positive integer")
    for name in ("z_opt_lr", "z_opt_sigma", "z_opt_lambda_prior", "z_opt_lambda_ent"):
        value = options[name]
        if not np.isfinite(value) or value < 0 or (name in ("z_opt_lr", "z_opt_sigma") and value == 0):
            raise ValueError("Invalid {}: {}".format(name, value))
    metadata = {
        "checkpoint": str(absolute_path(checkpoint_path)), "epoch": checkpoint.get("epoch"),
        "architecture": architecture, "z_optimization": options,
        "training_phase2_checkpoint": checkpoint.get("phase2_checkpoint"),
    }
    return score_net.to(device).eval().requires_grad_(False), metadata


def derived_seed(seed, label, image_offset):
    digest = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little")
    return (seed + digest + image_offset) % (2 ** 32)


@contextlib.contextmanager
def seeded_rng(seed):
    """Isolate native rounding, attacks and z optimization from one another."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def count_bit_errors(decoded, secret, name):
    if decoded.shape != secret.shape:
        raise ValueError("{} decoded shape {} != {}".format(name, decoded.shape, secret.shape))
    if not torch.isfinite(decoded).all():
        raise FloatingPointError("Non-finite decoded message: {}".format(name))
    return int(((decoded > 0.5) != (secret > 0.5)).sum().item())


def evaluate_models(model, dataloader, seen, unseen, device, seed=42,
                    baseline=None, z_optimizer=None, quiet=False):
    """Paired evaluation: one stego per model, one attack result per extraction pair.

    Do not decorate this function with no_grad: Phase 3 must differentiate the
    native inverse with respect to z. Encoding/attacks/random-z stay no_grad.
    """
    models = OrderedDict([("phase2_randz", model)])
    if baseline is not None:
        models["phase1_randz"] = baseline
    attacks = OrderedDict([("Clean", lambda image: image)])
    attacks.update(seen)
    attacks.update(unseen)
    mode_names = list(models) + (["phase3_zopt"] if z_optimizer is not None else [])
    errors = {name: OrderedDict((attack, 0) for attack in attacks) for name in mode_names}
    quality_sums = {name: {"psnr": 0.0, "ssim": 0.0, "overflow_pixels": 0, "pixels": 0} for name in models}
    num_images = num_bits = 0
    for cover, secret in tqdm(dataloader, desc="Grayscale evaluation", ncols=100, disable=quiet):
        cover, secret = cover.to(device), secret.to(device)
        batch_size, bits = secret.shape
        random_z = {}
        for attack_name in attacks:
            rows = [torch.randn(bits, generator=torch.Generator().manual_seed(
                derived_seed(seed, "rand-z/" + attack_name, num_images + index))) for index in range(batch_size)]
            random_z[attack_name] = torch.stack(rows).to(device=device, dtype=cover.dtype)
        for mode, network in models.items():
            network.eval()
            with torch.no_grad(), seeded_rng(derived_seed(seed, "encode", num_images)):
                stego, _unused_true_z = network.forward(cover, secret, True, False)
                if stego.shape != cover.shape or not torch.isfinite(stego).all():
                    raise ValueError("Invalid grayscale stego from {}".format(mode))
                quality_sums[mode]["psnr"] += float(compute_psnr_np(stego.cpu().numpy(), cover.cpu().numpy())) * batch_size
                quality_sums[mode]["ssim"] += float(compute_ssim_torch(stego, cover)) * batch_size
                quality_sums[mode]["overflow_pixels"] += int(((stego < 0) | (stego > 1)).sum().item())
                quality_sums[mode]["pixels"] += stego.numel()
                stego_quant = torch.round(stego.clamp(0.0, 1.0) * 255.0) / 255.0
            for attack_name, attack in attacks.items():
                with torch.no_grad(), seeded_rng(derived_seed(seed, "attack/" + attack_name, num_images)):
                    attacked = attack(stego_quant)
                    if attacked.shape != cover.shape or not torch.isfinite(attacked).all():
                        raise ValueError("Attack {} did not preserve finite N1HW grayscale images".format(attack_name))
                with torch.no_grad(), seeded_rng(derived_seed(seed, "inverse/" + attack_name, num_images)):
                    _, decoded = network.forward(attacked, random_z[attack_name], True, True)
                    errors[mode][attack_name] += count_bit_errors(decoded, secret, mode + "/" + attack_name)
                if mode == "phase2_randz" and z_optimizer is not None:
                    # Both Phase 2 and Phase 3 receive this exact attacked tensor.
                    for index in range(batch_size):
                        with torch.enable_grad(), seeded_rng(derived_seed(seed, "zopt/" + attack_name, num_images + index)):
                            _, decoded = z_optimizer.optimize(attacked[index:index + 1].detach())
                        errors["phase3_zopt"][attack_name] += count_bit_errors(
                            decoded.detach(), secret[index:index + 1], "phase3_zopt/" + attack_name)
        num_images += batch_size
        num_bits += secret.numel()
    if num_bits == 0:
        raise ValueError("No images were evaluated")
    summaries = OrderedDict()
    for mode in mode_names:
        source = "phase2_randz" if mode == "phase3_zopt" else mode
        quality = {key: float(quality_sums[source][key] / num_images) for key in ("psnr", "ssim")}
        quality["overflow_fraction"] = quality_sums[source]["overflow_pixels"] / quality_sums[source]["pixels"]
        all_results = OrderedDict((name, {"ber": count / num_bits, "accuracy": 1.0 - count / num_bits})
                                  for name, count in errors[mode].items())
        groups, averages = OrderedDict(), OrderedDict()
        for name, group in (("seen", seen), ("unseen", unseen)):
            if group:
                groups[name] = OrderedDict((key, all_results[key]) for key in group)
                avg_ber = float(np.mean([item["ber"] for item in groups[name].values()]))
                averages[name] = {"ber": avg_ber, "accuracy": 1.0 - avg_ber}
        summaries[mode] = {"quality": quality, "clean": all_results["Clean"],
                           "results": groups, "averages": averages,
                           "num_images": num_images, "num_bits": num_bits}
    return summaries


def build_parser():
    parser = argparse.ArgumentParser(description="ALCLO grayscale evaluation; specify paired Phase 2/3 checkpoints")
    parser.add_argument("--checkpoint", default=str(default_checkpoint()), help="Phase 2 iIWN weights (default: phase2_gray/model_best.pth)")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="Optional Phase 1 gray weights for comparison")
    parser.add_argument("--no_baseline", action="store_true", help="Only evaluate the selected Phase 2/3 model")
    parser.add_argument("--score_checkpoint", default=str(default_score_checkpoint()),
                        help="Phase 3 score weights (local default: phase3_gray_hybrid/score_model_53.pth; override for your run)")
    parser.add_argument("--skip_phase3", action="store_true", help="Evaluate Phase 2 only; no score checkpoint is required")
    parser.add_argument("--dataset_path", default=str(DEFAULT_DATASET), help="Recursively searched image directory")
    parser.add_argument("--num_images", type=int, default=24, help="0 = all images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--im_size", type=int, default=None, help="Read from checkpoint; fallback 512")
    parser.add_argument("--channel_dim", type=int, choices=[1], default=None, help="Gray only; fallback 1")
    parser.add_argument("--bit_length", type=int, default=None, help="Read from checkpoint; fallback 256")
    parser.add_argument("--k", type=int, default=None, help="Read from checkpoint; fallback 5")
    parser.add_argument("--min_size", type=int, default=None, help="Read from checkpoint; fallback 32")
    parser.add_argument("--fc", type=str2bool, default=None, help="Read from checkpoint; fallback False")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip_seen", action="store_true")
    parser.add_argument("--skip_unseen", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    for name in Z_OPT_DEFAULTS:
        parser.add_argument("--" + name, type=int if name == "z_opt_steps" else float, default=None,
                            help="Override score checkpoint value (gray fallback: {})".format(Z_OPT_DEFAULTS[name]))
    parser.add_argument("--output", default=None, help="JSON path; defaults beside the selected Phase 2/score checkpoint")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_images < 0 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("num_images/num_workers must be >= 0 and batch_size must be >= 1")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must be in [0, 2**32)")
    if args.skip_seen and args.skip_unseen:
        parser.error("At least one attack group must be enabled")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and (not torch.cuda.is_available() or not 0 <= args.gpu_id < torch.cuda.device_count()):
        parser.error("Requested CUDA device is unavailable")
    device = torch.device("cuda:{}".format(args.gpu_id) if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint_path = absolute_path(args.checkpoint)
    score_path = absolute_path(args.score_checkpoint) if args.score_checkpoint and not args.skip_phase3 else None
    baseline_path = absolute_path(args.baseline) if args.baseline and not args.no_baseline else None
    if args.output:
        output_path = absolute_path(args.output)
    elif score_path is not None:
        output_path = score_path.parent / ("evaluate_gray_" + score_path.stem + ".json")
    else:
        output_path = checkpoint_path.parent / ("evaluate_gray_" + checkpoint_path.stem + ".json")
    if output_path in (checkpoint_path, score_path, baseline_path):
        parser.error("output must not overwrite a model checkpoint")
    for path in (checkpoint_path, score_path, baseline_path):
        if path is not None and not path.is_file():
            parser.error("Checkpoint does not exist: {}. Set the actual --checkpoint/--baseline/--score_checkpoint path.".format(path))
    overrides = {key: getattr(args, "im_size" if key == "img_size" else key) for key in GRAY_DEFAULTS}
    try:
        model, model_info = load_model(checkpoint_path, device, overrides)
        config = model_info["config"]
        baseline, baseline_info = None, None
        if baseline_path is not None:
            baseline, baseline_info = load_model(baseline_path, device, config)
        z_optimizer, score_info = None, None
        if score_path is not None:
            score_net, score_info = load_score_network(score_path, device, config, args)
            options = score_info["z_optimization"]
            z_optimizer = load_local_module("z_optimizer").ZOptimizer(
                iiwn_model=model, score_network=score_net, bit_length=config["bit_length"],
                num_steps=options["z_opt_steps"], lr=options["z_opt_lr"], sigma=options["z_opt_sigma"],
                lambda_prior=options["z_opt_lambda_prior"], lambda_ent=options["z_opt_lambda_ent"], device=device)
            source = score_info["training_phase2_checkpoint"]
            if source and str(source).replace("\\", "/").split("/")[-1] != checkpoint_path.name:
                warnings.warn("Score was trained with {}; confirm it contains the same iIWN weights as {}".format(source, checkpoint_path))
        dataset = GrayEvaluationDataset(args.dataset_path, args.num_images, args.seed, config["img_size"], config["bit_length"])
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    seen = get_seen_attacks() if not args.skip_seen else OrderedDict()
    unseen = get_unseen_attacks() if not args.skip_unseen else OrderedDict()
    if any("WebP" in name for name in unseen) and not features.check("webp"):
        parser.error("Pillow WebP support is required for the enabled attacks")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, pin_memory=use_cuda)
    print("Device: {} | Gray {}x{} | {} raw bits | k={} | min_size={} | fc={}".format(
        device, config["img_size"], config["img_size"], config["bit_length"], config["k"], config["min_size"], config["fc"]))
    print("Phase 2: {} | epoch: {}".format(checkpoint_path, model_info["epoch"]))
    if baseline_info:
        print("Phase 1 baseline: {}".format(baseline_path))
    if score_info:
        print("Phase 3 score: {} | epoch: {} | architecture: {}".format(score_path, score_info["epoch"], score_info["architecture"]))
        print("z optimization: {}".format(score_info["z_optimization"]))
    print("Dataset: {} | selected: {}/{} | seed: {}".format(absolute_path(args.dataset_path), len(dataset), dataset.total_images, args.seed))
    print("Attacks: {} seen, {} unseen | Phase 3: {}".format(len(seen), len(unseen), "enabled" if z_optimizer else "disabled"))
    summaries = evaluate_models(model, loader, seen, unseen, device, args.seed, baseline, z_optimizer, args.quiet)
    for mode, summary in summaries.items():
        print("\n{} / Clean: BER={:.4f} | Accuracy={:.2%}".format(mode, summary["clean"]["ber"], summary["clean"]["accuracy"]))
        for group, results in summary["results"].items():
            print_results_table(results, summary["quality"], mode + " / " + group)
    for left, right in (("phase1_randz", "phase2_randz"), ("phase2_randz", "phase3_zopt")):
        if left in summaries and right in summaries:
            for group in summaries[right]["results"]:
                print_comparison_table(summaries[left]["results"][group], summaries[left]["quality"],
                                       summaries[right]["results"][group], summaries[right]["quality"], left, right)
    report = {
        "model": "ALCLO grayscale", "checkpoint": str(checkpoint_path), "model_config": config,
        "phase2": model_info, "baseline": baseline_info, "score": score_info,
        "dataset_path": str(absolute_path(args.dataset_path)), "selected_images": dataset.cover,
        "num_images": len(dataset), "seed": args.seed, "batch_size": args.batch_size, "device": str(device),
        "attack_groups_reference": "scripts/evaluate.py: 6 seen / 11 unseen",
        "preprocessing": "PIL L + ImageOps.fit; fixed messages per selected image",
        "quality_reference": "stego before clamping/8-bit quantization (same metric functions; batch weighted)",
        "quantize_before_attack": True, "clamp_before_quantization": True, "decode_threshold": 0.5,
        "paired_comparison": "same messages, shared Phase 2/3 stego+attacks, matched random-z and attack seeds for Phase 1/2",
        "true_z_used_for_extraction": False, "results": summaries,
    }
    # Serialize BEFORE opening; all metrics are native Python numbers.
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print("\nEvaluated {} images; results saved to {}".format(len(dataset), output_path))


if __name__ == "__main__":
    main()
