"""
ALCLO 评估脚本

功能:
1. 加载 Phase 1 或 Phase 2 的 checkpoint
2. 在已见攻击 (训练噪声池) 上测试
3. 在未见攻击 (训练时从未出现) 上测试
4. 输出对比表格

支持两种模式:
- 单模型评估:   python scripts/evaluate.py --checkpoint path/to/model.pth
- 双模型对比:   python scripts/evaluate.py --checkpoint path/to/phase2.pth --baseline path/to/phase1.pth

用法示例:
    python scripts/evaluate.py \
        --checkpoint checkpoints/phase2_adv_zreg/model_200.pth \
        --baseline checkpoints/phase1_baseline/model_400.pth \
        --dataset_path ../../DIV2K/val \
        --num_images 50
"""

import os
import sys
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crmark.nets import Model
from alclo.data import HideImage
from torch.utils.data import DataLoader

# Phase 3: score 网络 + z 优化器
from alclo.score.score_network import ScoreNetwork
from alclo.score.z_optimizer import ZOptimizer

warnings.filterwarnings("ignore")


# ==================== 攻击实现 ====================


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


# ==================== 评估函数 ====================


def compute_ber(w_hat, w):
    """计算 BER (Bit Error Rate)"""
    w_binary = (w > 0.5).float()
    w_hat_binary = (w_hat > 0.5).float()
    return torch.mean(torch.abs(w_binary - w_hat_binary)).item()


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


def load_model(checkpoint_path, device, bit_length=64, im_size=256,
               channel_dim=3, k=5, min_size=16, fc=False):
    """加载模型 checkpoint"""
    model = Model(im_size, channel_dim, bit_length, k, min_size, fc).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 读取 checkpoint 中的参数 (如果存在)
    if 'param_dict' in ckpt:
        info = ckpt['param_dict']
        print(f"  bit_length={info.get('bit_length')}, img_size={info.get('img_size')}, "
              f"k={info.get('k')}, channel_dim={info.get('channel_dim')}")

    return model


def load_score_network(checkpoint_path, device, channel_dim=3):
    """加载 Phase 3 训练的 score network checkpoint。

    checkpoint 里的 param_dict 记录了架构超参,优先使用;缺失时走默认值。
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = ckpt.get('param_dict', {}) or {}

    score_net = ScoreNetwork(
        in_channels=params.get('in_channels', channel_dim),
        base_channels=params.get('base_channels', 64),
        channel_mults=tuple(params.get('channel_mults', [1, 2, 4, 4])),
        embed_dim=params.get('embed_dim', 256),
    ).to(device)

    score_net.load_state_dict(ckpt['score_net_state_dict'])
    score_net.eval()

    n_params = sum(p.numel() for p in score_net.parameters())
    print(f"  score network 参数量: {n_params / 1e6:.2f}M, "
          f"来自 epoch {ckpt.get('epoch', '?')}")
    return score_net


def evaluate_model(model, dataloader, attacks, device, desc="Evaluating",
                   z_optimizer=None):
    """
    在一组攻击下评估模型。

    Args:
        z_optimizer: 若提供,则使用 Phase 3 的 score-based z 优化逐张提取水印;
                     否则走 DRRW 原始方法 (随机 z)。

    返回:
        results: dict[attack_name -> {"ber": float, "accuracy": float}]
        quality: dict {"psnr": float, "ssim": float} stego 图像质量
    """
    results = OrderedDict()
    psnr_list = []
    ssim_list = []

    # 先收集 stego 质量 (无攻击, 与 z 提取方式无关)
    for cover, secret in dataloader:
        cover, secret = cover.to(device), secret.to(device)
        with torch.no_grad():
            stego, z = model.forward(cover, secret, True, False)
            psnr_val = compute_psnr_np(
                stego.detach().cpu().numpy(),
                cover.detach().cpu().numpy()
            )
            psnr_list.append(psnr_val)
            ssim_list.append(compute_ssim_torch(stego.detach(), cover.detach()))

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    # 逐攻击评估
    for atk_name, atk_fn in tqdm(attacks.items(), desc=desc, ncols=100):
        ber_list = []

        for cover, secret in dataloader:
            cover, secret = cover.to(device), secret.to(device)

            # 正向 + 攻击 (不需要梯度)
            with torch.no_grad():
                stego, z = model.forward(cover, secret, True, False)
                stego_quant = torch.round(stego * 255.0) / 255.0
                attacked = atk_fn(stego_quant)

            # 水印提取: 两种路径
            if z_optimizer is not None:
                # Phase 3 路径: 逐张 score-based z 优化
                B = attacked.shape[0]
                w_hat_list = []
                for i in range(B):
                    _, w_hat_i = z_optimizer.optimize(attacked[i:i + 1])
                    w_hat_list.append(w_hat_i)
                w_hat = torch.cat(w_hat_list, dim=0)
            else:
                # DRRW 原始路径: 随机 z
                with torch.no_grad():
                    rand_z = torch.randn_like(z)
                    _, w_hat = model.forward(attacked, rand_z, True, True)

            ber = compute_ber(w_hat.detach(), secret)
            ber_list.append(ber)

        avg_ber = np.mean(ber_list)
        results[atk_name] = {
            "ber": avg_ber,
            "accuracy": 1.0 - avg_ber,
        }

    return results, {"psnr": avg_psnr, "ssim": avg_ssim}


# ==================== 结果展示 ====================


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


# ==================== 主函数 ====================


def _run_eval_category(cat_name, attacks, model, baseline, z_optimizer,
                       dataloader, device, checkpoint_basename):
    """对一组攻击 (seen 或 unseen) 完整评估并打印对比表。

    评估顺序:
        1. Phase2 rand-z (永远跑)
        2. Phase1 baseline (如提供) → 对比 Phase1 vs Phase2 rand-z
        3. Phase2 z-opt (如提供 score) → 对比 Phase2 rand-z vs Phase2 z-opt

    这样既保留原有 Phase1 vs Phase2 的对比, 又新增 Phase3 的增益验证。
    """
    print(f"\n{'#' * 60}")
    print(f"  {cat_name}")
    print(f"{'#' * 60}")

    # (1) Phase 2 rand-z
    r_randz, q_randz = evaluate_model(
        model, dataloader, attacks, device, f"{cat_name} rand-z")

    # (2) Phase1 对比
    if baseline is not None:
        r_base, q_base = evaluate_model(
            baseline, dataloader, attacks, device, f"{cat_name} Phase1")
        print_comparison_table(r_base, q_base, r_randz, q_randz,
                               "Phase1", "Phase2 rand-z")
    else:
        print_results_table(r_randz, q_randz, checkpoint_basename)

    # (3) Phase3 z-opt 对比
    if z_optimizer is not None:
        r_zopt, q_zopt = evaluate_model(
            model, dataloader, attacks, device,
            f"{cat_name} z-opt", z_optimizer=z_optimizer)
        print_comparison_table(r_randz, q_randz, r_zopt, q_zopt,
                               "Phase2 rand-z", "Phase2 z-opt")


def main():
    parser = argparse.ArgumentParser(description="ALCLO 模型评估")
    parser.add_argument("--checkpoint", type=str, required=False, default="./checkpoints/phase2_R0_baseline/model_63.pth",
                        help="要评估的模型 checkpoint 路径")
    # parser.add_argument("--baseline", type=str, default="./checkpoints/phase1_baseline_lambda_z1e-3/model_486.pth",
    #                     help="基线模型 checkpoint 路径 (用于对比)")
    parser.add_argument("--baseline", type=str, default="../pretrain/crmark_color_size_256_bit_64.pth",
                        help="基线模型 checkpoint 路径 (用于对比)")
    parser.add_argument("--dataset_path", type=str, default="../../DIV2K/val",
    #                     help="测试数据集路径")
    # parser.add_argument("--dataset_path", type=str, default="../../Kodak24",
    #                     help="测试数据集路径")
    # parser.add_argument("--dataset_path", type=str, default="../../AID",
    #                     help="测试数据集路径")
    # parser.add_argument("--dataset_path", type=str, default="../../DiffusionDB",
    #                     help="测试数据集路径")
    # parser.add_argument("--dataset_path", type=str, default="../../CelebA-HQ",
    #                     help="测试数据集路径")
    # parser.add_argument("--dataset_path", type=str, default="../../COCO/test",
                        help="测试数据集路径")
    parser.add_argument("--num_images", type=int, default=100,
                        help="测试图像数量 (0=全部)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机抽样种子")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--im_size", type=int, default=256)
    parser.add_argument("--channel_dim", type=int, default=3)
    parser.add_argument("--bit_length", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_size", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--skip_seen", action="store_true",
                        help="跳过已见攻击测试")
    parser.add_argument("--skip_unseen", action="store_true",
                        help="跳过未见攻击测试")
    parser.add_argument("--skip_phase3", action="store_true",
                        help="跳过 Phase 3 (score-based z 优化) 评估; 只测 Phase 1/2 rand-z 路径")

    # ---- Phase 3: score-based z 优化 ----
    parser.add_argument("--score_checkpoint", type=str, default="./checkpoints/phase3_hybrid/score_model_61.pth",
                        help="Phase 3 score network checkpoint 路径;不提供则跳过 z 优化评估")
    parser.add_argument("--z_opt_steps", type=int, default=30,
                        help="z 优化步数")
    parser.add_argument("--z_opt_lr", type=float, default=0.05,
                        help="z 优化学习率")
    parser.add_argument("--z_opt_sigma", type=float, default=0.05,
                        help="z 优化时 score network 的 σ")
    parser.add_argument("--z_opt_lambda_prior", type=float, default=0.5,
                        help="z 先验项权重")
    parser.add_argument("--z_opt_lambda_ent", type=float, default=0.3,
                        help="水印熵项权重")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 数据集
    dataset = HideImage(args.dataset_path, args.im_size, args.bit_length, args.channel_dim)
    if args.num_images > 0:
        if hasattr(dataset, "cover"):
            dataset.cover = sorted(dataset.cover)
        total_images = len(dataset)
        num_selected = min(args.num_images, total_images)
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(total_images, generator=generator)[:num_selected].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"随机抽样: {num_selected}/{total_images} 张, seed={args.seed}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(f"测试集: {len(dataset)} 张图像")

    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    model = load_model(args.checkpoint, device, args.bit_length, args.im_size,
                       args.channel_dim, args.k, args.min_size)

    baseline = None
    if args.baseline:
        print(f"加载基线: {args.baseline}")
        baseline = load_model(args.baseline, device, args.bit_length, args.im_size,
                              args.channel_dim, args.k, args.min_size)

    # ---- 加载 score 网络并构建 z 优化器 (如提供且未跳过) ----
    z_optimizer = None
    if args.skip_phase3:
        print("\n[跳过 Phase 3 评估] --skip_phase3 已设置, 不加载 score network")
    elif args.score_checkpoint:
        print(f"\n加载 score network: {args.score_checkpoint}")
        score_net = load_score_network(args.score_checkpoint, device, args.channel_dim)
        z_optimizer = ZOptimizer(
            iiwn_model=model,
            score_network=score_net,
            bit_length=args.bit_length,
            num_steps=args.z_opt_steps,
            lr=args.z_opt_lr,
            sigma=args.z_opt_sigma,
            lambda_prior=args.z_opt_lambda_prior,
            lambda_ent=args.z_opt_lambda_ent,
            device=device,
        )
        print(f"  z-opt 超参: steps={args.z_opt_steps}, lr={args.z_opt_lr}, "
              f"σ={args.z_opt_sigma}, λ_prior={args.z_opt_lambda_prior}, "
              f"λ_ent={args.z_opt_lambda_ent}")

    # 攻击配置
    seen_attacks = get_seen_attacks() if not args.skip_seen else OrderedDict()
    unseen_attacks = get_unseen_attacks() if not args.skip_unseen else OrderedDict()

    ckpt_name = os.path.basename(args.checkpoint)

    if seen_attacks:
        _run_eval_category("已见攻击 (Seen Attacks)", seen_attacks,
                           model, baseline, z_optimizer, dataloader, device, ckpt_name)

    if unseen_attacks:
        _run_eval_category("未见攻击 (Unseen Attacks) — 核心对比", unseen_attacks,
                           model, baseline, z_optimizer, dataloader, device, ckpt_name)

    print("\n评估完成。")


if __name__ == "__main__":
    main()
