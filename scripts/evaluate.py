"""ALCLO evaluation for base, adversarially trained, and score-guided models.

Evaluate configured seen/unseen attack groups and print comparison tables.
Pass --checkpoint for the target iIWN, --baseline for an optional base
network, and --score_checkpoint for optional score-guided latent optimization.
See README.md for complete commands.
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

# Phase 3 Score Network and latent optimizer.
from alclo.score.score_network import ScoreNetwork
from alclo.score.z_optimizer import ZOptimizer

warnings.filterwarnings("ignore")


# ==================== Distortion implementations ====================


class AttackLibrary:
    """Unified image distortion interface for evaluation.

    Each attack maps float tensors shaped (B, C, H, W) in [0, 1] to attacked
    tensors of the same shape.
    """

    # ---------- Seen attacks: six distortions from the training pool ----------

    @staticmethod
    def jpeg(x, quality=50):
        """Apply JPEG compression to each image using Pillow."""
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
        """Add Gaussian noise."""
        return torch.clamp(x + torch.randn_like(x) * sigma, 0, 1)

    @staticmethod
    def gaussian_blur(x, sigma=3.0):
        """Apply Gaussian blur."""
        kernel_size = int(6 * sigma + 1) | 1  # Ensure an odd kernel size.
        return transforms.GaussianBlur(kernel_size, sigma)(x)

    @staticmethod
    def salt_pepper(x, p=0.3):
        """Add salt-and-pepper noise."""
        mask = torch.rand_like(x)
        result = x.clone()
        result[mask < p / 2] = 0.0
        result[mask > 1 - p / 2] = 1.0
        return result

    @staticmethod
    def median_filter(x, kernel_size=9):
        """Apply a median filter to each image."""
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
        """Randomly set pixels to zero."""
        mask = (torch.rand_like(x) > p).float()
        return x * mask

    # ---------- Unseen attack benchmark group ----------

    @staticmethod
    def mean_filter(x, kernel_size=5):
        """Apply a mean filter."""
        C = x.shape[1]
        kernel = torch.ones(C, 1, kernel_size, kernel_size, device=x.device) / (kernel_size ** 2)
        pad = kernel_size // 2
        return torch.clamp(F.conv2d(x, kernel, padding=pad, groups=C), 0, 1)

    @staticmethod
    def sharpen(x, strength=1.0):
        """Sharpen the image."""
        B, C, H, W = x.shape
        result = torch.zeros_like(x)
        for i in range(B):
            img = x[i].detach().cpu()
            pil_img = transforms.ToPILImage()(img)
            sharp = pil_img.filter(ImageFilter.SHARPEN)
            sharp_t = transforms.ToTensor()(sharp).to(x.device)
            # Blend according to strength.
            result[i] = torch.clamp(x[i] + strength * (sharp_t - x[i]), 0, 1)
        return result

    @staticmethod
    def brightness(x, delta=20):
        """Adjust brightness using a pixel offset on the 0-255 scale."""
        return torch.clamp(x + delta / 255.0, 0, 1)

    @staticmethod
    def contrast(x, factor=0.8):
        """Adjust contrast."""
        mean = x.mean(dim=[2, 3], keepdim=True)
        return torch.clamp(mean + factor * (x - mean), 0, 1)

    @staticmethod
    def poisson_noise(x, lam=30.0):
        """Add Poisson noise."""
        # noisy = torch.poisson(x * lam) / lam
        noisy = torch.poisson(torch.clamp(x * lam,min=0.0)) / lam

        return torch.clamp(noisy, 0, 1)

    @staticmethod
    def webp(x, quality=50):
        """Apply WebP compression."""
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
        """Downsample and resize back to the original dimensions."""
        _, _, H, W = x.shape
        small = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        return F.interpolate(small, size=(H, W), mode='bilinear', align_corners=False)

    @staticmethod
    def rotation(x, angle=5.0):
        """Rotate by a small angle and center-crop to the original dimensions."""
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
        """Center-crop and resize back to the original dimensions."""
        _, _, H, W = x.shape
        ch, cw = int(H * crop_ratio), int(W * crop_ratio)
        top, left = (H - ch) // 2, (W - cw) // 2
        cropped = x[:, :, top:top + ch, left:left + cw]
        return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)

    @staticmethod
    def jpeg2000(x, quality=30):
        """Apply JPEG2000 compression using Pillow's JP2 codec."""
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
        """Apply horizontal motion blur."""
        C = x.shape[1]
        kernel = torch.zeros(C, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
        kernel[:, :, kernel_size // 2, :] = 1.0 / kernel_size
        pad = kernel_size // 2
        return torch.clamp(F.conv2d(x, kernel, padding=pad, groups=C), 0, 1)

    @staticmethod
    def hue_shift(x, shift=0.1):
        """Shift hue for three-channel RGB inputs."""
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
        """Apply multiple attacks in sequence."""
        for fn in fns:
            x = fn(x)
        return x


def get_seen_attacks():
    """Return the seen-attack configurations aligned with the training noise pool."""
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
    """Return the extended unseen-attack benchmark configurations.

    Stronger filters, additional distortion types, and cascaded attacks help
    avoid ceiling effects when comparing the base and Phase 2 networks.
    """
    atk = AttackLibrary
    return OrderedDict([
        # ---- Linear filtering with larger kernels ----
        # ("MeanFilter_7",      lambda x: atk.mean_filter(x, kernel_size=7)),
        # ("MeanFilter_9",      lambda x: atk.mean_filter(x, kernel_size=9)),
        # ("MotionBlur_9",      lambda x: atk.motion_blur(x, kernel_size=9)),
        ("MotionBlur_15",     lambda x: atk.motion_blur(x, kernel_size=15)),

        # ---- Stronger sharpening ----
        # ("Sharpen_1.5",       lambda x: atk.sharpen(x, strength=1.5)),
        ("Sharpen_2.0",       lambda x: atk.sharpen(x, strength=2.0)),

        # ---- Larger brightness and contrast changes ----
        # ("Brightness_-40",    lambda x: atk.brightness(x, delta=-40)),
        ("Brightness_+40",    lambda x: atk.brightness(x, delta=40)),
        # ("Contrast_0.5",      lambda x: atk.contrast(x, factor=0.5)),
        ("Contrast_1.6",      lambda x: atk.contrast(x, factor=1.6)),

        # ---- Additional hue shift ----
        # ("HueShift_0.1",      lambda x: atk.hue_shift(x, shift=0.1)),
        # ("HueShift_0.2",      lambda x: atk.hue_shift(x, shift=0.2)),

        # ---- Stronger Poisson noise: smaller lam produces more noise ----
        # ("Poisson_15",        lambda x: atk.poisson_noise(x, lam=15)),
        ("Poisson_8",         lambda x: atk.poisson_noise(x, lam=8)),

        # ---- Stronger WebP compression ----
        ("WebP_Q20",          lambda x: atk.webp(x, quality=20)),
        # ("WebP_Q10",          lambda x: atk.webp(x, quality=10)),

        # ---- Additional JPEG2000 compression ----
        # ("JPEG2000_Q30",      lambda x: atk.jpeg2000(x, quality=30)),
        # ("JPEG2000_Q15",      lambda x: atk.jpeg2000(x, quality=15)),

        # ---- More aggressive resizing: smaller scale loses more information ----
        ("Resize_0.5x",       lambda x: atk.resize_attack(x, scale=0.5)),
        # ("Resize_0.35x",      lambda x: atk.resize_attack(x, scale=0.35)),
        # ("Resize_0.25x",      lambda x: atk.resize_attack(x, scale=0.25)),

        # ---- Additional geometric transformations ----
        ("Rotation_5deg",     lambda x: atk.rotation(x, angle=5.0)),
        # ("Rotation_10deg",    lambda x: atk.rotation(x, angle=10.0)),
        ("CenterCrop_0.8",    lambda x: atk.center_crop_resize(x, crop_ratio=0.8)),
        # ("CenterCrop_0.7",    lambda x: atk.center_crop_resize(x, crop_ratio=0.7)),

        # ---- Cascaded attacks approximating image processing pipelines ----
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


# ==================== Evaluation helpers ====================


def compute_ber(w_hat, w):
    """Compute the bit error rate (BER)."""
    w_binary = (w > 0.5).float()
    w_hat_binary = (w_hat > 0.5).float()
    return torch.mean(torch.abs(w_binary - w_hat_binary)).item()


def compute_psnr_np(img1, img2):
    """Compute PSNR."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_ssim_torch(img1, img2, window_size=11, sigma=1.5):
    """Compute SSIM for NCHW image tensors with values in [0, 1]."""
    if img1.shape != img2.shape:
        raise ValueError(f"SSIM input shapes differ: {img1.shape} vs {img2.shape}")

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
    """Load a watermark network checkpoint."""
    model = Model(im_size, channel_dim, bit_length, k, min_size, fc).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Use checkpoint model settings when present.
    if 'param_dict' in ckpt:
        info = ckpt['param_dict']
        print(f"  bit_length={info.get('bit_length')}, img_size={info.get('img_size')}, "
              f"k={info.get('k')}, channel_dim={info.get('channel_dim')}")

    return model


def load_score_network(checkpoint_path, device, channel_dim=3):
    """Load a Phase 3 Score Network checkpoint.

    Prefer architecture settings recorded in param_dict, using defaults for
    settings not present in the checkpoint.
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
    print(f"  Score Network parameters: {n_params / 1e6:.2f}M, "
          f"from epoch {ckpt.get('epoch', '?')}")
    return score_net


def evaluate_model(model, dataloader, attacks, device, desc="Evaluating",
                   z_optimizer=None):
    """Evaluate a model under a collection of attacks.

    Use per-image score-guided latent optimization when z_optimizer is given;
    otherwise extract using random latent values. Return per-attack BER and
    accuracy together with stego PSNR and SSIM.
    """
    results = OrderedDict()
    psnr_list = []
    ssim_list = []

    # Collect unattacked stego quality, independent of the latent extraction method.
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

    # Evaluate each attack.
    for atk_name, atk_fn in tqdm(attacks.items(), desc=desc, ncols=100):
        ber_list = []

        for cover, secret in dataloader:
            cover, secret = cover.to(device), secret.to(device)

            # Embedding and attacks do not require gradients.
            with torch.no_grad():
                stego, z = model.forward(cover, secret, True, False)
                stego_quant = torch.round(stego * 255.0) / 255.0
                attacked = atk_fn(stego_quant)

            # Select the watermark extraction path.
            if z_optimizer is not None:
                # Phase 3: optimize z independently for each image.
                B = attacked.shape[0]
                w_hat_list = []
                for i in range(B):
                    _, w_hat_i = z_optimizer.optimize(attacked[i:i + 1])
                    w_hat_list.append(w_hat_i)
                w_hat = torch.cat(w_hat_list, dim=0)
            else:
                # Base extraction path: random z.
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


# ==================== Result display ====================


def print_results_table(results, quality, model_name="Model"):
    """Print a single-model results table."""
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
    """Print a comparison table for two models."""
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


# ==================== Main entry point ====================


def _run_eval_category(cat_name, attacks, model, baseline, z_optimizer,
                       dataloader, device, checkpoint_basename):
    """Evaluate one attack category and print the model comparisons.

    Always evaluate Phase 2 with random z, optionally compare the Phase 1
    baseline, then compare score-guided extraction when a score model is given.
    """
    print(f"\n{'#' * 60}")
    print(f"  {cat_name}")
    print(f"{'#' * 60}")

    # (1) Phase 2 rand-z
    r_randz, q_randz = evaluate_model(
        model, dataloader, attacks, device, f"{cat_name} rand-z")

    # (2) Compare with the Phase 1 baseline.
    if baseline is not None:
        r_base, q_base = evaluate_model(
            baseline, dataloader, attacks, device, f"{cat_name} Phase1")
        print_comparison_table(r_base, q_base, r_randz, q_randz,
                               "Phase1", "Phase2 rand-z")
    else:
        print_results_table(r_randz, q_randz, checkpoint_basename)

    # (3) Compare with Phase 3 latent optimization.
    if z_optimizer is not None:
        r_zopt, q_zopt = evaluate_model(
            model, dataloader, attacks, device,
            f"{cat_name} z-opt", z_optimizer=z_optimizer)
        print_comparison_table(r_randz, q_randz, r_zopt, q_zopt,
                               "Phase2 rand-z", "Phase2 z-opt")


def main():
    parser = argparse.ArgumentParser(description="ALCLO model evaluation")
    parser.add_argument("--checkpoint", type=str, required=False, default="./checkpoints/phase2_R0_baseline/model_63.pth",
                        help="Model checkpoint to evaluate")
    # parser.add_argument("--baseline", type=str, default="./checkpoints/phase1_baseline_lambda_z1e-3/model_486.pth",
    #                     help="Baseline model checkpoint for comparison")
    parser.add_argument("--baseline", type=str, default="../pretrain/crmark_color_size_256_bit_64.pth",
                        help="Baseline model checkpoint for comparison")
    parser.add_argument("--dataset_path", type=str, default="../../DIV2K/val",
    #                     help="Evaluation image directory")
    # parser.add_argument("--dataset_path", type=str, default="../../Kodak24",
    #                     help="Evaluation image directory")
    # parser.add_argument("--dataset_path", type=str, default="../../AID",
    #                     help="Evaluation image directory")
    # parser.add_argument("--dataset_path", type=str, default="../../DiffusionDB",
    #                     help="Evaluation image directory")
    # parser.add_argument("--dataset_path", type=str, default="../../CelebA-HQ",
    #                     help="Evaluation image directory")
    # parser.add_argument("--dataset_path", type=str, default="../../COCO/test",
                        help="Evaluation image directory")
    parser.add_argument("--num_images", type=int, default=100,
                        help="Number of evaluation images (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random sampling seed")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--im_size", type=int, default=256)
    parser.add_argument("--channel_dim", type=int, default=3)
    parser.add_argument("--bit_length", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_size", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--skip_seen", action="store_true",
                        help="Skip the seen-attack group")
    parser.add_argument("--skip_unseen", action="store_true",
                        help="Skip the unseen-attack group")
    parser.add_argument("--skip_phase3", action="store_true",
                        help="Skip Phase 3 score-guided optimization; evaluate Phase 1/2 random-z extraction only")

    # ---- Phase 3: score-guided latent optimization ----
    parser.add_argument("--score_checkpoint", type=str, default="./checkpoints/phase3_hybrid/score_model_61.pth",
                        help="Phase 3 Score Network checkpoint; omit to skip latent optimization")
    parser.add_argument("--z_opt_steps", type=int, default=30,
                        help="Number of latent optimization steps")
    parser.add_argument("--z_opt_lr", type=float, default=0.05,
                        help="Latent optimization learning rate")
    parser.add_argument("--z_opt_sigma", type=float, default=0.05,
                        help="Score Network sigma during latent optimization")
    parser.add_argument("--z_opt_lambda_prior", type=float, default=0.5,
                        help="Latent prior penalty weight")
    parser.add_argument("--z_opt_lambda_ent", type=float, default=0.3,
                        help="Watermark entropy penalty weight")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the evaluation dataset.
    dataset = HideImage(args.dataset_path, args.im_size, args.bit_length, args.channel_dim)
    if args.num_images > 0:
        if hasattr(dataset, "cover"):
            dataset.cover = sorted(dataset.cover)
        total_images = len(dataset)
        num_selected = min(args.num_images, total_images)
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(total_images, generator=generator)[:num_selected].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Random sample: {num_selected}/{total_images} images, seed={args.seed}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(f"Evaluation set: {len(dataset)} images")

    # Load the models.
    print(f"\nLoading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device, args.bit_length, args.im_size,
                       args.channel_dim, args.k, args.min_size)

    baseline = None
    if args.baseline:
        print(f"Loading baseline: {args.baseline}")
        baseline = load_model(args.baseline, device, args.bit_length, args.im_size,
                              args.channel_dim, args.k, args.min_size)

    # ---- Load the optional score network and construct the latent optimizer ----
    z_optimizer = None
    if args.skip_phase3:
        print("\n[Skipping Phase 3] --skip_phase3 is set; the Score Network will not be loaded")
    elif args.score_checkpoint:
        print(f"\nLoading Score Network: {args.score_checkpoint}")
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
        print(f"  z-opt settings: steps={args.z_opt_steps}, lr={args.z_opt_lr}, "
              f"σ={args.z_opt_sigma}, λ_prior={args.z_opt_lambda_prior}, "
              f"λ_ent={args.z_opt_lambda_ent}")

    # Attack configurations.
    seen_attacks = get_seen_attacks() if not args.skip_seen else OrderedDict()
    unseen_attacks = get_unseen_attacks() if not args.skip_unseen else OrderedDict()

    ckpt_name = os.path.basename(args.checkpoint)

    if seen_attacks:
        _run_eval_category("Seen Attacks", seen_attacks,
                           model, baseline, z_optimizer, dataloader, device, ckpt_name)

    if unseen_attacks:
        _run_eval_category("Unseen Attacks -- main comparison", unseen_attacks,
                           model, baseline, z_optimizer, dataloader, device, ckpt_name)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
