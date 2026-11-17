"""
Score network for ALCLO Phase 3.

The network predicts an image correction direction from I_rec toward I_cover.
ZOptimizer freezes it during lossy extraction and uses the predicted correction
as a learned prior for updating z.

Training uses supervised restoration:
    correction = score_net(I_rec, sigma)
    I_pred = clamp(I_rec + correction, 0, 1)
    L_restore = MSE(I_pred, I_cover)

The name ScoreNetwork is kept for compatibility with the existing pipeline, but
the output now means a correction vector, not a DSM score field.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmaEmbedding(nn.Module):
    """Encode sigma as a condition vector for the correction network."""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, sigma):
        """Return embedding e(sigma) with shape [B, embed_dim]."""
        if sigma.dim() > 1:
            sigma = sigma.squeeze(-1)

        # Use log(sigma) because sigma spans multiple noise scales.
        log_sigma = torch.log(sigma + 1e-8)
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=sigma.device, dtype=sigma.dtype) / half_dim
        )
        args = log_sigma[:, None] * freqs[None, :]
        embed = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(embed)


class ResBlock(nn.Module):
    """Residual block with additive sigma conditioning."""

    def __init__(self, channels, embed_dim=256, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.sigma_proj = nn.Linear(embed_dim, channels)

        # Start close to identity for stable score training.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, sigma_embed):
        """Apply residual block to x conditioned on sigma_embed."""
        h = self.norm1(x)
        h = F.silu(h)
        # Inject sigma condition as a channel bias.
        h = h + self.sigma_proj(sigma_embed)[:, :, None, None]
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class DownBlock(nn.Module):
    """ResBlock followed by stride-2 downsampling."""

    def __init__(self, in_ch, out_ch, embed_dim=256, num_groups=8):
        super().__init__()
        self.res = ResBlock(in_ch, embed_dim, num_groups)
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, sigma_embed):
        h = self.res(x, sigma_embed)
        return h, self.down(h)


class UpBlock(nn.Module):
    """Upsample, concatenate skip feature, then apply ResBlock."""

    def __init__(self, in_ch, skip_ch, out_ch, embed_dim=256, num_groups=8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.conv_reduce = nn.Conv2d(in_ch + skip_ch, out_ch, 1)
        self.res = ResBlock(out_ch, embed_dim, num_groups)

    def forward(self, x, skip, sigma_embed):
        h = self.up(x)
        # Align odd input sizes before concatenating skip features.
        if h.shape != skip.shape:
            h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, skip], dim=1)
        h = self.conv_reduce(h)
        h = self.res(h, sigma_embed)
        return h


class ScoreNetwork(nn.Module):
    """Lightweight U-Net predicting a correction direction for I_rec."""

    def __init__(self, in_channels=3, base_channels=64, channel_mults=(1, 2, 4, 4),
                 embed_dim=256, num_groups=8):
        super().__init__()
        self.sigma_embed = SigmaEmbedding(embed_dim)

        channels = [base_channels * m for m in channel_mults]
        # image channels -> base feature channels
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, 1, 1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                DownBlock(channels[i], channels[i + 1], embed_dim, num_groups)
            )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(channels[-1], embed_dim, num_groups),
            ResBlock(channels[-1], embed_dim, num_groups),
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_blocks.append(
                UpBlock(channels[i], channels[i - 1], channels[i - 1], embed_dim, num_groups)
            )

        # base feature channels -> image-shaped correction
        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_groups, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 3, 1, 1),
        )

        # Start with near-zero corrections.
        nn.init.zeros_(self.output_conv[-1].weight)
        nn.init.zeros_(self.output_conv[-1].bias)

    def forward(self, x, sigma):
        """Predict image-shaped correction vector."""
        # Normalize scalar / batched sigma input.
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma], device=x.device, dtype=x.dtype).expand(x.shape[0])
        elif sigma.dim() == 0:
            sigma = sigma.unsqueeze(0).expand(x.shape[0])

        sigma_embed = self.sigma_embed(sigma)

        # Encoder
        h = self.input_conv(x)
        skips = []
        for down_block in self.down_blocks:
            skip, h = down_block(h, sigma_embed)
            skips.append(skip)

        # Bottleneck blocks need sigma conditioning.
        for block in self.bottleneck:
            h = block(h, sigma_embed)

        # Decoder
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            h = up_block(h, skip, sigma_embed)

        correction = self.output_conv(h)
        return correction


def restoration_loss(score_net, I_rec, cover, sigma=0.01):
    """Supervised correction loss from I_rec toward cover.

    Formula:
        correction = score_net(I_rec, sigma)
        I_pred = clamp(I_rec + correction, 0, 1)
        L_restore = MSE(I_pred, cover)
    """
    correction = score_net(I_rec, sigma)
    I_pred = torch.clamp(I_rec + correction, 0.0, 1.0)

    loss = F.mse_loss(I_pred, cover)
    l1 = F.l1_loss(I_pred, cover)
    mse = F.mse_loss(I_pred, cover)
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse.detach(), min=1e-12))

    stats = {
        'restore_loss': loss.item(),
        'restore_l1': l1.item(),
        'restore_mse': mse.item(),
        'restore_psnr': psnr.item(),
        'correction_norm': correction.detach().flatten(1).norm(dim=1).mean().item(),
    }

    return loss, stats
