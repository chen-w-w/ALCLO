"""Adversarial attacker networks for ALCLO.

The active implementation is the residual CNN attacker. The commented
experimental implementation below adapts the model-parallel ensemble from:

    Huang et al., "Enhancing Robustness in Post-Processing Watermarking: An
    Ensemble Attack Network Using CNNs and Transformers", AISec 2025.
    arXiv: 2509.03006; https://github.com/aiiu-lab/DeepRobustWatermark

That ensemble randomly routes each forward pass to a DCT-Transformer
(probability p_dct, default 0.6) or a two-layer spatial CNN. Its interface
returns delta = clamp(adv_image - stego, -eps, eps) for HybridNoiseLayer.
The reference adaptation implements block DCT/IDCT and zigzag masks in
PyTorch and projects spatial patch features when num_patches differs from
the Transformer dimension. Both implementations are retained for ablations.
"""

import math
import torch
import torch.nn as nn


# ============================================================
# [LEGACY -- ALCLO v1, retained for fallback and ablation comparisons]
# ============================================================
class ResBlock(nn.Module):
    """Residual block: Conv -> ReLU -> Conv with a skip connection."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        return x + self.block(x)


class AdversarialAttacker(nn.Module):
    """Residual CNN attacker with four residual blocks by default.

    Accepts stego images shaped (B, C, H, W) in [0, 1] and returns perturbations
    of the same shape in [-eps, eps]. Tanh followed by epsilon scaling enforces
    the L-infinity perturbation bound.
    """

    def __init__(self, in_channels=3, hidden_channels=64, num_blocks=4):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
        ]
        for _ in range(num_blocks):
            layers.append(ResBlock(hidden_channels))
        layers.append(nn.Conv2d(hidden_channels, in_channels, 3, 1, 1, bias=True))
        self.net = nn.Sequential(*layers)
        self._init_last_layer()

    def _init_last_layer(self):
        last_conv = self.net[-1]
        nn.init.zeros_(last_conv.bias)
        nn.init.normal_(last_conv.weight, 0, 0.05)

    def forward(self, stego, eps):
        raw = self.net(stego)
        delta = torch.tanh(raw) * eps
        return delta


# ============================================================
# [NEW — ALCLO v2] Ensemble Attack Network (Model Parallel)
# ============================================================


# ---------- DCT / YUV utilities (pure PyTorch, no external dependencies) ----------

# def _rgb_to_yuv(img):
#     """ITU-R BT.601 RGB -> YUV for [B,3,H,W] inputs in [0,1]."""
#     r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
#     y = 0.299 * r + 0.587 * g + 0.114 * b
#     u = -0.14713 * r - 0.28886 * g + 0.436 * b
#     v = 0.615 * r - 0.51499 * g - 0.10001 * b
#     return torch.cat([y, u, v], dim=1)
#
#
# def _yuv_to_rgb(img):
#     """YUV -> RGB, the inverse of _rgb_to_yuv."""
#     y, u, v = img[:, 0:1], img[:, 1:2], img[:, 2:3]
#     r = y + 1.13983 * v
#     g = y - 0.39465 * u - 0.58060 * v
#     b = y + 2.03211 * u
#     return torch.cat([r, g, b], dim=1)
#
#
# def _dct_matrix(N, device, dtype):
#     """Build an orthonormal N x N DCT-II basis D: X = D x, x = D^T X."""
#     n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # [1, N]
#     k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # [N, 1]
#     D = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
#     D[0, :] = D[0, :] * math.sqrt(1.0 / N)
#     D[1:, :] = D[1:, :] * math.sqrt(2.0 / N)
#     return D
#
#
# def _block_dct(x, block_size=8):
#     """Apply nonoverlapping block DCT; H/W must be divisible by block_size."""
#     B, C, H, W = x.shape
#     bH, bW = H // block_size, W // block_size
#     blocks = x.reshape(B, C, bH, block_size, bW, block_size)
#     blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()  # [B,C,bH,bW,bs,bs]
#     D = _dct_matrix(block_size, x.device, x.dtype)
#     coeffs = D @ blocks @ D.transpose(-1, -2)
#     return coeffs  # [B,C,bH,bW,bs,bs]
#
#
# def _block_idct(coeffs, H, W, block_size=8):
#     """Invert [B,C,bH,bW,bs,bs] block coefficients into [B,C,H,W]."""
#     D = _dct_matrix(block_size, coeffs.device, coeffs.dtype)
#     blocks = D.transpose(-1, -2) @ coeffs @ D
#     B, C, bH, bW, bs, _ = blocks.shape
#     return blocks.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, C, H, W)
#
#
# def _zigzag_keep_mask(block_size, keep_count):
#     """Keep the first keep_count low-frequency coefficients in JPEG zigzag order."""
#     indices = sorted(
#         ((i, j) for i in range(block_size) for j in range(block_size)),
#         key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1])
#     )
#     mask = torch.zeros(block_size, block_size)
#     for i, j in indices[:keep_count]:
#         mask[i, j] = 1.0
#     return mask
#
#
# # ---------- CNN branch (generator.py in the reference implementation) ----------
#
# class _CNNGenerator(nn.Module):
#     """Spatial CNN attack: Conv(3->16) -> LeakyReLU -> Conv(16->3)."""
#
#     def __init__(self, in_channels=3, hidden=16):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, hidden, 3, 1, 1),
#             nn.LeakyReLU(inplace=False),
#             nn.Conv2d(hidden, in_channels, 3, 1, 1),
#         )
#         # [v3 fix] Zero-initialize the final layer so the initial residual is near zero.
#         # Avoid the strong initial distortion that destabilized iIWN in v2.
#         nn.init.zeros_(self.net[-1].weight)
#         nn.init.zeros_(self.net[-1].bias)
#
#     def forward(self, img):
#         # [v2] Direct adv_image output with random weights caused strong initial distortion.
#         # return self.net(img)
#         # [v3] Zero final layer -> net(img)=0 -> adv approximately img -> delta approximately 0.
#         return img + self.net(img)
#
#
# # ---------- DCT-Transformer branch (reference vit_mask.py) ----------
#
# class _MultiHeadAttention(nn.Module):
#     def __init__(self, dim, heads, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#         self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
#         self.attn_drop = nn.Dropout(dropout)
#
#     def forward(self, x):
#         B, N, _ = x.shape
#         q, k, v = self.to_qkv(x).chunk(3, dim=-1)
#         q = q.reshape(B, N, self.heads, -1).transpose(1, 2)
#         k = k.reshape(B, N, self.heads, -1).transpose(1, 2)
#         v = v.reshape(B, N, self.heads, -1).transpose(1, 2)
#         attn = (q @ k.transpose(-1, -2)) * self.scale
#         attn = self.attn_drop(attn.softmax(dim=-1))
#         out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#         return self.to_out(out)
#
#
# class _TransformerBlock(nn.Module):
#     def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = _MultiHeadAttention(dim, heads, dim_head, dropout)
#         self.norm2 = nn.LayerNorm(dim)
#         self.ff = nn.Sequential(
#             nn.Linear(dim, mlp_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_dim, dim),
#             nn.Dropout(dropout),
#         )
#
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ff(self.norm2(x))
#         return x
#
#
# class _DCTTransformer(nn.Module):
#     """DCT-Transformer attack adapted from model/vit_mask.py.
#
#     Pipeline:
#         RGB -> YUV -> 8 x 8 block DCT -> zigzag mask (retain 25/9/9 for Y/U/V) ->
#         reshape to [B, 192, num_patches] with frequency-channel tokens and spatial patches
#         -> optional Linear projection to dim -> Transformer(depth=6, heads=12) ->
#         inverse projection -> DCT coefficients -> block IDCT -> YUV -> RGB
#
#     Args:
#         image_size: Square input size, divisible by patch_size.
#         patch_size: DCT block size; default 8 matches JPEG and the reference.
#         dim/depth/heads/mlp_dim/dim_head: Transformer settings (reference: 256/6/12/256/64).
#         yuv_keep: Retained low-frequency counts for Y/U/V (reference: 25/9/9).
#     """
#
#     def __init__(self, image_size=256, patch_size=8, dim=256, depth=6, heads=12,
#                  mlp_dim=256, dim_head=64, dropout=0., yuv_keep=(25, 9, 9)):
#         super().__init__()
#         assert image_size % patch_size == 0
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.num_patches = (image_size // patch_size) ** 2  # 128 -> 256; 256 -> 1024.
#         self.num_freq_tokens = 3 * patch_size * patch_size  # 192 in the reference.
#
#         # At 128 x 128, dim=num_patches=256; at 256 x 256, use paired Linear projections.
#         # Bound parameter count while preserving token semantics.
#         self.proj_in = nn.Linear(self.num_patches, dim)
#         self.proj_out = nn.Linear(dim, self.num_patches)
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_freq_tokens, dim))
#         self.dropout = nn.Dropout(dropout)
#         self.blocks = nn.ModuleList([
#             _TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
#             for _ in range(depth)
#         ])
#         # Final 3 x 3 refinement convolution, zero-initialized as in the residual design.
#         self.refine = nn.Conv2d(3, 3, 3, padding=1)
#         nn.init.zeros_(self.refine.weight)
#         nn.init.zeros_(self.refine.bias)
#         # [v3 fix] Also zero-initialize proj_out so the main Transformer signal starts at zero.
#         # With residual forward, initial adv is approximately img and delta approximately zero.
#         # In v2 only refine was zero-initialized; proj_out -> IDCT -> YUV-to-RGB remained random.
#         # Initial PSNR was about 23 dB, destabilizing iIWN when unfrozen at epoch 11.
#         nn.init.zeros_(self.proj_out.weight)
#         nn.init.zeros_(self.proj_out.bias)
#
#         # Precompute JPEG zigzag masks [3, 1, 1, ps, ps], broadcast over
#         # DCT coefficients shaped [B, 3, bH, bW, ps, ps].
#         masks = torch.stack([
#             _zigzag_keep_mask(patch_size, keep) for keep in yuv_keep
#         ])  # [3, ps, ps]
#         self.register_buffer("dct_mask",
#                              masks.unsqueeze(1).unsqueeze(1))  # [3, 1, 1, ps, ps]
#
#     def forward(self, img):
#         B, C, H, W = img.shape
#         assert C == 3, "DCT-Transformer only supports RGB"
#         ps = self.patch_size
#
#         # 1. RGB → YUV
#         yuv = _rgb_to_yuv(img)
#         # 2. block DCT → [B, 3, bH, bW, ps, ps]
#         coeffs = _block_dct(yuv, ps)
#         # 3. Retain a channel-specific number of low-frequency zigzag coefficients.
#         coeffs = coeffs * self.dct_mask.unsqueeze(0)
#         bH, bW = coeffs.shape[2], coeffs.shape[3]
#         # 4. tokenize → [B, 192, num_patches]
#         # Each of the 192 frequency-channel tokens has num_patches spatial features.
#         tokens = coeffs.permute(0, 1, 4, 5, 2, 3).reshape(
#             B, self.num_freq_tokens, bH * bW)
#         # 5. Projection and positional embeddings -> Transformer.
#         x = self.proj_in(tokens) + self.pos_embed
#         x = self.dropout(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.proj_out(x)  # [B, 192, num_patches]
#         # 6. Detokenize into DCT coefficients [B, 3, bH, bW, ps, ps].
#         x = x.reshape(B, 3, ps, ps, bH, bW).permute(0, 1, 4, 5, 2, 3).contiguous()
#         # 7. block IDCT → YUV → RGB
#         yuv_out = _block_idct(x, H, W, ps)
#         rgb_out = _yuv_to_rgb(yuv_out)
#         rgb_out = rgb_out + self.refine(rgb_out)
#         # [v2] return rgb_out  # Direct random adv_image output gave initial PSNR near 23 dB.
#         # [v3] Zero-initialize proj_out and refine so rgb_out starts at zero.
#         #      → adv = img + 0 = img → delta = clamp(0, -eps, eps) = 0
#         # Start from the identity mapping and learn attacks gradually to stabilize iIWN.
#         return img + rgb_out
#
#
# # ---------- Model Parallel (reference model/parallel.py) ----------
#
# class AdversarialAttacker(nn.Module):
#     """Model-parallel ensemble, the highest average bit-accuracy configuration in Table 3.
#
#     Route each forward pass to DCT-Transformer with probability p_dct (default 0.6), else CNN.
#     Return delta = clamp(adv_image - stego, -eps, eps), preserving the ALCLO L-infinity bound.
#
#     Args:
#         in_channels: Input channels; default 3 (RGB).
#         image_size: Input image size; default 256 for ALCLO.
#         patch_size: DCT block size; default 8 matches JPEG.
#         dim/depth/heads/mlp_dim/dim_head: Transformer settings (reference: 256/6/12/256/64).
#         p_dct: DCT-Transformer routing probability; default 0.6.
#         yuv_keep: Retained low-frequency counts per channel; default (25, 9, 9).
#     """
#
#     def __init__(self, in_channels=3, image_size=256, patch_size=8,
#                  dim=256, depth=6, heads=12, mlp_dim=256, dim_head=64,
#                  p_dct=0.6, yuv_keep=(25, 9, 9)):
#         super().__init__()
#         assert in_channels == 3, "Ensemble attacker currently requires RGB input"
#         self.dct_transformer = _DCTTransformer(
#             image_size=image_size, patch_size=patch_size,
#             dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
#             dim_head=dim_head, yuv_keep=yuv_keep,
#         )
#         self.cnn = _CNNGenerator(in_channels=in_channels)
#         self.p_dct = p_dct
#
#     def forward(self, stego, eps):
#         """Generate adversarial perturbations delta in [-eps, eps].
#
#         Randomly select one branch per forward pass, matching single-path MP routing.
#         Explicitly clamp the L-infinity bound outside the branch, unlike v1 tanh scaling.
#         """
#         if torch.rand(1, device=stego.device).item() < self.p_dct:
#             adv = self.dct_transformer(stego)
#         else:
#             adv = self.cnn(stego)
#         delta = torch.clamp(adv - stego, -eps, eps)
#         return delta