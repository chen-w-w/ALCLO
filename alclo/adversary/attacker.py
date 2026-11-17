"""
对抗攻击器网络 (Adversarial Attacker Network)

[2026-05-XX 升级 v2] 将攻击器从 4-ResBlock CNN 替换为 **Ensemble Attack Network
(Model Parallel)**, 参考论文:

    Huang et al., "Enhancing Robustness in Post-Processing Watermarking: An
    Ensemble Attack Network Using CNNs and Transformers", AISec 2025
    arXiv: 2509.03006   repo: https://github.com/aiiu-lab/DeepRobustWatermark

论文在 4 种 ensemble 配置 (Cascade / Parallel / Random Blend / Aggregate Blend) 中,
**Model Parallel (MP) 平均 bit accuracy 最高**, 因此本实现采用 MP:
    每次 forward 以概率 p_dct (默认 0.6) 用 DCT-Transformer (频域 ViT) 攻击,
    否则用 CNN (空域 2 层卷积) 攻击.

与论文/官方 repo 的差异 (做了最小化适配, 语义保持一致):
1. **接口对齐 ALCLO**: 论文输出 adv_image (整张攻击后图), 这里返回
   delta = clamp(adv_image - stego, -eps, eps), 让 HybridNoiseLayer 沿用
   stego + delta 的现有接口与 eps 渐进调度.
2. **去依赖**: 原版用 `torchjpeg.dct` 和 `einops`, 这里用纯 PyTorch 自实现
   8×8 block DCT/IDCT 与 zigzag mask, 避免引入额外依赖.
3. **尺寸自适应**: 论文默认 image_size=128 时 num_patches=256 与 transformer
   dim=256 等长. 我们 256×256 输入时 num_patches=1024, 引入一对 Linear
   投影 (num_patches ↔ dim) 控制参数量, 保留 192 频率 token 的 ViT 语义.

旧实现 (ALCLO v1, 4-ResBlock CNN attacker) 整体注释保留在下方, 方便 ablation 回退.
"""

import math
import torch
import torch.nn as nn


# ============================================================
# [LEGACY — ALCLO v1, 已停用, 保留供回退 / ablation 对照]
# ============================================================
class ResBlock(nn.Module):
    """残差块: Conv → ReLU → Conv + skip connection"""

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
    """
    对抗攻击器网络 (v1: 4-ResBlock CNN).

    输入: stego 图像 (B, C, H, W)，像素范围 [0, 1]
    输出: 扰动 delta (B, C, H, W)，范围 [-eps, eps]

    通过 tanh 激活 + eps 缩放硬性保证 L_inf 约束。
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


# ---------- DCT / YUV 工具 (纯 PyTorch, 无外部依赖) ----------

# def _rgb_to_yuv(img):
#     """ITU-R BT.601 RGB → YUV. 输入 [B,3,H,W], 范围 [0,1]."""
#     r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
#     y = 0.299 * r + 0.587 * g + 0.114 * b
#     u = -0.14713 * r - 0.28886 * g + 0.436 * b
#     v = 0.615 * r - 0.51499 * g - 0.10001 * b
#     return torch.cat([y, u, v], dim=1)
#
#
# def _yuv_to_rgb(img):
#     """YUV → RGB (与 _rgb_to_yuv 严格互逆)."""
#     y, u, v = img[:, 0:1], img[:, 1:2], img[:, 2:3]
#     r = y + 1.13983 * v
#     g = y - 0.39465 * u - 0.58060 * v
#     b = y + 2.03211 * u
#     return torch.cat([r, g, b], dim=1)
#
#
# def _dct_matrix(N, device, dtype):
#     """生成 N×N 的 1D DCT-II 正交基矩阵 D. 满足 X = D x, x = D^T X."""
#     n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # [1, N]
#     k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # [N, 1]
#     D = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
#     D[0, :] = D[0, :] * math.sqrt(1.0 / N)
#     D[1:, :] = D[1:, :] * math.sqrt(2.0 / N)
#     return D
#
#
# def _block_dct(x, block_size=8):
#     """对 [B,C,H,W] 做非重叠 block×block 的 2D-DCT. H/W 必须可整除 block_size."""
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
#     """对 [B,C,bH,bW,bs,bs] DCT 系数做逆块 DCT 并拼回 [B,C,H,W]."""
#     D = _dct_matrix(block_size, coeffs.device, coeffs.dtype)
#     blocks = D.transpose(-1, -2) @ coeffs @ D
#     B, C, bH, bW, bs, _ = blocks.shape
#     return blocks.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, C, H, W)
#
#
# def _zigzag_keep_mask(block_size, keep_count):
#     """JPEG zigzag 顺序保留前 keep_count 个低频系数的 mask [block_size, block_size]."""
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
# # ---------- CNN 分支 (论文 generator.py) ----------
#
# class _CNNGenerator(nn.Module):
#     """空域 CNN 攻击 (论文 model/generator.py): Conv(3→16) → LeakyReLU → Conv(16→3)."""
#
#     def __init__(self, in_channels=3, hidden=16):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, hidden, 3, 1, 1),
#             nn.LeakyReLU(inplace=False),
#             nn.Conv2d(hidden, in_channels, 3, 1, 1),
#         )
#         # [v3 修正] 末层零初始化, 配合下方 residual forward, 初始 delta ≈ 0
#         # 避免 v2 训练中 attacker 第 0 步就把 PSNR 打到 23 dB 导致 iIWN 崩
#         nn.init.zeros_(self.net[-1].weight)
#         nn.init.zeros_(self.net[-1].bias)
#
#     def forward(self, img):
#         # [v2] 直接输出 adv_image; 初始随机权重导致扰动强, PSNR 立刻崩
#         # return self.net(img)
#         # [v3] residual: net 末层零初始化 → 初始 net(img)=0 → adv≈img → delta≈0
#         return img + self.net(img)
#
#
# # ---------- DCT-Transformer 分支 (论文 vit_mask.py) ----------
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
#     """DCT-Transformer 攻击 (论文 model/vit_mask.py 复刻).
#
#     工作流:
#         RGB → YUV → 8×8 block DCT → JPEG zigzag mask (Y/U/V 各保留 25/9/9) →
#         重排为 [B, 192, num_patches] (192 个频率-通道 token, num_patches 个空间块)
#         → (可选) Linear 投影到 dim → Transformer(depth=6, heads=12) →
#         反投影 → 重排回 DCT 系数 → block IDCT → YUV → RGB
#
#     Args:
#         image_size: 输入图像尺寸 (H=W, 须可被 patch_size 整除)
#         patch_size: DCT 块大小, 默认 8 (与 JPEG 对齐, 论文默认)
#         dim/depth/heads/mlp_dim/dim_head: Transformer 超参 (论文默认 256/6/12/256/64)
#         yuv_keep: Y/U/V 各通道 zigzag 保留低频系数数 (论文默认 25/9/9)
#     """
#
#     def __init__(self, image_size=256, patch_size=8, dim=256, depth=6, heads=12,
#                  mlp_dim=256, dim_head=64, dropout=0., yuv_keep=(25, 9, 9)):
#         super().__init__()
#         assert image_size % patch_size == 0
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.num_patches = (image_size // patch_size) ** 2  # 论文 128: 256; 256: 1024
#         self.num_freq_tokens = 3 * patch_size * patch_size  # 论文恒为 192
#
#         # 论文 128×128 时 dim=num_patches=256 直通; 256×256 时引入一对 Linear 投影
#         # 控制参数量, 不改变 token 语义
#         self.proj_in = nn.Linear(self.num_patches, dim)
#         self.proj_out = nn.Linear(dim, self.num_patches)
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_freq_tokens, dim))
#         self.dropout = nn.Dropout(dropout)
#         self.blocks = nn.ModuleList([
#             _TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
#             for _ in range(depth)
#         ])
#         # 末端再过一层 3×3 Conv (论文 return_patch_embedding 末尾的 Conv2d), 零初始化
#         self.refine = nn.Conv2d(3, 3, 3, padding=1)
#         nn.init.zeros_(self.refine.weight)
#         nn.init.zeros_(self.refine.bias)
#         # [v3 修正] proj_out 也零初始化, 让 Transformer 主信号路径初始全 0,
#         # 配合下方 residual forward 实现 "初始 adv ≈ img → delta ≈ 0".
#         # v2 中只有 refine 零初始化, 但 proj_out → idct → yuv2rgb 这条主路径
#         # 初始随机, 导致初始扰动 PSNR ~23 dB, ep11 解冻时 iIWN 立刻被打崩.
#         nn.init.zeros_(self.proj_out.weight)
#         nn.init.zeros_(self.proj_out.bias)
#
#         # 预计算 JPEG zigzag mask. shape [3, 1, 1, ps, ps], broadcast 到
#         # DCT 系数 [B, 3, bH, bW, ps, ps]
#         masks = torch.stack([
#             _zigzag_keep_mask(patch_size, keep) for keep in yuv_keep
#         ])  # [3, ps, ps]
#         self.register_buffer("dct_mask",
#                              masks.unsqueeze(1).unsqueeze(1))  # [3, 1, 1, ps, ps]
#
#     def forward(self, img):
#         B, C, H, W = img.shape
#         assert C == 3, "DCT-Transformer 仅支持 RGB"
#         ps = self.patch_size
#
#         # 1. RGB → YUV
#         yuv = _rgb_to_yuv(img)
#         # 2. block DCT → [B, 3, bH, bW, ps, ps]
#         coeffs = _block_dct(yuv, ps)
#         # 3. JPEG zigzag mask (按通道保留不同数量的低频)
#         coeffs = coeffs * self.dct_mask.unsqueeze(0)
#         bH, bW = coeffs.shape[2], coeffs.shape[3]
#         # 4. tokenize → [B, 192, num_patches]
#         #    192 个频率-通道 token, 每个 token 的 "feature" 是 num_patches 个空间块
#         tokens = coeffs.permute(0, 1, 4, 5, 2, 3).reshape(
#             B, self.num_freq_tokens, bH * bW)
#         # 5. 投影 + pos_embed → Transformer
#         x = self.proj_in(tokens) + self.pos_embed
#         x = self.dropout(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.proj_out(x)  # [B, 192, num_patches]
#         # 6. detokenize → DCT 系数 [B, 3, bH, bW, ps, ps]
#         x = x.reshape(B, 3, ps, ps, bH, bW).permute(0, 1, 4, 5, 2, 3).contiguous()
#         # 7. block IDCT → YUV → RGB
#         yuv_out = _block_idct(x, H, W, ps)
#         rgb_out = _yuv_to_rgb(yuv_out)
#         rgb_out = rgb_out + self.refine(rgb_out)
#         # [v2] return rgb_out  # 直接输出 adv_image, 初始随机 → PSNR 23 dB
#         # [v3] residual: proj_out + refine 双零初始化 → rgb_out 初始 0
#         #      → adv = img + 0 = img → delta = clamp(0, -eps, eps) = 0
#         #      让 attacker 从恒等映射开始, 慢慢学攻击, 不会一上来打崩 iIWN
#         return img + rgb_out
#
#
# # ---------- Model Parallel (论文 model/parallel.py) ----------
#
# class AdversarialAttacker(nn.Module):
#     """Model Parallel 集成攻击器 — 论文表 3 中 average bit accuracy 最高的配置.
#
#     每次 forward 以概率 p_dct (默认 0.6) 用 DCT-Transformer 攻击, 否则用 CNN 攻击.
#     输出 delta = clamp(adv_image - stego, -eps, eps), 保留 ALCLO 的 L_inf 约束.
#
#     Args:
#         in_channels:   输入通道, 默认 3 (RGB)
#         image_size:    输入图像尺寸, 默认 256 (ALCLO 标准配置)
#         patch_size:    DCT 块大小, 默认 8 (与 JPEG 对齐, 论文默认)
#         dim/depth/heads/mlp_dim/dim_head: Transformer 超参 (论文默认 256/6/12/256/64)
#         p_dct:         选用 DCT-Transformer 的概率, 默认 0.6 (论文 parallel.py 默认)
#         yuv_keep:      JPEG zigzag 各通道保留低频系数数, 默认 (25, 9, 9) (论文默认)
#     """
#
#     def __init__(self, in_channels=3, image_size=256, patch_size=8,
#                  dim=256, depth=6, heads=12, mlp_dim=256, dim_head=64,
#                  p_dct=0.6, yuv_keep=(25, 9, 9)):
#         super().__init__()
#         assert in_channels == 3, "Ensemble Attacker 当前要求 RGB 输入"
#         self.dct_transformer = _DCTTransformer(
#             image_size=image_size, patch_size=patch_size,
#             dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
#             dim_head=dim_head, yuv_keep=yuv_keep,
#         )
#         self.cnn = _CNNGenerator(in_channels=in_channels)
#         self.p_dct = p_dct
#
#     def forward(self, stego, eps):
#         """生成 [-eps, eps] 范围的对抗扰动 delta.
#
#         随机路由每次走一条分支 (论文 MP 的单路径语义).
#         L_inf 约束在外层显式 clamp, 区别于 v1 的 tanh*eps 软约束.
#         """
#         if torch.rand(1, device=stego.device).item() < self.p_dct:
#             adv = self.dct_transformer(stego)
#         else:
#             adv = self.cnn(stego)
#         delta = torch.clamp(adv - stego, -eps, eps)
#         return delta