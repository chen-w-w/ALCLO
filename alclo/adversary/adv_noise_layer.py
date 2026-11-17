"""
混合噪声层 (Hybrid Noise Layer)

设计思路:
    在 Phase 2 训练中替换原始的固定噪声池。
    每个 batch 以概率 p_adv 使用对抗攻击器生成的攻击，
    以概率 1-p_adv 使用原始固定噪声池中随机选择的攻击。

    这样做的原因:
    1. 对抗攻击提供"当前模型最薄弱方向"的攻击，提升泛化鲁棒性
    2. 保留固定噪声确保已知攻击的鲁棒性不退化（防止"遗忘"）
    3. p_adv=0.3 是经验最优平衡点

    混合策略是 per-sample 的: batch 中的每张图独立决定用哪种噪声，
    而不是整个 batch 统一选择。这样每个 batch 内同时包含两种攻击，
    梯度信号更丰富，训练更稳定。

Epsilon 调度:
    eps 在训练过程中线性递增（而非阶跃），从 eps_min 到 eps_max。
    渐进式增强避免攻击器在早期就太强导致 iIWN 训练崩溃。
"""

import torch
import torch.nn as nn


def get_eps_schedule(epoch, total_epochs, eps_min=5.0 / 255, eps_max=20.0 / 255):
    """
    线性 epsilon 调度。

    参数:
        epoch: 当前 epoch (从 0 开始)
        total_epochs: Phase 2 总 epoch 数
        eps_min: 初始 epsilon
        eps_max: 最终 epsilon

    返回:
        当前 epoch 的 epsilon 值
    """
    progress = min(epoch / max(total_epochs - 1, 1), 1.0)
    return eps_min + (eps_max - eps_min) * progress


class HybridNoiseLayer(nn.Module):
    """
    混合噪声层: 对抗攻击 + 固定噪声池的概率混合。

    参数:
        attacker: AdversarialAttacker 实例
        fixed_noise_layer: 原始固定噪声层 (watermarklab.DigitalDistortion)
        p_adv: 使用对抗攻击的概率，默认 0.3
    """

    def __init__(self, attacker, fixed_noise_layer, p_adv=0.3):
        super().__init__()
        self.attacker = attacker
        self.fixed_noise_layer = fixed_noise_layer
        self.p_adv = p_adv

    def forward(self, stego, cover, now_step, eps):
        """
        对 stego 施加混合噪声。

        参数:
            stego: 含水印图像 (B, C, H, W)，范围 [0, 1]
            cover: 原始图像 (B, C, H, W)，用于固定噪声层
            now_step: 当前训练步数，传给固定噪声层
            eps: 当前 epsilon 约束

        返回:
            noised: 加噪后的图像 (B, C, H, W)
            is_adv_mask: 布尔掩码 (B,)，标记哪些样本用了对抗攻击
        """
        B = stego.shape[0]
        device = stego.device

        # 为每个样本独立采样: True=对抗攻击, False=固定噪声
        is_adv_mask = torch.rand(B, device=device) < self.p_adv

        # 先用固定噪声处理整个 batch（保证所有样本都有一个基础噪声结果）
        noised_fixed = self.fixed_noise_layer(stego, cover, now_step)

        if is_adv_mask.any():
            # 对抗攻击: 攻击器生成扰动
            # detach stego 是为了让攻击器的梯度不直接回传到 iIWN 的正向传播
            # （攻击器有自己的优化步骤）
            delta = self.attacker(stego.detach(), eps)
            noised_adv = torch.clamp(stego + delta, 0.0, 1.0)

            # per-sample 混合: 将对抗攻击的结果填入对应位置
            mask = is_adv_mask.view(B, 1, 1, 1).float()
            noised = mask * noised_adv + (1 - mask) * noised_fixed
        else:
            noised = noised_fixed

        return noised, is_adv_mask
