"""
对抗攻击器损失函数 (Adversarial Attacker Loss)

设计思路:
    攻击器的目标是"在约束内最大化水印提取误差"，即:
        max_phi  MSE(w_hat_adv, w)
        s.t.     PSNR(stego + delta, stego) >= tau_psnr

    实现为最小化问题:
        min_phi  -MSE(w_hat_adv, w) + lambda_constraint * max(0, tau_psnr - PSNR)

    其中:
    - 第一项: 最大化水印误差（取负号变成最小化）
    - 第二项: PSNR 约束的软惩罚。当 PSNR < tau_psnr 时产生正的惩罚，
      迫使攻击器不要过度破坏图像质量

    为什么用软约束而非硬截断:
    硬截断（如直接 clamp delta 到满足 PSNR 的范围）会切断梯度流，
    攻击器无法从"越界"中学到教训。软约束保持梯度连续，
    攻击器能平滑地学会在约束边界附近操作。
"""

import torch
import torch.nn as nn


def compute_psnr_batch(img1, img2):
    """
    计算 batch 内每张图的 PSNR。

    参数:
        img1, img2: (B, C, H, W)，范围 [0, 1]

    返回:
        psnr: (B,) 每张图的 PSNR (dB)
    """
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    # 避免除零: mse=0 时 PSNR 为无穷大，用一个很大的值代替
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    return psnr


class AdversarialLoss(nn.Module):
    """
    攻击器损失: 最大化水印误差 + PSNR 约束惩罚。

    参数:
        tau_psnr: PSNR 下限 (dB)，默认 25.0
        lambda_constraint: 约束惩罚权重，默认 10.0
    """

    def __init__(self, tau_psnr=25.0, lambda_constraint=10.0):
        super().__init__()
        self.tau_psnr = tau_psnr
        self.lambda_constraint = lambda_constraint

    def forward(self, w_hat_adv, w, stego, noised_adv):
        """
        参数:
            w_hat_adv: 从对抗攻击后提取的水印 (B_adv, bit_length)
            w: 原始水印 (B_adv, bit_length)
            stego: 原始 stego 图像 (B_adv, C, H, W)
            noised_adv: 攻击后图像 (B_adv, C, H, W)

        返回:
            loss: 攻击器总损失 (标量)
            stats: 字典，包含攻击统计量
        """
        # 水印误差 (取负号: 攻击器要最大化误差，优化器做最小化)
        watermark_error = torch.mean((w_hat_adv - w) ** 2)
        loss_attack = -watermark_error

        # PSNR 约束惩罚
        psnr_values = compute_psnr_batch(stego, noised_adv)
        # 当 PSNR < tau_psnr 时产生惩罚
        psnr_violation = torch.clamp(self.tau_psnr - psnr_values, min=0.0)
        loss_constraint = torch.mean(psnr_violation)

        loss = loss_attack + self.lambda_constraint * loss_constraint

        stats = {
            "attacker_watermark_error": watermark_error.item(),
            "attacker_psnr_mean": psnr_values.mean().item(),
            "attacker_psnr_min": psnr_values.min().item(),
            "attacker_psnr_violation": loss_constraint.item(),
        }

        return loss, stats
