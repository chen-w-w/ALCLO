"""
z 分布正则化损失 (z Distribution Regularization Loss)

设计思路:
    DRRW 原始的 Lz = MSE(z, 0) 只约束 z 的均值接近零，
    但不关心 z 的方差和形状。这导致:
    1. z 的分布偏离高斯 → 算术编码的概率模型不匹配 → 辅助比特流变长
    2. z 的分布不规则 → 推理时 z 优化的搜索空间无结构 → 优化效果差

    ALCLO 的 Lz_dist 用三项联合约束 z 逼近 N(0,1):
    - (E[z])^2:       约束均值 → 0 (注意不是 MSE(z,0)=E[z^2], 见下方说明)
    - (std(z)-1)^2:   约束标准差 → 1
    - |skewness(z)|:  约束偏度 → 0 (对称性)

    为什么均值项不能写成 torch.mean(z**2):
        torch.mean(z**2) = E[z^2] = Var(z) + (E[z])^2
        它同时在压 "均值→0" 和 "方差→0", 会和 loss_std 正交冲突.
        正确的纯均值约束应该是 (torch.mean(z))**2 — 先求均值再平方.

    统计量在整个 batch 的所有元素上计算 (全局 scalar 统计量), 对应的假设是
    "z 所有元素整体服从 N(0,1)", 这与 arithmetic coding 对 z 做熵编码时假设的
    全局分布一致.
"""

import torch
import torch.nn as nn


class ZDistributionLoss(nn.Module):
    """
    z 分布正则化损失。

    将 z 推向标准正态分布 N(0, 1)。

    参数:
        lambda_mean: 均值约束权重，默认 1e-3
        lambda_std: 方差约束权重，默认 1e-2
        lambda_skew: 偏度约束权重，默认 1e-3
    """

    def __init__(self, lambda_mean=1e-3, lambda_std=1e-2, lambda_skew=1e-3):
        super().__init__()
        self.lambda_mean = lambda_mean
        self.lambda_std = lambda_std
        self.lambda_skew = lambda_skew

    def forward(self, z):
        """
        参数:
            z: 潜变量 (B, bit_length)

        返回:
            loss: 标量损失
            stats: 字典，包含 z 的统计量 (用于日志)
        """
        # 注意 loss_mean 与 loss_std 必须解耦, 否则会互相打架:
        # 若写成 torch.mean(z**2) = E[z^2] = Var(z) + E[z]^2, 则该项同时在压
        # 均值到 0 和方差到 0, 会与 loss_std (推 std → 1) 产生直接冲突,
        # 导致 z 分布迁移过程反复震荡无法收敛. 因此这里用 (E[z])^2 做"纯均值约束".

        # 均值约束: E[z] → 0, 即 (E[z])^2 → 0, 不影响方差
        z_mean = torch.mean(z)
        loss_mean = z_mean ** 2

        # 标准差约束: std(z) → 1 (在整个 batch 的所有元素上计算)
        z_std = torch.std(z)
        loss_std = (z_std - 1.0) ** 2

        # 偏度约束: skewness(z) → 0
        z_centered = z - z_mean
        skewness = torch.mean(z_centered ** 3) / (z_std ** 3 + 1e-8)
        loss_skew = torch.abs(skewness)

        loss = (self.lambda_mean * loss_mean
                + self.lambda_std * loss_std
                + self.lambda_skew * loss_skew)

        stats = {
            "z_mean": z_mean.item(),
            "z_std": z_std.item(),
            "z_skewness": skewness.item(),
            "loss_z_mean": loss_mean.item(),
            "loss_z_std": loss_std.item(),
            "loss_z_skew": loss_skew.item(),
        }

        return loss, stats