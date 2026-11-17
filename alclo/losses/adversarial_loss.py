"""Adversarial attacker loss with a soft PSNR constraint.

The attacker maximizes watermark extraction MSE while limiting image
distortion. The minimized objective is:

    -MSE(w_hat_adv, w) + lambda_constraint * max(0, tau_psnr - PSNR)

The PSNR penalty is positive below tau_psnr and preserves a gradient signal
that discourages excessive distortion.
"""

import torch
import torch.nn as nn


def compute_psnr_batch(img1, img2):
    """Return per-image PSNR in dB for (B, C, H, W) tensors in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    # Avoid division by zero; use a large finite PSNR when MSE is zero.
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    return psnr


class AdversarialLoss(nn.Module):
    """Maximize watermark error while penalizing violations of a PSNR floor.

    Args:
        tau_psnr: Minimum desired PSNR in dB; defaults to 25.0.
        lambda_constraint: PSNR penalty weight; defaults to 10.0.
    """

    def __init__(self, tau_psnr=25.0, lambda_constraint=10.0):
        super().__init__()
        self.tau_psnr = tau_psnr
        self.lambda_constraint = lambda_constraint

    def forward(self, w_hat_adv, w, stego, noised_adv):
        """Return the scalar attacker loss and a dictionary of attack statistics.

        Args:
            w_hat_adv: Extracted watermark, shaped (B_adv, bit_length).
            w: Ground-truth watermark of the same shape.
            stego: Unattacked stego images, shaped (B_adv, C, H, W).
            noised_adv: Attacked images of the same shape.
        """
        # Negate watermark MSE because the attacker maximizes error while the optimizer minimizes loss.
        watermark_error = torch.mean((w_hat_adv - w) ** 2)
        loss_attack = -watermark_error

        # PSNR constraint penalty.
        psnr_values = compute_psnr_batch(stego, noised_adv)
        # Penalize attacks whose PSNR falls below tau_psnr.
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
