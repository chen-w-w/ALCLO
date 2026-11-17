"""Latent distribution regularization for ALCLO.

Three terms encourage z to approach a standard normal distribution:
    (E[z])**2: Push the mean toward zero.
    (std(z) - 1)**2: Push the standard deviation toward one.
    abs(skewness(z)): Encourage symmetry.

The mean term must square the mean, rather than average squared values:
E[z**2] = Var(z) + E[z]**2 would also shrink the variance and conflict with
the standard-deviation objective. Statistics are computed over all elements
of the batch, matching the global distribution used for arithmetic coding.
Regularizing the distribution also structures the latent optimization space.
"""

import torch
import torch.nn as nn


class ZDistributionLoss(nn.Module):
    """Regularize z toward N(0, 1) using mean, standard deviation, and skewness.

    Args:
        lambda_mean: Mean penalty weight; defaults to 1e-3.
        lambda_std: Standard-deviation penalty weight; defaults to 1e-2.
        lambda_skew: Skewness penalty weight; defaults to 1e-3.
    """

    def __init__(self, lambda_mean=1e-3, lambda_std=1e-2, lambda_skew=1e-3):
        super().__init__()
        self.lambda_mean = lambda_mean
        self.lambda_std = lambda_std
        self.lambda_skew = lambda_skew

    def forward(self, z):
        """Return a scalar distribution loss and latent statistics for logging.

        Args:
            z: Latent tensor, typically shaped (B, bit_length).
        """
        # Decouple the mean and standard-deviation objectives:
        # mean(z**2) = E[z**2] = Var(z) + E[z]**2 would shrink both
        # the mean and variance toward zero, conflicting with the target std of one.
        # Use E[z]**2 to constrain only the mean and avoid oscillatory distribution updates.

        # Mean constraint: E[z] -> 0 without directly shrinking the variance.
        z_mean = torch.mean(z)
        loss_mean = z_mean ** 2

        # Standard-deviation constraint: std(z) -> 1 across all elements in the batch.
        z_std = torch.std(z)
        loss_std = (z_std - 1.0) ** 2

        # Skewness constraint: skewness(z) -> 0.
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