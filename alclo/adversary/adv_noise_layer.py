"""Hybrid noise layers for Phase 2 training.

Each sample independently uses an adversarial attack with probability p_adv,
or a random attack from the fixed noise pool otherwise. Adversarial examples
target the current model's weaknesses, while fixed distortions maintain
robustness to known attacks. The default mixture probability is 0.3.

Epsilon increases linearly from eps_min to eps_max to avoid destabilizing
the watermark network with strong attacks at the start of training.
"""

import torch
import torch.nn as nn


def get_eps_schedule(epoch, total_epochs, eps_min=5.0 / 255, eps_max=20.0 / 255):
    """Return the linearly scheduled epsilon for a zero-based epoch.

    Args:
        epoch: Current zero-based epoch.
        total_epochs: Total number of Phase 2 epochs.
        eps_min: Initial perturbation bound.
        eps_max: Final perturbation bound.
    """
    progress = min(epoch / max(total_epochs - 1, 1), 1.0)
    return eps_min + (eps_max - eps_min) * progress


class HybridNoiseLayer(nn.Module):
    """Mix adversarial and fixed distortions independently for each sample.

    Args:
        attacker: AdversarialAttacker instance.
        fixed_noise_layer: Fixed distortion module.
        p_adv: Probability of using an adversarial attack; defaults to 0.3.
    """

    def __init__(self, attacker, fixed_noise_layer, p_adv=0.3):
        super().__init__()
        self.attacker = attacker
        self.fixed_noise_layer = fixed_noise_layer
        self.p_adv = p_adv

    def forward(self, stego, cover, now_step, eps):
        """Apply hybrid noise to a batch of stego images.

        Args:
            stego: Watermarked images, shaped (B, C, H, W), in [0, 1].
            cover: Original images used by the fixed noise layer.
            now_step: Training step passed to the fixed noise layer.
            eps: Current perturbation bound.

        Returns:
            The distorted images and a boolean (B,) mask of adversarial samples.
        """
        B = stego.shape[0]
        device = stego.device

        # Sample independently: True selects adversarial noise; False selects fixed noise.
        is_adv_mask = torch.rand(B, device=device) < self.p_adv

        # Apply fixed noise to the entire batch to provide a result for every sample.
        noised_fixed = self.fixed_noise_layer(stego, cover, now_step)

        if is_adv_mask.any():
            # Generate adversarial perturbations.
            # Detach stego to keep attacker gradients out of the iIWN forward graph.
            # The attacker has a separate optimization step.
            delta = self.attacker(stego.detach(), eps)
            noised_adv = torch.clamp(stego + delta, 0.0, 1.0)

            # Replace the selected samples with their adversarially attacked versions.
            mask = is_adv_mask.view(B, 1, 1, 1).float()
            noised = mask * noised_adv + (1 - mask) * noised_fixed
        else:
            noised = noised_fixed

        return noised, is_adv_mask
