"""
Score-guided latent optimizer for lossy ALCLO extraction.

When the auxiliary bitstream is unavailable, the true latent z cannot be read.
This module treats z as the only trainable variable and optimizes it online while
freezing both iIWN and the score network.

Objective:
    L(z) = L_score + L_prior + L_entropy
    I_rec, w_hat = iIWN_inverse(I_noised, z)
    correction = score_net(I_rec, sigma)
    I_target = clamp(I_rec + correction, 0, 1).detach()
    L_score   = MSE(I_rec, I_target)
    L_prior   = lambda_prior * 0.5 * ||z||^2
    p_w       = sigmoid(w_hat)
    H(p_w)    = -p_w * log(p_w) - (1 - p_w) * log(1 - p_w)
    L_entropy = lambda_ent * mean(H(p_w))

This module is only for lossy extraction; it is not used in the lossless recovery
path, so it does not affect exact reversibility.
"""

import torch
import torch.nn.functional as F


class ZOptimizer:
    """Optimize z for one attacked stego image during lossy extraction."""

    def __init__(self, iiwn_model, score_network, bit_length=64,
                 num_steps=15, lr=0.1, sigma=0.01,
                 lambda_prior=1.0, lambda_ent=0.01, device='cuda'):
        self.iiwn = iiwn_model
        self.score_net = score_network
        self.bit_length = bit_length
        self.num_steps = num_steps
        self.lr = lr
        self.sigma = sigma
        self.lambda_prior = lambda_prior
        self.lambda_ent = lambda_ent
        self.device = device

        # Inference-time optimizer uses frozen networks.
        self.iiwn.eval()
        self.score_net.eval()

    def _binary_entropy(self, p):
        """Binary entropy: H(p) = -p log p - (1-p) log(1-p)."""
        p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
        return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

    @torch.no_grad()
    def extract_baseline(self, noised_image):
        """Baseline extraction with random z, matching the original DRRW path."""
        z_rand = torch.randn(1, self.bit_length, device=self.device)
        _, w_hat = self.iiwn.forward(noised_image, z_rand, True, True)
        watermark = torch.round(torch.clip(w_hat, 0., 1.))
        return watermark

    def optimize(self, noised_image, return_trajectory=False):
        """Run K-step Adam optimization over z and return extracted watermark."""
        noised_image = noised_image.to(self.device)

        # Initialize z at the prior mean.
        z = torch.zeros(1, self.bit_length, device=self.device,
                        dtype=noised_image.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=self.lr)

        # Restore these flags after optimization to avoid side effects.
        iiwn_grad_states = [p.requires_grad for p in self.iiwn.parameters()]
        score_grad_states = [p.requires_grad for p in self.score_net.parameters()]

        trajectory = []

        try:
            # Build gradients only for z.
            for p in self.iiwn.parameters():
                p.requires_grad_(False)
            for p in self.score_net.parameters():
                p.requires_grad_(False)

            for step in range(self.num_steps):
                optimizer.zero_grad()

                # iIWN inverse: (noised, z) -> (I_rec, w_hat).
                I_rec, w_hat = self.iiwn.forward(noised_image, z, True, True)

                # L_score = MSE(I_rec, clamp(I_rec + correction, 0, 1)).
                # The target is detached so gradients update z through iIWN only.
                correction = self.score_net(I_rec, self.sigma)
                I_target = torch.clamp(I_rec + correction, 0.0, 1.0).detach()
                loss_score = F.mse_loss(I_rec, I_target)

                # L_prior = lambda_prior * 0.5 * ||z||^2.
                loss_prior = self.lambda_prior * 0.5 * torch.sum(z ** 2)

                # L_entropy = lambda_ent * mean(H(sigmoid(w_hat))).
                p_w = torch.sigmoid(w_hat)
                loss_entropy = self.lambda_ent * torch.mean(self._binary_entropy(p_w))

                loss = loss_score + loss_prior + loss_entropy
                loss.backward()
                optimizer.step()

                if return_trajectory:
                    trajectory.append({
                        'step': step,
                        'loss_total': loss.item(),
                        'loss_score': loss_score.item(),
                        'loss_prior': loss_prior.item(),
                        'loss_entropy': loss_entropy.item(),
                        'z_norm': z.detach().norm().item(),
                    })

            # Final extraction with optimized z.
            with torch.no_grad():
                _, w_final = self.iiwn.forward(noised_image, z, True, True)
                watermark = torch.round(torch.clip(w_final, 0., 1.))

            z_opt = z.detach()
        finally:
            # Restore original requires_grad states.
            for p, s in zip(self.iiwn.parameters(), iiwn_grad_states):
                p.requires_grad_(s)
            for p, s in zip(self.score_net.parameters(), score_grad_states):
                p.requires_grad_(s)

        if return_trajectory:
            return z_opt, watermark, trajectory
        return z_opt, watermark

    def optimize_batch(self, noised_images, batch_size=1):
        """Optimize each image independently and concatenate watermarks."""
        N = noised_images.shape[0]
        watermarks = []

        for i in range(N):
            _, wm = self.optimize(noised_images[i:i + 1])
            watermarks.append(wm)

        return torch.cat(watermarks, dim=0)
