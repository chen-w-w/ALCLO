# ==================== Imports ====================
import os
import re
import string

import torch
import lpips      # Perceptual loss library.
import bchlib     # BCH error-correcting code library.
import kornia     # Computer vision library.
import hashlib    # Hashing library.
import numpy as np
from torch import nn, Tensor


# ==================== Loss functions ====================
class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss using pretrained AlexNet features."""

    def __init__(self):
        super().__init__()
        # Use AlexNet as the feature extractor.
        self.lpips = lpips.LPIPS(net="alex", verbose=False)

    def __call__(self, input_image, target_image):
        """Compute LPIPS loss for input and target images in [0, 1]."""
        # Repeat grayscale inputs across three channels.
        if input_image.shape[1] == 1:
            input_image = input_image.repeat(1, 3, 1, 1)
            target_image = target_image.repeat(1, 3, 1, 1)

        # Normalize to [-1, 1], as required by LPIPS.
        normalized_input = input_image.clamp(0, 1.) * 2 - 1
        normalized_encoded = target_image.clamp(0, 1.) * 2 - 1

        lpips_loss = self.lpips(normalized_input, normalized_encoded).mean()
        return lpips_loss


# ==================== Stochastic rounding ====================
class StochasticRound:
    """Stochastic rounding with a straight-through gradient path.

    Soft rounding adds noise during training; hard rounding produces
    deterministic quantized values for inference.
    """

    def __init__(self, scale=1 / 255.):
        """Initialize the rounding scale; 1/255 corresponds to 8-bit image precision."""
        super().__init__()
        self.scale = scale

    def __call__(self, x, hard_round):
        """Round tensor x; hard_round selects deterministic inference or noisy training."""
        # Rescale the input.
        scale_x = x / self.scale

        # Use detach to preserve a straight-through gradient through rounding.
        round_out = scale_x + (torch.round(scale_x) - scale_x).detach()

        # Add random noise for stochastic rounding during training.
        out = round_out + torch.rand_like(x) - 0.5

        # Select the result for the requested mode.
        if hard_round:
            return round_out * self.scale  # Hard rounding produces deterministic values.
        return out * self.scale  # Soft rounding adds noise while preserving gradients.


# ==================== Pixel-range penalty ====================
class PenalityLoss(nn.Module):
    """Penalize pixel values outside the valid interval [0, max_value]."""

    def __init__(self, max_value=1.):
        """Configure the maximum permitted pixel value for the range penalty."""
        super().__init__()
        self.max_value = max_value
        self.MSE = nn.MSELoss(reduce=True)

    def __call__(self, input_tensor):
        """Return the range-violation penalty for an input tensor."""
        # Penalize underflow below zero.
        loss_0 = self.MSE(torch.relu(-input_tensor), torch.zeros_like(input_tensor))

        # Penalize overflow above max_value.
        loss_255 = self.MSE(torch.relu(input_tensor - self.max_value), torch.zeros_like(input_tensor))

        # Sum both penalties.
        loss = loss_0 + loss_255
        return loss


# ==================== Image processing utilities ====================
def normalize(input_image):
    """Apply min-max normalization to map an image tensor into [0, 1]."""
    min_vals = input_image.amin(dim=(1, 2, 3), keepdim=True)
    max_vals = input_image.amax(dim=(1, 2, 3), keepdim=True)
    normalized_img = (input_image - min_vals) / (max_vals - min_vals + 1e-5)  # Avoid division by zero.
    return normalized_img


def extract_accuracy(ext_secret, secret, max_value=1.):
    """Return extraction accuracy between predicted and ground-truth secrets."""
    acc = 1.0 - (torch.abs(torch.round(ext_secret.clamp(0., max_value)) - secret).mean())
    return acc.item()


def overflow_num(stego, mode, min_value=0., max_value=1.):
    """Return the mean number of underflow or overflow pixels in a stego batch."""
    assert mode in [0, 255]
    if mode == 0:
        overflow_pixel_n = torch.sum(StochasticRound()(stego, True) < min_value, dim=(1, 2, 3)).float().mean()
    else:
        overflow_pixel_n = torch.sum(StochasticRound()(stego, True) > max_value, dim=(1, 2, 3)).float().mean()
    return overflow_pixel_n.item()


def compute_psnr(input_image, target_image, max_value=1.):
    """Return mean PSNR in dB between image tensors with the given maximum value."""
    # Apply stochastic rounding and clamp to [0, 1].
    input_image = StochasticRound()(input_image.clamp(0., 1.), True)
    target_image = StochasticRound()(target_image.clamp(0., 1.), True)

    average_psnr = kornia.metrics.psnr(input_image, target_image, max_value).mean()
    return average_psnr.item()


def quantize_image(input_image):
    """Quantize an image tensor in [0, 1] using hard stochastic rounding."""
    # Apply deterministic hard rounding.
    input_image = StochasticRound()(input_image.clamp(0., 1.), True)
    return input_image


def quantize_residual_image(input_image, target_image):
    """Quantize an amplified residual image for visualizing embedding changes."""
    # Amplify the image residual tenfold for visualization.
    shift = (input_image - target_image) * 10.
    # Normalize and apply stochastic rounding.
    input_image = StochasticRound()(normalize(shift), True)
    return input_image


# ==================== Checkpoint utilities ====================
def find_latest_model(model_dir):
    """Find model_<epoch>.pth with the largest epoch, or return None if absent."""
    max_epoch = -1
    latest_model_path = None

    # Match filenames following the model_<epoch>.pth pattern.
    pattern = re.compile(r"model_(\d+)\.pth")

    for file_name in os.listdir(model_dir):
        match = pattern.match(file_name)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_model_path = os.path.join(model_dir, file_name)

    return latest_model_path


# ==================== Hash utilities ====================
def sha256_of_image_array(img_array):
    """Return the 64-character hexadecimal SHA-256 hash of a NumPy image array."""
    # Serialize the image array as bytes.
    img_bytes = img_array.tobytes()
    # Return its SHA-256 hash as a hexadecimal string.
    sha256_hash = hashlib.sha256(img_bytes).hexdigest()
    return sha256_hash


def sha256_to_bitstream(sha256_str):
    """Convert a 64-character hexadecimal SHA-256 hash into 256 big-endian bits."""
    bits = np.zeros(256, dtype=np.uint8)

    # Process each hexadecimal hash character.
    for i, hex_char in enumerate(sha256_str):
        # Convert the character to an integer in [0, 15].
        val = int(hex_char, 16)
        # Write the four-bit value in big-endian order.
        for j in range(4):
            bits[i * 4 + (3 - j)] = (val >> j) & 1

    return bits.tolist()


# ==================== BCH error correction ====================
class BCH:
    """BCH error-correcting code wrapper for robust watermark messages."""

    def __init__(self, BCH_POLYNOMIAL_=501, BCH_BITS_=12):
        """Initialize BCH with the requested polynomial and error-correction capacity."""
        self.bch = bchlib.BCH(BCH_BITS_, BCH_POLYNOMIAL_)

    def Encode(self, data_: str):
        """Encode a UTF-8 string into message bits followed by BCH parity bits."""
        # Encode the string as UTF-8 bytes.
        data_ = bytearray(data_, 'utf-8')
        # Compute error-correction parity bytes.
        ecc = self.bch.encode(data_)
        # Append parity bytes to the message.
        packet = data_ + ecc
        # Convert the packet to a binary string.
        packet_binary = ''.join(format(x, '08b') for x in packet)
        # Convert the string to a bit list.
        secret_ = [int(x) for x in packet_binary]
        return secret_

    def Decode(self, secret_: list):
        """Correct bit errors and return (success, decoded_data)."""
        # Convert the bit list to a binary string.
        packet_binary = "".join([str(int(bit)) for bit in secret_])
        # Convert the string to bytes.
        packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        # Separate message and parity bytes.
        data_, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]

        # Attempt decoding and error correction.
        bit_flips = self.bch.decode(data_, ecc)

        if bit_flips != -1:
            # Correct the errors after successful decoding.
            self.bch.correct(data_, ecc)
            return True, data_.decode('utf-8')
        else:
            # Decoding failed because the errors could not be corrected.
            return False, data_.decode('utf-8')


# ==================== Standalone entry point ====================
if __name__ == "__main__":
    import random

    def flip_n_bits(bits: list, n: int) -> list:
        """Randomly flip n bits to simulate transmission errors."""
        flipped = bits.copy()
        bit_indices = random.sample(range(len(bits)), n)
        for idx in bit_indices:
            flipped[idx] ^= 1
        return flipped

    def generate_random_string(n: int) -> str:
        """Generate a random string."""
        characters = string.ascii_letters + string.digits
        return ''.join(random.choices(characters, k=n))

    # Exercise BCH encoding and decoding.
    bch_codec = BCH()
    original_data = generate_random_string(20)
    print("Original message:", original_data)

    # Encode the message.
    secret = bch_codec.Encode(original_data)
    print(f"Encoded bit count: {len(secret)}")

    # Simulate errors by flipping n bits.
    n = 0  # Number of bits to flip.
    noisy_secret = flip_n_bits(secret, n)
    print(f"Introducing {n} bit errors")

    # Decode the message.
    isdecode, decoded_data = bch_codec.Decode(noisy_secret)

    if isdecode:
        print("Decoded message:", decoded_data)
        print("Success!" if decoded_data == original_data else "Mismatch!")
    else:
        print("Decoding failed: too many errors to correct")
