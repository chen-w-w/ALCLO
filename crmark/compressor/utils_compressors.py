# ==================== Imports ====================
import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from .arithmeticcoder import ArithmeticEncoder


# ==================== Array arithmetic coder ====================
class CustomArithmeticEncoder:
    """Array arithmetic coder supporting fixed and adaptive frequency models."""

    def __init__(self, level_bits_len: int = 10, freq_bits_len: int = 20):
        """Configure frequency-table header width and signed data value width in bits."""
        self.buffer_bits = 40  # Arithmetic coder precision in bits.
        self.freq_bits_len = freq_bits_len
        self.level_bits_len = level_bits_len

    def ndarray2strlist(self, data: ndarray) -> list:
        """Flatten a NumPy array and convert its values to strings."""
        data_list = data.flatten().tolist()
        data_str_list = [str(data_item) for data_item in data_list]
        return data_str_list

    def strlist2ndarray(self, data_str_list: list) -> list:
        """Convert numeric strings to an integer list."""
        data_list = [int(data_str_item) for data_str_item in data_str_list]
        return data_list

    def datastr2bits(self, data_str_list):
        """Encode numeric strings as fixed-width two's-complement bits."""
        encoded_bits = []
        for item in data_str_list:
            num = int(item)
            if num < 0:
                num = (1 << self.freq_bits_len) + num  # Convert negative values to two's complement.
            binary_representation = bin(num)[2:]  # Convert to a binary string.
            padded_binary = binary_representation.zfill(self.freq_bits_len)  # Pad to the configured width with leading zeros.
            if len(padded_binary) > self.freq_bits_len:
                raise ValueError(f"Value {num} cannot be represented with {self.freq_bits_len} bits.")
            encoded_bits.extend(int(bit) for bit in padded_binary)
        return encoded_bits

    def bits2datastr(self, encoded_bits):
        """Decode fixed-width signed bits into numeric strings."""
        decoded_data = []
        for i in range(0, len(encoded_bits), self.freq_bits_len):
            bit_str = ''.join(map(str, encoded_bits[i:i + self.freq_bits_len]))
            num = int(bit_str, 2)
            if num >= (1 << (self.freq_bits_len - 1)):
                num -= (1 << self.freq_bits_len)  # Convert two's complement back to a signed value.
            decoded_data.append(str(num))
        return decoded_data

    def integer2bits(self, integer: int):
        """Encode an integer as a fixed-width bit list."""
        if integer < 0:
            integer = (1 << self.level_bits_len) + integer
        if integer >= (1 << self.level_bits_len):
            raise ValueError(f"Value {integer} cannot be represented with {self.level_bits_len} bits.")
        binary_representation = bin(integer)[2:]
        padded_binary = binary_representation.zfill(self.level_bits_len)
        return [int(bit) for bit in padded_binary]

    def bits2integer(self, bits: list):
        """Decode a bit list into an integer."""
        if len(bits) != self.level_bits_len:
            raise ValueError(f"The bit list must contain {self.level_bits_len} bits.")
        bit_str = ''.join(str(bit) for bit in bits)
        num = int(bit_str, 2)
        if num >= (1 << (self.level_bits_len - 1)):
            num -= (1 << self.level_bits_len)
        return num

    def compress(self, data: ndarray, frequencies=None) -> list:
        """Compress an array using fixed frequencies or an adaptive model."""
        data_str_list = self.ndarray2strlist(data)

        if frequencies is None:
            # Build the adaptive symbol model from unique input values.
            _frequencies = list(set(data_str_list))
            freqs_bits = self.datastr2bits(_frequencies)
            auxbits = freqs_bits + self.integer2bits(len(_frequencies))
            frequencies_input = _frequencies + ["<EOM>"]
        else:
            frequencies_input = frequencies + ["<EOM>"]
            auxbits = []

        coder = ArithmeticEncoder(frequencies=frequencies_input, bits=self.buffer_bits)
        data_bits = list(coder.encode(data_str_list + ["<EOM>"]))

        if frequencies is None:
            return data_bits + auxbits
        else:
            return data_bits

    def decompress(self, data_freqs_bits: list, frequencies=None) -> ndarray:
        """Decompress bits using supplied frequencies or the serialized frequency table."""
        if frequencies is None:
            # Read frequency-table information from the bitstream suffix.
            len_bits_freqs = data_freqs_bits[-self.level_bits_len:]
            retain_bits = data_freqs_bits[:-self.level_bits_len]
            len_freqs = self.bits2integer(len_bits_freqs)
            freqs_bits = retain_bits[-len_freqs * self.freq_bits_len:]
            retain_bits = retain_bits[:-len_freqs * self.freq_bits_len]
            frequencies = self.bits2datastr(freqs_bits)
            data_bits = retain_bits
            frequencies += ["<EOM>"]
        else:
            data_bits = data_freqs_bits
            frequencies += ["<EOM>"]

        coder = ArithmeticEncoder(frequencies=frequencies, bits=self.buffer_bits)
        data_str_list = list(coder.decode(data_bits))
        data = [int(data_item) for data_item in data_str_list[:-1]]
        return np.asarray(data)


# ==================== Arithmetic compression ====================
class ACCompress:
    """Arithmetic-code latent values and stego overflow data into one bitstream."""

    def __init__(self, im_size, z_size, level_bits_len, freq_bits_len, device: str = "cpu"):
        """Configure arithmetic coding for image shape (H, W, C) and latent shape (H, W)."""
        self.mark_len = 20  # Width of the latent bitstream length header.
        self.im_size = im_size  # (H, W, C)
        self.z_size = z_size  # (H, W)
        self.device = device
        self.im_len = im_size[0] * im_size[1] * im_size[2]  # Total image pixel count.
        self.z_len = z_size[0] * z_size[1]  # Number of latent elements.
        self.coder = CustomArithmeticEncoder(level_bits_len=level_bits_len, freq_bits_len=freq_bits_len)

    def combine_bits(self, z_bits: list, stego_bits: list):
        """Combine a 20-bit latent length header, latent bits, and overflow bits."""
        length_bits = format(len(z_bits), f'0{self.mark_len}b')
        length_bits = [int(b) for b in length_bits]
        combined_bits = length_bits + z_bits + stego_bits
        return combined_bits

    def split_bits(self, combined_bits: list):
        """Split a combined bitstream into (z_bits, stego_bits)."""
        length_bits = combined_bits[:self.mark_len]
        z_length = int(''.join(map(str, length_bits)), 2)
        z_bits = combined_bits[self.mark_len:self.mark_len + z_length]
        stego_bits = combined_bits[self.mark_len + z_length:]
        return z_bits, stego_bits

    def encode(self, stego_img, drop_z):
        """Encode a stego tensor (1, C, H, W) and its latent tensor.

        Return the clipped image and (combined_bits, z_bits, overflow_bits).
        """
        overflow_bits_list = []
        clip_stego_img = None

        if stego_img is not None:
            # Clip pixels to [0, 255] and identify overflow.
            clip_stego_img = torch.clip(stego_img, 0, 255)
            steg_img_numpy = stego_img.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
            overflow = np.zeros_like(steg_img_numpy)

            # Compute overflow values outside [0, 255].
            overflow[steg_img_numpy > 255] = steg_img_numpy[steg_img_numpy > 255] - 255
            overflow[steg_img_numpy < 0] = 0 - steg_img_numpy[steg_img_numpy < 0]
            overflow = overflow.astype(int)

            # Arithmetic-code the overflow data.
            overflow_bits_list = self.coder.compress(overflow.flatten())

        drop_z_bits_list = []
        if drop_z is not None:
            # Clip latent values to the permitted range and compress them.
            drop_z_numpy = drop_z.squeeze(0).detach().cpu().numpy()
            drop_z_numpy = np.clip(drop_z_numpy, -2 ** self.coder.freq_bits_len + 1, 2 ** self.coder.freq_bits_len - 1)
            drop_z_numpy_clip = drop_z_numpy.astype(np.int64)
            drop_z_bits_list = self.coder.compress(drop_z_numpy_clip.flatten())

        # Combine available auxiliary bitstreams.
        if stego_img is None:
            data_list = drop_z_bits_list
        elif drop_z is None:
            data_list = overflow_bits_list
        else:
            data_list = self.combine_bits(drop_z_bits_list, overflow_bits_list)
            # Optionally verify bitstream combination.
            drop_z_bits_list, overflow_bits_list = self.split_bits(data_list)

        return clip_stego_img, (data_list, drop_z_bits_list, overflow_bits_list)

    def decode(self, clip_stego_img: Tensor, data_bits: list):
        """Decode the auxiliary bitstream to restore the stego and latent tensors."""
        # Mark pixels equal to zero or 255 for overflow correction.
        mask_0 = (clip_stego_img == 0).int()
        mask_255 = (clip_stego_img == 255).int()

        # Split the bitstream.
        drop_z_bits_list, overflow_bits_list = self.split_bits(data_bits)

        # Decompress both components.
        drop_z_list = self.coder.decompress(drop_z_bits_list)
        overflow_list = self.coder.decompress(overflow_bits_list)

        # Reconstruct the overflow map and correct pixels.
        overflow = torch.as_tensor(overflow_list).reshape(self.im_size).to(self.device).permute(2, 0, 1).unsqueeze(0)
        stego_img = (clip_stego_img - mask_0 * overflow + mask_255 * overflow).float()

        # Reconstruct the latent tensor z.
        rec_z = torch.as_tensor(drop_z_list).reshape(self.z_size).to(self.device).float()

        return stego_img, rec_z


# ==================== Sparse tensor compression ====================
class SparseTensorCompressor:
    """Compress sparse overflow values by storing their coordinates and values.

    Each nonzero pixel uses row, column, channel, and value bit fields.
    """

    def __init__(self, image_shape: tuple, z_shape: tuple, level_bits_len: int, freq_bits_len: int, val_bits: int = 10):
        """Configure coordinate widths, latent shape, and signed overflow value width."""
        self.height, self.width, self.channels = image_shape
        self.mark_len = 20
        self.z_shape = z_shape

        # Compute the bit width required by each dimension.
        self.row_bits = int(np.ceil(np.log2(self.height)))  # Row index width.
        self.col_bits = int(np.ceil(np.log2(self.width)))  # Column index width.
        self.channel_bits = 0 if self.channels == 1 else int(np.ceil(np.log2(self.channels)))  # Channel index width.
        self.value_bits = int(val_bits)  # Signed value width.

        self.coder = CustomArithmeticEncoder(level_bits_len=level_bits_len, freq_bits_len=freq_bits_len)

    def combine_bits(self, z_bits: list, stego_bits: list) -> list:
        """Combine latent and overflow bitstreams with a latent length header."""
        length_bits = format(len(z_bits), f'0{self.mark_len}b')
        length_bits = [int(b) for b in length_bits]
        return length_bits + z_bits + stego_bits

    def split_bits(self, combined_bits: list) -> tuple:
        """Split a combined bitstream into (z_bits, stego_bits)."""
        length_bits = combined_bits[:self.mark_len]
        z_length = int(''.join(map(str, length_bits)), 2)
        z_bits = combined_bits[self.mark_len:self.mark_len + z_length]
        stego_bits = combined_bits[self.mark_len + z_length:]
        return z_bits, stego_bits

    def compress(self, stego_img: torch.Tensor, drop_z: torch.Tensor) -> tuple:
        """Compress a stego tensor and latent values using sparse overflow coordinates."""
        overflow_bits_list = []
        clip_stego_img = None

        if stego_img is not None:
            clip_stego_img = torch.clip(stego_img, 0, 255)
            steg_img_numpy = stego_img.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()

            # Compute overflow values.
            overflow = np.zeros_like(steg_img_numpy)
            overflow[steg_img_numpy > 255] = steg_img_numpy[steg_img_numpy > 255] - 255
            overflow[steg_img_numpy < 0] = 0 - steg_img_numpy[steg_img_numpy < 0]
            overflow = overflow.astype(int)

            # Locate all nonzero overflow pixels.
            pos_1, pos_2, pos_3 = (overflow != 0).nonzero()
            non_zero_values = overflow[pos_1, pos_2, pos_3]

            # Encode each nonzero pixel's coordinates and value.
            compressed_bitstream = ''
            for row, col, channel, val in zip(pos_1.tolist(), pos_2.tolist(), pos_3.tolist(), non_zero_values.tolist()):
                row_bin = f'{row:0{self.row_bits}b}'
                col_bin = f'{col:0{self.col_bits}b}'
                channel_bin = f'{channel:0{self.channel_bits}b}' if self.channels > 1 else ''
                value_bin = f'{val:0{self.value_bits}b}'
                compressed_bitstream += row_bin + col_bin + channel_bin + value_bin

            overflow_bits_list = [int(bit) for bit in compressed_bitstream]

        # Compress drop_z.
        drop_z_bits_list = []
        if drop_z is not None:
            drop_z_numpy = drop_z.squeeze(0).detach().cpu().numpy()
            drop_z_numpy = np.clip(drop_z_numpy, -2 ** self.coder.freq_bits_len + 1, 2 ** self.coder.freq_bits_len - 1)
            drop_z_numpy = drop_z_numpy.astype(np.int64)
            drop_z_bits_list = self.coder.compress(drop_z_numpy.flatten())

        # Combine the bitstreams.
        if stego_img is None:
            data_list = drop_z_bits_list
        elif drop_z is None:
            data_list = overflow_bits_list
        else:
            data_list = self.combine_bits(drop_z_bits_list, overflow_bits_list)

        return clip_stego_img, (data_list, drop_z_bits_list, overflow_bits_list)

    def decompress(self, clip_stego_img: torch.Tensor, data_bits: list) -> tuple:
        """Restore a clipped stego tensor and latent values from the encoded bits."""
        # Split the bitstream.
        drop_z_bits_list, overflow_bits_list = self.split_bits(data_bits)

        # Decompress drop_z.
        drop_z_list = self.coder.decompress(drop_z_bits_list)
        rec_z = torch.as_tensor(drop_z_list, dtype=torch.float).reshape(self.z_shape)

        # Decompress overflow data.
        compressed_bitstream = ''.join(str(bit) for bit in overflow_bits_list)
        overflow = np.zeros((self.height, self.width, self.channels))
        index = 0

        while index < len(compressed_bitstream):
            row = int(compressed_bitstream[index:index + self.row_bits], 2)
            col = int(compressed_bitstream[index + self.row_bits:index + self.row_bits + self.col_bits], 2)
            channel_start = index + self.row_bits + self.col_bits

            if self.channels > 1:
                channel = int(compressed_bitstream[channel_start:channel_start + self.channel_bits], 2)
                index += self.row_bits + self.col_bits + self.channel_bits
            else:
                channel = 0
                index += self.row_bits + self.col_bits

            value = int(compressed_bitstream[index:index + self.value_bits], 2)
            overflow[row, col, channel] = value
            index += self.value_bits

        # Apply overflow correction.
        mask_0 = (clip_stego_img == 0.).float()
        mask_255 = (clip_stego_img == 255.).float()
        overflow_tensor = torch.as_tensor(overflow, dtype=clip_stego_img.dtype).permute(2, 0, 1).unsqueeze(0)

        stego_img = clip_stego_img - mask_0 * overflow_tensor + mask_255 * overflow_tensor
        return stego_img, rec_z


# ==================== Automatic tensor coding strategy ====================
class TensorCoder:
    """Choose the shorter of arithmetic coding and sparse coordinate coding.

    Prefix the selected bitstream with a flag identifying its decoding strategy.
    """

    def __init__(self, image_shape, drop_z_shape, level_bits_len, freq_bits_len):
        """Configure image and latent shapes and arithmetic coding bit widths."""
        self.sparsetensorcompressor = SparseTensorCompressor(image_shape, drop_z_shape, level_bits_len, freq_bits_len)
        self.accompress = ACCompress(image_shape, drop_z_shape, level_bits_len, freq_bits_len)

    def compress(self, input_tensor, drop_z):
        """Return the clipped image and the shorter encoded auxiliary bitstream."""
        # Use arithmetic coding.
        clip_stego_img_a, data_list_a = self.accompress.encode(input_tensor, drop_z)
        # Use sparse tensor coding.
        clip_stego_img_s, data_list_s = self.sparsetensorcompressor.compress(input_tensor, drop_z)

        # Choose the shorter bitstream.
        if len(data_list_a[0]) > len(data_list_s[0]):
            # Prefix sparse tensor coding with flag zero.
            data_list = [0] + data_list_s[0]
        else:
            # Prefix arithmetic coding with flag one.
            data_list = [1] + data_list_a[0]

        clip_stego_img = clip_stego_img_a
        return clip_stego_img, data_list

    def decompress(self, clip_stego_img, data_list):
        """Decode the strategy flag and restore the image and latent tensors."""
        # Read the strategy flag.
        data_list_ = data_list[1:]
        if data_list[0] == 1:
            # Use arithmetic decoding.
            rec_img, rec_drop_z = self.accompress.decode(clip_stego_img, data_list_)
        else:
            # Use sparse tensor decoding.
            rec_img, rec_drop_z = self.sparsetensorcompressor.decompress(clip_stego_img, data_list_)
        return rec_img, rec_drop_z


# ==================== Standalone entry point ====================
if __name__ == "__main__":
    # Exercise TensorCoder.
    print("=" * 50)
    print("Testing TensorCoder encoding and decoding")
    print("=" * 50)

    # Define image and latent dimensions.
    im_size = (400, 400, 3)  # Image shape (H, W, C).
    z_size = (1, 100)  # Latent shape (H, W).
    device = 'cpu'

    # Generate random example data.
    stego_img = torch.randint(-10, 280, (1, im_size[2], im_size[0], im_size[1]), dtype=torch.float32, device=device)
    drop_z = torch.randint(44631230, 44631236, (1, *z_size), dtype=torch.float32, device=device)

    print(f"Image shape: {im_size}")
    print(f"Latent shape: {z_size}")
    print(f"Image value range: [{stego_img.min():.0f}, {stego_img.max():.0f}]")
    print(f"Latent value range: [{drop_z.min():.0f}, {drop_z.max():.0f}]")

    # Initialize the coder.
    tensor_coder = TensorCoder(im_size, z_size, 10, 40)

    # Encode the data.
    print("\nStarting encoding...")
    clip_encoded_img, data_bits = tensor_coder.compress(stego_img, drop_z)
    print(f"Encoded bit count: {len(data_bits)}")
    print(f"Strategy: {'Arithmetic coding' if data_bits[0] == 1 else 'Sparse coding'}")

    # Decode the data.
    print("\nStarting decoding...")
    decoded_img, decoded_z = tensor_coder.decompress(clip_encoded_img, data_bits)

    # Verify the results.
    print("\nVerification results:")
    print(f"First 10 original image values: {stego_img.flatten()[:10].tolist()}")
    print(f"First 10 decoded image values: {decoded_img.flatten()[:10].tolist()}")
    print(f"First 10 original latent values: {drop_z.flatten()[:10].tolist()}")
    print(f"First 10 decoded latent values: {decoded_z.flatten()[:10].tolist()}")

    # Check reconstruction precision.
    img_match = torch.allclose(stego_img, decoded_img, atol=1e-8)
    z_match = torch.allclose(drop_z, decoded_z, atol=1e-8)

    print(f"\nImage matches exactly: {img_match}")
    print(f"Latent matches exactly: {z_match}")

    if img_match and z_match:
        print("\nEncoding/decoding check passed.")
    else:
        print("\nEncoding/decoding check failed.")
