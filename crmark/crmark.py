# ==================== Imports ====================
import os
import torch
import random
import platform
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from numpy import ndarray
from crmark.nets import Model
from torchvision import transforms
from typing import Tuple, Union, Any
from crmark.compressor.rdh import CustomRDH
from crmark.compressor.utils import sha256_of_image_array
from crmark.compressor.utils_compressors import TensorCoder
from crmark.compressor.utils import sha256_to_bitstream, BCH
import warnings

# Suppress warnings to keep console output concise.
warnings.filterwarnings("ignore")

# Public module interface.
__all__ = ["CRMark"]

# ==================== Pretrained model URLs ====================
# Grayscale model: 512 x 512 images, 256 bits (about 20 characters).
_gray_512_256_MODEL_URL = "https://github.com/chenoly/CRMark/releases/download/v1.0/crmark_gray_size_512_bit_256.pth"
# Color model: 256 x 256 images, 64 bits (about 5 characters).
_color_256_64_MODEL_URL = "https://github.com/chenoly/CRMark/releases/download/v1.0/crmark_color_size_256_bit_64.pth"
# Color model: 256 x 256 images, 100 bits (about 7 characters).
_color_256_100_MODEL_URL = "https://github.com/chenoly/CRMark/releases/download/v1.0/crmark_color_size_256_bit_100.pth"


# ==================== Model download utilities ====================
def _download_from_github_release(url: str, save_path: str) -> bool:
    """Download a release asset to save_path and validate its byte count.

    Write to a temporary file and rename it after validation. Return True on
    success and False on download or validation failure.
    """
    temp_path = None
    file_handle = None
    try:
        print("Starting download:", url)
        # Authenticate with a GitHub token when the environment variable is set.
        headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}"}
        response = requests.get(url, stream=True, allow_redirects=True, headers=headers, timeout=30)

        # Check the response status.
        if response.status_code != 200:
            print(f"Download failed with HTTP status {response.status_code}")
            print(f"\nDownload manually from:\n{url}\nand place the file in: {os.path.dirname(save_path)}")
            return False

        # Read the expected file size.
        total_size = int(response.headers.get('content-length', 0))
        if total_size < 1024:
            print(f"Unexpected file size: {total_size} bytes")
            return False

        # Create the temporary download path.
        temp_path = save_path + '.tmp'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Stream into a temporary file with a progress indicator.
        with open(temp_path, 'wb') as file_handle:
            with tqdm(
                    desc=save_path,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    file_handle.write(data)
                    bar.update(len(data))

        # Verify the downloaded byte count.
        downloaded_size = os.path.getsize(temp_path)
        if downloaded_size != total_size:
            print(f"Incomplete download: received {downloaded_size} bytes, expected {total_size} bytes")
            return False

        # Rename the completed download to its final path.
        os.rename(temp_path, save_path)
        print(f"Download complete: {save_path}")
        return True

    except (requests.RequestException, KeyboardInterrupt) as e:
        print(f"Download failed or was interrupted: {e}")
        print(f"\nDownload manually from:\n{url}\nand place the file in:\n{os.path.dirname(save_path)}")
        return False

    finally:
        # Ensure the file handle is closed.
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception as e:
                print(f"Could not close the file handle: {e}")

        # Remove the temporary file.
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"Removed temporary file: {temp_path}")
            except OSError as e:
                print(f"Could not remove temporary file {temp_path}: {e}")


def _download_models(model_mode):
    """Download the selected pretrained weights if they are not cached.

    Supported modes are gray_512_256, color_256_64, and color_256_100. Weights
    are stored in the platform-specific user cache directory.
    """
    # Select the cache directory for the current operating system.
    if platform.system() == "Windows":
        base_cache_dir = os.path.join(os.environ["USERPROFILE"], ".cache", "crmark")
    else:
        base_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "crmark")
    os.makedirs(base_cache_dir, exist_ok=True)

    # Download grayscale weights if necessary.
    if model_mode == "gray_512_256":
        gray_model_path = os.path.join(base_cache_dir, "crmark_gray_size_512_bit_256.pth")
        if not os.path.exists(gray_model_path):
            _download_from_github_release(_gray_512_256_MODEL_URL, gray_model_path)

    # Download 64-bit color weights if necessary.
    if model_mode == "color_256_64":
        model_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_64.pth")
        if not os.path.exists(model_model_path):
            _download_from_github_release(_color_256_64_MODEL_URL, model_model_path)

    # Download 100-bit color weights if necessary.
    if model_mode == "color_256_100":
        model_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_100.pth")
        if not os.path.exists(model_model_path):
            _download_from_github_release(_color_256_100_MODEL_URL, model_model_path)


# ==================== Watermarking wrapper ====================
class CRMark(object):
    """Reversible image watermarking wrapper combining iIWN and RDH.

    Embed text or raw bits, extract messages, and recover the original cover
    through lossless channels. Supports color and grayscale model variants.
    """

    def __init__(self, model_mode="color_256_64", model_path: str = None, level_bits_len=10, freq_bits_len=10,
                 device="cpu", float64=False):
        """Load model weights and initialize compression and error-correction tools.

        Args:
            model_mode: color_256_64, color_256_100, or gray_512_256.
            model_path: Custom weights, or None to use cached/downloaded weights.
            level_bits_len: Tensor quantization header width; defaults to 10.
            freq_bits_len: Tensor quantization value width; defaults to 10.
            device: Target computation device, such as cpu or cuda.
            float64: Enable double precision instead of the default float32.

        The color modes use 256 x 256 RGB images with 64 or 100 bits and BCH
        correction capacities of 3 or 5 bits. The grayscale mode uses 512 x 512
        images with 256 bits and a BCH correction capacity of 12 bits.
        """
        # Validate the requested model mode.
        assert model_mode in ["color_256_64", "color_256_100", "gray_512_256"], \
            "model_mode must be 'color_256_64', 'color_256_100', or 'gray_512_256'"

        # ==================== Instance state ====================
        self.k = None              # Number of invertible blocks, controlling network depth.
        self.fc = None             # Fully connected layer configuration.
        self.rdh = None            # Reversible data hiding module.
        self.iIWN = None           # Integer invertible watermark network.
        self.device = device       # Computation device.
        self.float64 = float64     # Precision flag.
        self.img_size = None       # Expected image size.
        self.bit_length = None     # Watermark bit length.
        self.model_mode = model_mode  # Model mode.

        # ==================== Model-specific configuration ====================
        if model_mode == "color_256_64":
            self.channel_dim = 3           # Three-channel RGB.
            BCH_POLYNOMIAL_ = 137          # BCH polynomial.
            BCH_BITS_ = 3                  # BCH correction capacity: three bit errors.
        elif model_mode == "color_256_100":
            self.channel_dim = 3           # Three-channel RGB.
            BCH_POLYNOMIAL_ = 137          # BCH polynomial.
            BCH_BITS_ = 5                  # BCH correction capacity: five bit errors.
        else:  # gray_512_256
            self.channel_dim = 1           # Single-channel grayscale.
            BCH_POLYNOMIAL_ = 501          # BCH polynomial.
            BCH_BITS_ = 12                 # BCH correction capacity: twelve bit errors.

        # ==================== Initialize supporting components ====================
        self.tensorcoder = None            # Tensor compression module.
        self.net_min_size = None           # Minimum network feature size.
        self.freq_bits_len = freq_bits_len # Frequency encoding width in bits.
        self.level_bits_len = level_bits_len  # Level encoding width in bits.
        self.bch = BCH(BCH_POLYNOMIAL_, BCH_BITS_)  # BCH encoder instance.
        self.transform = transforms.Compose([transforms.ToTensor()])  # Image-to-tensor transform.

        # ==================== Model cache directory ====================
        if platform.system() == "Windows":
            base_cache_dir = os.path.join(os.environ["USERPROFILE"], ".cache", "crmark")
        else:
            base_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "crmark")
        os.makedirs(base_cache_dir, exist_ok=True)

        # Set the model file paths.
        gray_model_path = os.path.join(base_cache_dir, "crmark_gray_size_512_bit_256.pth")
        color_256_64_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_64.pth")
        color_256_100_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_100.pth")

        # ==================== Load model weights ====================
        if model_path is None:
            # Download required model weights automatically.
            _download_models(model_mode)
            # Load the weights corresponding to the selected mode.
            if model_mode == "color_256_64":
                self.load_model(color_256_64_model_path)
            elif model_mode == "color_256_100":
                self.load_model(color_256_100_model_path)
            else:
                self.load_model(gray_model_path)
        else:
            # Load user-specified weights.
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a checkpoint and configure the watermark network, RDH, and tensor coder."""
        # Read the checkpoint dictionary from disk.
        load_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        # ==================== Checkpoint model configuration ====================
        self.k = load_dict['param_dict']['k']                    # Number of invertible blocks.
        self.net_min_size = load_dict['param_dict']['min_size']  # Minimum feature size.
        self.fc = load_dict['param_dict']['fc']                  # Fully connected layer configuration.
        self.bit_length = load_dict['param_dict']['bit_length']  # Watermark bit length.
        self.img_size = load_dict['param_dict']['img_size']      # Expected image size.
        self.channel_dim = load_dict['param_dict']['channel_dim']  # Channel count.

        # Set the default dtype when double precision is requested.
        if self.float64:
            torch.set_default_dtype(torch.float64)

        # ==================== Initialize RDH ====================
        # RDH embeds auxiliary information and the cover image hash.
        self.rdh = CustomRDH((self.img_size, self.img_size, self.channel_dim), self.device)

        # ==================== Initialize tensor compression ====================
        # TensorCoder compresses floating-point tensors produced by the network.
        self.tensorcoder = TensorCoder(
            (self.img_size, self.img_size, self.channel_dim),  # Image shape.
            (1, self.bit_length),                              # Watermark shape.
            self.level_bits_len,                               # Level encoding width in bits.
            self.freq_bits_len                                 # Frequency encoding width in bits.
        )

        # ==================== Initialize the watermark network ====================
        # iIWN is the core invertible image watermark network.
        self.iIWN = Model(self.img_size, self.channel_dim, self.bit_length, self.k, self.net_min_size, self.fc)
        self.iIWN.load_state_dict(load_dict['model_state_dict'])  # Load the weights.
        self.iIWN.to(self.device)                                   # Move the model to the configured device.
        self.iIWN.eval()                                            # Switch to evaluation mode.

    def encode(self, cover_img: ndarray, message: str) -> Tuple[bool, Any]:
        """Embed a text message into a NumPy cover image.

        Encode the message with BCH, watermark with iIWN, compress the auxiliary
        tensors, and embed them with the cover hash through RDH. Message length
        depends on the model mode: 5, 7, or 20 characters. Return (success, stego_image).
        """
        # ==================== Validate inputs ====================
        cover_img = np.uint8(cover_img)

        # Validate message length and image dimensions.
        if self.model_mode == "color_256_64":
            assert len(message) == 5 and cover_img.ndim == 3, \
                "color_256_64 requires a 5-character message and an RGB image"
        if self.model_mode == "color_256_100":
            assert len(message) == 7 and cover_img.ndim == 3, \
                "color_256_100 requires a 7-character message and an RGB image"
        if self.model_mode == "gray_512_256":
            assert len(message) == 20 and cover_img.ndim == 2, \
                "gray_512_256 requires a 20-character message and a grayscale image"

        # ==================== Hash the cover image ====================
        # The hash later verifies whether the recovered cover has been altered.
        cover_img_hash = sha256_of_image_array(cover_img)
        cover_img_hash_bitstream = sha256_to_bitstream(cover_img_hash)

        # ==================== Preprocess the image ====================
        # Convert the NumPy image to a PyTorch tensor.
        cover_img_tensor = self.transform(cover_img).unsqueeze(0).to(self.device)

        # ==================== Encode the message ====================
        # Add BCH error correction to improve robustness.
        watermark = self.bch.Encode(message)
        # Pad with random bits to the configured watermark length.
        watermark += [random.randint(0, 1) for _ in range(self.bit_length - len(watermark))]
        secret_tensor = torch.as_tensor(watermark, dtype=torch.float32).unsqueeze(0).to(self.device)

        # ==================== Neural watermark embedding ====================
        with torch.no_grad():
            if self.float64:
                cover_img_tensor = cover_img_tensor.to(torch.float64)
                secret_tensor = secret_tensor.to(torch.float64)
            # Forward iIWN: cover + watermark -> overflow stego image + auxiliary tensor z.
            overflow_stego, z = self.iIWN.forward(cover_img_tensor, secret_tensor, True, False)

        # ==================== Compress auxiliary tensors ====================
        # Convert floating-point tensors into integer data suitable for storage.
        stego_255 = torch.round(overflow_stego * 255.)
        z_round = torch.round(z)
        clip_stego_img, auxbits = self.tensorcoder.compress(stego_255, z_round)

        # ==================== Embed auxiliary data with RDH ====================
        # Embed the auxiliary information and cover hash into the stego image.
        issuccessful, rdh_stego_img = self.rdh.embed(clip_stego_img, auxbits + cover_img_hash_bitstream)
        if not issuccessful:
            print("Reversible embedding failed")

        return issuccessful, Image.fromarray(np.uint8(rdh_stego_img))

    def recover(self, stego_img: ndarray) -> Union[Tuple[bool, Any, Any], Tuple[bool, None, None]]:
        """Recover the cover and text message from a lossless stego image.

        Extract RDH data, decompress auxiliary tensors, invert iIWN, decode BCH,
        and check the recovered cover hash. Return (is_attacked, cover, message),
        or (True, None, None) when recovery fails.
        """
        # ==================== Load the stego image ====================
        stego_img = np.uint8(stego_img)

        # ==================== Extract RDH data ====================
        # Extract the auxiliary information and cover hash.
        issuccessful, clipped_stego_img, ext_auxbits = self.rdh.extract(stego_img)
        if issuccessful:
            # Separate the hash from the auxiliary information.
            cover_img_hash_bitstream = ext_auxbits[-256:]  # The last 256 bits contain the hash.
            rec_overflow_stego, rec_z = self.tensorcoder.decompress(clipped_stego_img, ext_auxbits[:-256])

            # ==================== Invert the watermark network ====================
            with torch.no_grad():
                if self.float64:
                    rec_z = rec_z.to(torch.float64)
                    rec_overflow_stego = rec_overflow_stego.to(torch.float64)
                # Inverse iIWN: overflow stego + z -> recovered cover + watermark.
                rec_cover_tensor, rec_watermark = self.iIWN.forward(rec_overflow_stego / 255., rec_z, True, True)

            # ==================== Convert the recovered image ====================
            # Convert the tensor back to a NumPy image.
            if rec_cover_tensor.shape[1] == 1:  # Grayscale image.
                rec_cover = torch.round(rec_cover_tensor * 255.)[0][0].detach().cpu().numpy()
            else:  # RGB image.
                rec_cover = torch.round(rec_cover_tensor * 255.)[0].permute(1, 2, 0).detach().cpu().numpy()
            rec_cover = np.uint8(rec_cover)

            # ==================== Verify integrity ====================
            # Compare the recovered cover's hash with the embedded hash.
            rec_cover_img_hash = sha256_of_image_array(rec_cover)
            rec_cover_img_hash_bitstream = sha256_to_bitstream(rec_cover_img_hash)

            # ==================== Decode the message ====================
            # Process and decode the recovered watermark.
            rec_watermark = torch.round(torch.clip(rec_watermark, 0., 1.))
            rec_watermark = rec_watermark[0].detach().cpu().numpy().astype(int).tolist()
            # Truncate to a multiple of eight bits for byte alignment.
            valid_part = rec_watermark[:(len(rec_watermark) // 8) * 8]
            isdecode, decoded_data = self.bch.Decode(valid_part)

            # ==================== Determine the attack flag ====================
            if np.array_equal(rec_cover_img_hash_bitstream, cover_img_hash_bitstream):
                return False, Image.fromarray(rec_cover), decoded_data
            else:
                return True, Image.fromarray(rec_cover), decoded_data
        else:
            # RDH extraction failed.
            return True, None, None

    def decode(self, stego_img: ndarray) -> Tuple[bool, str]:
        """Extract and decode a text message without recovering the cover.

        Use random latent noise instead of stored auxiliary information and return
        (success, message). Use recover when lossless cover reconstruction is needed.
        """
        # ==================== Load and preprocess the image ====================
        stego_img = np.uint8(stego_img)
        stego_img_tensor = self.transform(stego_img).unsqueeze(0).to(self.device)

        # ==================== Generate random latent noise ====================
        # Message-only decoding substitutes random noise for auxiliary latent z.
        sampled_z = torch.randn(size=(1, self.bit_length)).to(self.device)

        # ==================== Extract the watermark ====================
        with torch.no_grad():
            # Decode: stego image + random latent noise -> extracted watermark.
            _, _ext_watermark = self.iIWN.forward(stego_img_tensor, sampled_z, True, True)

        # ==================== Decode the message ====================
        _ext_watermark = torch.round(torch.clip(_ext_watermark, 0., 1.))
        ext_watermark = _ext_watermark[0].detach().cpu().numpy().astype(int).tolist()
        # Truncate to a multiple of eight bits for byte alignment.
        valid_part = ext_watermark[:(len(ext_watermark) // 8) * 8]
        isdecode, decoded_data = self.bch.Decode(valid_part)
        return isdecode, decoded_data

    def encode_bits(self, cover_img: ndarray, watermark: list) -> Tuple[bool, Any]:
        """Embed raw watermark bits into a NumPy cover image.

        The bit list must match the model's bit_length. BCH encoding is not applied;
        add error correction externally if needed. Return (success, stego_image).
        """
        # ==================== Load the cover image ====================
        cover_img = np.uint8(cover_img)

        # ==================== Hash the cover image ====================
        cover_img_hash = sha256_of_image_array(cover_img)
        cover_img_hash_bitstream = sha256_to_bitstream(cover_img_hash)

        # ==================== Preprocess the image ====================
        cover_img_tensor = self.transform(cover_img).unsqueeze(0).to(self.device)
        # Convert the raw watermark bits to a tensor without BCH encoding.
        secret_tensor = torch.as_tensor(watermark, dtype=torch.float32).unsqueeze(0).to(self.device)

        # ==================== Neural watermark embedding ====================
        with torch.no_grad():
            if self.float64:
                cover_img_tensor = cover_img_tensor.to(torch.float64)
                secret_tensor = secret_tensor.to(torch.float64)
            overflow_stego, z = self.iIWN.forward(cover_img_tensor, secret_tensor, True, False)

        # ==================== Compress auxiliary tensors ====================
        # TODO: Review the tensor compression pipeline in detail.
        stego_255 = torch.round(overflow_stego * 255.)
        z_round = torch.round(z)
        clip_stego_img, auxbits = self.tensorcoder.compress(stego_255, z_round)

        # ==================== Embed auxiliary data with RDH ====================
        # TODO: Review RDH auxiliary embedding and hash-based integrity verification.
        issuccessful, rdh_stego_img = self.rdh.embed(clip_stego_img, auxbits + cover_img_hash_bitstream)
        if not issuccessful:
            print("Reversible embedding failed")
        return issuccessful, Image.fromarray(np.uint8(rdh_stego_img))

    def recover_bits(self, stego_img: ndarray) -> Union[Tuple[bool, Any, Any], Tuple[bool, None, None]]:
        """Recover the cover image and raw bits embedded through encode_bits.

        Return (is_attacked, cover, bits), with the flag based on cover hash
        verification, or (True, None, None) on failure. No BCH decoding is applied.
        """
        # ==================== Load the stego image ====================
        stego_img = np.uint8(stego_img)

        # ==================== Extract RDH data ====================
        issuccessful, clipped_stego_img, ext_auxbits = self.rdh.extract(stego_img)
        if issuccessful:
            # Separate the cover hash from the auxiliary data.
            cover_img_hash_bitstream = ext_auxbits[-256:]
            rec_overflow_stego, rec_z = self.tensorcoder.decompress(clipped_stego_img, ext_auxbits[:-256])

            # ==================== Invert the watermark network ====================
            with torch.no_grad():
                if self.float64:
                    rec_z = rec_z.to(torch.float64)
                    rec_overflow_stego = rec_overflow_stego.to(torch.float64)
                rec_cover_tensor, rec_watermark = self.iIWN.forward(rec_overflow_stego / 255., rec_z, True, True)

            # ==================== Convert the recovered image ====================
            if rec_cover_tensor.shape[1] == 1:  # Grayscale image.
                rec_cover = torch.round(rec_cover_tensor * 255.)[0][0].detach().cpu().numpy()
            else:  # RGB image.
                rec_cover = torch.round(rec_cover_tensor * 255.)[0].permute(1, 2, 0).detach().cpu().numpy()
            rec_cover = np.uint8(rec_cover)

            # ==================== Verify integrity ====================
            rec_cover_img_hash = sha256_of_image_array(rec_cover)
            rec_cover_img_hash_bitstream = sha256_to_bitstream(rec_cover_img_hash)

            # ==================== Process recovered watermark bits ====================
            rec_watermark = torch.round(torch.clip(rec_watermark, 0., 1.))
            # Return raw bits without BCH decoding.
            rec_watermark = rec_watermark.squeeze(0).detach().cpu().numpy().astype(int).tolist()

            # ==================== Determine the attack flag ====================
            if np.array_equal(rec_cover_img_hash_bitstream, cover_img_hash_bitstream):
                return False, Image.fromarray(rec_cover), rec_watermark
            else:
                return True, Image.fromarray(rec_cover), rec_watermark
        else:
            return True, None, None

    def decode_bits(self, stego_img: ndarray) -> list:
        """Extract raw watermark bits using random latent noise.

        Return a binary list without BCH decoding or cover recovery. Use
        recover_bits when lossless reconstruction is needed.
        """
        # ==================== Load and preprocess the image ====================
        stego_img = np.uint8(stego_img)
        stego_img_tensor = self.transform(stego_img).unsqueeze(0).to(self.device)

        # ==================== Generate random latent noise ====================
        sampled_z = torch.randn(size=(1, self.bit_length)).to(self.device)

        # ==================== Extract the watermark ====================
        with torch.no_grad():
            _, _ext_watermark = self.iIWN.forward(stego_img_tensor, sampled_z, True, True)

        # ==================== Process extracted watermark bits ====================
        _ext_watermark = torch.round(torch.clip(_ext_watermark, 0., 1.))
        ext_watermark = _ext_watermark[0].detach().cpu().numpy().astype(int).tolist()
        return ext_watermark
