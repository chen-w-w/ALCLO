# ==================== Imports ====================
import argparse
from crmark.compressor.utils import *


# ==================== Convolution blocks ====================
class ConvTP2d(nn.Module):
    """Upsampling block consisting of ConvTranspose2d and LeakyReLU."""

    def __init__(self, input_dim, output_dim, kernel_size=2, stride=2, padding=0, bias=True):
        super(ConvTP2d, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Conv2D(nn.Module):
    """Convolution block consisting of Conv2d and LeakyReLU."""

    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True):
        super(Conv2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


# ==================== Spatial attention ====================
class SpatialAttention(nn.Module):
    """CBAM-style spatial attention.

    Concatenate channel-wise maximum and average pooled features, produce a
    sigmoid attention map, and multiply it with the input features.
    """

    def __init__(self, input_dim, bias):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.ave_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(input_dim * 2, input_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        # Maximum and average pooling.
        max_out = self.max_pool(x)
        avg_out = self.ave_pool(x)
        # Concatenate pooled features and convolve to generate attention weights.
        cat_out = torch.cat([max_out, avg_out], dim=1)
        attention = torch.sigmoid(self.conv(cat_out))
        # Apply attention weights.
        return attention * x


# ==================== Downsampling encoder ====================
class BaseFuncDown(nn.Module):
    """Downsample image features into a watermark vector.

    Use convolution, spatial attention, and downsampling stages followed by
    either a fully connected output (fc=True) or a convolutional output.
    """

    def __init__(self, img_size, input_dim, bit_length, min_size=16, base_conv=8, bias=True, fc=True):
        """Configure image dimensions, channels, bit length, downsampling, and output mode."""
        super(BaseFuncDown, self).__init__()
        self.min_size = min_size
        self.activation = nn.LeakyReLU(inplace=True)

        # Initial convolution.
        self.first_conv = Conv2D(input_dim, base_conv, 3, 1, 1, bias=bias)
        conv_input_dim = base_conv

        # Build the downsampling stages.
        conv_list = []
        conv_num = int(np.log2(img_size // self.min_size))  # Compute the number of downsampling layers.
        for i in range(1, conv_num + 1):
            conv_list += [
                Conv2D(conv_input_dim, (i + 1) * base_conv, 3, 1, 1, bias=bias),
                SpatialAttention((i + 1) * base_conv, bias=bias),
                Conv2D((i + 1) * base_conv, (i + 1) * base_conv, 3, 2, 1, bias=bias),  # Downsample features.
            ]
            conv_input_dim = (i + 1) * base_conv

        # Output projection: fully connected or convolutional.
        if fc:
            # Fully connected mode: flatten and project.
            self.feature_down = nn.Sequential(
                nn.Flatten(),
                nn.Linear(conv_input_dim * min_size * min_size, bit_length, bias=bias),
                nn.LeakyReLU(inplace=True),
            )
        else:
            # Convolutional mode: adjust the feature dimensions by convolution.
            sqrt_bit_length = int(np.sqrt(bit_length))
            stride = min_size // sqrt_bit_length
            kernel_size = min_size - sqrt_bit_length * stride + stride
            self.feature_down = nn.Sequential(
                Conv2D((conv_num + 1) * base_conv, 1, kernel_size, stride, 0, bias=bias),
                nn.Flatten()
            )

        self.downsampleblock = nn.Sequential(*conv_list)

    def forward(self, x):
        """Map image features [B, C, H, W] to a vector [B, bit_length]."""
        x = self.first_conv(x)
        x = self.downsampleblock(x)
        x = self.feature_down(x)
        return x


# ==================== Upsampling decoder ====================
class BaseFuncUp(nn.Module):
    """Upsample a watermark vector into image features.

    A fully connected or convolutional input projection is followed by
    transposed convolutions, spatial attention, and convolution blocks.
    """

    def __init__(self, img_size, output_dim, bit_length, min_size=16, base_conv=8, bias=True, fc=True):
        """Configure target image dimensions, bit length, upsampling, and input mode."""
        super(BaseFuncUp, self).__init__()
        self.min_size = min_size
        self.activation = nn.LeakyReLU(inplace=True)

        feature_dim = (int(np.log2(img_size // self.min_size)) + 1) * base_conv

        # Input projection: fully connected or convolutional.
        if fc:
            # Fully connected mode: project the vector into a feature map.
            self.feature_up = nn.Sequential(
                nn.Linear(bit_length, feature_dim * min_size * min_size, bias=bias),
                nn.Unflatten(1, (feature_dim, min_size, min_size)),
                nn.LeakyReLU(inplace=True),
            )
        else:
            # Convolutional mode: upsample using a transposed convolution.
            sqrt_bit_length = int(np.sqrt(bit_length))
            stride = min_size // sqrt_bit_length
            kernel_size = min_size - sqrt_bit_length * stride + stride
            self.feature_up = nn.Sequential(
                nn.Unflatten(1, (1, sqrt_bit_length, sqrt_bit_length)),
                ConvTP2d(1, feature_dim, kernel_size, stride, 0, bias=bias)
            )

        # Build the upsampling stages.
        upsample_list = []
        upsample_num = int(np.log2(img_size // self.min_size))
        for i in range(upsample_num, 0, -1):
            upsample_list += [
                ConvTP2d(feature_dim, i * base_conv, 4, 2, 1, bias=bias),  # Upsample features.
                SpatialAttention(i * base_conv, bias=bias),
                Conv2D(i * base_conv, i * base_conv, 3, 1, 1, bias=bias),
            ]
            feature_dim = i * base_conv

        self.final_conv = Conv2D(feature_dim, output_dim, 3, 1, 1, bias=bias)
        self.upsampleblock = nn.Sequential(*upsample_list)

    def forward(self, x):
        """Map bits [B, bit_length] to image features [B, output_dim, H, W]."""
        x = self.feature_up(x)
        x = self.upsampleblock(x)
        x = self.final_conv(x)
        return x


# ==================== Invertible coupling block ====================
class InvertibleBlock(nn.Module):
    """Invertible affine coupling block for watermark embedding and recovery.

    Forward mapping:
        y1 = x1 + ui(x2)
        y2 = x2 * exp(sigmoid_scale(si(y1))) + qi(y1)

    The inverse recovers the watermark component before subtracting its
    upsampled features from the image. ui upsamples bits; si and qi predict
    the scale and shift from image features.
    """

    def __init__(self, img_size, input_dim, bit_length, min_size=16, base_conv=8, bias=True, clamp=4., fc=False):
        """Initialize up/downsampling networks, affine scale bounds, and rounding modules."""
        super().__init__()
        self.clamp = clamp

        # ui upsamples the bit vector into image features (phi).
        self.ui = BaseFuncUp(img_size, input_dim, bit_length, min_size, base_conv, bias, fc)

        # si downsamples image features to predict scale (s).
        self.si = BaseFuncDown(img_size, input_dim, bit_length, min_size, base_conv, bias, fc)

        # qi downsamples image features to predict shift (t).
        self.qi = BaseFuncDown(img_size, input_dim, bit_length, min_size, base_conv, bias, fc)

        # Rounding modules retain a gradient path during training.
        self.round_img = StochasticRound(scale=1. / 255.)  # Image rounding.
        self.round_bit = StochasticRound(scale=1.)         # Watermark bit rounding.

    def sigmoid_scale(self, s):
        """Apply the scaled sigmoid used by the affine coupling transform."""
        return self.clamp * torch.sigmoid(s)

    def forward(self, x1, x2, hard_round, reverse=False):
        """Apply the forward or inverse coupling transform to x1 and x2.

        x1 contains image features and x2 contains watermark bits. hard_round
        selects deterministic rounding, and reverse selects recovery over embedding.
        Return the two transformed components.
        """
        if not reverse:
            # ============ Forward: embed the watermark ============
            # y1 = x1 + phi(x2): add upsampled watermark features to the image.
            y1 = x1 + self.round_img(self.ui(x2), hard_round)

            # y2 = x2 * exp(s(y1)) + t(y1): scale and shift the watermark.
            scale = self.sigmoid_scale(self.si(y1))
            y2 = x2 * self.round_bit(torch.exp(scale), hard_round) + self.round_bit(self.qi(y1), hard_round)
        else:
            # ============ Inverse: extract the watermark ============
            # y2 = (x2 - t(x1)) / exp(s(x1))
            scale = self.sigmoid_scale(self.si(x1))
            y2 = (x2 - self.round_bit(self.qi(x1), hard_round)) / self.round_bit(torch.exp(scale), hard_round)

            # y1 = x1 - φ(y2)
            y1 = x1 - self.round_img(self.ui(y2), hard_round)

        return y1, y2


# ==================== Stacked invertible network ====================
class Model(nn.Module):
    """Stack k invertible blocks for watermark embedding and cover recovery."""

    def __init__(self, img_size, channel_dim, bit_length, k, min_size=16, fc=False):
        """Initialize k invertible blocks and the image, overflow, and perceptual losses."""
        super().__init__()
        self.k = k
        self.fc = fc
        self.min_size = min_size
        self.img_size = img_size
        self.bit_length = bit_length
        self.channel_dim = channel_dim

        # Stack k invertible blocks.
        self.inn_blocks = nn.Sequential(
            *[InvertibleBlock(img_size, channel_dim, bit_length, min_size=min_size, fc=fc) for _ in range(k)]
        )

        # Loss functions.
        self.MSE = nn.MSELoss(reduce=True)      # Mean squared error.
        self.PMSE = PenalityLoss()               # Pixel overflow penalty.
        self.LLoss = LPIPSLoss()                 # Perceptual loss.

    def forward(self, x1, x2, hard_round, reverse=False):
        """Apply all invertible blocks in forward order for embedding or reverse order for recovery."""
        if not reverse:
            # Embedding traverses the blocks in forward order.
            for block in self.inn_blocks:
                x1, x2 = block(x1, x2, hard_round, False)
        else:
            # Recovery traverses the blocks in reverse order.
            for block in self.inn_blocks[::-1]:
                x1, x2 = block(x1, x2, hard_round, True)
        return x1, x2

    def load_model(self, model_path, optim_blocks=None, scheduler_blocks=None):
        """Load model weights and optional optimizer and scheduler state.

        Return (start_epoch, global_step, lambda_secret), or None if loading fails.
        """
        try:
            if os.path.exists(model_path):
                save_dict = torch.load(model_path, weights_only=False)

                # Load training progress.
                start_epoch = save_dict['param_dict']['epoch']
                global_step = save_dict['param_dict']['global_step']
                lambda_secret = save_dict['param_dict']['lambda_secret']

                # Load model weights.
                self.load_state_dict(save_dict['model_state_dict'])

                # Load optimizer state.
                if optim_blocks is not None and 'optimizer_state_dict' in save_dict:
                    optim_blocks.load_state_dict(save_dict['optimizer_state_dict'])

                # Load scheduler state.
                if scheduler_blocks is not None and 'scheduler_state_dict' in save_dict:
                    scheduler_blocks.load_state_dict(save_dict['scheduler_state_dict'])

                print(f"Model loaded successfully: {model_path}")
                return start_epoch, global_step, lambda_secret
            else:
                print(f"Model file does not exist: {model_path}")
                return None
        except Exception as e:
            print(f"Model loading failed: {e}")
            return None

    def save_model(self, args, optim_blocks=None, scheduler_blocks=None, now_global_step=0, epoch=0):
        """Save model parameters, training progress, and optional optimizer/scheduler state."""
        # Create the checkpoint directory.
        model_dir = f"{args.checkpoint_path}/{args.train_name}"
        os.makedirs(model_dir, exist_ok=True)

        # Build the checkpoint dictionary.
        save_dict = {
            'model_state_dict': self.state_dict(),
            'param_dict': {
                'epoch': epoch,
                'global_step': now_global_step,
                'lambda_secret': args.lambda_secret,
                'bit_length': self.bit_length,
                'img_size': self.img_size,
                'channel_dim': self.channel_dim,
                'min_size': self.min_size,
                'fc': self.fc,
                'k': self.k
            }
        }

        # Optionally include optimizer and scheduler state.
        if optim_blocks is not None:
            save_dict['optimizer_state_dict'] = optim_blocks.state_dict()
        if scheduler_blocks is not None:
            save_dict['scheduler_state_dict'] = scheduler_blocks.state_dict()

        # Save the checkpoint to disk.
        save_path = f"{model_dir}/model_{epoch}.pth"
        torch.save(save_dict, save_path)
        print(f"Model saved: {save_path}")


# ==================== Standalone model example ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Invertible watermark network: model example")
    print("=" * 60)

    # -------------------- Argument configuration --------------------
    parser = argparse.ArgumentParser(description="Invertible watermark network check")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--im_size", type=int, default=400, help="Image size")
    parser.add_argument("--hard_round", type=bool, default=False, help="Enable hard rounding")
    parser.add_argument("--bit_length", type=int, default=100, help="Watermark bit length")
    parser.add_argument("--min_size", type=int, default=25, help="Minimum feature map size")
    parser.add_argument("--k", type=int, default=5, help="Number of invertible blocks")
    parser.add_argument("--channel_dim", type=int, default=1, help="Image channels: 1 for grayscale, 3 for RGB")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    args = parser.parse_args()

    # -------------------- Device selection --------------------
    print("\n[Device information]")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set the computation device.
    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
    else:
        args.device = torch.device("cpu")
    print(f"  Selected device: {args.device}")

    # -------------------- Environment initialization --------------------
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    print(f"\n[Environment]")
    print(f"  Random seed: {args.seed}")
    print(f"  Data type: float64")

    # -------------------- Model construction --------------------
    print(f"\n[Model configuration]")
    print(f"  Image size: {args.im_size}x{args.im_size}")
    print(f"  Channels: {args.channel_dim}")
    print(f"  Watermark bits: {args.bit_length}")
    print(f"  Invertible blocks: {args.k}")
    print(f"  Minimum feature map: {args.min_size}x{args.min_size}")

    model = Model(
        img_size=args.im_size,
        channel_dim=args.channel_dim,
        bit_length=args.bit_length,
        k=args.k,
        min_size=args.min_size
    ).to(args.device)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # -------------------- Performance profiling --------------------
    print(f"\n[Performance profiling]")
    cover = torch.randint(0, 256, (args.batch_size, args.channel_dim, args.im_size, args.im_size)).to(args.device) / 255.
    secret = torch.randint(0, 2, (args.batch_size, args.bit_length)).to(args.device).float()

    try:
        from thop import profile
        flops, params = profile(model, inputs=(cover, secret, True, False))
        print(f"  FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"  Parameters: {params / 1e6:.2f} M")
    except ImportError:
        print("  Install thop to inspect FLOPs: pip install thop")

    # -------------------- Reversibility check --------------------
    print(f"\n[Reversibility check]")

    # Forward: embed the watermark.
    stego, z = model(cover, secret, hard_round=True, reverse=False)
    print(f"  Forward complete: cover + secret -> stego + z")

    # Quantize to simulate stored image values.
    stego_quant = torch.round(stego * 255.) / 255.
    z_quant = torch.round(z)

    # Inverse: recover the cover and watermark.
    cover_rec, secret_rec = model(stego_quant, z_quant, hard_round=True, reverse=True)
    print(f"  Inverse complete: stego + z -> cover' + secret'")

    # Compute reconstruction errors.
    img_error = torch.mean(torch.abs(torch.round(cover_rec * 255) - torch.round(cover * 255)))
    secret_error = torch.mean(torch.abs(torch.round(secret_rec) - torch.round(secret)))

    print(f"\n[Recovery errors]")
    print(f"  Image error: {img_error.item():.6f} (ideal=0)")
    print(f"  Watermark error: {secret_error.item():.6f} (ideal=0)")

    if img_error < 1e-5 and secret_error < 1e-5:
        print(f"\nCheck passed: the network is reversible.")
    else:
        print(f"\nCheck failed: large reconstruction errors; inspect the model implementation.")
