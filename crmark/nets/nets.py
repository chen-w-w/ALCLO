# ==================== 导入依赖库 ====================
import argparse
from crmark.compressor.utils import *


# ==================== 基础卷积模块 ====================
class ConvTP2d(nn.Module):
    """
    转置卷积模块（用于上采样）。

    结构：ConvTranspose2d + LeakyReLU
    """

    def __init__(self, input_dim, output_dim, kernel_size=2, stride=2, padding=0, bias=True):
        super(ConvTP2d, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Conv2D(nn.Module):
    """
    标准卷积模块。

    结构：Conv2d + LeakyReLU
    """

    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True):
        super(Conv2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


# ==================== 空间注意力模块 ====================
class SpatialAttention(nn.Module):
    """
    空间注意力模块（CBAM风格）。

    结构：
    1. 对输入特征分别进行最大池化和平均池化
    2. 拼接两个池化结果
    3. 通过卷积和Sigmoid生成注意力权重
    4. 与原特征相乘

    作用：让网络关注重要的空间位置。
    """

    def __init__(self, input_dim, bias):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.ave_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(input_dim * 2, input_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        # 最大池化 + 平均池化
        max_out = self.max_pool(x)
        avg_out = self.ave_pool(x)
        # 拼接后卷积生成注意力图
        cat_out = torch.cat([max_out, avg_out], dim=1)
        attention = torch.sigmoid(self.conv(cat_out))
        # 加权
        return attention * x


# ==================== 下采样编码器 ====================
class BaseFuncDown(nn.Module):
    """
    下采样编码器，将图像特征编码为比特向量。

    支持两种输出模式：
    1. FC模式（fc=True）：使用全连接层输出比特向量
    2. Conv模式（fc=False）：使用卷积直接输出方形特征

    结构：
    输入 → [Conv + SpatialAttention + DownConv]×N → FC/Conv → 比特向量
    """

    def __init__(self, img_size, input_dim, bit_length, min_size=16, base_conv=8, bias=True, fc=True):
        """
        初始化下采样编码器。

        参数:
            img_size: 输入特征图尺寸
            input_dim: 输入通道数
            bit_length: 输出比特长度
            min_size: 最小特征图尺寸
            base_conv: 基础卷积核数量
            bias: 是否使用偏置
            fc: 是否使用全连接层输出
        """
        super(BaseFuncDown, self).__init__()
        self.min_size = min_size
        self.activation = nn.LeakyReLU(inplace=True)

        # 初始卷积
        self.first_conv = Conv2D(input_dim, base_conv, 3, 1, 1, bias=bias)
        conv_input_dim = base_conv

        # 构建下采样模块
        conv_list = []
        conv_num = int(np.log2(img_size // self.min_size))  # 计算下采样层数
        for i in range(1, conv_num + 1):
            conv_list += [
                Conv2D(conv_input_dim, (i + 1) * base_conv, 3, 1, 1, bias=bias),
                SpatialAttention((i + 1) * base_conv, bias=bias),
                Conv2D((i + 1) * base_conv, (i + 1) * base_conv, 3, 2, 1, bias=bias),  # 下采样
            ]
            conv_input_dim = (i + 1) * base_conv

        # 输出层：FC模式或Conv模式
        if fc:
            # FC模式：展平后全连接
            self.feature_down = nn.Sequential(
                nn.Flatten(),
                nn.Linear(conv_input_dim * min_size * min_size, bit_length, bias=bias),
                nn.LeakyReLU(inplace=True),
            )
        else:
            # Conv模式：通过卷积调整尺寸
            sqrt_bit_length = int(np.sqrt(bit_length))
            stride = min_size // sqrt_bit_length
            kernel_size = min_size - sqrt_bit_length * stride + stride
            self.feature_down = nn.Sequential(
                Conv2D((conv_num + 1) * base_conv, 1, kernel_size, stride, 0, bias=bias),
                nn.Flatten()
            )

        self.downsampleblock = nn.Sequential(*conv_list)

    def forward(self, x):
        """
        前向传播。

        参数:
            x: 输入特征图 [B, C, H, W]

        返回:
            比特向量 [B, bit_length]
        """
        x = self.first_conv(x)
        x = self.downsampleblock(x)
        x = self.feature_down(x)
        return x


# ==================== 上采样解码器 ====================
class BaseFuncUp(nn.Module):
    """
    上采样解码器，将比特向量解码为图像特征。

    支持两种输入模式：
    1. FC模式（fc=True）：全连接层接收比特向量
    2. Conv模式（fc=False）：卷积接收方形特征

    结构：
    比特向量 → FC/Conv → 特征图 → [ConvTP + SpatialAttention + Conv]×N → 输出
    """

    def __init__(self, img_size, output_dim, bit_length, min_size=16, base_conv=8, bias=True, fc=True):
        """
        初始化上采样解码器。

        参数:
            img_size: 输出图像尺寸
            output_dim: 输出通道数
            bit_length: 输入比特长度
            min_size: 最小特征图尺寸
            base_conv: 基础卷积核数量
            bias: 是否使用偏置
            fc: 是否使用全连接层输入
        """
        super(BaseFuncUp, self).__init__()
        self.min_size = min_size
        self.activation = nn.LeakyReLU(inplace=True)

        feature_dim = (int(np.log2(img_size // self.min_size)) + 1) * base_conv

        # 输入层：FC模式或Conv模式
        if fc:
            # FC模式：全连接展开为特征图
            self.feature_up = nn.Sequential(
                nn.Linear(bit_length, feature_dim * min_size * min_size, bias=bias),
                nn.Unflatten(1, (feature_dim, min_size, min_size)),
                nn.LeakyReLU(inplace=True),
            )
        else:
            # Conv模式：通过转置卷积上采样
            sqrt_bit_length = int(np.sqrt(bit_length))
            stride = min_size // sqrt_bit_length
            kernel_size = min_size - sqrt_bit_length * stride + stride
            self.feature_up = nn.Sequential(
                nn.Unflatten(1, (1, sqrt_bit_length, sqrt_bit_length)),
                ConvTP2d(1, feature_dim, kernel_size, stride, 0, bias=bias)
            )

        # 构建上采样模块
        upsample_list = []
        upsample_num = int(np.log2(img_size // self.min_size))
        for i in range(upsample_num, 0, -1):
            upsample_list += [
                ConvTP2d(feature_dim, i * base_conv, 4, 2, 1, bias=bias),  # 上采样
                SpatialAttention(i * base_conv, bias=bias),
                Conv2D(i * base_conv, i * base_conv, 3, 1, 1, bias=bias),
            ]
            feature_dim = i * base_conv

        self.final_conv = Conv2D(feature_dim, output_dim, 3, 1, 1, bias=bias)
        self.upsampleblock = nn.Sequential(*upsample_list)

    def forward(self, x):
        """
        前向传播。

        参数:
            x: 比特向量 [B, bit_length]

        返回:
            图像特征 [B, output_dim, H, W]
        """
        x = self.feature_up(x)
        x = self.upsampleblock(x)
        x = self.final_conv(x)
        return x


# ==================== 可逆块（核心组件）====================
class InvertibleBlock(nn.Module):
    """
    可逆神经网络块（Invertible Block），CRMark的核心组件。

    基于仿射耦合层（Affine Coupling Layer）实现可逆变换：
    - 正向：将封面图像和水印信息融合为隐写图像和辅助信息
    - 逆向：从隐写图像和辅助信息中恢复封面图像和水印

    数学原理：
    正向：
        y1 = x1 + φ(x2)          # 图像 + 水印变换
        y2 = x2 ⊙ exp(s(y1)) + t(y1)  # 水印 * 缩放 + 偏移

    逆向：
        y2 = (x2 - t(x1)) / exp(s(x1))
        y1 = x1 - φ(y2)

    其中：
    - φ = ui（上采样网络）：将比特向量转换为图像特征
    - s = si（下采样网络）：生成缩放因子
    - t = qi（下采样网络）：生成偏移量
    """

    def __init__(self, img_size, input_dim, bit_length, min_size=16, base_conv=8, bias=True, clamp=4., fc=False):
        """
        初始化可逆块。

        参数:
            img_size: 输入图像尺寸
            input_dim: 输入通道数
            bit_length: 水印比特长度
            min_size: 最小特征图尺寸
            base_conv: 基础卷积核数量
            bias: 是否使用偏置
            clamp: Sigmoid输出缩放系数（控制变换强度）
            fc: 是否使用全连接层
        """
        super().__init__()
        self.clamp = clamp

        # ui: 上采样网络，将比特向量转换为图像特征（φ函数）
        self.ui = BaseFuncUp(img_size, input_dim, bit_length, min_size, base_conv, bias, fc)

        # si: 下采样网络，生成缩放因子（s函数）
        self.si = BaseFuncDown(img_size, input_dim, bit_length, min_size, base_conv, bias, fc)

        # qi: 下采样网络，生成偏移量（t函数）
        self.qi = BaseFuncDown(img_size, input_dim, bit_length, min_size, base_conv, bias, fc)

        # 随机舍入模块（用于训练时保持梯度）
        self.round_img = StochasticRound(scale=1. / 255.)  # 图像舍入
        self.round_bit = StochasticRound(scale=1.)         # 比特舍入

    def sigmoid_scale(self, s):
        """带缩放的Sigmoid激活。"""
        return self.clamp * torch.sigmoid(s)

    def forward(self, x1, x2, hard_round, reverse=False):
        """
        可逆变换的前向/逆向传播。

        参数:
            x1: 第一部分输入（图像）[B, C, H, W]
            x2: 第二部分输入（水印比特）[B, bit_length]
            hard_round: 是否使用硬舍入（True=推理，False=训练）
            reverse: 是否执行逆向变换（True=恢复，False=嵌入）

        返回:
            y1, y2: 变换后的输出
        """
        if not reverse:
            # ============ 正向：嵌入水印 ============
            # y1 = x1 + φ(x2): 图像 + 上采样后的水印特征
            y1 = x1 + self.round_img(self.ui(x2), hard_round)

            # y2 = x2 * exp(s(y1)) + t(y1): 水印经缩放和偏移
            scale = self.sigmoid_scale(self.si(y1))
            y2 = x2 * self.round_bit(torch.exp(scale), hard_round) + self.round_bit(self.qi(y1), hard_round)
        else:
            # ============ 逆向：提取水印 ============
            # y2 = (x2 - t(x1)) / exp(s(x1))
            scale = self.sigmoid_scale(self.si(x1))
            y2 = (x2 - self.round_bit(self.qi(x1), hard_round)) / self.round_bit(torch.exp(scale), hard_round)

            # y1 = x1 - φ(y2)
            y1 = x1 - self.round_img(self.ui(y2), hard_round)

        return y1, y2


# ==================== 完整模型（堆叠可逆块）====================
class Model(nn.Module):
    """
    CRMark完整模型，堆叠多个可逆块实现深度可逆变换。

    结构：
    输入 → [InvertibleBlock]×k → 输出

    特点：
    - 每个可逆块都是双射（bijective）的
    - 整个网络也是可逆的
    - 正向：嵌入水印
    - 逆向：提取水印并恢复图像
    """

    def __init__(self, img_size, channel_dim, bit_length, k, min_size=16, fc=False):
        """
        初始化模型。

        参数:
            img_size: 输入图像尺寸
            channel_dim: 图像通道数（1=灰度，3=RGB）
            bit_length: 水印比特长度
            k: 可逆块堆叠数量（网络深度）
            min_size: 最小特征图尺寸
            fc: 是否使用全连接层
        """
        super().__init__()
        self.k = k
        self.fc = fc
        self.min_size = min_size
        self.img_size = img_size
        self.bit_length = bit_length
        self.channel_dim = channel_dim

        # 堆叠k个可逆块
        self.inn_blocks = nn.Sequential(
            *[InvertibleBlock(img_size, channel_dim, bit_length, min_size=min_size, fc=fc) for _ in range(k)]
        )

        # 损失函数
        self.MSE = nn.MSELoss(reduce=True)      # 均方误差
        self.PMSE = PenalityLoss()               # 溢出惩罚
        self.LLoss = LPIPSLoss()                 # 感知损失

    def forward(self, x1, x2, hard_round, reverse=False):
        """
        模型前向/逆向传播。

        参数:
            x1: 图像 [B, C, H, W]
            x2: 水印比特 [B, bit_length]
            hard_round: 是否硬舍入
            reverse: 是否逆向（True=恢复，False=嵌入）

        返回:
            x1, x2: 变换后的输出
        """
        if not reverse:
            # 正向：依次通过每个可逆块
            for block in self.inn_blocks:
                x1, x2 = block(x1, x2, hard_round, False)
        else:
            # 逆向：反向依次通过每个可逆块
            for block in self.inn_blocks[::-1]:
                x1, x2 = block(x1, x2, hard_round, True)
        return x1, x2

    def load_model(self, model_path, optim_blocks=None, scheduler_blocks=None):
        """
        加载预训练模型。

        参数:
            model_path: 模型文件路径
            optim_blocks: 优化器（可选）
            scheduler_blocks: 学习率调度器（可选）

        返回:
            (start_epoch, global_step, lambda_secret) 或 None
        """
        try:
            if os.path.exists(model_path):
                save_dict = torch.load(model_path, weights_only=False)

                # 加载训练状态
                start_epoch = save_dict['param_dict']['epoch']
                global_step = save_dict['param_dict']['global_step']
                lambda_secret = save_dict['param_dict']['lambda_secret']

                # 加载模型参数
                self.load_state_dict(save_dict['model_state_dict'])

                # 加载优化器状态
                if optim_blocks is not None and 'optimizer_state_dict' in save_dict:
                    optim_blocks.load_state_dict(save_dict['optimizer_state_dict'])

                # 加载调度器状态
                if scheduler_blocks is not None and 'scheduler_state_dict' in save_dict:
                    scheduler_blocks.load_state_dict(save_dict['scheduler_state_dict'])

                print(f"✓ 模型加载成功: {model_path}")
                return start_epoch, global_step, lambda_secret
            else:
                print(f"✗ 模型文件不存在: {model_path}")
                return None
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            return None

    def save_model(self, args, optim_blocks=None, scheduler_blocks=None, now_global_step=0, epoch=0):
        """
        保存模型。

        参数:
            args: 配置参数
            optim_blocks: 优化器（可选）
            scheduler_blocks: 学习率调度器（可选）
            now_global_step: 当前全局步数
            epoch: 当前轮数
        """
        # 创建保存目录
        model_dir = f"{args.checkpoint_path}/{args.train_name}"
        os.makedirs(model_dir, exist_ok=True)

        # 构建保存字典
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

        # 可选：保存优化器和调度器状态
        if optim_blocks is not None:
            save_dict['optimizer_state_dict'] = optim_blocks.state_dict()
        if scheduler_blocks is not None:
            save_dict['scheduler_state_dict'] = scheduler_blocks.state_dict()

        # 保存到文件
        save_path = f"{model_dir}/model_{epoch}.pth"
        torch.save(save_dict, save_path)
        print(f"✓ 模型已保存: {save_path}")


# ==================== 主程序入口（测试脚本）====================
if __name__ == "__main__":
    print("=" * 60)
    print("CRMark 可逆水印网络 - 模型测试")
    print("=" * 60)

    # -------------------- 参数配置 --------------------
    parser = argparse.ArgumentParser(description="CRMark可逆水印网络测试")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU设备ID")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--im_size", type=int, default=400, help="图像尺寸")
    parser.add_argument("--hard_round", type=bool, default=False, help="是否硬舍入")
    parser.add_argument("--bit_length", type=int, default=100, help="水印比特长度")
    parser.add_argument("--min_size", type=int, default=25, help="最小特征图尺寸")
    parser.add_argument("--k", type=int, default=5, help="可逆块数量")
    parser.add_argument("--channel_dim", type=int, default=1, help="图像通道数（1=灰度，3=RGB）")
    parser.add_argument("--seed", type=int, default=100, help="随机种子")
    args = parser.parse_args()

    # -------------------- 设备设置 --------------------
    print("\n[设备信息]")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    print(f"  GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 设置计算设备
    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
    else:
        args.device = torch.device("cpu")
    print(f"  使用设备: {args.device}")

    # -------------------- 环境初始化 --------------------
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    print(f"\n[环境设置]")
    print(f"  随机种子: {args.seed}")
    print(f"  数据类型: float64")

    # -------------------- 模型创建 --------------------
    print(f"\n[模型配置]")
    print(f"  图像尺寸: {args.im_size}x{args.im_size}")
    print(f"  通道数: {args.channel_dim}")
    print(f"  水印比特: {args.bit_length}")
    print(f"  可逆块数: {args.k}")
    print(f"  最小特征图: {args.min_size}x{args.min_size}")

    model = Model(
        img_size=args.im_size,
        channel_dim=args.channel_dim,
        bit_length=args.bit_length,
        k=args.k,
        min_size=args.min_size
    ).to(args.device)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # -------------------- 性能分析 --------------------
    print(f"\n[性能分析]")
    cover = torch.randint(0, 256, (args.batch_size, args.channel_dim, args.im_size, args.im_size)).to(args.device) / 255.
    secret = torch.randint(0, 2, (args.batch_size, args.bit_length)).to(args.device).float()

    try:
        from thop import profile
        flops, params = profile(model, inputs=(cover, secret, True, False))
        print(f"  FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"  参数量: {params / 1e6:.2f} M")
    except ImportError:
        print("  提示: 安装thop库可查看FLOPs (pip install thop)")

    # -------------------- 可逆性测试 --------------------
    print(f"\n[可逆性测试]")

    # 正向：嵌入水印
    stego, z = model(cover, secret, hard_round=True, reverse=False)
    print(f"  正向完成: cover + secret → stego + z")

    # 量化（模拟实际存储）
    stego_quant = torch.round(stego * 255.) / 255.
    z_quant = torch.round(z)

    # 逆向：恢复
    cover_rec, secret_rec = model(stego_quant, z_quant, hard_round=True, reverse=True)
    print(f"  逆向完成: stego + z → cover' + secret'")

    # 计算误差
    img_error = torch.mean(torch.abs(torch.round(cover_rec * 255) - torch.round(cover * 255)))
    secret_error = torch.mean(torch.abs(torch.round(secret_rec) - torch.round(secret)))

    print(f"\n[恢复误差]")
    print(f"  图像误差: {img_error.item():.6f} (理想=0)")
    print(f"  水印误差: {secret_error.item():.6f} (理想=0)")

    if img_error < 1e-5 and secret_error < 1e-5:
        print(f"\n✓ 测试通过！网络具有良好的可逆性。")
    else:
        print(f"\n✗ 测试未通过！误差较大，请检查模型实现。")
