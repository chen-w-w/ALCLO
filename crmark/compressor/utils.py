# ==================== 导入依赖库 ====================
import os
import re
import string

import torch
import lpips      # 感知损失库
import bchlib     # BCH 纠错码库
import kornia     # 计算机视觉库
import hashlib    # 哈希算法库
import numpy as np
from torch import nn, Tensor


# ==================== 损失函数类 ====================
class LPIPSLoss(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) 感知损失。

    使用预训练的 AlexNet 提取特征，计算输入图像和目标图像在特征空间的距离，
    比传统的 MSE 更符合人眼感知。
    """

    def __init__(self):
        super().__init__()
        # 使用 AlexNet 作为特征提取器
        self.lpips = lpips.LPIPS(net="alex", verbose=False)

    def __call__(self, input_image, target_image):
        """
        计算 LPIPS 损失。

        参数:
            input_image: 输入图像，范围 [0, 1]。
            target_image: 目标图像，范围 [0, 1]。

        返回:
            LPIPS 损失值。
        """
        # 如果是灰度图，复制为三通道
        if input_image.shape[1] == 1:
            input_image = input_image.repeat(1, 3, 1, 1)
            target_image = target_image.repeat(1, 3, 1, 1)

        # 归一化到 [-1, 1] 范围（LPIPS 要求）
        normalized_input = input_image.clamp(0, 1.) * 2 - 1
        normalized_encoded = target_image.clamp(0, 1.) * 2 - 1

        lpips_loss = self.lpips(normalized_input, normalized_encoded).mean()
        return lpips_loss


# ==================== 随机舍入类 ====================
class StochasticRound:
    """
    随机舍入（Stochastic Rounding）操作类。

    在神经网络量化中，随机舍入可以在训练时提供梯度信号，
    在推理时提供确定性的硬舍入结果。

    原理：
    - 软舍入（训练）：添加随机噪声，保持梯度流动
    - 硬舍入（推理）：直接四舍五入到最近整数
    """

    def __init__(self, scale=1 / 255.):
        """
        初始化随机舍入操作。

        参数:
            scale: 缩放因子，默认 1/255（对应8位图像精度）。
        """
        super().__init__()
        self.scale = scale

    def __call__(self, x, hard_round):
        """
        执行随机舍入。

        参数:
            x: 输入张量。
            hard_round: 是否使用硬舍入（True=推理模式，False=训练模式）。

        返回:
            舍入后的张量。
        """
        # 按缩放因子缩放输入
        scale_x = x / self.scale

        # 执行舍入操作（使用 detach 保持梯度）
        round_out = scale_x + (torch.round(scale_x) - scale_x).detach()

        # 添加随机噪声（用于训练时的随机舍入）
        out = round_out + torch.rand_like(x) - 0.5

        # 根据模式返回结果
        if hard_round:
            return round_out * self.scale  # 硬舍入：确定性结果
        return out * self.scale  # 软舍入：带噪声，保持梯度


# ==================== 溢出惩罚损失类 ====================
class PenalityLoss(nn.Module):
    """
    溢出像素惩罚损失。

    用于惩罚超出有效范围 [0, max_value] 的像素值，
    防止神经网络生成溢出（>255）或下溢（<0）的像素。
    """

    def __init__(self, max_value=1.):
        """
        初始化惩罚损失。

        参数:
            max_value: 最大允许像素值，默认 1（归一化范围）。
        """
        super().__init__()
        self.max_value = max_value
        self.MSE = nn.MSELoss(reduce=True)

    def __call__(self, input_tensor):
        """
        计算溢出惩罚损失。

        参数:
            input_tensor: 输入张量。

        返回:
            惩罚损失值。
        """
        # 惩罚小于0的像素（下溢）
        loss_0 = self.MSE(torch.relu(-input_tensor), torch.zeros_like(input_tensor))

        # 惩罚大于max_value的像素（溢出）
        loss_255 = self.MSE(torch.relu(input_tensor - self.max_value), torch.zeros_like(input_tensor))

        # 总损失为两者之和
        loss = loss_0 + loss_255
        return loss


# ==================== 图像处理工具函数 ====================
def normalize(input_image):
    """
    将输入图像张量归一化到 [0, 1] 范围。

    使用 min-max 归一化：
    normalized = (x - min) / (max - min)

    参数:
        input_image: 输入图像张量。

    返回:
        归一化后的图像张量。
    """
    min_vals = input_image.amin(dim=(1, 2, 3), keepdim=True)
    max_vals = input_image.amax(dim=(1, 2, 3), keepdim=True)
    normalized_img = (input_image - min_vals) / (max_vals - min_vals + 1e-5)  # 防止除零
    return normalized_img


def extract_accuracy(ext_secret, secret, max_value=1.):
    """
    计算提取的秘密图像的准确率。

    参数:
        ext_secret: 提取的秘密图像。
        secret: 真实的秘密图像（ground truth）。
        max_value: 最大像素值，默认 1。

    返回:
        准确率值（0-1之间）。
    """
    acc = 1.0 - (torch.abs(torch.round(ext_secret.clamp(0., max_value)) - secret).mean())
    return acc.item()


def overflow_num(stego, mode, min_value=0., max_value=1.):
    """
    计算隐写图像中溢出像素的数量。

    参数:
        stego: 隐写图像张量。
        mode: 溢出模式（0=下溢，255=溢出）。
        min_value: 最小允许像素值，默认 0。
        max_value: 最大允许像素值，默认 1。

    返回:
        平均溢出像素数量。
    """
    assert mode in [0, 255]
    if mode == 0:
        overflow_pixel_n = torch.sum(StochasticRound()(stego, True) < min_value, dim=(1, 2, 3)).float().mean()
    else:
        overflow_pixel_n = torch.sum(StochasticRound()(stego, True) > max_value, dim=(1, 2, 3)).float().mean()
    return overflow_pixel_n.item()


def compute_psnr(input_image, target_image, max_value=1.):
    """
    计算两幅图像之间的 PSNR（峰值信噪比）。

    参数:
        input_image: 输入图像张量。
        target_image: 目标图像张量。
        max_value: 最大像素值，默认 1。

    返回:
        平均 PSNR 值（dB）。
    """
    # 应用随机舍入并限制到 [0, 1] 范围
    input_image = StochasticRound()(input_image.clamp(0., 1.), True)
    target_image = StochasticRound()(target_image.clamp(0., 1.), True)

    average_psnr = kornia.metrics.psnr(input_image, target_image, max_value).mean()
    return average_psnr.item()


def quantize_image(input_image):
    """
    使用随机舍入对输入图像进行量化。

    参数:
        input_image: 输入图像张量，值范围 [0, 1]。

    返回:
        量化后的图像张量。
    """
    # 应用随机舍入（硬舍入模式）
    input_image = StochasticRound()(input_image.clamp(0., 1.), True)
    return input_image


def quantize_residual_image(input_image, target_image):
    """
    量化残差图像（输入图像与目标图像的差值）。

    用于可视化嵌入造成的像素变化。

    参数:
        input_image: 输入图像张量。
        target_image: 目标图像张量。

    返回:
        量化后的残差图像。
    """
    # 计算残差（差值），放大10倍以便可视化
    shift = (input_image - target_image) * 10.
    # 归一化并应用随机舍入
    input_image = StochasticRound()(normalize(shift), True)
    return input_image


# ==================== 模型管理工具函数 ====================
def find_latest_model(model_dir):
    """
    在给定目录中查找最新的模型文件（epoch 号最大的 .pth 文件）。

    参数:
        model_dir: 包含模型文件的目录路径。

    返回:
        最新模型文件的路径，如果没有找到则返回 None。
    """
    max_epoch = -1
    latest_model_path = None

    # 正则表达式匹配 "model_{epoch}.pth" 格式
    pattern = re.compile(r"model_(\d+)\.pth")

    for file_name in os.listdir(model_dir):
        match = pattern.match(file_name)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_model_path = os.path.join(model_dir, file_name)

    return latest_model_path


# ==================== 哈希工具函数 ====================
def sha256_of_image_array(img_array):
    """
    计算 NumPy 图像数组的 SHA-256 哈希值。

    用于图像完整性验证，检测图像是否被篡改。

    参数:
        img_array: 输入图像数组。

    返回:
        64字符的十六进制 SHA-256 哈希字符串。
    """
    # 将图像数组转换为字节序列
    img_bytes = img_array.tobytes()
    # 计算 SHA-256 哈希并返回十六进制表示
    sha256_hash = hashlib.sha256(img_bytes).hexdigest()
    return sha256_hash


def sha256_to_bitstream(sha256_str):
    """
    将 64 字符的十六进制 SHA-256 字符串转换为 256 位比特流。

    每个十六进制字符代表 4 位（16 = 2^4），所以 64 个字符 = 256 位。

    参数:
        sha256_str: 64 字符的十六进制 SHA-256 哈希字符串。

    返回:
        256 个二进制值（0 或 1）的列表。
    """
    bits = np.zeros(256, dtype=np.uint8)

    # 遍历 SHA-256 字符串的每个字符
    for i, hex_char in enumerate(sha256_str):
        # 将十六进制字符转换为整数（0-15）
        val = int(hex_char, 16)
        # 将 4 位值转换为二进制，按大端序存储
        for j in range(4):
            bits[i * 4 + (3 - j)] = (val >> j) & 1

    return bits.tolist()


# ==================== BCH 纠错码类 ====================
class BCH:
    """
    BCH (Bose-Chaudhuri-Hocquenghem) 纠错码封装类。

    BCH 是一种循环纠错码，可以纠正多个比特错误。
    用于增强水印的鲁棒性，即使部分比特被破坏也能正确恢复消息。
    """

    def __init__(self, BCH_POLYNOMIAL_=501, BCH_BITS_=12):
        """
        初始化 BCH 编码器。

        参数:
            BCH_POLYNOMIAL_: BCH 多项式，默认 501。
            BCH_BITS_: 纠错能力（可纠正的比特数），默认 12。
        """
        self.bch = bchlib.BCH(BCH_BITS_, BCH_POLYNOMIAL_)

    def Encode(self, data_: str):
        """
        编码字符串为带纠错码的比特流。

        参数:
            data_: 要编码的字符串。

        返回:
            编码后的比特列表（数据 + 纠错码）。
        """
        # 将字符串转换为字节数组
        data_ = bytearray(data_, 'utf-8')
        # 计算纠错码（ECC）
        ecc = self.bch.encode(data_)
        # 合并数据和纠错码
        packet = data_ + ecc
        # 转换为二进制字符串
        packet_binary = ''.join(format(x, '08b') for x in packet)
        # 转换为比特列表
        secret_ = [int(x) for x in packet_binary]
        return secret_

    def Decode(self, secret_: list):
        """
        解码比特流并纠正错误。

        参数:
            secret_: 接收到的比特列表（可能包含错误）。

        返回:
            元组 (success, decoded_data):
            - success: 是否成功解码（True/False）
            - decoded_data: 解码后的字符串
        """
        # 将比特列表转换为二进制字符串
        packet_binary = "".join([str(int(bit)) for bit in secret_])
        # 转换为字节
        packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        # 分离数据和纠错码
        data_, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]

        # 尝试解码并纠正错误
        bit_flips = self.bch.decode(data_, ecc)

        if bit_flips != -1:
            # 解码成功，纠正错误
            self.bch.correct(data_, ecc)
            return True, data_.decode('utf-8')
        else:
            # 解码失败（错误太多无法纠正）
            return False, data_.decode('utf-8')


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    import random

    def flip_n_bits(bits: list, n: int) -> list:
        """随机翻转 n 个比特（模拟传输错误）。"""
        flipped = bits.copy()
        bit_indices = random.sample(range(len(bits)), n)
        for idx in bit_indices:
            flipped[idx] ^= 1
        return flipped

    def generate_random_string(n: int) -> str:
        """生成随机字符串。"""
        characters = string.ascii_letters + string.digits
        return ''.join(random.choices(characters, k=n))

    # 测试 BCH 编解码
    bch_codec = BCH()
    original_data = generate_random_string(20)
    print("原始数据:", original_data)

    # 编码
    secret = bch_codec.Encode(original_data)
    print(f"编码后比特数: {len(secret)}")

    # 模拟错误（翻转 n 个比特）
    n = 0  # 设置错误比特数
    noisy_secret = flip_n_bits(secret, n)
    print(f"添加 {n} 个错误比特")

    # 解码
    isdecode, decoded_data = bch_codec.Decode(noisy_secret)

    if isdecode:
        print("解码数据:", decoded_data)
        print("成功!" if decoded_data == original_data else "不匹配!")
    else:
        print("解码失败（错误太多无法纠正）")
