# ==================== 导入依赖库 ====================
import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from .arithmeticcoder import ArithmeticEncoder


# ==================== 自定义算术编码器 ====================
class CustomArithmeticEncoder:
    """
    针对NumPy数组优化的算术编码器。

    用于压缩RDH产生的预测误差、drop_z等整数数据。
    支持自适应频率模型和固定频率模型。
    """

    def __init__(self, level_bits_len: int = 10, freq_bits_len: int = 20):
        """
        初始化编码器。

        参数:
            level_bits_len: 整数编码的比特长度（用于编码频率表长度）。
            freq_bits_len: 频率编码的比特长度（用于编码数据值）。
        """
        self.buffer_bits = 40  # 算术编码器使用的比特数（高精度）
        self.freq_bits_len = freq_bits_len
        self.level_bits_len = level_bits_len

    def ndarray2strlist(self, data: ndarray) -> list:
        """
        将NumPy数组展平并转换为字符串列表。

        参数:
            data: 输入数组。

        返回:
            字符串列表。
        """
        data_list = data.flatten().tolist()
        data_str_list = [str(data_item) for data_item in data_list]
        return data_str_list

    def strlist2ndarray(self, data_str_list: list) -> list:
        """
        将字符串列表转换为整数列表。

        参数:
            data_str_list: 字符串列表。

        返回:
            整数列表。
        """
        data_list = [int(data_str_item) for data_str_item in data_str_list]
        return data_list

    def datastr2bits(self, data_str_list):
        """
        将字符串列表编码为固定长度的二进制比特流。

        支持有符号整数（使用补码表示负数）。

        参数:
            data_str_list: 字符串列表。

        返回:
            比特列表（0和1）。
        """
        encoded_bits = []
        for item in data_str_list:
            num = int(item)
            if num < 0:
                num = (1 << self.freq_bits_len) + num  # 负数转补码
            binary_representation = bin(num)[2:]  # 转为二进制字符串
            padded_binary = binary_representation.zfill(self.freq_bits_len)  # 零填充到固定位数
            if len(padded_binary) > self.freq_bits_len:
                raise ValueError(f"值 {num} 无法用 {self.freq_bits_len} 比特表示。")
            encoded_bits.extend(int(bit) for bit in padded_binary)
        return encoded_bits

    def bits2datastr(self, encoded_bits):
        """
        将比特流解码回字符串列表。

        参数:
            encoded_bits: 比特列表。

        返回:
            字符串列表。
        """
        decoded_data = []
        for i in range(0, len(encoded_bits), self.freq_bits_len):
            bit_str = ''.join(map(str, encoded_bits[i:i + self.freq_bits_len]))
            num = int(bit_str, 2)
            if num >= (1 << (self.freq_bits_len - 1)):
                num -= (1 << self.freq_bits_len)  # 补码转回有符号数
            decoded_data.append(str(num))
        return decoded_data

    def integer2bits(self, integer: int):
        """
        将整数编码为固定长度的比特列表。

        参数:
            integer: 要编码的整数。

        返回:
            比特列表。
        """
        if integer < 0:
            integer = (1 << self.level_bits_len) + integer
        if integer >= (1 << self.level_bits_len):
            raise ValueError(f"值 {integer} 无法用 {self.level_bits_len} 比特表示。")
        binary_representation = bin(integer)[2:]
        padded_binary = binary_representation.zfill(self.level_bits_len)
        return [int(bit) for bit in padded_binary]

    def bits2integer(self, bits: list):
        """
        将比特列表解码为整数。

        参数:
            bits: 比特列表。

        返回:
            解码后的整数。
        """
        if len(bits) != self.level_bits_len:
            raise ValueError(f"比特列表长度必须是 {self.level_bits_len}。")
        bit_str = ''.join(str(bit) for bit in bits)
        num = int(bit_str, 2)
        if num >= (1 << (self.level_bits_len - 1)):
            num -= (1 << self.level_bits_len)
        return num

    def compress(self, data: ndarray, frequencies=None) -> list:
        """
        压缩NumPy数组。

        参数:
            data: 要压缩的数组。
            frequencies: 预定义的符号频率（None=自适应模型）。

        返回:
            编码后的比特列表。
        """
        data_str_list = self.ndarray2strlist(data)

        if frequencies is None:
            # 自适应模型：从数据中自动提取唯一值作为频率表
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
        """
        解压缩比特流。

        参数:
            data_freqs_bits: 编码的比特列表。
            frequencies: 预定义的符号频率（None=从比特流中读取）。

        返回:
            解压后的NumPy数组。
        """
        if frequencies is None:
            # 从比特流末尾提取频率表信息
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


# ==================== AC压缩器（算术编码压缩）====================
class ACCompress:
    """
    使用算术编码压缩隐写图像和潜在向量。

    功能：
    - 压缩drop_z（潜在向量）
    - 压缩stego_img的溢出像素信息
    - 将两部分比特流合并存储
    """

    def __init__(self, im_size, z_size, level_bits_len, freq_bits_len, device: str = "cpu"):
        """
        初始化AC压缩器。

        参数:
            im_size: 图像尺寸 (H, W, C)。
            z_size: 潜在向量尺寸 (H, W)。
            level_bits_len: 层级比特长度。
            freq_bits_len: 频率比特长度。
            device: 计算设备 ('cpu' 或 'cuda')。
        """
        self.mark_len = 20  # 用于编码z_bits长度的比特数
        self.im_size = im_size  # (H, W, C)
        self.z_size = z_size  # (H, W)
        self.device = device
        self.im_len = im_size[0] * im_size[1] * im_size[2]  # 图像总像素数
        self.z_len = z_size[0] * z_size[1]  # 潜在向量元素数
        self.coder = CustomArithmeticEncoder(level_bits_len=level_bits_len, freq_bits_len=freq_bits_len)

    def combine_bits(self, z_bits: list, stego_bits: list):
        """
        将z_bits和stego_bits合并为单个比特流。

        格式：[20位长度标记] + [z_bits] + [stego_bits]

        参数:
            z_bits: drop_z的编码比特。
            stego_bits: 溢出像素的编码比特。

        返回:
            合并后的比特列表。
        """
        length_bits = format(len(z_bits), f'0{self.mark_len}b')
        length_bits = [int(b) for b in length_bits]
        combined_bits = length_bits + z_bits + stego_bits
        return combined_bits

    def split_bits(self, combined_bits: list):
        """
        将合并的比特流拆分为z_bits和stego_bits。

        参数:
            combined_bits: 合并的比特列表。

        返回:
            (z_bits, stego_bits) 元组。
        """
        length_bits = combined_bits[:self.mark_len]
        z_length = int(''.join(map(str, length_bits)), 2)
        z_bits = combined_bits[self.mark_len:self.mark_len + z_length]
        stego_bits = combined_bits[self.mark_len + z_length:]
        return z_bits, stego_bits

    def encode(self, stego_img, drop_z):
        """
        编码隐写图像和潜在向量。

        参数:
            stego_img: 隐写图像张量 (1, C, H, W)。
            drop_z: 潜在向量张量 (1, H, W)。

        返回:
            (裁剪后的图像, (合并比特流, z_bits, 溢出bits))
        """
        overflow_bits_list = []
        clip_stego_img = None

        if stego_img is not None:
            # 裁剪像素值到[0, 255]并检测溢出
            clip_stego_img = torch.clip(stego_img, 0, 255)
            steg_img_numpy = stego_img.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
            overflow = np.zeros_like(steg_img_numpy)

            # 计算溢出值（超出[0, 255]范围的像素）
            overflow[steg_img_numpy > 255] = steg_img_numpy[steg_img_numpy > 255] - 255
            overflow[steg_img_numpy < 0] = 0 - steg_img_numpy[steg_img_numpy < 0]
            overflow = overflow.astype(int)

            # 使用算术编码压缩溢出数据
            overflow_bits_list = self.coder.compress(overflow.flatten())

        drop_z_bits_list = []
        if drop_z is not None:
            # 裁剪z值到允许范围并压缩
            drop_z_numpy = drop_z.squeeze(0).detach().cpu().numpy()
            drop_z_numpy = np.clip(drop_z_numpy, -2 ** self.coder.freq_bits_len + 1, 2 ** self.coder.freq_bits_len - 1)
            drop_z_numpy_clip = drop_z_numpy.astype(np.int64)
            drop_z_bits_list = self.coder.compress(drop_z_numpy_clip.flatten())

        # 根据输入可用性合并比特流
        if stego_img is None:
            data_list = drop_z_bits_list
        elif drop_z is None:
            data_list = overflow_bits_list
        else:
            data_list = self.combine_bits(drop_z_bits_list, overflow_bits_list)
            # 可选：验证合并正确性
            drop_z_bits_list, overflow_bits_list = self.split_bits(data_list)

        return clip_stego_img, (data_list, drop_z_bits_list, overflow_bits_list)

    def decode(self, clip_stego_img: Tensor, data_bits: list):
        """
        解码比特流，恢复隐写图像和潜在向量。

        参数:
            clip_stego_img: 裁剪后的隐写图像 (1, C, H, W)。
            data_bits: 编码的二进制比特列表。

        返回:
            (恢复的隐写图像, 恢复的潜在向量)。
        """
        # 创建0和255像素的掩码（用于溢出校正）
        mask_0 = (clip_stego_img == 0).int()
        mask_255 = (clip_stego_img == 255).int()

        # 拆分比特流
        drop_z_bits_list, overflow_bits_list = self.split_bits(data_bits)

        # 解压两部分数据
        drop_z_list = self.coder.decompress(drop_z_bits_list)
        overflow_list = self.coder.decompress(overflow_bits_list)

        # 重建溢出图并应用校正
        overflow = torch.as_tensor(overflow_list).reshape(self.im_size).to(self.device).permute(2, 0, 1).unsqueeze(0)
        stego_img = (clip_stego_img - mask_0 * overflow + mask_255 * overflow).float()

        # 重建潜在向量z
        rec_z = torch.as_tensor(drop_z_list).reshape(self.z_size).to(self.device).float()

        return stego_img, rec_z


# ==================== 稀疏张量压缩器 ====================
class SparseTensorCompressor:
    """
    稀疏张量压缩器。

    与ACCompress不同，该类只存储非零溢出像素的位置和值，
    适用于溢出像素较少的情况（稀疏数据）。

    压缩格式（每个非零像素）：
    [row_bits] + [col_bits] + [channel_bits] + [value_bits]
    """

    def __init__(self, image_shape: tuple, z_shape: tuple, level_bits_len: int, freq_bits_len: int, val_bits: int = 10):
        """
        初始化稀疏张量压缩器。

        参数:
            image_shape: 图像形状 (height, width, channels)。
            z_shape: drop_z张量形状。
            level_bits_len: 层级比特长度。
            freq_bits_len: 频率比特长度。
            val_bits: 溢出值编码比特数（默认10）。
        """
        self.height, self.width, self.channels = image_shape
        self.mark_len = 20
        self.z_shape = z_shape

        # 计算各维度所需的比特数
        self.row_bits = int(np.ceil(np.log2(self.height)))  # 行索引比特数
        self.col_bits = int(np.ceil(np.log2(self.width)))  # 列索引比特数
        self.channel_bits = 0 if self.channels == 1 else int(np.ceil(np.log2(self.channels)))  # 通道索引比特数
        self.value_bits = int(val_bits)  # 值编码比特数

        self.coder = CustomArithmeticEncoder(level_bits_len=level_bits_len, freq_bits_len=freq_bits_len)

    def combine_bits(self, z_bits: list, stego_bits: list) -> list:
        """
        合并z_bits和stego_bits。

        参数:
            z_bits: drop_z的比特流。
            stego_bits: 溢出数据的比特流。

        返回:
            合并后的比特列表。
        """
        length_bits = format(len(z_bits), f'0{self.mark_len}b')
        length_bits = [int(b) for b in length_bits]
        return length_bits + z_bits + stego_bits

    def split_bits(self, combined_bits: list) -> tuple:
        """
        拆分合并的比特流。

        参数:
            combined_bits: 合并的比特列表。

        返回:
            (z_bits, stego_bits) 元组。
        """
        length_bits = combined_bits[:self.mark_len]
        z_length = int(''.join(map(str, length_bits)), 2)
        z_bits = combined_bits[self.mark_len:self.mark_len + z_length]
        stego_bits = combined_bits[self.mark_len + z_length:]
        return z_bits, stego_bits

    def compress(self, stego_img: torch.Tensor, drop_z: torch.Tensor) -> tuple:
        """
        压缩隐写图像和drop_z张量。

        参数:
            stego_img: 图像张量 [1, C, H, W]。
            drop_z: 潜在向量张量。

        返回:
            (裁剪后的图像, (合并比特流, drop_z_bits, 溢出bits))。
        """
        overflow_bits_list = []
        clip_stego_img = None

        if stego_img is not None:
            clip_stego_img = torch.clip(stego_img, 0, 255)
            steg_img_numpy = stego_img.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()

            # 计算溢出值
            overflow = np.zeros_like(steg_img_numpy)
            overflow[steg_img_numpy > 255] = steg_img_numpy[steg_img_numpy > 255] - 255
            overflow[steg_img_numpy < 0] = 0 - steg_img_numpy[steg_img_numpy < 0]
            overflow = overflow.astype(int)

            # 找到所有非零溢出像素
            pos_1, pos_2, pos_3 = (overflow != 0).nonzero()
            non_zero_values = overflow[pos_1, pos_2, pos_3]

            # 编码每个非零像素的位置和值
            compressed_bitstream = ''
            for row, col, channel, val in zip(pos_1.tolist(), pos_2.tolist(), pos_3.tolist(), non_zero_values.tolist()):
                row_bin = f'{row:0{self.row_bits}b}'
                col_bin = f'{col:0{self.col_bits}b}'
                channel_bin = f'{channel:0{self.channel_bits}b}' if self.channels > 1 else ''
                value_bin = f'{val:0{self.value_bits}b}'
                compressed_bitstream += row_bin + col_bin + channel_bin + value_bin

            overflow_bits_list = [int(bit) for bit in compressed_bitstream]

        # 压缩drop_z
        drop_z_bits_list = []
        if drop_z is not None:
            drop_z_numpy = drop_z.squeeze(0).detach().cpu().numpy()
            drop_z_numpy = np.clip(drop_z_numpy, -2 ** self.coder.freq_bits_len + 1, 2 ** self.coder.freq_bits_len - 1)
            drop_z_numpy = drop_z_numpy.astype(np.int64)
            drop_z_bits_list = self.coder.compress(drop_z_numpy.flatten())

        # 合并比特流
        if stego_img is None:
            data_list = drop_z_bits_list
        elif drop_z is None:
            data_list = overflow_bits_list
        else:
            data_list = self.combine_bits(drop_z_bits_list, overflow_bits_list)

        return clip_stego_img, (data_list, drop_z_bits_list, overflow_bits_list)

    def decompress(self, clip_stego_img: torch.Tensor, data_bits: list) -> tuple:
        """
        解压缩比特流。

        参数:
            clip_stego_img: 裁剪后的隐写图像 [1, C, H, W]。
            data_bits: 编码的比特列表。

        返回:
            (恢复的隐写图像, 恢复的drop_z)。
        """
        # 拆分比特流
        drop_z_bits_list, overflow_bits_list = self.split_bits(data_bits)

        # 解压drop_z
        drop_z_list = self.coder.decompress(drop_z_bits_list)
        rec_z = torch.as_tensor(drop_z_list, dtype=torch.float).reshape(self.z_shape)

        # 解压溢出数据
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

        # 应用溢出校正
        mask_0 = (clip_stego_img == 0.).float()
        mask_255 = (clip_stego_img == 255.).float()
        overflow_tensor = torch.as_tensor(overflow, dtype=clip_stego_img.dtype).permute(2, 0, 1).unsqueeze(0)

        stego_img = clip_stego_img - mask_0 * overflow_tensor + mask_255 * overflow_tensor
        return stego_img, rec_z


# ==================== 张量编码器（自动选择最优压缩策略）====================
class TensorCoder:
    """
    张量编码器，自动选择最优压缩策略。

    结合两种压缩方法：
    1. ACCompress: 算术编码，适用于一般情况
    2. SparseTensorCompressor: 稀疏编码，适用于溢出像素较少的情况

    自动选择比特数更少的方法。
    """

    def __init__(self, image_shape, drop_z_shape, level_bits_len, freq_bits_len):
        """
        初始化张量编码器。

        参数:
            image_shape: 图像张量形状 (H, W, C)。
            drop_z_shape: 潜在向量形状 (H, W)。
            level_bits_len: 层级比特长度。
            freq_bits_len: 频率比特长度。
        """
        self.sparsetensorcompressor = SparseTensorCompressor(image_shape, drop_z_shape, level_bits_len, freq_bits_len)
        self.accompress = ACCompress(image_shape, drop_z_shape, level_bits_len, freq_bits_len)

    def compress(self, input_tensor, drop_z):
        """
        压缩输入张量和drop_z，自动选择最优策略。

        参数:
            input_tensor: 要压缩的图像张量。
            drop_z: 要压缩的潜在向量。

        返回:
            (裁剪后的图像, 编码比特流)。
        """
        # 使用算术编码
        clip_stego_img_a, data_list_a = self.accompress.encode(input_tensor, drop_z)
        # 使用稀疏张量编码
        clip_stego_img_s, data_list_s = self.sparsetensorcompressor.compress(input_tensor, drop_z)

        # 比较比特流长度，选择更短的
        if len(data_list_a[0]) > len(data_list_s[0]):
            # 使用稀疏张量编码，前置标志0
            data_list = [0] + data_list_s[0]
        else:
            # 使用算术编码，前置标志1
            data_list = [1] + data_list_a[0]

        clip_stego_img = clip_stego_img_a
        return clip_stego_img, data_list

    def decompress(self, clip_stego_img, data_list):
        """
        解压缩比特流。

        参数:
            clip_stego_img: 裁剪后的隐写图像张量。
            data_list: 编码的比特流（首位为策略标志）。

        返回:
            (恢复的图像张量, 恢复的潜在向量)。
        """
        # 提取标志位
        data_list_ = data_list[1:]
        if data_list[0] == 1:
            # 使用算术解码
            rec_img, rec_drop_z = self.accompress.decode(clip_stego_img, data_list_)
        else:
            # 使用稀疏张量解码
            rec_img, rec_drop_z = self.sparsetensorcompressor.decompress(clip_stego_img, data_list_)
        return rec_img, rec_drop_z


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 测试 TensorCoder
    print("=" * 50)
    print("测试 TensorCoder 编解码功能")
    print("=" * 50)

    # 定义图像和潜在向量尺寸
    im_size = (400, 400, 3)  # 图像尺寸 (H, W, C)
    z_size = (1, 100)  # 潜在向量尺寸 (H, W)
    device = 'cpu'

    # 创建随机测试数据
    stego_img = torch.randint(-10, 280, (1, im_size[2], im_size[0], im_size[1]), dtype=torch.float32, device=device)
    drop_z = torch.randint(44631230, 44631236, (1, *z_size), dtype=torch.float32, device=device)

    print(f"图像尺寸: {im_size}")
    print(f"潜在向量尺寸: {z_size}")
    print(f"图像值范围: [{stego_img.min():.0f}, {stego_img.max():.0f}]")
    print(f"潜在向量值范围: [{drop_z.min():.0f}, {drop_z.max():.0f}]")

    # 初始化编码器
    tensor_coder = TensorCoder(im_size, z_size, 10, 40)

    # 编码
    print("\n开始编码...")
    clip_encoded_img, data_bits = tensor_coder.compress(stego_img, drop_z)
    print(f"编码后比特数: {len(data_bits)}")
    print(f"使用策略: {'算术编码' if data_bits[0] == 1 else '稀疏编码'}")

    # 解码
    print("\n开始解码...")
    decoded_img, decoded_z = tensor_coder.decompress(clip_encoded_img, data_bits)

    # 验证结果
    print("\n验证结果:")
    print(f"原始图像前10值: {stego_img.flatten()[:10].tolist()}")
    print(f"解码图像前10值: {decoded_img.flatten()[:10].tolist()}")
    print(f"原始潜在向量前10值: {drop_z.flatten()[:10].tolist()}")
    print(f"解码潜在向量前10值: {decoded_z.flatten()[:10].tolist()}")

    # 检查精度
    img_match = torch.allclose(stego_img, decoded_img, atol=1e-8)
    z_match = torch.allclose(drop_z, decoded_z, atol=1e-8)

    print(f"\n图像完全匹配: {img_match}")
    print(f"潜在向量完全匹配: {z_match}")

    if img_match and z_match:
        print("\n✓ 编解码测试通过！")
    else:
        print("\n✗ 编解码测试失败！")
