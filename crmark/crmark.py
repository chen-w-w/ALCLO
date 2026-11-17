# ==================== 导入依赖库 ====================
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

# 忽略警告信息，保持控制台输出整洁
warnings.filterwarnings("ignore")

# 定义模块对外暴露的接口
__all__ = ["CRMark"]

# ==================== 预训练模型下载地址 ====================
# 灰度图模型：512×512 尺寸，可嵌入 256 位（约 20 个字符）
_gray_512_256_MODEL_URL = "https://github.com/chenoly/CRMark/releases/download/v1.0/crmark_gray_size_512_bit_256.pth"
# 彩色图模型：256×256 尺寸，可嵌入 64 位（约 5 个字符）
_color_256_64_MODEL_URL = "https://github.com/chenoly/CRMark/releases/download/v1.0/crmark_color_size_256_bit_64.pth"
# 彩色图模型：256×256 尺寸，可嵌入 100 位（约 7 个字符）
_color_256_100_MODEL_URL = "https://github.com/chenoly/CRMark/releases/download/v1.0/crmark_color_size_256_bit_100.pth"


# ==================== 模型下载工具函数 ====================
def _download_from_github_release(url: str, save_path: str) -> bool:
    """
    从 GitHub Release 下载文件，支持断点续传和完整性校验。

    参数:
        url (str): 要下载的文件 URL（如模型权重文件）。
        save_path (str): 本地保存路径。

    返回:
        bool: 下载成功返回 True，失败返回 False。
    """
    temp_path = None
    file_handle = None
    try:
        print("开始下载:", url)
        # 使用 GitHub Token 进行认证（如果设置了环境变量）
        headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}"}
        response = requests.get(url, stream=True, allow_redirects=True, headers=headers, timeout=30)

        # 检查响应状态
        if response.status_code != 200:
            print(f"下载失败，状态码: {response.status_code}")
            print(f"\n请手动从以下链接下载:\n{url}\n并放置到: {os.path.dirname(save_path)}")
            return False

        # 获取文件总大小
        total_size = int(response.headers.get('content-length', 0))
        if total_size < 1024:
            print(f"文件大小异常: {total_size} 字节")
            return False

        # 创建临时文件路径
        temp_path = save_path + '.tmp'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 使用进度条下载到临时文件
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

        # 验证下载完整性
        downloaded_size = os.path.getsize(temp_path)
        if downloaded_size != total_size:
            print(f"下载不完整。已下载: {downloaded_size} 字节, 预期: {total_size} 字节")
            return False

        # 将临时文件重命名为最终路径
        os.rename(temp_path, save_path)
        print(f"下载成功，保存至: {save_path}")
        return True

    except (requests.RequestException, KeyboardInterrupt) as e:
        print(f"下载失败或中断: {e}")
        print(f"\n请手动从以下链接下载:\n{url}\n并放置到:\n{os.path.dirname(save_path)}")
        return False

    finally:
        # 确保文件句柄已关闭
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception as e:
                print(f"关闭文件句柄失败: {e}")

        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"已清理临时文件: {temp_path}")
            except OSError as e:
                print(f"删除临时文件失败 {temp_path}: {e}")


def _download_models(model_mode):
    """
    根据指定的模型模式下载预训练模型权重（如果尚未缓存）。

    缓存路径:
        - Windows: %USERPROFILE%\.cache\crmark\
        - Linux/Mac: ~/.cache/crmark/

    参数:
        model_mode (str): 模型模式，可选 "gray_512_256", "color_256_64", "color_256_100"。

    返回:
        None: 函数会自动下载并缓存模型权重。
    """
    # 根据操作系统确定缓存目录
    if platform.system() == "Windows":
        base_cache_dir = os.path.join(os.environ["USERPROFILE"], ".cache", "crmark")
    else:
        base_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "crmark")
    os.makedirs(base_cache_dir, exist_ok=True)

    # 下载灰度图模型（如果需要）
    if model_mode == "gray_512_256":
        gray_model_path = os.path.join(base_cache_dir, "crmark_gray_size_512_bit_256.pth")
        if not os.path.exists(gray_model_path):
            _download_from_github_release(_gray_512_256_MODEL_URL, gray_model_path)

    # 下载彩色图模型 64 位（如果需要）
    if model_mode == "color_256_64":
        model_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_64.pth")
        if not os.path.exists(model_model_path):
            _download_from_github_release(_color_256_64_MODEL_URL, model_model_path)

    # 下载彩色图模型 100 位（如果需要）
    if model_mode == "color_256_100":
        model_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_100.pth")
        if not os.path.exists(model_model_path):
            _download_from_github_release(_color_256_100_MODEL_URL, model_model_path)


# ==================== CRMark 主类 ====================
class CRMark(object):
    """
    CRMark 可逆图像水印主类。

    基于深度学习（iIWN 网络）和可逆数据隐藏（RDH）技术，
    支持将文本消息或二进制水印嵌入图像，并能完全恢复原始图像和水印。
    支持灰度图和彩色图像，提供三种预训练模型可选。
    """

    def __init__(self, model_mode="color_256_64", model_path: str = None, level_bits_len=10, freq_bits_len=10,
                 device="cpu", float64=False):
        """
        初始化 CRMark 实例。

        参数:
            model_mode (str): 模型模式，可选：
                - "color_256_64":  RGB 彩色图，256×256，嵌入 64 位（约 5 字符），BCH 3位纠错
                - "color_256_100": RGB 彩色图，256×256，嵌入 100 位（约 7 字符），BCH 5位纠错
                - "gray_512_256":  灰度图，512×512，嵌入 256 位（约 20 字符），BCH 12位纠错
            model_path (str): 自定义预训练模型路径，默认 None 则自动下载
            level_bits_len (int): 张量量化级别编码比特数，默认 10
            freq_bits_len (int): 张量量化频率编码比特数，默认 10
            device (str): 计算设备，"cpu" 或 "cuda"，默认 "cpu"
            float64 (bool): 是否使用 float64 精度，默认 False（使用 float32）

        返回:
            None: 构造函数加载模型权重并准备编解码环境

        异常:
            AssertionError: 如果 model_mode 不是支持的选项
        """
        # 验证模型模式
        assert model_mode in ["color_256_64", "color_256_100", "gray_512_256"], \
            "model_mode 必须是 'color_256_64', 'color_256_100' 或 'gray_512_256'"

        # ==================== 初始化实例变量 ====================
        self.k = None              # 可逆块数量（网络深度参数）
        self.fc = None             # 全连接层配置
        self.rdh = None            # RDH 可逆数据隐藏模块
        self.iIWN = None           # iIWN 神经网络模型
        self.device = device       # 计算设备
        self.float64 = float64     # 精度标志
        self.img_size = None       # 期望图像尺寸
        self.bit_length = None     # 水印比特长度
        self.model_mode = model_mode  # 模型模式

        # ==================== 根据模型模式设置参数 ====================
        if model_mode == "color_256_64":
            self.channel_dim = 3           # RGB 三通道
            BCH_POLYNOMIAL_ = 137          # BCH 多项式
            BCH_BITS_ = 3                  # BCH 纠错能力（可纠正 3 位错误）
        elif model_mode == "color_256_100":
            self.channel_dim = 3           # RGB 三通道
            BCH_POLYNOMIAL_ = 137          # BCH 多项式
            BCH_BITS_ = 5                  # BCH 纠错能力（可纠正 5 位错误）
        else:  # gray_512_256
            self.channel_dim = 1           # 灰度单通道
            BCH_POLYNOMIAL_ = 501          # BCH 多项式
            BCH_BITS_ = 12                 # BCH 纠错能力（可纠正 12 位错误）

        # ==================== 初始化其他组件 ====================
        self.tensorcoder = None            # 张量压缩模块
        self.net_min_size = None           # 神经网络最小输入尺寸
        self.freq_bits_len = freq_bits_len # 频率编码比特数
        self.level_bits_len = level_bits_len  # 级别编码比特数
        self.bch = BCH(BCH_POLYNOMIAL_, BCH_BITS_)  # BCH 纠错编码实例
        self.transform = transforms.Compose([transforms.ToTensor()])  # 图像转张量

        # ==================== 设置模型缓存目录 ====================
        if platform.system() == "Windows":
            base_cache_dir = os.path.join(os.environ["USERPROFILE"], ".cache", "crmark")
        else:
            base_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "crmark")
        os.makedirs(base_cache_dir, exist_ok=True)

        # 定义模型文件路径
        gray_model_path = os.path.join(base_cache_dir, "crmark_gray_size_512_bit_256.pth")
        color_256_64_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_64.pth")
        color_256_100_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_100.pth")

        # ==================== 加载模型 ====================
        if model_path is None:
            # 自动下载所需模型权重
            _download_models(model_mode)
            # 根据模型模式加载对应模型
            if model_mode == "color_256_64":
                self.load_model(color_256_64_model_path)
            elif model_mode == "color_256_100":
                self.load_model(color_256_100_model_path)
            else:
                self.load_model(gray_model_path)
        else:
            # 加载用户指定的模型
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        加载预训练模型权重并配置水印模型。

        参数:
            model_path (str): 预训练模型权重文件路径。

        返回:
            None: 配置模型及相关组件。
        """
        # 从文件加载模型字典
        load_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        # ==================== 提取模型参数 ====================
        self.k = load_dict['param_dict']['k']                    # 可逆块数量
        self.net_min_size = load_dict['param_dict']['min_size']  # 最小输入尺寸
        self.fc = load_dict['param_dict']['fc']                  # 全连接层配置
        self.bit_length = load_dict['param_dict']['bit_length']  # 水印比特长度
        self.img_size = load_dict['param_dict']['img_size']      # 期望图像尺寸
        self.channel_dim = load_dict['param_dict']['channel_dim']  # 通道数

        # 如果指定了 float64 精度，设置默认数据类型
        if self.float64:
            torch.set_default_dtype(torch.float64)

        # ==================== 初始化 RDH 模块 ====================
        # RDH（可逆数据隐藏）用于嵌入辅助信息和图像哈希
        self.rdh = CustomRDH((self.img_size, self.img_size, self.channel_dim), self.device)

        # ==================== 初始化张量压缩器 ====================
        # TensorCoder 用于压缩神经网络输出的浮点张量
        self.tensorcoder = TensorCoder(
            (self.img_size, self.img_size, self.channel_dim),  # 图像形状
            (1, self.bit_length),                              # 水印形状
            self.level_bits_len,                               # 级别编码比特数
            self.freq_bits_len                                 # 频率编码比特数
        )

        # ==================== 初始化神经网络 ====================
        # iIWN（可逆图像水印网络）是核心深度学习模型
        self.iIWN = Model(self.img_size, self.channel_dim, self.bit_length, self.k, self.net_min_size, self.fc)
        self.iIWN.load_state_dict(load_dict['model_state_dict'])  # 加载权重
        self.iIWN.to(self.device)                                   # 移至指定设备
        self.iIWN.eval()                                            # 设置为评估模式

    def encode(self, cover_img: ndarray, message: str) -> Tuple[bool, Any]:
        """
        将文本消息嵌入图像作为水印（文本模式）。

        参数:
            cover_img (ndarray): 封面图像（载体图像），消息将嵌入其中。
            message (str): 要嵌入的文本消息。
                消息长度要求取决于模型模式：
                - "color_256_64":  长度必须为 5 个字符
                - "color_256_100": 长度必须为 7 个字符
                - "gray_512_256":  长度必须为 20 个字符

        返回:
            Tuple[bool, PIL.Image]: (成功标志, 隐写图像)
                - bool: 嵌入成功返回 True，失败返回 False
                - PIL.Image: 包含嵌入水印的隐写图像

        流程:
            1. 计算封面图像的 SHA-256 哈希用于完整性验证
            2. 使用 BCH 编码对消息进行纠错编码
            3. 通过 iIWN 网络将水印嵌入图像
            4. 使用 TensorCoder 压缩网络输出
            5. 使用 RDH 嵌入辅助信息和图像哈希
        """
        # ==================== 输入验证 ====================
        cover_img = np.uint8(cover_img)

        # 验证消息长度和图像维度
        if self.model_mode == "color_256_64":
            assert len(message) == 5 and cover_img.ndim == 3, \
                "color_256_64 模式要求消息长度为 5 且图像为 RGB 格式"
        if self.model_mode == "color_256_100":
            assert len(message) == 7 and cover_img.ndim == 3, \
                "color_256_100 模式要求消息长度为 7 且图像为 RGB 格式"
        if self.model_mode == "gray_512_256":
            assert len(message) == 20 and cover_img.ndim == 2, \
                "gray_512_256 模式要求消息长度为 20 且图像为灰度格式"

        # ==================== 计算图像哈希 ====================
        # 用于后续验证图像是否被篡改
        cover_img_hash = sha256_of_image_array(cover_img)
        cover_img_hash_bitstream = sha256_to_bitstream(cover_img_hash)

        # ==================== 图像预处理 ====================
        # 将 numpy 图像转换为 PyTorch 张量
        cover_img_tensor = self.transform(cover_img).unsqueeze(0).to(self.device)

        # ==================== 消息编码 ====================
        # 使用 BCH 纠错码编码消息，增强鲁棒性
        watermark = self.bch.Encode(message)
        # 填充随机比特到指定长度
        watermark += [random.randint(0, 1) for _ in range(self.bit_length - len(watermark))]
        secret_tensor = torch.as_tensor(watermark, dtype=torch.float32).unsqueeze(0).to(self.device)

        # ==================== 水印嵌入（神经网络）====================
        with torch.no_grad():
            if self.float64:
                cover_img_tensor = cover_img_tensor.to(torch.float64)
                secret_tensor = secret_tensor.to(torch.float64)
            # iIWN 前向传播：封面图 + 水印 → 溢出隐写图 + 辅助张量 z
            overflow_stego, z = self.iIWN.forward(cover_img_tensor, secret_tensor, True, False)

        # ==================== 张量压缩 ====================
        # 将浮点张量转换为可存储的整型数据
        stego_255 = torch.round(overflow_stego * 255.)
        z_round = torch.round(z)
        clip_stego_img, auxbits = self.tensorcoder.compress(stego_255, z_round)

        # ==================== RDH 嵌入 ====================
        # 将辅助信息和图像哈希嵌入到隐写图像中
        issuccessful, rdh_stego_img = self.rdh.embed(clip_stego_img, auxbits + cover_img_hash_bitstream)
        if not issuccessful:
            print("可逆嵌入失败")

        return issuccessful, Image.fromarray(np.uint8(rdh_stego_img))

    def recover(self, stego_img: ndarray) -> Union[Tuple[bool, Any, Any], Tuple[bool, None, None]]:
        """
        从隐写图像中恢复原始封面图像和嵌入的消息（完全可逆）。

        参数:
            stego_img (ndarray): 包含水印的隐写图像。

        返回:
            Union[Tuple[bool, PIL.Image, str], Tuple[bool, None, None]]:
                - bool: 图像是否被攻击（哈希不匹配返回 True）
                - PIL.Image or None: 恢复的封面图像，失败返回 None
                - str or None: 恢复的消息，失败返回 None

        流程:
            1. 使用 RDH 提取辅助信息和图像哈希
            2. 使用 TensorCoder 解压张量数据
            3. 通过 iIWN 网络逆向恢复封面图和水印
            4. 使用 BCH 解码恢复原始消息
            5. 比对哈希值判断图像是否被篡改
        """
        # ==================== 加载隐写图像 ====================
        stego_img = np.uint8(stego_img)

        # ==================== RDH 提取 ====================
        # 提取辅助信息和图像哈希
        issuccessful, clipped_stego_img, ext_auxbits = self.rdh.extract(stego_img)
        if issuccessful:
            # 分离图像哈希和辅助信息
            cover_img_hash_bitstream = ext_auxbits[-256:]  # 最后 256 位是哈希
            rec_overflow_stego, rec_z = self.tensorcoder.decompress(clipped_stego_img, ext_auxbits[:-256])

            # ==================== 神经网络恢复 ====================
            with torch.no_grad():
                if self.float64:
                    rec_z = rec_z.to(torch.float64)
                    rec_overflow_stego = rec_overflow_stego.to(torch.float64)
                # iIWN 逆向传播：溢出隐写图 + z → 恢复封面图 + 水印
                rec_cover_tensor, rec_watermark = self.iIWN.forward(rec_overflow_stego / 255., rec_z, True, True)

            # ==================== 转换恢复图像 ====================
            # 将张量转换回 numpy 图像
            if rec_cover_tensor.shape[1] == 1:  # 灰度图
                rec_cover = torch.round(rec_cover_tensor * 255.)[0][0].detach().cpu().numpy()
            else:  # 彩色图
                rec_cover = torch.round(rec_cover_tensor * 255.)[0].permute(1, 2, 0).detach().cpu().numpy()
            rec_cover = np.uint8(rec_cover)

            # ==================== 完整性验证 ====================
            # 计算恢复图像的哈希并与嵌入的哈希比对
            rec_cover_img_hash = sha256_of_image_array(rec_cover)
            rec_cover_img_hash_bitstream = sha256_to_bitstream(rec_cover_img_hash)

            # ==================== 消息解码 ====================
            # 处理恢复的水印并解码
            rec_watermark = torch.round(torch.clip(rec_watermark, 0., 1.))
            rec_watermark = rec_watermark[0].detach().cpu().numpy().astype(int).tolist()
            # 截取 8 的整数倍长度（字节对齐）
            valid_part = rec_watermark[:(len(rec_watermark) // 8) * 8]
            isdecode, decoded_data = self.bch.Decode(valid_part)

            # ==================== 判断是否被攻击 ====================
            if np.array_equal(rec_cover_img_hash_bitstream, cover_img_hash_bitstream):
                return False, Image.fromarray(rec_cover), decoded_data
            else:
                return True, Image.fromarray(rec_cover), decoded_data
        else:
            # RDH 提取失败
            return True, None, None

    def decode(self, stego_img: ndarray) -> Tuple[bool, str]:
        """
        仅从隐写图像中提取并解码消息（不可逆，不恢复封面图）。

        与 recover 方法不同，decode 只提取消息而不恢复原始封面图像，
        适用于只需要获取水印信息的场景。

        参数:
            stego_img (ndarray): 包含水印的隐写图像。

        返回:
            Tuple[bool, str]:
                - bool: 解码成功返回 True，失败返回 False
                - str: 从水印中提取的解码消息

        注意:
            此方法使用随机噪声作为输入，仅适用于提取消息，
            无法恢复原始封面图像。如需完整恢复，请使用 recover 方法。
        """
        # ==================== 加载并预处理图像 ====================
        stego_img = np.uint8(stego_img)
        stego_img_tensor = self.transform(stego_img).unsqueeze(0).to(self.device)

        # ==================== 生成随机噪声 ====================
        # decode 不需要辅助信息 z，使用随机噪声作为替代
        sampled_z = torch.randn(size=(1, self.bit_length)).to(self.device)

        # ==================== 提取水印 ====================
        with torch.no_grad():
            # 前向传播：隐写图 + 随机噪声 → 提取水印
            _, _ext_watermark = self.iIWN.forward(stego_img_tensor, sampled_z, True, True)

        # ==================== 解码消息 ====================
        _ext_watermark = torch.round(torch.clip(_ext_watermark, 0., 1.))
        ext_watermark = _ext_watermark[0].detach().cpu().numpy().astype(int).tolist()
        # 截取 8 的整数倍长度（字节对齐）
        valid_part = ext_watermark[:(len(ext_watermark) // 8) * 8]
        isdecode, decoded_data = self.bch.Decode(valid_part)
        return isdecode, decoded_data

    def encode_bits(self, cover_img: ndarray, watermark: list) -> Tuple[bool, Any]:
        """
        将二进制水印嵌入图像（比特模式）。

        与 encode 方法不同，此方法直接接收二进制比特列表而非文本消息，
        适用于需要自定义水印内容的场景。

        参数:
            cover_img (ndarray): 封面图像（载体图像）。
            watermark (list): 要嵌入的二进制水印，值为 0 或 1 的列表。
                列表长度必须等于模型的 bit_length。

        返回:
            Tuple[bool, PIL.Image]: (成功标志, 隐写图像)
                - bool: 嵌入成功返回 True，失败返回 False
                - PIL.Image: 包含嵌入水印的隐写图像

        注意:
            此方法不对水印进行 BCH 编码，直接嵌入原始比特。
            如需纠错编码，请在调用前自行处理。
        """
        # ==================== 加载封面图像 ====================
        cover_img = np.uint8(cover_img)

        # ==================== 计算图像哈希 ====================
        cover_img_hash = sha256_of_image_array(cover_img)
        cover_img_hash_bitstream = sha256_to_bitstream(cover_img_hash)

        # ==================== 图像预处理 ====================
        cover_img_tensor = self.transform(cover_img).unsqueeze(0).to(self.device)
        # 直接将水印列表转换为张量（不进行 BCH 编码）
        secret_tensor = torch.as_tensor(watermark, dtype=torch.float32).unsqueeze(0).to(self.device)

        # ==================== 水印嵌入（神经网络）====================
        with torch.no_grad():
            if self.float64:
                cover_img_tensor = cover_img_tensor.to(torch.float64)
                secret_tensor = secret_tensor.to(torch.float64)
            overflow_stego, z = self.iIWN.forward(cover_img_tensor, secret_tensor, True, False)

        # ==================== 张量压缩 ====================
        # todo: 了解张量压缩的详细过程
        stego_255 = torch.round(overflow_stego * 255.)
        z_round = torch.round(z)
        clip_stego_img, auxbits = self.tensorcoder.compress(stego_255, z_round)

        # ==================== RDH 嵌入 ====================
        # todo: 了解 RDH 嵌入的详细过程，图像哈希如何校验完整性
        issuccessful, rdh_stego_img = self.rdh.embed(clip_stego_img, auxbits + cover_img_hash_bitstream)
        if not issuccessful:
            print("可逆嵌入失败")
        return issuccessful, Image.fromarray(np.uint8(rdh_stego_img))

    def recover_bits(self, stego_img: ndarray) -> Union[Tuple[bool, Any, Any], Tuple[bool, None, None]]:
        """
        从隐写图像中恢复原始封面图像和二进制水印（比特模式）。

        与 recover 方法类似，但返回原始比特列表而非解码后的文本消息，
        适用于使用 encode_bits 嵌入水印的场景。

        参数:
            stego_img (ndarray): 包含水印的隐写图像。

        返回:
            Union[Tuple[bool, PIL.Image, list], Tuple[bool, None, None]]:
                - bool: 图像是否被攻击（哈希不匹配返回 True）
                - PIL.Image or None: 恢复的封面图像，失败返回 None
                - list or None: 恢复的水印比特列表，失败返回 None

        注意:
            此方法不对水印进行 BCH 解码，直接返回原始比特。
            如需解码消息，请在返回后自行处理或改用 recover 方法。
        """
        # ==================== 加载隐写图像 ====================
        stego_img = np.uint8(stego_img)

        # ==================== RDH 提取 ====================
        issuccessful, clipped_stego_img, ext_auxbits = self.rdh.extract(stego_img)
        if issuccessful:
            # 分离图像哈希和辅助信息
            cover_img_hash_bitstream = ext_auxbits[-256:]
            rec_overflow_stego, rec_z = self.tensorcoder.decompress(clipped_stego_img, ext_auxbits[:-256])

            # ==================== 神经网络恢复 ====================
            with torch.no_grad():
                if self.float64:
                    rec_z = rec_z.to(torch.float64)
                    rec_overflow_stego = rec_overflow_stego.to(torch.float64)
                rec_cover_tensor, rec_watermark = self.iIWN.forward(rec_overflow_stego / 255., rec_z, True, True)

            # ==================== 转换恢复图像 ====================
            if rec_cover_tensor.shape[1] == 1:  # 灰度图
                rec_cover = torch.round(rec_cover_tensor * 255.)[0][0].detach().cpu().numpy()
            else:  # 彩色图
                rec_cover = torch.round(rec_cover_tensor * 255.)[0].permute(1, 2, 0).detach().cpu().numpy()
            rec_cover = np.uint8(rec_cover)

            # ==================== 完整性验证 ====================
            rec_cover_img_hash = sha256_of_image_array(rec_cover)
            rec_cover_img_hash_bitstream = sha256_to_bitstream(rec_cover_img_hash)

            # ==================== 处理恢复的水印 ====================
            rec_watermark = torch.round(torch.clip(rec_watermark, 0., 1.))
            # 返回原始比特列表（不进行 BCH 解码）
            rec_watermark = rec_watermark.squeeze(0).detach().cpu().numpy().astype(int).tolist()

            # ==================== 判断是否被攻击 ====================
            if np.array_equal(rec_cover_img_hash_bitstream, cover_img_hash_bitstream):
                return False, Image.fromarray(rec_cover), rec_watermark
            else:
                return True, Image.fromarray(rec_cover), rec_watermark
        else:
            return True, None, None

    def decode_bits(self, stego_img: ndarray) -> list:
        """
        仅从隐写图像中提取二进制水印（比特模式，不可逆）。

        与 decode 方法类似，但返回原始比特列表而非解码后的文本消息，
        适用于使用 encode_bits 嵌入水印且只需提取水印的场景。

        参数:
            stego_img (ndarray): 包含水印的隐写图像。

        返回:
            list: 提取的水印比特列表，值为 0 或 1。

        注意:
            此方法使用随机噪声作为输入，仅适用于提取水印，
            不进行 BCH 解码，也不恢复原始封面图像。
            如需完整恢复，请使用 recover_bits 方法。
        """
        # ==================== 加载并预处理图像 ====================
        stego_img = np.uint8(stego_img)
        stego_img_tensor = self.transform(stego_img).unsqueeze(0).to(self.device)

        # ==================== 生成随机噪声 ====================
        sampled_z = torch.randn(size=(1, self.bit_length)).to(self.device)

        # ==================== 提取水印 ====================
        with torch.no_grad():
            _, _ext_watermark = self.iIWN.forward(stego_img_tensor, sampled_z, True, True)

        # ==================== 处理提取的水印 ====================
        _ext_watermark = torch.round(torch.clip(_ext_watermark, 0., 1.))
        ext_watermark = _ext_watermark[0].detach().cpu().numpy().astype(int).tolist()
        return ext_watermark
