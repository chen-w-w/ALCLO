# ==================== 导入依赖库 ====================
import math
import torch
import random
import numpy as np
from PIL import Image
from torch import Tensor
from typing import Tuple, List
from numpy import ndarray
from .arithmeticcoder import CustomArithmeticEncoder  # 自定义算术编码器


# ==================== RDH 核心类 ====================
class RDH:
    """
    RDH (Reversible Data Hiding) 可逆数据隐藏核心类。

    基于预测误差扩展（Prediction Error Expansion）技术实现可逆水印嵌入，
    能够在嵌入水印的同时保证原始图像可以完全无损恢复。
    """

    def __init__(self, img_size: Tuple[int, int, int], bit_plane: int = 3, level_bits_len: int = 10, freq_bits_len: int = 10):
        """
        初始化 RDH 类。

        参数:
            img_size: 图像尺寸元组 (高度, 宽度, 通道数)。
            bit_plane: 用于存储辅助信息的位平面数，默认 3。
            level_bits_len: 算术编码级别比特数，默认 10。
            freq_bits_len: 算术编码频率比特数，默认 10。
        """
        self.grayscale_bit = 8  # 灰度图像素位数
        self.img_size = img_size
        self.bit_plane = bit_plane  # LSB 位平面数
        # 计算存储容量：宽度 × 通道数 × 位平面数
        self.storage_len = self.img_size[1] * self.img_size[2] * self.bit_plane

        # 计算各维度的编码长度（用于坐标编码）
        self.c_len = math.ceil(np.log2(img_size[2])) + 1  # 通道索引所需比特数
        self.w_len = math.ceil(np.log2(img_size[1])) + 1  # 宽度索引所需比特数
        self.h_len = math.ceil(np.log2(img_size[0])) + 1  # 高度索引所需比特数
        self.freq_len = self.c_len + self.w_len + self.h_len  # 总频率长度

        # 初始化算术编码器（用于压缩辅助信息）
        self.customAC = CustomArithmeticEncoder(level_bits_len, freq_bits_len)

    def set_mask(self, split_height: int):
        """
        设置棋盘掩码，用于将像素分为两组（O组和X组）进行嵌入。

        采用棋盘模式（checkerboard pattern）分离像素，使得嵌入时相邻像素不会相互干扰，
        从而提高嵌入容量并减少失真。

        参数:
            split_height: 图像分割高度，用于确定掩码尺寸。
        """
        # 创建棋盘模式掩码：(h + w) % 2 产生 0/1 交替模式
        base_mask = np.fromfunction(
            lambda h, w: (h + w) % 2,
            (self.img_size[0] - split_height, self.img_size[1]),
            dtype=int
        )
        # 扩展到所有通道
        mask = np.repeat(base_mask[:, :, np.newaxis], self.img_size[2], axis=2)
        self.mask_o = mask      # O组掩码（值为1的位置）
        self.mask_x = 1 - mask  # X组掩码（值为0的位置）

    def prediect(self, cover_img: ndarray, h: int, w: int, c: int):
        """
        使用 MED（Median Edge Detector）算法预测像素值。

        MED 是一种简单的上下文预测器，利用当前像素的四个邻域像素进行预测。

        参数:
            cover_img: 封面图像数组。
            h: 像素高度索引。
            w: 像素宽度索引。
            c: 通道索引。

        返回:
            预测值（四舍五入到整数）。
        """
        # 定义 3×3 掩码，中心为0，四邻域为1
        mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        # 提取 3×3 邻域块
        block = cover_img[h - 1: h + 2, w - 1: w + 2, c]
        # 计算邻域平均值作为预测值
        predict_value = np.round(np.sum(mask * block) / 4.)
        return predict_value

    def predicting_error(self, cover_img: ndarray):
        """
        计算图像的预测误差。

        预测误差 = 实际像素值 - 预测像素值。
        预测误差通常服从拉普拉斯分布，集中在0附近，适合用于可逆嵌入。

        参数:
            cover_img: 封面图像数组。

        返回:
            元组 (pe_o, pe_x, pv_o, pv_x):
            - pe_o: O组的预测误差
            - pe_x: X组的预测误差
            - pv_o: O组的预测值
            - pv_x: X组的预测值
        """
        H, W, C = cover_img.shape
        pv = np.copy(cover_img)

        # 对所有内部像素计算预测值（边界像素不处理）
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    pv[h, w, c] = self.prediect(cover_img, h, w, c)

        # 分离O组和X组的预测值
        pv_o = self.mask_o * pv
        pv_x = self.mask_x * pv

        # 计算预测误差
        pe_o = cover_img * self.mask_o - pv_o
        pe_x = cover_img * self.mask_x - pv_x

        return pe_o, pe_x, pv_o, pv_x

    def split_img(self, cover_img: ndarray, height_end: int):
        """
        将图像分割为两部分：用于嵌入的主体部分和用于存储位置图的顶部区域。

        参数:
            cover_img: 封面图像数组。
            height_end: 分割高度，顶部用于存储位置图，底部用于嵌入水印。

        返回:
            元组 (img4embed, img4locmap):
            - img4embed: 用于嵌入水印的主体图像部分
            - img4locmap: 用于存储位置图的顶部图像部分
        """
        img4locmap = cover_img[:height_end, :, :]   # 顶部：存储位置图
        img4embed = cover_img[height_end:, :, :]    # 底部：嵌入水印的主体
        return np.float32(img4embed), np.float32(img4locmap)

    def merge_img(self, watermarked_img4embed: ndarray, marked_img4locmap: ndarray, split_height: int):
        """
        将嵌入后的主体图像和位置图区域合并为完整的隐写图像。

        参数:
            watermarked_img4embed: 嵌入水印后的主体图像。
            marked_img4locmap: 存储辅助信息的位置图区域。
            split_height: 分割高度。

        返回:
            合并后的完整隐写图像。
        """
        watermarked_img = np.zeros(shape=self.img_size)
        watermarked_img[:split_height, :, :] = marked_img4locmap      # 顶部：位置图
        watermarked_img[split_height:, :, :] = watermarked_img4embed  # 底部：嵌入区域
        return watermarked_img

    def get_top_two_frequent_values(self, pe: ndarray, set_mask: ndarray):
        """
        找出预测误差中出现频率最高的两个值。

        这两个值将用于预测误差扩展嵌入，选择频率最高的值可以最大化嵌入容量。

        参数:
            pe: 预测误差数组。
            set_mask: 二进制掩码，指示有效嵌入区域。

        返回:
            元组 (first_most, first_count, second_most, second_count):
            - first_most: 出现频率最高的值
            - first_count: 最高频率
            - second_most: 出现频率第二高的值
            - second_count: 第二高频率
        """
        # 提取中心区域（排除边界像素）
        pe_central = pe[1:-1, 1:-1, :]
        mask_central = set_mask[1:-1, 1:-1, :]

        # 展平数组，只选择掩码为1的元素
        pe_flat = pe_central[mask_central == 1]

        # 统计各值的出现次数
        unique, counts = np.unique(pe_flat, return_counts=True)

        # 按频率降序排序
        sorted_indices = np.argsort(counts)[::-1]

        # 获取前两个最频繁的值
        top_two_indices = sorted_indices[:2]
        first_most = unique[top_two_indices[0]] if len(top_two_indices) > 0 else None
        first_count = counts[top_two_indices[0]] if len(top_two_indices) > 0 else None
        second_most = unique[top_two_indices[1]] if len(top_two_indices) > 1 else None
        second_count = counts[top_two_indices[1]] if len(top_two_indices) > 1 else None

        return first_most, first_count, second_most, second_count

    def shift_and_embed(self, pe: ndarray, set_mask: ndarray, wm_list: list):
        """
        预测误差平移与嵌入（核心算法）。

        基于预测误差扩展（PEE）技术：
        1. 找到预测误差中出现频率最高的两个值（bin）
        2. 将这两个bin之间的误差值向外平移，腾出空间
        3. 在这两个bin的位置嵌入水印比特（0保持，1扩展）

        参数:
            pe: 预测误差数组。
            set_mask: 二进制掩码，指示有效嵌入区域。
            wm_list: 要嵌入的水印比特列表。

        返回:
            元组 (shifted_pe, min_value, max_value, capacity, stopcoordinate, remaining_wm_list):
            - shifted_pe: 修改后的预测误差数组
            - min_value: 嵌入下限阈值
            - max_value: 嵌入上限阈值
            - capacity: 可用嵌入容量
            - stopcoordinate: 嵌入停止坐标
            - remaining_wm_list: 剩余未嵌入的水印比特
        """
        # ==================== 步骤1：计算最频繁的两个值 ====================
        first_most, first_count, second_most, second_count = self.get_top_two_frequent_values(pe, set_mask)
        capacity = first_count + second_count  # 可用容量为两个bin的总频率

        # ==================== 步骤2：定义嵌入阈值 ====================
        min_value = int(min(first_most, second_most))  # 较小的bin值
        max_value = int(max(first_most, second_most))  # 较大的bin值

        # ==================== 步骤3：初始化 ====================
        shifted_pe = pe.copy()
        H, W, C = pe.shape
        wm_index = 0  # 水印比特索引
        stopcoordinate = (0, 0, 0)  # 默认停止坐标

        # ==================== 步骤4：平移并嵌入水印 ====================
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if set_mask[h, w, c] == 1 and wm_index < len(wm_list):
                        stopcoordinate = (h, w, c)  # 记录当前位置

                        if pe[h, w, c] < min_value:
                            # 小于min_value的误差向左平移（-1）
                            shifted_pe[h, w, c] = pe[h, w, c] - 1.
                        elif pe[h, w, c] > max_value:
                            # 大于max_value的误差向右平移（+1）
                            shifted_pe[h, w, c] = pe[h, w, c] + 1.
                        elif pe[h, w, c] == max_value:
                            # 在max_value处嵌入：0保持，1扩展为max_value+1
                            shifted_pe[h, w, c] += wm_list[wm_index]
                            wm_index += 1
                        elif pe[h, w, c] == min_value:
                            # 在min_value处嵌入：0保持，1扩展为min_value-1
                            shifted_pe[h, w, c] -= wm_list[wm_index]
                            wm_index += 1

                        # 如果所有水印比特都已嵌入，提前返回
                        if wm_index == len(wm_list):
                            return shifted_pe, min_value, max_value, capacity, stopcoordinate, []

        # ==================== 步骤5：返回结果 ====================
        return shifted_pe, min_value, max_value, capacity, stopcoordinate, wm_list[wm_index:]

    def extract_and_shift(self, embedded_pe: ndarray, min_value: int, max_value: int, set_mask: ndarray,
                          stopcoordinate: tuple):
        """
        提取水印并恢复原始预测误差（shift_and_embed 的逆操作）。

        提取过程：
        1. 遍历嵌入后的预测误差
        2. 检测扩展值（max_value+1 或 min_value-1）提取水印比特1
        3. 检测原始值（max_value 或 min_value）提取水印比特0
        4. 将平移的误差值恢复原始位置

        参数:
            embedded_pe: 嵌入水印后的预测误差数组。
            min_value: 嵌入时使用的下限阈值。
            max_value: 嵌入时使用的上限阈值。
            set_mask: 二进制掩码，指示有效嵌入区域。
            stopcoordinate: 嵌入停止坐标。

        返回:
            元组 (restored_pe, extracted_wm_list):
            - restored_pe: 恢复后的原始预测误差
            - extracted_wm_list: 提取的水印比特列表
        """
        H, W, C = embedded_pe.shape
        restored_pe = embedded_pe.copy()
        extracted_wm_list = []
        stop_reached = False

        # ==================== 第一步：提取水印 ====================
        for h in range(1, H - 1):
            if stop_reached:
                break
            for w in range(1, W - 1):
                if stop_reached:
                    break
                for c in range(C):
                    if stop_reached:
                        break
                    if set_mask[h, w, c] == 1:
                        # 检测 max_value+1：表示嵌入的是比特1
                        if embedded_pe[h, w, c] == max_value + 1:
                            extracted_wm_list.append(1)
                            restored_pe[h, w, c] = max_value  # 恢复为原始值
                        # 检测 max_value：表示嵌入的是比特0
                        elif embedded_pe[h, w, c] == max_value:
                            extracted_wm_list.append(0)
                        # 检测 min_value-1：表示嵌入的是比特1
                        elif embedded_pe[h, w, c] == min_value - 1:
                            extracted_wm_list.append(1)
                            restored_pe[h, w, c] = min_value  # 恢复为原始值
                        # 检测 min_value：表示嵌入的是比特0
                        elif embedded_pe[h, w, c] == min_value:
                            extracted_wm_list.append(0)

                        # 到达停止坐标，结束提取
                        if (h, w, c) == stopcoordinate:
                            stop_reached = True
                            break

        # ==================== 第二步：恢复平移的误差值 ====================
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if set_mask[h, w, c] == 1:
                        # 将向左平移的误差恢复（+1）
                        if restored_pe[h, w, c] < min_value:
                            restored_pe[h, w, c] += 1
                        # 将向右平移的误差恢复（-1）
                        elif restored_pe[h, w, c] > max_value:
                            restored_pe[h, w, c] -= 1

                    # 到达停止坐标，提前返回
                    if (h, w, c) == stopcoordinate:
                        return restored_pe, extracted_wm_list

        return restored_pe, extracted_wm_list

    def compute_stego_img(self, pe_o: ndarray, pe_x: ndarray, pv_o: ndarray, pv_x: ndarray):
        """
        根据预测误差和预测值计算隐写图像。

        公式: 隐写图像 = (预测误差 + 预测值) * 掩码

        参数:
            pe_o: O组的预测误差。
            pe_x: X组的预测误差。
            pv_o: O组的预测值。
            pv_x: X组的预测值。

        返回:
            计算得到的隐写图像。
        """
        # 分别计算O组和X组的像素值，然后用掩码合并
        stego_img = (pe_o + pv_o) * self.mask_o + (pe_x + pv_x) * self.mask_x
        return stego_img

    def compute_overflow_map(self, stego_img: ndarray, mask: ndarray, stopcoordinate: Tuple) -> list:
        """
        计算溢出/下溢位置图（Location Map）。

        在预测误差扩展过程中，某些像素值可能超出 [0, 255] 范围，
        需要记录这些位置以便后续恢复。

        参数:
            stego_img: 嵌入后的隐写图像。
            mask: 有效嵌入区域的掩码。
            stopcoordinate: 提前停止坐标。

        返回:
            位置图列表，1表示溢出/下溢，0表示正常。
        """
        H, W, C = stego_img.shape
        if stego_img.dtype is not np.float32:
            stego_img = stego_img.astype(np.float32)

        location_map = []
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if mask[h, w, c] == 1:
                        pixel_value = stego_img[h, w, c]
                        # 检测溢出（>255）或下溢（<0）
                        if pixel_value > 255 or pixel_value < 0:
                            location_map.append(1)
                        else:
                            location_map.append(0)

                    # 到达停止坐标，提前返回
                    if (h, w, c) == stopcoordinate:
                        return location_map
        return location_map

    def recovery_overflow_stego_image(self, clipped_stego_img: ndarray, mask: ndarray, overflow_map: list) -> ndarray:
        """
        根据位置图恢复溢出/下溢像素，还原截断前的隐写图像。

        在嵌入过程中，超出 [0, 255] 范围的像素被截断（clip），
        通过位置图可以将这些像素恢复到原始值（-1 或 256）。

        参数:
            clipped_stego_img: 截断后的隐写图像。
            mask: 有效嵌入区域的掩码。
            overflow_map: 溢出位置图，1表示溢出/下溢位置。

        返回:
            恢复后的隐写图像（float32类型）。
        """
        H, W, C = clipped_stego_img.shape
        recovered_img = clipped_stego_img.astype(np.float32)
        location_idx = 0

        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if mask[h, w, c] == 1 and location_idx < len(overflow_map):
                        if overflow_map[location_idx] == 1:
                            # 恢复下溢像素（原值为 -1，截断后为 0）
                            if clipped_stego_img[h, w, c] == 0:
                                recovered_img[h, w, c] = -1.0
                            # 恢复溢出像素（原值为 256，截断后为 255）
                            elif clipped_stego_img[h, w, c] == 255:
                                recovered_img[h, w, c] = 256.0
                        location_idx += 1
                        if location_idx == len(overflow_map):
                            return recovered_img
        return recovered_img

    def encodeIntegerbyGivenLength(self, n: int, length: int) -> list:
        """
        将整数编码为指定长度的二进制列表（支持正负数）。

        使用补码表示负数，确保可以正确编码和解码有符号整数。

        参数:
            n: 要编码的整数。
            length: 二进制表示的目标长度。

        返回:
            整数编码后的比特列表。
        """
        # 如果是负数，转换为补码表示
        if n < 0:
            n = (1 << length) + n

        # 获取二进制表示（去掉 '0b' 前缀）
        binary_representation = bin(n)[2:]

        # 如果长度不足，左侧补零
        if len(binary_representation) < length:
            binary_representation = binary_representation.zfill(length)
        elif len(binary_representation) > length:
            raise ValueError("数值超出给定长度的表示范围")

        return [int(bit) for bit in binary_representation]

    def decodeIntegerbyGivenBits(self, bits: list) -> int:
        """
        将二进制列表解码为整数（支持正负数）。

        检测符号位（最高位），如果是1则表示负数，进行补码转换。

        参数:
            bits: 二进制比特列表。

        返回:
            解码后的整数。
        """
        bit_string = ''.join(map(str, bits))
        value = int(bit_string, 2)

        # 如果符号位为1，转换为负数（补码）
        if bits[0] == 1:
            value -= (1 << len(bits))

        return value

    def encode_auxiliary_information(self, overflow_map_o, min_v_o, max_v_o, stop_coor_o,
                                     overflow_map_x, min_v_x, max_v_x, stop_coor_x):
        """
        编码辅助信息，用于恢复时重建原始图像。

        辅助信息包括：
        - 溢出位置图（压缩存储）
        - 最小/最大像素值（用于确定嵌入bin）
        - 停止坐标（指示嵌入结束位置）

        参数:
            overflow_map_o: O组的溢出位置图。
            min_v_o, max_v_o: O组的最小/最大像素值。
            stop_coor_o: O组的停止坐标。
            overflow_map_x: X组的溢出位置图。
            min_v_x, max_v_x: X组的最小/最大像素值。
            stop_coor_x: X组的停止坐标。

        返回:
            编码后的辅助信息比特列表。
        """
        # ==================== 编码溢出位置图长度 ====================
        l_mask_o = int(self.c_len + self.h_len + self.w_len)
        l_mask_o_bits = [int(b) for b in format(len(overflow_map_o), f'0{l_mask_o}b')]

        # 合并O组和X组的溢出位置图
        overflow_map_bitstream = l_mask_o_bits + overflow_map_o + overflow_map_x

        # 使用算术编码压缩位置图
        compressed_overflow_map_bitstream = self.customAC.compress(np.asarray(overflow_map_bitstream), ["1", "0"])

        # ==================== 编码最小/最大值 ====================
        min_v_o_bits = self.encodeIntegerbyGivenLength(min_v_o, self.grayscale_bit)
        max_v_o_bits = self.encodeIntegerbyGivenLength(max_v_o, self.grayscale_bit)
        min_v_x_bits = self.encodeIntegerbyGivenLength(min_v_x, self.grayscale_bit)
        max_v_x_bits = self.encodeIntegerbyGivenLength(max_v_x, self.grayscale_bit)

        # ==================== 编码停止坐标 ====================
        h_bits = self.encodeIntegerbyGivenLength(stop_coor_o[0], self.h_len)
        w_bits = self.encodeIntegerbyGivenLength(stop_coor_o[1], self.w_len)
        c_bits = self.encodeIntegerbyGivenLength(stop_coor_o[2], self.c_len)
        coor_o_bits = h_bits + w_bits + c_bits

        h_bits = self.encodeIntegerbyGivenLength(stop_coor_x[0], self.h_len)
        w_bits = self.encodeIntegerbyGivenLength(stop_coor_x[1], self.w_len)
        c_bits = self.encodeIntegerbyGivenLength(stop_coor_x[2], self.c_len)
        coor_x_bits = h_bits + w_bits + c_bits

        # ==================== 合并所有辅助信息 ====================
        all_bits = (compressed_overflow_map_bitstream + min_v_o_bits + max_v_o_bits +
                    min_v_x_bits + max_v_x_bits + coor_o_bits + coor_x_bits)

        return all_bits

    def decode_auxiliary_information(self, bits4auxinfo: list):
        """
        Decomposes the auxiliary information bitstream to retrieve the overflow maps, min/max values, and stop coordinates.

        Args:
            bits4auxinfo (list): A list of bits representing the encoded auxiliary information.

        Returns:
            tuple: A tuple containing:
                - overflow_map_mask_o (list): A bit list representing the overflow location map for the original image.
                - min_v_o (int): The minimum value for the original image.
                - max_v_o (int): The maximum value for the original image.
                - stop_coor_o (tuple): The stop coordinate (h, w, c) for the original image.
                - overflow_map_mask_x (list): A bit list representing the overflow location map for the attacked image.
                - min_v_x (int): The minimum value for the attacked image.
                - max_v_x (int): The maximum value for the attacked image.
                - stop_coor_x (tuple): The stop coordinate (h, w, c) for the attacked image.
        """
        # Extract the bits representing the coordinates and min/max values
        total_coor_bits = int(2 * (self.h_len + self.w_len + self.c_len))
        total_min_max_bits = int(4 * self.grayscale_bit)
        corr_bits = bits4auxinfo[-(total_coor_bits + total_min_max_bits):]
        compressed_bitstream = bits4auxinfo[:-(total_coor_bits + total_min_max_bits)]

        # Length of the bitstream encoding the map mask dimensions (h_len, w_len, c_len)
        l_mask_o = int(self.c_len + self.h_len + self.w_len)

        # Decompress the bitstream containing the overflow map information
        compressed_overflow_map_bitstream = list(self.customAC.decompress(compressed_bitstream, ["1", "0"]))

        # Extract the bits representing the length of the overflow map
        l_mask_o_bits = compressed_overflow_map_bitstream[:l_mask_o]
        len_mask_o = int(''.join(map(str, l_mask_o_bits)), 2)

        # Split the overflow map into mask_o and mask_x based on the extracted length
        overflow_map_mask_o = compressed_overflow_map_bitstream[l_mask_o:][:len_mask_o]
        overflow_map_mask_x = compressed_overflow_map_bitstream[l_mask_o:][len_mask_o:]

        # Decode the min and max values for the original and attacked images
        min_v_o_bits = corr_bits[:self.grayscale_bit]
        max_v_o_bits = corr_bits[self.grayscale_bit:2 * self.grayscale_bit]
        min_v_x_bits = corr_bits[2 * self.grayscale_bit:3 * self.grayscale_bit]
        max_v_x_bits = corr_bits[3 * self.grayscale_bit:4 * self.grayscale_bit]

        min_v_o = self.decodeIntegerbyGivenBits(min_v_o_bits)
        max_v_o = self.decodeIntegerbyGivenBits(max_v_o_bits)
        min_v_x = self.decodeIntegerbyGivenBits(min_v_x_bits)
        max_v_x = self.decodeIntegerbyGivenBits(max_v_x_bits)

        # Decode the stop coordinates
        coor_bits = corr_bits[4 * self.grayscale_bit:]
        coor_o_bits = coor_bits[:int(self.h_len + self.w_len + self.c_len)]
        coor_x_bits = coor_bits[int(self.h_len + self.w_len + self.c_len):]

        h_o = self.decodeIntegerbyGivenBits(coor_o_bits[:self.h_len])
        w_o = self.decodeIntegerbyGivenBits(coor_o_bits[self.h_len:self.h_len + self.w_len])
        c_o = self.decodeIntegerbyGivenBits(coor_o_bits[self.h_len + self.w_len:])
        stop_coor_o = (h_o, w_o, c_o)

        h_x = self.decodeIntegerbyGivenBits(coor_x_bits[:self.h_len])
        w_x = self.decodeIntegerbyGivenBits(coor_x_bits[self.h_len:self.h_len + self.w_len])
        c_x = self.decodeIntegerbyGivenBits(coor_x_bits[self.h_len + self.w_len:])
        stop_coor_x = (h_x, w_x, c_x)

        return (
            overflow_map_mask_o, min_v_o, max_v_o, stop_coor_o,
            overflow_map_mask_x, min_v_x, max_v_x, stop_coor_x
        )

    def extract_lsb(self, image: ndarray) -> list:
        """
        Extract the least significant bits (LSBs) from the image, based on the object's bit_plane attribute.
        """
        if image.dtype is not np.uint8:
            image = image.astype(np.uint8)
        if image.ndim != 3:
            raise ValueError("Input image must have 3 dimensions.")
        if self.bit_plane < 1 or self.bit_plane > 8:
            raise ValueError("Bit plane must be between 1 and 8.")
        mask = (1 << self.bit_plane) - 1
        lsb_bits = []
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                for c in range(image.shape[2]):
                    pixel_value = image[h, w, c]
                    lsb_bits_for_pixel = pixel_value & mask
                    bits_str = format(lsb_bits_for_pixel, f'0{self.bit_plane}b')[::-1]
                    lsb_bits.extend([int(bit) for bit in bits_str])
        return lsb_bits

    def insert_lsb(self, image: ndarray, lsb_bits: list) -> ndarray:
        """
        Insert the least significant bits (LSBs) back into the image, based on the object's bit_plane attribute.
        """
        if image.dtype is not np.uint8:
            image = image.astype(np.uint8)
        if image.ndim != 3:
            raise ValueError("Input image must have 3 dimensions.")
        if self.bit_plane < 1 or self.bit_plane > 8:
            raise ValueError("Bit plane must be between 1 and 8.")
        mask = (1 << self.bit_plane) - 1
        num_bits = self.bit_plane * image.shape[0] * image.shape[1] * image.shape[2]
        if len(lsb_bits) != num_bits:
            raise ValueError(f"Expected {num_bits} LSB bits, but got {len(lsb_bits)}.")
        lsb_idx = 0
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                for c in range(image.shape[2]):
                    # 将像素值转为 Python int，避免 numpy uint8 的溢出问题

                    pixel_value = int(image[h, w, c])
                    # 清除低位：使用 & (255 - mask) 替代 & ~mask
                    # ~mask 在 Python 中会产生负数（无限位补码），导致溢出错误
                    pixel_value &= (255 - mask)
                    lsb_bits_for_pixel = 0
                    for b in range(self.bit_plane):
                        lsb_bits_for_pixel |= (lsb_bits[lsb_idx] << b)
                        lsb_idx += 1
                    pixel_value |= lsb_bits_for_pixel
                    image[h, w, c] = np.uint8(pixel_value)
        return image

    def encode_bitswithlength(self, bitstream: list, pos: str = "prefix"):
        """
        Encode a bitstream with its length information.

        Args:
            bitstream (list): A list of binary bits (e.g., [1, 0, 1, 1]).
            pos (str): Position of the length information, either "prefix" or "suffix".

        Returns:
            list: A new bitstream with length information added.
        """
        assert pos in ["prefix", "suffix"], ValueError("pos must be either 'prefix' or 'suffix'")
        # Calculate the number of bits needed to represent the length
        bits_len = int(self.c_len + self.h_len + self.w_len)
        # Convert the length to a fixed-length binary string
        length_bits = format(len(bitstream), f'0{bits_len}b')
        # Convert the binary string to a list of integers
        length_bits = list(map(int, length_bits))
        # Combine length bits and the original bitstream
        if pos == "prefix":
            return length_bits + bitstream
        else:
            return bitstream + length_bits

    def decode_bitswithlength(self, encoded_bits: list, pos: str = "prefix"):
        """
        Decode a bitstream by its length information.

        Args:
            encoded_bits (list): A list of binary bits containing multiple encoded bitstreams.
            pos (str): Position of the length information, either "prefix" or "suffix".

        Returns:
            list: The decoded bitstream (list of 0s and 1s).
        """
        assert pos in ["prefix", "suffix"], ValueError("pos must be either 'prefix' or 'suffix'")
        # Calculate the number of bits needed to represent the length
        bits_len = int(self.c_len + self.h_len + self.w_len)
        # Extract length bits and determine the length of the bitstream
        if pos == "prefix":
            length_bits = encoded_bits[:bits_len]  # Length bits at the beginning
            length = int(''.join(map(str, length_bits)), 2)
            decoded_bitstream = encoded_bits[bits_len:bits_len + length]
            other_bits = encoded_bits[bits_len + length:]
        else:
            length_bits = encoded_bits[-bits_len:]  # Length bits at the end
            length = int(''.join(map(str, length_bits)), 2)
            decoded_bitstream = encoded_bits[:-bits_len][-length:]
            other_bits = encoded_bits[:-(bits_len + length)]
        return decoded_bitstream, other_bits

    def encode_auxbitslist(self, auxbits: list):
        """
        Encode a list of auxiliary bitstreams into a single bitstream.

        This function concatenates the length of each bitstream (in binary) followed by the bitstream itself
        into a single bitstream. The length of each bitstream is encoded using a fixed-length binary representation
        defined by `self.c_len + self.h_len + self.w_len`.

        Args:
            auxbits (list): A list of lists, where each inner list represents a bitstream of 0s and 1s.

        Returns:
            list: A single list representing the encoded bitstream, where each bitstream is prefixed with its length
                  (encoded as binary).
        """
        encoded_bits = []
        for bitstream in auxbits:
            encoded_bits.extend(self.encode_bitswithlength(bitstream, pos="prefix"))
        return encoded_bits

    def decode_auxbitslist(self, encoded_bits):
        """
        Decode a single encoded bitstream back into a list of auxiliary bitstreams.

        This function reads the length of each bitstream (in binary) from the encoded bitstream, extracts the corresponding
        bitstream, and returns the original list of bitstreams.

        Args:
            encoded_bits (list): A single list representing the encoded bitstream, where each bitstream is prefixed with its
                                  length (encoded as binary).

        Returns:
            list: A list of lists, where each inner list represents a decoded bitstream of 0s and 1s.
        """
        decoded_bitstreams = []
        idx = 0
        # Define the number of bits used to represent the length of each bitstream
        bits_len = int(self.c_len + self.h_len + self.w_len)
        # Process the encoded bitstream by extracting bitstreams one by one
        while idx < len(encoded_bits):
            # Extract the binary length of the current bitstream
            length_bits = encoded_bits[idx: idx + bits_len]
            # Convert the binary length bits to an integer
            length = int(''.join(map(str, length_bits)), 2)
            # Move the index forward by the length of the binary length bits
            idx += bits_len
            # Extract the actual bitstream using the length value
            bitstream = encoded_bits[idx: idx + length]
            # Add the decoded bitstream to the list
            decoded_bitstreams.append(bitstream)
            # Move the index forward by the length of the extracted bitstream
            idx += length
        return decoded_bitstreams

    def embed_watermark(self, cover_img: ndarray, watermark_list: list):
        """
        Embed a watermark into the cover image using reversible data hiding techniques.

        :param cover_img: The cover image as a NumPy array.
        :param watermark_list: The list of watermark bits to embed.
        :return: A tuple indicating success and the resulting watermarked image.
        """
        now_watermark_list = watermark_list.copy()
        pe_o, pe_x, pv_o, pv_x = self.predicting_error(cover_img)
        result_o = self.shift_and_embed(pe_o, self.mask_o, now_watermark_list)
        shifted_stego_pe_o, min_v_o, max_v_o, capacity_o, stop_coor_o, rest_wm_list = result_o
        stego_img4embed_o = self.compute_stego_img(shifted_stego_pe_o, pe_x, pv_o, pv_x)
        overflow_map_mask_o = self.compute_overflow_map(stego_img4embed_o, self.mask_o, stop_coor_o)
        auxbit = self.encode_auxiliary_information(overflow_map_mask_o, min_v_o, max_v_o, stop_coor_o, [], 0, 0,
                                                   (0, 0, 0))
        stego_img = np.clip(stego_img4embed_o, 0, 255)
        if len(rest_wm_list) > 0:
            pe_o, pe_x, pv_o, pv_x = self.predicting_error(stego_img4embed_o)
            result_x = self.shift_and_embed(pe_x, self.mask_x, rest_wm_list)
            shifted_stego_pe_x, min_v_x, max_v_x, capacity_x, stop_coor_x, rest_wm_list = result_x
            stego_img4embed_x = self.compute_stego_img(pe_o, shifted_stego_pe_x, pv_o, pv_x)
            overflow_map_mask_x = self.compute_overflow_map(stego_img4embed_x, self.mask_x, stop_coor_x)
            auxbit = self.encode_auxiliary_information(overflow_map_mask_o, min_v_o, max_v_o, stop_coor_o,
                                                       overflow_map_mask_x, min_v_x, max_v_x, stop_coor_x)
            stego_img = np.clip(stego_img4embed_x, 0, 255)
        return stego_img, auxbit, rest_wm_list

    def embed_once(self, cover_img: ndarray, watermark_list: list, time_index: int):
        """
        Embed a watermark into the cover image through iterative embedding.

        :param cover_img: The original cover image to embed the watermark into (H, W, C).
        :param watermark_list: A list of bits representing the watermark to embed.
        :param time_index: The time of embedding.
        :return:
            True: Indicating the embedding process has completed.
            marked_img: The image with the embedded watermark (H, W, C).
            time_index: The time of embedding
        """
        try_index = 0
        lsb_used_length = 500
        while True:
            split_height = int(np.ceil(lsb_used_length / self.storage_len))
            self.set_mask(split_height)
            img4embed, img4locmap = self.split_img(cover_img, split_height)
            lsb_bits = self.extract_lsb(img4locmap)
            used_lsb_bits = self.encode_bitswithlength(lsb_bits[:lsb_used_length], pos="prefix")
            watermark_list_now = used_lsb_bits + watermark_list
            img4embed, auxbits, rest_wm_list = self.embed_watermark(img4embed, watermark_list_now)
            flatten_auxbits = self.encode_bitswithlength(auxbits, pos="prefix")
            split_height_bits = format(split_height, f'0{self.h_len}b')
            split_height_bits = list(map(int, split_height_bits))
            if time_index == 0:
                flatten_auxbits_with_stop_flag = [0] + split_height_bits + flatten_auxbits
            else:
                flatten_auxbits_with_stop_flag = [1] + split_height_bits + flatten_auxbits
            if len(flatten_auxbits_with_stop_flag) <= lsb_used_length:
                lsb_bits[:len(flatten_auxbits_with_stop_flag)] = flatten_auxbits_with_stop_flag
                marked_img4locmap = self.insert_lsb(img4locmap, lsb_bits)
                marked_img = self.merge_img(img4embed, marked_img4locmap, split_height)
                return np.uint8(np.clip(marked_img, 0, 255)), rest_wm_list
            else:
                lsb_used_length = int(np.ceil(len(flatten_auxbits_with_stop_flag) / 500.) * 500)
                try_index += 1

    def embed(self, cover_img: ndarray, watermark_list: list):
        """
        Embeds a given watermark into the cover image iteratively until all watermark bits are embedded.

        Args:
            cover_img (ndarray): The original cover image where the watermark will be embedded.
                                 It can be a grayscale or color image with shape (H, W) or (H, W, C).
            watermark_list (list): A list of binary watermark bits to be embedded into the image.

        Returns:
            ndarray: The resulting stego image with the embedded watermark.
        """
        issuccessful = True
        stego_img = cover_img.copy()
        # Ensure the image has three dimensions (H, W, C) for consistency
        if stego_img.ndim == 2:
            stego_img = stego_img.reshape((*stego_img.shape, 1))
        # Iteratively embed watermark bits until the list is empty
        time_index = 0
        while len(watermark_list) > 0 and issuccessful:
            stego_img, next_watermark_list = self.embed_once(stego_img, watermark_list, time_index)
            issuccessful = len(next_watermark_list) < len(watermark_list)
            watermark_list = next_watermark_list
            time_index += 1
        # Convert back to 2D if the original image was grayscale
        if stego_img.shape[2] == 1:
            stego_img = stego_img[:, :, 0]
        return issuccessful, stego_img

    def extract(self, stego_img: ndarray):
        """
        Extracts the embedded watermark bits from the given stego image.

        Args:
            stego_img (ndarray): The stego image containing the embedded watermark.
                                 It can be a grayscale or color image with shape (H, W) or (H, W, C).

        Returns:
            list: A list of extracted binary watermark bits, in the order they were embedded.
        """
        ext_all_bits = []  # List to store all extracted watermark bits
        iscontinue = True  # Flag to indicate whether more bits are available for extraction
        issuccessful = True
        # Ensure the image has three dimensions (H, W, C) for consistency
        if stego_img.ndim == 2:
            stego_img = stego_img.reshape((*stego_img.shape, 1))
        # Iteratively extract watermark bits until no more bits are found
        while iscontinue:
            try:
                iscontinue, stego_img, ext_bits = self.extract_once(stego_img)
                ext_all_bits = ext_bits + ext_all_bits
            except Exception as e:
                iscontinue = False
                issuccessful = False
            # Prepend the newly extracted bits to maintain the correct order
        if stego_img.shape[2] == 1:
            stego_img = stego_img[:, :, 0]
        return issuccessful, np.uint8(np.clip(stego_img, 0, 255)), ext_all_bits

    def extract_once(self, stego_img: ndarray):
        """
        Extract the embedded watermark bits from the stego image through iterative extraction.

        :param stego_img: The stego image from which the watermark is to be extracted (H, W, C).
        :return:
            stego_img: The stego image after processing (H, W, C).
            wm_list: The list of extracted watermark bits (in reverse order of extraction).
        """

        stego4ext, img4locmap = self.split_img(stego_img, 1)
        lsb_bits = self.extract_lsb(img4locmap)
        iscontinue = bool(lsb_bits[0])
        split_height_bits = lsb_bits[1:1 + self.h_len]
        split_height = int(''.join(map(str, split_height_bits)), 2)
        self.set_mask(split_height)
        stego4ext, img4locmap = self.split_img(stego_img, split_height)
        lsb_bits = self.extract_lsb(img4locmap)
        flatten_auxbits, _ = self.decode_bitswithlength(lsb_bits[1 + self.h_len:], pos="prefix")
        rec_img4emb, wm_list = self.extract_watermark(stego4ext, flatten_auxbits)
        used_lsb_bits, ext_bits = self.decode_bitswithlength(wm_list, pos="prefix")
        lsb_bits[:len(used_lsb_bits)] = used_lsb_bits
        rec_img4locmap = self.insert_lsb(img4locmap, lsb_bits)
        rec_cover = self.merge_img(rec_img4emb, rec_img4locmap, split_height)
        return iscontinue, rec_cover, ext_bits

    def extract_watermark(self, stego4ext: ndarray, auxbits: List[List]):
        """
        Extract the watermark from the stego image by recovering auxiliary information
        and using reversible error prediction techniques.

        Args:
            stego4ext: The stego image from which the watermark is extracted.
            auxbits: A list of auxiliary bits used for the extraction process.

        Returns:
            stego4ext: The final stego image after watermark extraction.
            wm_list: The extracted watermark bits.
        """
        if stego4ext.dtype == np.float32:
            stego4ext = stego4ext.astype(np.float32)
        wm_list = []  # Initialize an empty list to store the extracted watermark
        # Decode the auxiliary information for the current bit
        decode_info = self.decode_auxiliary_information(auxbits)
        overflow_map_mask_o, min_v_o, max_v_o, stop_coor_o, overflow_map_mask_x, min_v_x, max_v_x, stop_coor_x = decode_info
        # Recover the overflow maps from the stego image
        stego4ext = self.recovery_overflow_stego_image(stego4ext, self.mask_o, overflow_map_mask_o)
        stego4ext = self.recovery_overflow_stego_image(stego4ext, self.mask_x, overflow_map_mask_x)
        if stop_coor_x != (0, 0, 0):
            # Predict the errors (pe) and predictions (pv) for the recovered stego image
            pe_o, pe_x, pv_o, pv_x = self.predicting_error(stego4ext)
            # Extract the watermark bits from the error predictions (pe) for both overflow regions
            recovered_pe_x, wm_list_x = self.extract_and_shift(pe_x, min_v_x, max_v_x, self.mask_x, stop_coor_x)
            # Recompute the stego image using the recovered predicting errors (pe) and predictions (pv)
            stego4ext = self.compute_stego_img(pe_o, recovered_pe_x, pv_o, pv_x)
            wm_list = wm_list_x + wm_list
        # Predict the errors and predictions for the new stego image after watermark recovery
        pe_o, pe_x, pv_o, pv_x = self.predicting_error(stego4ext)
        # Extract the watermark bits again for the other overflow region
        recovered_pe_o, wm_list_o = self.extract_and_shift(pe_o, min_v_o, max_v_o, self.mask_o, stop_coor_o)
        # Recompute the stego image using the recovered predicting errors (pe) for both overflow regions
        recovered_img4embed = self.compute_stego_img(recovered_pe_o, pe_x, pv_o, pv_x)
        # Combine the watermark bits extracted from both overflow regions
        wm_list = wm_list_o + wm_list
        return recovered_img4embed, wm_list


# ==================== CustomRDH 包装类 ====================
class CustomRDH:
    """
    CustomRDH 是 RDH 的 PyTorch 张量包装类。

    提供与 PyTorch 张量兼容的接口，自动处理张量与 NumPy 数组之间的转换，
    便于与神经网络模型集成。
    """

    def __init__(self, img_size: Tuple[int, int, int], device: str = "cpu"):
        """
        初始化 CustomRDH 类。

        参数:
            img_size: 图像尺寸 (高度, 宽度, 通道数)，如 (256, 256, 3)。
            device: 计算设备，"cuda" 或 "cpu"。
        """
        self.device = device
        self.rdh = RDH(img_size)

    def embed(self, cover_img: Tensor, watermark_list: list):
        """
        将水印嵌入封面图像（张量接口）。

        参数:
            cover_img: 封面图像张量，形状为 (N, C, H, W)，N=1。
            watermark_list: 要嵌入的水印比特列表。

        返回:
            元组 (issuccessful, rdh_stego_img):
            - issuccessful: 嵌入是否成功
            - rdh_stego_img: 输出的隐写图像（NumPy数组）
        """
        N, C, H, W = cover_img.shape
        assert N == 1, "只支持单张图像处理"

        # 将 PyTorch 张量转换为 NumPy 数组 (H, W, C)
        cover_img_numpy = cover_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # 使用 RDH 方法嵌入水印
        issuccessful, rdh_stego_img = self.rdh.embed(cover_img_numpy, watermark_list)

        return issuccessful, np.uint8(rdh_stego_img)

    def extract(self, rdh_stego_img: ndarray):
        """
        提取水印并重建封面图像（张量接口）。

        参数:
            rdh_stego_img: 隐写图像（NumPy数组）。

        返回:
            元组 (issuccessful, rec_stego_img_tensor, wm_list):
            - issuccessful: 提取是否成功
            - rec_stego_img_tensor: 重建的图像张量（已移至指定设备）
            - wm_list: 提取的水印比特列表
        """
        # 使用 RDH 方法提取水印
        issuccessful, rec_stego_img, wm_list = self.rdh.extract(rdh_stego_img)

        # 将 NumPy 数组转换回 PyTorch 张量
        if rdh_stego_img.ndim == 3:  # 彩色图
            rec_stego_img_tensor = torch.as_tensor(rec_stego_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        else:  # 灰度图
            rec_stego_img_tensor = torch.as_tensor(rec_stego_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return issuccessful, rec_stego_img_tensor.to(self.device), wm_list


# ==================== 工具函数 ====================
def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    计算两幅图像之间的 PSNR（峰值信噪比）。

    PSNR 是衡量图像质量的常用指标，值越大表示图像质量越好。
    通常 PSNR > 40dB 表示人眼无法察觉差异。

    参数:
        original: 原始图像。
        compressed: 压缩/重建后的图像。

    返回:
        PSNR 值（单位：dB）。
    """
    # 确保两幅图像尺寸相同
    assert original.shape == compressed.shape, "输入图像必须具有相同的尺寸"

    # 计算 MSE（均方误差）
    mse = np.mean((original - compressed) ** 2)

    # 如果 MSE 为0，返回无穷大（完全相同）
    if mse == 0:
        return float('inf')

    # 8位图像的最大像素值
    max_value = 255.0

    # 计算 PSNR
    psnr = 10 * np.log10(max_value ** 2 / mse)
    return psnr


def test4grayimage(img_path):
    """
    灰度图像 RDH 测试函数。

    演示完整的嵌入-提取流程，验证可逆性。
    """
    # 读取灰度图像
    cover_img = Image.open(img_path).convert("L")
    cover_img = np.float32(cover_img)
    cover_img_np = cover_img.reshape(cover_img.shape[0], cover_img.shape[1], 1)

    # 初始化 RDH
    rdh = RDH(img_size=cover_img_np.shape, height_end=10)

    # 生成随机水印
    watermark_length = 50000
    watermark_list = [random.randint(0, 1) for _ in range(watermark_length)]

    # 嵌入水印
    issuccessful, stego_img = rdh.embed(cover_img_np, watermark_list)

    # 保存隐写图像
    Image.fromarray(np.uint8(stego_img)).save('images/stego_img_gray.png')
    reload_stego_img = np.float32(Image.open('images/stego_img_gray.png'))

    # 提取水印
    recovered_cover_img, wm_list = rdh.extract(reload_stego_img)

    # 验证可逆性
    print("图像完全恢复:", np.array_equal(recovered_cover_img, cover_img_np))
    print("水印完全恢复:", np.array_equal(watermark_list, wm_list))
    print(f"PSNR: {calculate_psnr(cover_img, stego_img):.2f} dB")


def test4rgbimage(img_path):
    """
    RGB 彩色图像 RDH 测试函数。

    演示完整的嵌入-提取流程，并测试对轻微攻击的鲁棒性。
    """
    # 读取彩色图像
    cover_img = Image.open(img_path)
    cover_img_np = np.float32(cover_img)

    # 初始化 RDH
    rdh = RDH(img_size=cover_img_np.shape)

    # 生成随机水印
    watermark_length = 10000
    watermark_list = [random.randint(0, 1) for _ in range(watermark_length)]

    # 嵌入水印
    issuccessful, stego_img = rdh.embed(cover_img_np, watermark_list)

    # 保存隐写图像
    Image.fromarray(np.uint8(stego_img)).save('stego_img_rgb.tif')
    reload_stego_img = np.float32(Image.open('stego_img_rgb.tif'))

    # 测试对轻微攻击的鲁棒性（修改一个像素）
    reload_stego_img_attacked = reload_stego_img.copy()
    reload_stego_img_attacked[100, 12, 0] += 1

    # 提取水印（原始版本和攻击版本）
    issuccessful, recovered_cover_img_attacked, wm_list_attacked = rdh.extract(reload_stego_img_attacked)
    issuccessful, recovered_cover_img, wm_list = rdh.extract(reload_stego_img)

    # 验证结果
    print("攻击前后恢复图像相同:", np.array_equal(recovered_cover_img_attacked, recovered_cover_img))
    print("攻击前后水印相同:", np.array_equal(wm_list_attacked, wm_list))
    print("图像完全恢复:", np.array_equal(recovered_cover_img, cover_img_np))
    print("水印完全恢复:", np.array_equal(watermark_list, wm_list))
    print(f"PSNR: {calculate_psnr(cover_img_np, stego_img):.2f} dB")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 运行 RGB 图像测试
    test4rgbimage('images/BaboonRGB.tif')
    # 运行灰度图像测试
    # test4grayimage('compressor/images/BaboonRGB.tif')
