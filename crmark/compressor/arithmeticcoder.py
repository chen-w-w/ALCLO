#!/usr/bin/env python3
"""
算术编码器 - 累积和与Fenwick树实现
=====================================

本模块实现算术编码（Arithmetic Coding），用于高效压缩数据。

核心组件：
- FenwickTree: 树状数组，支持O(log n)的累积和操作
- CumulativeSum: 基于Fenwick树的累积频率表
- ArithmeticEncoder: 算术编码器/解码器
- CustomArithmeticEncoder: 针对NumPy数组优化的编码器

算术编码原理：
- 将整个消息编码为一个[0,1)区间内的浮点数
- 高频符号使用较大的子区间，低频符号使用较小的子区间
- 相比霍夫曼编码，可以达到分数比特的编码效率

自适应模型：
在自适应模型中，我们不使用固定的频率，而是：
1. 初始时每个符号的频率设为1
2. 编码过程中动态更新频率
3. 解码时同步更新，保持编解码器状态一致

示例：
>>> frequencies = {'a': 4, 'b': 1, 'c': 3}
>>> cumsum = CumulativeSum(frequencies)
>>> cumsum.get_low_high('a')
(0, 4)
>>> cumsum.get_low_high('b')
(4, 5)

使用Fenwick树可以将累积和操作从O(n)优化到O(log n)，
在符号数量较大时性能提升明显。
"""

# ==================== 导入依赖库 ====================
import copy
import random
import itertools

import numpy as np
from numpy import ndarray


# ==================== Fenwick树（树状数组）====================
class FenwickTree:
    """
    Fenwick树（Binary Indexed Tree），用于高效维护前缀和。

    核心操作：
    - prefix_sum(i): 查询[0, i)的前缀和，O(log n)
    - add(i, k): 在位置i增加k，O(log n)
    - range_sum(l, r): 查询[l, r)区间和，O(log n)

    原理：
    利用二进制表示，每个节点存储一段区间的和。
    例如：tree[12] = tree[1100b] 存储区间 [1100b - 100b, 1100b) = [8, 12) 的和

    参考实现：https://github.com/dstein64/fenwick

    示例：
    >>> frequencies = [1, 0, 2, 1, 1, 3, 0, 4]
    >>> ft = FenwickTree(frequencies)
    >>> ft.prefix_sum(3)  # 1 + 0 + 2 = 3
    3
    >>> ft[2]  # 获取索引2的值
    2
    """

    def __init__(self, frequencies):
        """
        初始化Fenwick树。

        参数:
            frequencies: 初始频率列表。
        """
        self._v = list(frequencies)

        # O(n)初始化：利用子节点更新父节点
        for idx in range(1, len(self) + 1):
            parent_idx = idx + (idx & -idx)  # 父节点索引
            if parent_idx <= len(self):
                self._v[parent_idx - 1] += self._v[idx - 1]

    def __len__(self):
        return len(self._v)

    def prefix_sum(self, stop):
        """
        计算[0, stop)区间的前缀和（不包含stop）。

        算法：利用lowbit不断减去最低位的1
        例如：prefix_sum(13) = prefix_sum(1101b)
        = tree[1101b] + tree[1100b] + tree[1000b]

        参数:
            stop: 结束位置（不包含）。

        返回:
            前缀和。
        """
        if stop <= 0 or stop > len(self):
            raise IndexError("index out of range")
        _sum = 0
        while stop > 0:
            _sum += self._v[stop - 1]
            stop &= stop - 1  # 减去lowbit
        return _sum

    def range_sum(self, start, stop):
        """
        计算[start, stop)区间的和。

        参数:
            start: 起始位置（包含）。
            stop: 结束位置（不包含）。

        返回:
            区间和。
        """
        if start < 0 or start >= len(self):
            raise IndexError("index out of range")
        if stop <= start or stop > len(self):
            raise IndexError("index out of range")
        result = self.prefix_sum(stop)
        if start > 0:
            result -= self.prefix_sum(start)
        return result

    def __getitem__(self, idx):
        """
        获取索引idx处的值（不是累积和）。

        参数:
            idx: 索引（支持负数索引）。

        返回:
            该位置的值。
        """
        if isinstance(idx, int):
            idx = idx % len(self)
            return self.range_sum(idx, idx + 1)
        else:
            raise IndexError(f"Indexing only works with integers, got {idx}")

    def frequencies(self):
        """
        获取所有频率值，O(n)。

        返回:
            频率列表。
        """
        _frequencies = [0] * len(self)
        for idx in range(1, len(self) + 1):
            _frequencies[idx - 1] += self._v[idx - 1]
            parent_idx = idx + (idx & -idx)
            if parent_idx <= len(self):
                _frequencies[parent_idx - 1] -= self._v[idx - 1]
        return _frequencies

    def add(self, idx, k):
        """
        在索引idx处增加k。

        参数:
            idx: 索引（0-based）。
            k: 要增加的值。
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")
        idx += 1
        while idx <= len(self):
            self._v[idx - 1] += k
            idx += idx & -idx  # 加上lowbit，更新所有相关父节点

    def __setitem__(self, idx, value):
        """直接设置值（通过add实现）。"""
        self.add(idx, value - self[idx])

    def bisect_left(self, value):
        """
        二分查找：返回最小的索引i，使得前缀和 >= value。

        用于根据累积值查找对应的符号。
        例如：在算术解码时，根据编码值查找对应的符号。

        参数:
            value: 目标累积值。

        返回:
            满足条件的最小索引，如果不存在则返回len(self)。
        """
        # 参考：https://stackoverflow.com/questions/34699616/fenwick-trees-to-determine-which-interval-a-point-falls-in
        j = 2 ** len(self)

        i = -1
        while j > 0:
            if i + j < len(self) and value >= self._v[i + j]:
                value -= self._v[i + j]
                i += j
            j >>= 1
        return i + 1

    def __eq__(self, other):
        return isinstance(other, FenwickTree) and self._v == other._v


# ==================== 累积和类 ====================
class NaiveCumulativeSum:
    """
    简单累积和实现（O(n)操作）。

    用于对比和测试，实际使用CumulativeSum（基于Fenwick树）。
    """

    def __init__(self, frequencies, update=True):
        """
        初始化累积和。

        参数:
            frequencies: 符号频率字典。
            update: 是否允许更新频率。
        """
        self.frequencies = dict(frequencies)
        self.ranges = dict(self.ranges_from_frequencies(self.frequencies))
        self.update = update

    def get_low_high(self, symbol):
        """获取符号的累积范围 (low, high)。"""
        return self.ranges[symbol]

    def add_count(self, symbol, value):
        """增加符号的计数（O(n)）。"""
        if self.update:
            self.frequencies[symbol] += value
            self.ranges = dict(self.ranges_from_frequencies(self.frequencies))

    def total_count(self):
        """获取总频率。"""
        return sum(self.frequencies.values())

    def reset(self):
        """重置所有频率为1。"""
        self.frequencies = {frequency: 1 for frequency in self.frequencies}
        self.ranges = dict(self.ranges_from_frequencies(self.frequencies))

    @staticmethod
    def ranges_from_frequencies(frequencies):
        """
        从频率字典构建范围字典。

        例如：{'a': 5, 'b': 3, 'c': 2} -> {'a': (0, 5), 'b': (5, 8), 'c': (8, 10)}
        """
        cumsum = 0
        for symbol, frequency in sorted(frequencies.items()):
            yield (symbol, (cumsum, cumsum + frequency))
            cumsum += frequency

    def search_ranges(self, value):
        """
        根据累积值查找对应的符号。

        参数:
            value: 累积值。

        返回:
            对应的符号。
        """
        for symbol, (low, high) in self.ranges.items():
            if low <= value < high:
                return symbol
        raise ValueError("Could not locate value in ranges.")


class CumulativeSum:
    """
    基于Fenwick树的快速累积和实现。

    用于算术编码中的符号频率管理，支持O(log n)的查询和更新。
    """

    def __init__(self, frequencies, update=True):
        """
        初始化累积和。

        参数:
            frequencies: 符号频率字典。
            update: 是否允许更新频率（自适应模型）。
        """
        symbols = sorted(frequencies.keys())
        self.idx_to_symbol = dict(enumerate(symbols))  # 索引到符号的映射
        self.symbol_to_idx = {s: i for (i, s) in self.idx_to_symbol.items()}  # 符号到索引的映射
        self.fenwick_tree = FenwickTree([frequencies[s] for s in symbols])
        self.update = update

    def get_low_high(self, symbol):
        """
        获取符号的累积范围 [low, high)。

        在算术编码中，这个范围决定了符号对应的子区间。

        参数:
            symbol: 符号。

        返回:
            (low, high) 元组。
        """
        idx = self.symbol_to_idx[symbol]
        if idx == 0:
            return (0, self.fenwick_tree[idx])

        sum_upto = self.fenwick_tree.prefix_sum(idx)
        return (sum_upto, sum_upto + self.fenwick_tree[idx])

    def add_count(self, symbol, value):
        """
        增加符号的计数。

        用于自适应模型：编码/解码过程中动态更新频率。

        参数:
            symbol: 符号。
            value: 增加的值。
        """
        if self.update:
            idx = self.symbol_to_idx[symbol]
            self.fenwick_tree.add(idx, value)

    def total_count(self):
        """获取总频率。"""
        return self.fenwick_tree.prefix_sum(len(self.fenwick_tree))

    def reset(self):
        """重置所有频率为1。"""
        self.fenwick_tree = FenwickTree([1] * len(self.fenwick_tree))

    def search_ranges(self, value):
        """
        根据累积值查找符号。

        在算术解码时使用：根据编码值确定对应的符号。

        参数:
            value: 累积值。

        返回:
            对应的符号。
        """
        idx = self.fenwick_tree.bisect_left(value)
        return self.idx_to_symbol[idx]


# ==================== 算术编码器 ====================
"""
算术编码器实现说明
===================

算术编码是一种熵编码方法，将整个消息编码为一个[0,1)区间内的浮点数。

核心思想：
- 根据符号频率将区间划分为子区间
- 高频符号获得较大的子区间
- 低频符号获得较小的子区间
- 相比霍夫曼编码，可以达到分数比特的编码效率

基本使用示例：
>>> message = ['A', 'B', 'B', 'B', '<EOM>']
>>> frequencies = {'A': 1, 'B': 3, '<EOM>': 1}
>>> encoder = ArithmeticEncoder(frequencies=frequencies, bits=6)
>>> bits = list(encoder.encode(message))
>>> list(encoder.decode(bits))
['A', 'B', 'B', 'B', '<EOM>']

压缩效果示例：
- 符号'e'的频率是'q'的136倍
- 10000个符号压缩后仅需676比特
- 平均每个符号仅需0.0676比特
"""


class BitQueue:
    """
    比特队列，用于处理算术编码中的"比特跟随"情况。

    在算术编码中，当区间跨越中点时，需要延迟输出比特，
    直到区间收敛到某一侧。此时需要记录需要跟随的相反比特数量。

    示例：
    >>> bitqueue = BitQueue()
    >>> bitqueue += 3  # 有3个比特需要跟随
    >>> list(bitqueue.bit_plus_follow(0))  # 输出0，然后跟随3个1
    [0, 1, 1, 1]
    """

    bits_to_follow = 0  # 需要跟随的比特计数器

    def __add__(self, bits):
        """增加需要跟随的比特数量。"""
        self.bits_to_follow += bits
        return self

    def bit_plus_follow(self, bit):
        """
        输出比特，然后输出所有需要跟随的相反比特。

        参数:
            bit: 当前比特（0或1）。

        生成:
            当前比特，后跟bits_to_follow个相反比特。
        """
        yield bit
        yield from itertools.repeat(int(not bit), times=self.bits_to_follow)
        self.bits_to_follow = 0  # 重置计数器


class ArithmeticEncoder:
    """
    算术编码器/解码器实现。

    参考论文：
    - Witten, Neal, Cleary (1987). Arithmetic coding for data compression.
    - https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html

    核心概念：
    - 使用整数区间 [0, TOP_VALUE] 代替浮点数 [0, 1)
    - 通过不断缩小区间来编码符号
    - 当区间足够小时，输出确定的比特并扩展区间

    区间划分：
    - [0, FIRST_QUARTER): 下四分之一
    - [FIRST_QUARTER, HALF): 中下半区
    - [HALF, THIRD_QUARTER): 中上半区
    - [THIRD_QUARTER, TOP_VALUE]: 上四分之一
    """

    def __init__(self, frequencies: list, *, bits=16, verbose=0, EOM="<EOM>"):
        """
        初始化算术编码器。

        参数:
            frequencies: 符号频率字典或符号列表。
                        如果是字典，使用固定频率；
                        如果是列表，使用自适应模型（初始频率为1）。
            bits: 编码器使用的比特数，决定精度（默认16）。
            verbose: 调试信息级别（0=无，>0=详细）。
            EOM: 消息结束标记符号。

        示例:
            >>> encoder = ArithmeticEncoder(frequencies={'A': 1, 'B': 3}, bits=6)
            >>> bits = list(encoder.encode(['A', 'B', '<EOM>']))
            >>> list(encoder.decode(bits))
            ['A', 'B', '<EOM>']
        """
        self.EOM = EOM
        self.frequencies = frequencies.copy()
        self.bits = bits
        self.verbose = verbose

        # 从频率构建累积和
        if isinstance(frequencies, dict):
            assert all(isinstance(freq, int) for freq in self.frequencies.values())
            assert self.EOM in self.frequencies.keys()
            self.cumsum = CumulativeSum(dict(frequencies), update=False)
        elif isinstance(frequencies, (list, set)):
            assert self.EOM in self.frequencies
            frequencies = {symbol: 1 for symbol in self.frequencies}
            self.cumsum = CumulativeSum(frequencies, update=True)

        # 定义区间边界（以4比特为例）：
        self.TOP_VALUE = (1 << self.bits) - 1      # 0b1111 = 15，最大值
        self.FIRST_QUARTER = (self.TOP_VALUE >> 2) + 1  # 0b0100 = 4，四分之一处
        self.HALF = self.FIRST_QUARTER * 2         # 0b1000 = 8，中点
        self.THIRD_QUARTER = self.FIRST_QUARTER * 3     # 0b1100 = 12，四分之三处

        # 检查精度是否足够（论文第533页公式）
        if self.cumsum.total_count() > int((self.TOP_VALUE + 1) / 4) + 1:
            msg = "精度不足，无法编码低概率符号。"
            msg += "\n请增加 `bits` 参数的值。"
            raise Exception(msg)

        if self.verbose > 0:
            print("初始化参数:")
            print(f" bits          = {self.bits}")
            print(f" TOP_VALUE     = 0b{self.TOP_VALUE:0{self.bits}b} ({self.TOP_VALUE})")
            print(f" THIRD_QUARTER = 0b{self.THIRD_QUARTER:0{self.bits}b} ({self.THIRD_QUARTER})")
            print(f" HALF          = 0b{self.HALF:0{self.bits}b} ({self.HALF})")
            print(f" FIRST_QUARTER = 0b{self.FIRST_QUARTER:0{self.bits}b} ({self.FIRST_QUARTER})")

    def _print_state(self, low, high, value=None, *, prefix=" ", end="\n"):
        """打印编码器状态（用于调试）。"""
        range_ = high - low + 1
        print(prefix + f"High value: 0b{high:0{self.bits}b} ({high})")
        if value is not None:
            print(prefix + f"Value:      0b{value:0{self.bits}b} ({value})")
        print(prefix + f"Low value:  0b{low:0{self.bits}b} ({low})")
        print(prefix + f"Range: [{low}, {high + 1}) Width: {range_}", end=end)

    def encode(self, iterable):
        """
        编码符号序列，生成比特流。

        算术编码核心算法：
        1. 根据符号概率缩小编码区间 [low, high]
        2. 当区间落入某一区域时，输出确定的比特
        3. 扩展区间，继续编码

        参数:
            iterable: 符号序列。

        生成:
            比特（0或1）。
        """
        if self.verbose:
            print("------------------------ 编码中 ------------------------")

        # 复制累积和对象，防止编解码器共享状态
        cumsum = copy.deepcopy(self.cumsum)

        bit_queue = BitQueue()  # 比特跟随队列

        # 初始化编码区间 [low, high]
        low = 0
        high = self.TOP_VALUE

        # 遍历每个符号
        for i, symbol in enumerate(iterable, 1):
            if self.verbose > 0:
                print(f"\n处理第 {i} 个符号: {repr(symbol)}")
                print("-" * 32)

            # 当前区间宽度
            range_ = high - low + 1

            # 算法不变量检查
            if range_ < cumsum.total_count():
                msg = "精度不足，无法编码低概率符号。"
                msg += "\n请增加 `bits` 参数的值。"
                raise Exception(msg)
            assert 0 <= low <= high <= self.TOP_VALUE
            assert low < self.HALF <= high
            assert high - low > self.FIRST_QUARTER

            if self.verbose > 0:
                self._print_state(low, high, prefix="")

            # 获取符号的累积频率范围
            symbol_low, symbol_high = cumsum.get_low_high(symbol)

            # 根据符号概率缩小区间
            # 注意：运算顺序必须与解码器完全一致
            total_count = cumsum.total_count()
            high = low + int(range_ * symbol_high / total_count) - 1
            low = low + int(range_ * symbol_low / total_count)

            if self.verbose > 0:
                prob = (symbol_high - symbol_low) / total_count
                print(f"\n缩小区间（符号 '{symbol}' 概率: {prob:.4f}）:")
                self._print_state(low, high, prefix="", end="\n\n")

            # 处理区间收敛的三种情况：
            # (1) 区间完全在下半区 [0, HALF) - 首位都是0
            # (2) 区间完全在上半区 [HALF, TOP] - 首位都是1
            # (3) 区间跨越中点 [FIRST_QUARTER, THIRD_QUARTER) - 首位相反
            while True:
                # 情况1：区间在下半区
                if high < self.HALF:
                    if self.verbose > 0:
                        print(" 区间在下半区 - 首位都是0")
                        self._print_state(low, high, prefix="   ")
                    # 输出0，然后跟随之前积累的相反比特
                    yield from bit_queue.bit_plus_follow(bit=0)

                # 情况2：区间在上半区
                elif low >= self.HALF:
                    if self.verbose > 0:
                        print(" 区间在上半区 - 首位都是1")
                        self._print_state(low, high, prefix="   ")
                    # 输出1，然后跟随之前积累的相反比特
                    yield from bit_queue.bit_plus_follow(bit=1)
                    # 去掉首位1，将区间移到下半区
                    low -= self.HALF
                    high -= self.HALF

                # 情况3：区间跨越中点
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    if self.verbose > 0:
                        print(" 区间在中区 - 首位相反")
                        self._print_state(low, high, prefix="   ")
                    # 此时 low 以01开头，high 以10开头
                    # 去掉次高位，延迟输出（记录到跟随队列）
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                    bit_queue += 1  # 增加需要跟随的比特数
                else:
                    break  # 区间不够小，继续编码

                # 扩展区间（左移一位，low补0，high补1）
                low = 2 * low
                high = 2 * high + 1
                if self.verbose > 0:
                    print(" 扩展区间后:")
                    self._print_state(low, high, prefix="   ")

            # 自适应模型：更新符号频率
            cumsum.add_count(symbol, 1)

        # 检查最后一个符号是否是结束标记
        if symbol != self.EOM:
            raise ValueError(f"最后一个符号必须是 {repr(self.EOM)}，实际是 {repr(symbol)}")

        # 结束编码：输出最终的比特
        assert low < self.HALF
        bit_queue += 1
        yield from bit_queue.bit_plus_follow(int(low >= self.FIRST_QUARTER))

    def decode(self, iterable):
        """
        解码比特流，生成符号序列。

        解码是编码的逆过程：
        1. 根据当前值在区间中的位置确定符号
        2. 按照编码时的方式缩小区间
        3. 读取新的比特，扩展区间

        参数:
            iterable: 比特序列。

        生成:
            解码后的符号。
        """
        if self.verbose:
            print("------------------------ 解码中 ------------------------")

        cumsum = copy.deepcopy(self.cumsum)  # 复制累积和

        # 初始化区间和当前值
        low = 0
        value = 0
        high = self.TOP_VALUE

        # 读取前 `self.bits` 个比特作为初始值
        iterable = enumerate(itertools.chain(iter(iterable), itertools.repeat(0)), 1)
        first_bits = itertools.islice(iterable, self.bits)
        for i, input_bit in first_bits:
            value = (value << 1) + input_bit

        if self.verbose:
            print(f"读取初始 {i} 个比特: 0b{value:0{self.bits}b}")

        # 主循环
        while True:
            if self.verbose:
                print("当前状态:")
                self._print_state(low, high, value, prefix=" ", end="\n")

            # 计算当前值在区间中的相对位置
            range_ = high - low + 1
            total_count = cumsum.total_count()
            scaled_value = ((value - low + 1) * total_count - 1) / range_

            # 根据累积值查找对应的符号
            symbol = cumsum.search_ranges(scaled_value)
            yield symbol

            # 按照编码时的方式缩小区间
            symbol_low, symbol_high = cumsum.get_low_high(symbol)
            total_count = cumsum.total_count()
            high = low + int(range_ * symbol_high / total_count) - 1
            low = low + int(range_ * symbol_low / total_count)

            # 自适应模型：更新频率
            cumsum.add_count(symbol, 1)

            if self.verbose:
                print(f"输出符号 '{symbol}' 后:")
                self._print_state(low, high, value, prefix=" ", end="\n\n")

            # 如果是结束标记，停止解码
            if symbol == self.EOM:
                break

            # 处理区间收敛（与编码器同步）
            while True:
                if high < self.HALF:
                    # 区间在下半区
                    if self.verbose > 0:
                        print(" 区间在下半区")
                    pass
                elif low >= self.HALF:
                    # 区间在上半区
                    if self.verbose > 0:
                        print(" 区间在上半区")
                    value -= self.HALF
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    # 区间跨越中点
                    if self.verbose > 0:
                        print(" 区间在中区")
                    value -= self.FIRST_QUARTER
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                else:
                    break

                if self.verbose > 0:
                    self._print_state(low, high, value, prefix="   ", end="\n")

                # 扩展区间并读取新比特
                low = 2 * low
                high = 2 * high + 1
                i, input_bit = next(iterable, 1)
                value = 2 * value + input_bit
                assert low <= value <= high

                if self.verbose > 0:
                    print(f"  读取比特 {i}: {input_bit}")
                    self._print_state(low, high, value, prefix="   ", end="\n\n")


# ==================== 自定义算术编码器（针对NumPy数组）====================
class CustomArithmeticEncoder:
    """
    针对NumPy数组优化的算术编码器。

    用于压缩RDH产生的预测误差等整数数据。
    支持将数组元素作为符号进行算术编码。
    """

    def __init__(self, level_bits_len: int = 10, freq_bits_len: int = 10):
        """
        初始化编码器。

        参数:
            level_bits_len: 整数编码的比特长度。
            freq_bits_len: 频率编码的比特长度。
        """
        self.buffer_bits = 30  # 算术编码器使用的比特数
        self.freq_bits_len = freq_bits_len
        self.level_bits_len = level_bits_len

    def ndarray2strlist(self, data: ndarray) -> list:
        """
        将NumPy数组转换为字符串列表。

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

        支持有符号整数（使用补码表示）。

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
            padded_binary = binary_representation.zfill(self.freq_bits_len)  # 零填充
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
            frequencies: 预定义的符号频率（可选）。

        返回:
            编码后的比特列表。
        """
        data_str_list = self.ndarray2strlist(data)

        if frequencies is None:
            # 自适应模型：从数据中自动提取频率
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
            frequencies: 预定义的符号频率（可选）。

        返回:
            解压后的NumPy数组。
        """
        if frequencies is None:
            # 从比特流末尾提取频率信息
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


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 测试 CustomArithmeticEncoder
    print("测试 CustomArithmeticEncoder:")
    message = np.round(np.random.randn(256, 256, 3)).astype(int)
    encoder = CustomArithmeticEncoder()

    # 压缩
    bits = encoder.compress(message)
    print(f"原始数据大小: {message.size} 个整数")
    print(f"压缩后比特数: {len(bits)}")
    print(f"压缩率: {len(bits) / (message.size * 8) * 100:.2f}%")

    # 解压
    ext_data = encoder.decompress(bits).reshape(message.shape)
    print(f"解压成功: {np.array_equal(message, ext_data)}")

    # 测试固定频率模型
    # message = np.random.randint(0, 2, size=(256, 256, 3))
    # bits = encoder.compress(message, ["1", "0"])
    # ext_data = encoder.decompress(bits, ["1", "0"]).reshape(message.shape)
    # print("固定频率解压成功:", np.array_equal(message, ext_data))
