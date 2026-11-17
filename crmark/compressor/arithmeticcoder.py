#!/usr/bin/env python3
"""Arithmetic coding with Fenwick-tree cumulative frequency tables.

FenwickTree supports prefix sums and frequency updates in O(log n).
CumulativeSum maps symbols to cumulative intervals. ArithmeticEncoder
encodes and decodes symbol streams using integer interval arithmetic, and
CustomArithmeticEncoder provides array-oriented serialization.

Adaptive models initialize each symbol count to one and update matching
encoder and decoder frequency tables after each symbol.
"""

# ==================== Imports ====================
import copy
import random
import itertools

import numpy as np
from numpy import ndarray


# ==================== Fenwick tree ====================
class FenwickTree:
    """Fenwick tree supporting O(log n) prefix sums and updates.

    prefix_sum(i) covers [0, i), range_sum(l, r) covers [l, r), and add(i, k)
    increments an element. Each tree node stores a range determined by its
    least significant set bit.

    Reference: https://github.com/dstein64/fenwick
    """

    def __init__(self, frequencies):
        """Initialize the Fenwick tree from the supplied frequency list."""
        self._v = list(frequencies)

        # Initialize in O(n) by propagating child sums to their parents.
        for idx in range(1, len(self) + 1):
            parent_idx = idx + (idx & -idx)  # Parent index.
            if parent_idx <= len(self):
                self._v[parent_idx - 1] += self._v[idx - 1]

    def __len__(self):
        return len(self._v)

    def prefix_sum(self, stop):
        """Return the prefix sum over [0, stop), excluding stop."""
        if stop <= 0 or stop > len(self):
            raise IndexError("index out of range")
        _sum = 0
        while stop > 0:
            _sum += self._v[stop - 1]
            stop &= stop - 1  # Subtract the least significant set bit.
        return _sum

    def range_sum(self, start, stop):
        """Return the sum over the half-open interval [start, stop)."""
        if start < 0 or start >= len(self):
            raise IndexError("index out of range")
        if stop <= start or stop > len(self):
            raise IndexError("index out of range")
        result = self.prefix_sum(stop)
        if start > 0:
            result -= self.prefix_sum(start)
        return result

    def __getitem__(self, idx):
        """Return the element at idx, supporting negative indices."""
        if isinstance(idx, int):
            idx = idx % len(self)
            return self.range_sum(idx, idx + 1)
        else:
            raise IndexError(f"Indexing only works with integers, got {idx}")

    def frequencies(self):
        """Return all element frequencies in O(n) time."""
        _frequencies = [0] * len(self)
        for idx in range(1, len(self) + 1):
            _frequencies[idx - 1] += self._v[idx - 1]
            parent_idx = idx + (idx & -idx)
            if parent_idx <= len(self):
                _frequencies[parent_idx - 1] -= self._v[idx - 1]
        return _frequencies

    def add(self, idx, k):
        """Add k to the element at zero-based index idx."""
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")
        idx += 1
        while idx <= len(self):
            self._v[idx - 1] += k
            idx += idx & -idx  # Add the least significant set bit to update each relevant parent.

    def __setitem__(self, idx, value):
        """Set an element by applying the corresponding increment."""
        self.add(idx, value - self[idx])

    def bisect_left(self, value):
        """Return the smallest index whose cumulative sum reaches value.

        This locates the symbol containing an arithmetic decoder value. Return
        len(self) when no index satisfies the search condition.
        """
        # Reference: https://stackoverflow.com/questions/34699616/fenwick-trees-to-determine-which-interval-a-point-falls-in
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


# ==================== Cumulative frequency tables ====================
class NaiveCumulativeSum:
    """Naive O(n) cumulative frequency table for comparison with CumulativeSum."""

    def __init__(self, frequencies, update=True):
        """Initialize a frequency dictionary, optionally enabling adaptive updates."""
        self.frequencies = dict(frequencies)
        self.ranges = dict(self.ranges_from_frequencies(self.frequencies))
        self.update = update

    def get_low_high(self, symbol):
        """Return the cumulative (low, high) bounds for a symbol."""
        return self.ranges[symbol]

    def add_count(self, symbol, value):
        """Increment a symbol frequency using the naive O(n) implementation."""
        if self.update:
            self.frequencies[symbol] += value
            self.ranges = dict(self.ranges_from_frequencies(self.frequencies))

    def total_count(self):
        """Return the total symbol frequency."""
        return sum(self.frequencies.values())

    def reset(self):
        """Reset every symbol frequency to one."""
        self.frequencies = {frequency: 1 for frequency in self.frequencies}
        self.ranges = dict(self.ranges_from_frequencies(self.frequencies))

    @staticmethod
    def ranges_from_frequencies(frequencies):
        """Convert symbol frequencies to contiguous cumulative (low, high) ranges."""
        cumsum = 0
        for symbol, frequency in sorted(frequencies.items()):
            yield (symbol, (cumsum, cumsum + frequency))
            cumsum += frequency

    def search_ranges(self, value):
        """Return the symbol containing the supplied cumulative value."""
        for symbol, (low, high) in self.ranges.items():
            if low <= value < high:
                return symbol
        raise ValueError("Could not locate value in ranges.")


class CumulativeSum:
    """Fenwick-tree cumulative frequencies with O(log n) queries and updates."""

    def __init__(self, frequencies, update=True):
        """Initialize a Fenwick-tree frequency table with optional adaptive updates."""
        symbols = sorted(frequencies.keys())
        self.idx_to_symbol = dict(enumerate(symbols))  # Map indices to symbols.
        self.symbol_to_idx = {s: i for (i, s) in self.idx_to_symbol.items()}  # Map symbols to indices.
        self.fenwick_tree = FenwickTree([frequencies[s] for s in symbols])
        self.update = update

    def get_low_high(self, symbol):
        """Return the half-open cumulative interval [low, high) for a symbol."""
        idx = self.symbol_to_idx[symbol]
        if idx == 0:
            return (0, self.fenwick_tree[idx])

        sum_upto = self.fenwick_tree.prefix_sum(idx)
        return (sum_upto, sum_upto + self.fenwick_tree[idx])

    def add_count(self, symbol, value):
        """Increment a symbol frequency during adaptive encoding or decoding."""
        if self.update:
            idx = self.symbol_to_idx[symbol]
            self.fenwick_tree.add(idx, value)

    def total_count(self):
        """Return the total symbol frequency."""
        return self.fenwick_tree.prefix_sum(len(self.fenwick_tree))

    def reset(self):
        """Reset every symbol frequency to one."""
        self.fenwick_tree = FenwickTree([1] * len(self.fenwick_tree))

    def search_ranges(self, value):
        """Locate the symbol associated with a cumulative decoder value."""
        idx = self.fenwick_tree.bisect_left(value)
        return self.idx_to_symbol[idx]


# ==================== Arithmetic coder ====================
"""Arithmetic coding represents a message as an interval within [0, 1). Symbol frequencies partition the interval: frequent symbols receive larger subintervals. Integer interval renormalization supports fractional-bit coding efficiency. Construct ArithmeticEncoder with a frequency model and precision, encode a sequence ending with EOM, and decode the resulting bitstream with the same model."""


class BitQueue:
    """Track deferred complement bits during arithmetic coding.

    When an interval straddles the midpoint, defer the output until a leading
    bit is known, then emit its complement for every deferred bit.
    """

    bits_to_follow = 0  # Number of deferred complement bits.

    def __add__(self, bits):
        """Increase the number of deferred complement bits."""
        self.bits_to_follow += bits
        return self

    def bit_plus_follow(self, bit):
        """Yield bit followed by the complement for each deferred output bit."""
        yield bit
        yield from itertools.repeat(int(not bit), times=self.bits_to_follow)
        self.bits_to_follow = 0  # Reset the counter.


class ArithmeticEncoder:
    """Integer arithmetic encoder and decoder.

    The coding interval [0, TOP_VALUE] represents [0, 1). Symbol frequencies
    progressively narrow it; renormalization emits resolved bits and expands
    the remaining interval.

    References:
        Witten, Neal, and Cleary (1987), Arithmetic coding for data compression.
        https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
    """

    def __init__(self, frequencies: list, *, bits=16, verbose=0, EOM="<EOM>"):
        """Initialize the arithmetic coder.

        Args:
            frequencies: A fixed frequency dictionary or an adaptive symbol list.
            bits: Integer interval precision; defaults to 16.
            verbose: Diagnostic verbosity; zero disables diagnostic output.
            EOM: End-of-message symbol.
        """
        self.EOM = EOM
        self.frequencies = frequencies.copy()
        self.bits = bits
        self.verbose = verbose

        # Build cumulative ranges from frequencies.
        if isinstance(frequencies, dict):
            assert all(isinstance(freq, int) for freq in self.frequencies.values())
            assert self.EOM in self.frequencies.keys()
            self.cumsum = CumulativeSum(dict(frequencies), update=False)
        elif isinstance(frequencies, (list, set)):
            assert self.EOM in self.frequencies
            frequencies = {symbol: 1 for symbol in self.frequencies}
            self.cumsum = CumulativeSum(frequencies, update=True)

        # Interval boundaries, illustrated with 4-bit precision:
        self.TOP_VALUE = (1 << self.bits) - 1      # 0b1111 = 15, the maximum value.
        self.FIRST_QUARTER = (self.TOP_VALUE >> 2) + 1  # 0b0100 = 4, the first quarter.
        self.HALF = self.FIRST_QUARTER * 2         # 0b1000 = 8, the midpoint.
        self.THIRD_QUARTER = self.FIRST_QUARTER * 3     # 0b1100 = 12, the third quarter.

        # Check that the interval precision is sufficient (formula on page 533 of the paper).
        if self.cumsum.total_count() > int((self.TOP_VALUE + 1) / 4) + 1:
            msg = "Insufficient precision to encode low-probability symbols."
            msg += "\nIncrease the bits parameter."
            raise Exception(msg)

        if self.verbose > 0:
            print("Initialization parameters:")
            print(f" bits          = {self.bits}")
            print(f" TOP_VALUE     = 0b{self.TOP_VALUE:0{self.bits}b} ({self.TOP_VALUE})")
            print(f" THIRD_QUARTER = 0b{self.THIRD_QUARTER:0{self.bits}b} ({self.THIRD_QUARTER})")
            print(f" HALF          = 0b{self.HALF:0{self.bits}b} ({self.HALF})")
            print(f" FIRST_QUARTER = 0b{self.FIRST_QUARTER:0{self.bits}b} ({self.FIRST_QUARTER})")

    def _print_state(self, low, high, value=None, *, prefix=" ", end="\n"):
        """Print the coder state for debugging."""
        range_ = high - low + 1
        print(prefix + f"High value: 0b{high:0{self.bits}b} ({high})")
        if value is not None:
            print(prefix + f"Value:      0b{value:0{self.bits}b} ({value})")
        print(prefix + f"Low value:  0b{low:0{self.bits}b} ({low})")
        print(prefix + f"Range: [{low}, {high + 1}) Width: {range_}", end=end)

    def encode(self, iterable):
        """Yield encoded bits for an iterable of symbols.

        Narrow the coding interval using symbol frequencies, emit resolved leading
        bits, and renormalize the interval. Encoder and decoder arithmetic must use
        the same operation order.
        """
        if self.verbose:
            print("------------------------ Encoding ------------------------")

        # Copy cumulative frequencies so encoder and decoder states remain independent.
        cumsum = copy.deepcopy(self.cumsum)

        bit_queue = BitQueue()  # Queue of deferred complement bits.

        # Initialize the coding interval [low, high].
        low = 0
        high = self.TOP_VALUE

        # Process each symbol.
        for i, symbol in enumerate(iterable, 1):
            if self.verbose > 0:
                print(f"\nProcessing symbol {i}: {repr(symbol)}")
                print("-" * 32)

            # Current interval width.
            range_ = high - low + 1

            # Check algorithm invariants.
            if range_ < cumsum.total_count():
                msg = "Insufficient precision to encode low-probability symbols."
                msg += "\nIncrease the bits parameter."
                raise Exception(msg)
            assert 0 <= low <= high <= self.TOP_VALUE
            assert low < self.HALF <= high
            assert high - low > self.FIRST_QUARTER

            if self.verbose > 0:
                self._print_state(low, high, prefix="")

            # Get the cumulative frequency range for this symbol.
            symbol_low, symbol_high = cumsum.get_low_high(symbol)

            # Narrow the interval according to the symbol probability.
            # The operation order must exactly match the decoder.
            total_count = cumsum.total_count()
            high = low + int(range_ * symbol_high / total_count) - 1
            low = low + int(range_ * symbol_low / total_count)

            if self.verbose > 0:
                prob = (symbol_high - symbol_low) / total_count
                print(f"\nNarrowed interval (symbol '{symbol}', probability {prob:.4f}):")
                self._print_state(low, high, prefix="", end="\n\n")

            # Handle the three interval-renormalization cases:
            # (1) Entirely in [0, HALF): all values have leading bit zero.
            # (2) Entirely in [HALF, TOP]: all values have leading bit one.
            # (3) Straddling the midpoint within [FIRST_QUARTER, THIRD_QUARTER).
            while True:
                # Case 1: the interval lies in the lower half.
                if high < self.HALF:
                    if self.verbose > 0:
                        print(" Interval in the lower half: leading bit zero")
                        self._print_state(low, high, prefix="   ")
                    # Emit zero followed by the deferred complement bits.
                    yield from bit_queue.bit_plus_follow(bit=0)

                # Case 2: the interval lies in the upper half.
                elif low >= self.HALF:
                    if self.verbose > 0:
                        print(" Interval in the upper half: leading bit one")
                        self._print_state(low, high, prefix="   ")
                    # Emit one followed by the deferred complement bits.
                    yield from bit_queue.bit_plus_follow(bit=1)
                    # Remove the leading one and shift the interval into the lower half.
                    low -= self.HALF
                    high -= self.HALF

                # Case 3: the interval straddles the midpoint.
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    if self.verbose > 0:
                        print(" Interval straddles the midpoint: leading bits differ")
                        self._print_state(low, high, prefix="   ")
                    # Here low starts with 01 and high starts with 10.
                    # Remove the second most significant bit and defer its output.
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                    bit_queue += 1  # Increment the deferred-bit count.
                else:
                    break  # The interval is not narrow enough; continue encoding.

                # Expand by shifting left, appending zero to low and one to high.
                low = 2 * low
                high = 2 * high + 1
                if self.verbose > 0:
                    print(" After interval expansion:")
                    self._print_state(low, high, prefix="   ")

            # Update symbol frequencies in the adaptive model.
            cumsum.add_count(symbol, 1)

        # Verify that the last symbol is the end-of-message marker.
        if symbol != self.EOM:
            raise ValueError(f"The final symbol must be {repr(self.EOM)}; got {repr(symbol)}")

        # Finish encoding by emitting the final bits.
        assert low < self.HALF
        bit_queue += 1
        yield from bit_queue.bit_plus_follow(int(low >= self.FIRST_QUARTER))

    def decode(self, iterable):
        """Yield decoded symbols from an iterable of bits.

        Locate each symbol using the current interval value, narrow the interval,
        and read new bits while renormalizing in sync with the encoder.
        """
        if self.verbose:
            print("------------------------ Decoding ------------------------")

        cumsum = copy.deepcopy(self.cumsum)  # Copy cumulative frequencies.

        # Initialize the interval and current value.
        low = 0
        value = 0
        high = self.TOP_VALUE

        # Read the first self.bits bits as the initial decoder value.
        iterable = enumerate(itertools.chain(iter(iterable), itertools.repeat(0)), 1)
        first_bits = itertools.islice(iterable, self.bits)
        for i, input_bit in first_bits:
            value = (value << 1) + input_bit

        if self.verbose:
            print(f"Read the initial {i} bits: 0b{value:0{self.bits}b}")

        # Main decoding loop.
        while True:
            if self.verbose:
                print("Current state:")
                self._print_state(low, high, value, prefix=" ", end="\n")

            # Compute the current value's relative location in the interval.
            range_ = high - low + 1
            total_count = cumsum.total_count()
            scaled_value = ((value - low + 1) * total_count - 1) / range_

            # Find the symbol containing this cumulative value.
            symbol = cumsum.search_ranges(scaled_value)
            yield symbol

            # Narrow the interval using the same arithmetic as the encoder.
            symbol_low, symbol_high = cumsum.get_low_high(symbol)
            total_count = cumsum.total_count()
            high = low + int(range_ * symbol_high / total_count) - 1
            low = low + int(range_ * symbol_low / total_count)

            # Update adaptive model frequencies.
            cumsum.add_count(symbol, 1)

            if self.verbose:
                print(f"After emitting symbol '{symbol}':")
                self._print_state(low, high, value, prefix=" ", end="\n\n")

            # Stop at the end-of-message marker.
            if symbol == self.EOM:
                break

            # Renormalize in sync with the encoder.
            while True:
                if high < self.HALF:
                    # The interval lies in the lower half.
                    if self.verbose > 0:
                        print(" Interval in the lower half")
                    pass
                elif low >= self.HALF:
                    # The interval lies in the upper half.
                    if self.verbose > 0:
                        print(" Interval in the upper half")
                    value -= self.HALF
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    # The interval straddles the midpoint.
                    if self.verbose > 0:
                        print(" Interval straddles the midpoint")
                    value -= self.FIRST_QUARTER
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                else:
                    break

                if self.verbose > 0:
                    self._print_state(low, high, value, prefix="   ", end="\n")

                # Expand the interval and read another bit.
                low = 2 * low
                high = 2 * high + 1
                i, input_bit = next(iterable, 1)
                value = 2 * value + input_bit
                assert low <= value <= high

                if self.verbose > 0:
                    print(f"  Read bit {i}: {input_bit}")
                    self._print_state(low, high, value, prefix="   ", end="\n\n")


# ==================== Array-oriented arithmetic coder ====================
class CustomArithmeticEncoder:
    """Array-oriented arithmetic coding for integer data and prediction errors."""

    def __init__(self, level_bits_len: int = 10, freq_bits_len: int = 10):
        """Configure the integer header width and signed data value width in bits."""
        self.buffer_bits = 30  # Arithmetic coder precision in bits.
        self.freq_bits_len = freq_bits_len
        self.level_bits_len = level_bits_len

    def ndarray2strlist(self, data: ndarray) -> list:
        """Flatten a NumPy array and convert its values to strings."""
        data_list = data.flatten().tolist()
        data_str_list = [str(data_item) for data_item in data_list]
        return data_str_list

    def strlist2ndarray(self, data_str_list: list) -> list:
        """Convert a list of numeric strings to a list of integers."""
        data_list = [int(data_str_item) for data_str_item in data_str_list]
        return data_list

    def datastr2bits(self, data_str_list):
        """Encode numeric strings as fixed-width bits using two's complement."""
        encoded_bits = []
        for item in data_str_list:
            num = int(item)
            if num < 0:
                num = (1 << self.freq_bits_len) + num  # Convert negative values to two's complement.
            binary_representation = bin(num)[2:]  # Convert to a binary string.
            padded_binary = binary_representation.zfill(self.freq_bits_len)  # Pad with leading zeros.
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
        """Compress a NumPy array using supplied frequencies or an adaptive model."""
        data_str_list = self.ndarray2strlist(data)

        if frequencies is None:
            # Build the adaptive symbol model from the input data.
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
        """Decompress bits using supplied frequencies or a serialized frequency table."""
        if frequencies is None:
            # Extract the serialized frequency information from the bitstream suffix.
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


# ==================== Standalone entry point ====================
if __name__ == "__main__":
    # Exercise CustomArithmeticEncoder.
    print("Testing CustomArithmeticEncoder:")
    message = np.round(np.random.randn(256, 256, 3)).astype(int)
    encoder = CustomArithmeticEncoder()

    # Compress the data.
    bits = encoder.compress(message)
    print(f"Original data size: {message.size} integers")
    print(f"Compressed bit count: {len(bits)}")
    print(f"Compression ratio: {len(bits) / (message.size * 8) * 100:.2f}%")

    # Decompress the data.
    ext_data = encoder.decompress(bits).reshape(message.shape)
    print(f"Decompression successful: {np.array_equal(message, ext_data)}")

    # Exercise the fixed-frequency model.
    # message = np.random.randint(0, 2, size=(256, 256, 3))
    # bits = encoder.compress(message, ["1", "0"])
    # ext_data = encoder.decompress(bits, ["1", "0"]).reshape(message.shape)
    # print("Fixed-frequency decompression successful:", np.array_equal(message, ext_data))
