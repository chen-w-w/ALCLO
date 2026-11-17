# ==================== Imports ====================
import math
import torch
import random
import numpy as np
from PIL import Image
from torch import Tensor
from typing import Tuple, List
from numpy import ndarray
from .arithmeticcoder import CustomArithmeticEncoder  # Array-oriented arithmetic encoder.


# ==================== RDH core ====================
class RDH:
    """Reversible data hiding through prediction-error expansion.

    Embed watermark bits while retaining the auxiliary information needed for
    lossless reconstruction of the original image.
    """

    def __init__(self, img_size: Tuple[int, int, int], bit_plane: int = 3, level_bits_len: int = 10, freq_bits_len: int = 10):
        """Configure reversible data hiding and auxiliary information encoding.

        Args:
            img_size: Image dimensions (height, width, channels).
            bit_plane: Number of LSB planes reserved for auxiliary information.
            level_bits_len: Arithmetic coding header width in bits.
            freq_bits_len: Arithmetic coding value width in bits.
        """
        self.grayscale_bit = 8  # Bits per grayscale pixel.
        self.img_size = img_size
        self.bit_plane = bit_plane  # Number of LSB planes.
        # Storage capacity: width x channels x bit planes.
        self.storage_len = self.img_size[1] * self.img_size[2] * self.bit_plane

        # Compute coordinate encoding widths for each dimension.
        self.c_len = math.ceil(np.log2(img_size[2])) + 1  # Channel index width in bits.
        self.w_len = math.ceil(np.log2(img_size[1])) + 1  # Column index width in bits.
        self.h_len = math.ceil(np.log2(img_size[0])) + 1  # Row index width in bits.
        self.freq_len = self.c_len + self.w_len + self.h_len  # Total coordinate width.

        # Initialize arithmetic coding for auxiliary information.
        self.customAC = CustomArithmeticEncoder(level_bits_len, freq_bits_len)

    def set_mask(self, split_height: int):
        """Create checkerboard O/X pixel masks for the specified split height."""
        # Checkerboard pattern: (h + w) % 2 alternates between zero and one.
        base_mask = np.fromfunction(
            lambda h, w: (h + w) % 2,
            (self.img_size[0] - split_height, self.img_size[1]),
            dtype=int
        )
        # Extend the pattern to all channels.
        mask = np.repeat(base_mask[:, :, np.newaxis], self.img_size[2], axis=2)
        self.mask_o = mask      # O-group mask: positions equal to one.
        self.mask_x = 1 - mask  # X-group mask: positions equal to zero.

    def prediect(self, cover_img: ndarray, h: int, w: int, c: int):
        """Predict a pixel from its four neighbors and round to an integer."""
        # Use a 3 x 3 mask with the four direct neighbors set to one and the center to zero.
        mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        # Extract 3 x 3 neighborhoods.
        block = cover_img[h - 1: h + 2, w - 1: w + 2, c]
        # Predict from the neighborhood mean.
        predict_value = np.round(np.sum(mask * block) / 4.)
        return predict_value

    def predicting_error(self, cover_img: ndarray):
        """Return prediction errors and predicted values for both checkerboard groups.

        The result is (pe_o, pe_x, pv_o, pv_x); errors equal actual minus predicted
        pixel values. Boundary pixels are excluded from prediction.
        """
        H, W, C = cover_img.shape
        pv = np.copy(cover_img)

        # Predict all interior pixels, leaving boundary pixels untouched.
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    pv[h, w, c] = self.prediect(cover_img, h, w, c)

        # Separate predictions for the O and X groups.
        pv_o = self.mask_o * pv
        pv_x = self.mask_x * pv

        # Compute prediction errors.
        pe_o = cover_img * self.mask_o - pv_o
        pe_x = cover_img * self.mask_x - pv_x

        return pe_o, pe_x, pv_o, pv_x

    def split_img(self, cover_img: ndarray, height_end: int):
        """Split the cover into a payload region and a top auxiliary-map region.

        Return (img4embed, img4locmap), with height_end specifying the split.
        """
        img4locmap = cover_img[:height_end, :, :]   # Top region: location-map storage.
        img4embed = cover_img[height_end:, :, :]    # Bottom region: watermark payload.
        return np.float32(img4embed), np.float32(img4locmap)

    def merge_img(self, watermarked_img4embed: ndarray, marked_img4locmap: ndarray, split_height: int):
        """Merge the watermarked payload and auxiliary-map regions into one image."""
        watermarked_img = np.zeros(shape=self.img_size)
        watermarked_img[:split_height, :, :] = marked_img4locmap      # Top region: location map.
        watermarked_img[split_height:, :, :] = watermarked_img4embed  # Bottom region: embedding area.
        return watermarked_img

    def get_top_two_frequent_values(self, pe: ndarray, set_mask: ndarray):
        """Return the two most frequent masked prediction errors and their counts.

        The result is (first_most, first_count, second_most, second_count). These
        histogram bins determine the embedding capacity.
        """
        # Extract the interior region without boundary pixels.
        pe_central = pe[1:-1, 1:-1, :]
        mask_central = set_mask[1:-1, 1:-1, :]

        # Flatten and retain only masked elements.
        pe_flat = pe_central[mask_central == 1]

        # Count occurrences of each value.
        unique, counts = np.unique(pe_flat, return_counts=True)

        # Sort by decreasing frequency.
        sorted_indices = np.argsort(counts)[::-1]

        # Select the two most frequent values.
        top_two_indices = sorted_indices[:2]
        first_most = unique[top_two_indices[0]] if len(top_two_indices) > 0 else None
        first_count = counts[top_two_indices[0]] if len(top_two_indices) > 0 else None
        second_most = unique[top_two_indices[1]] if len(top_two_indices) > 1 else None
        second_count = counts[top_two_indices[1]] if len(top_two_indices) > 1 else None

        return first_most, first_count, second_most, second_count

    def shift_and_embed(self, pe: ndarray, set_mask: ndarray, wm_list: list):
        """Shift prediction errors and embed bits by expanding two histogram bins.

        Bits equal to zero retain the selected bin value; bits equal to one expand
        it outward. Return the modified errors, minimum and maximum bins, capacity,
        stopping coordinate, and any watermark bits that remain unembedded.
        """
        # ==================== Step 1: select the two most frequent bins ====================
        first_most, first_count, second_most, second_count = self.get_top_two_frequent_values(pe, set_mask)
        capacity = first_count + second_count  # Capacity is the sum of the two bin frequencies.

        # ==================== Step 2: define embedding thresholds ====================
        min_value = int(min(first_most, second_most))  # Lower bin value.
        max_value = int(max(first_most, second_most))  # Upper bin value.

        # ==================== Step 3: initialize state ====================
        shifted_pe = pe.copy()
        H, W, C = pe.shape
        wm_index = 0  # Watermark bit index.
        stopcoordinate = (0, 0, 0)  # Default stopping coordinate.

        # ==================== Step 4: shift errors and embed bits ====================
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if set_mask[h, w, c] == 1 and wm_index < len(wm_list):
                        stopcoordinate = (h, w, c)  # Record the current position.

                        if pe[h, w, c] < min_value:
                            # Shift errors below min_value left by one.
                            shifted_pe[h, w, c] = pe[h, w, c] - 1.
                        elif pe[h, w, c] > max_value:
                            # Shift errors above max_value right by one.
                            shifted_pe[h, w, c] = pe[h, w, c] + 1.
                        elif pe[h, w, c] == max_value:
                            # At max_value, zero keeps the bin and one expands it to max_value + 1.
                            shifted_pe[h, w, c] += wm_list[wm_index]
                            wm_index += 1
                        elif pe[h, w, c] == min_value:
                            # At min_value, zero keeps the bin and one expands it to min_value - 1.
                            shifted_pe[h, w, c] -= wm_list[wm_index]
                            wm_index += 1

                        # Return early once all watermark bits have been embedded.
                        if wm_index == len(wm_list):
                            return shifted_pe, min_value, max_value, capacity, stopcoordinate, []

        # ==================== Step 5: return results ====================
        return shifted_pe, min_value, max_value, capacity, stopcoordinate, wm_list[wm_index:]

    def extract_and_shift(self, embedded_pe: ndarray, min_value: int, max_value: int, set_mask: ndarray,
                          stopcoordinate: tuple):
        """Extract watermark bits and reverse prediction-error shifting.

        Expanded bins encode one, unchanged bins encode zero. Restore shifted
        values up to stopcoordinate and return (restored_pe, extracted_wm_list).
        """
        H, W, C = embedded_pe.shape
        restored_pe = embedded_pe.copy()
        extracted_wm_list = []
        stop_reached = False

        # ==================== Step 1: extract watermark bits ====================
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
                        # A value of max_value + 1 encodes bit one.
                        if embedded_pe[h, w, c] == max_value + 1:
                            extracted_wm_list.append(1)
                            restored_pe[h, w, c] = max_value  # Restore the original bin value.
                        # A value of max_value encodes bit zero.
                        elif embedded_pe[h, w, c] == max_value:
                            extracted_wm_list.append(0)
                        # A value of min_value - 1 encodes bit one.
                        elif embedded_pe[h, w, c] == min_value - 1:
                            extracted_wm_list.append(1)
                            restored_pe[h, w, c] = min_value  # Restore the original bin value.
                        # A value of min_value encodes bit zero.
                        elif embedded_pe[h, w, c] == min_value:
                            extracted_wm_list.append(0)

                        # Stop extraction at the saved stopping coordinate.
                        if (h, w, c) == stopcoordinate:
                            stop_reached = True
                            break

        # ==================== Step 2: restore shifted errors ====================
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if set_mask[h, w, c] == 1:
                        # Undo a left shift by adding one.
                        if restored_pe[h, w, c] < min_value:
                            restored_pe[h, w, c] += 1
                        # Undo a right shift by subtracting one.
                        elif restored_pe[h, w, c] > max_value:
                            restored_pe[h, w, c] -= 1

                    # Return early at the stopping coordinate.
                    if (h, w, c) == stopcoordinate:
                        return restored_pe, extracted_wm_list

        return restored_pe, extracted_wm_list

    def compute_stego_img(self, pe_o: ndarray, pe_x: ndarray, pv_o: ndarray, pv_x: ndarray):
        """Reconstruct the stego image from masked prediction errors and predictions."""
        # Reconstruct each checkerboard group and combine them using their masks.
        stego_img = (pe_o + pv_o) * self.mask_o + (pe_x + pv_x) * self.mask_x
        return stego_img

    def compute_overflow_map(self, stego_img: ndarray, mask: ndarray, stopcoordinate: Tuple) -> list:
        """Record overflow or underflow positions before the stopping coordinate.

        Return a location-map list: one marks a pixel outside [0, 255], zero marks
        a pixel that remains within the valid range.
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
                        # Detect overflow above 255 and underflow below zero.
                        if pixel_value > 255 or pixel_value < 0:
                            location_map.append(1)
                        else:
                            location_map.append(0)

                    # Return early at the stopping coordinate.
                    if (h, w, c) == stopcoordinate:
                        return location_map
        return location_map

    def recovery_overflow_stego_image(self, clipped_stego_img: ndarray, mask: ndarray, overflow_map: list) -> ndarray:
        """Restore clipped pixels from an overflow location map.

        Recover -1 or 256 at marked locations and return a float32 stego image.
        """
        H, W, C = clipped_stego_img.shape
        recovered_img = clipped_stego_img.astype(np.float32)
        location_idx = 0

        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if mask[h, w, c] == 1 and location_idx < len(overflow_map):
                        if overflow_map[location_idx] == 1:
                            # Restore an underflow pixel from clipped zero to its original value of -1.
                            if clipped_stego_img[h, w, c] == 0:
                                recovered_img[h, w, c] = -1.0
                            # Restore an overflow pixel from clipped 255 to its original value of 256.
                            elif clipped_stego_img[h, w, c] == 255:
                                recovered_img[h, w, c] = 256.0
                        location_idx += 1
                        if location_idx == len(overflow_map):
                            return recovered_img
        return recovered_img

    def encodeIntegerbyGivenLength(self, n: int, length: int) -> list:
        """Encode a signed integer as a fixed-width two's-complement bit list."""
        # Represent a negative integer using two's complement.
        if n < 0:
            n = (1 << length) + n

        # Get the binary representation without the 0b prefix.
        binary_representation = bin(n)[2:]

        # Pad shorter representations with leading zeros.
        if len(binary_representation) < length:
            binary_representation = binary_representation.zfill(length)
        elif len(binary_representation) > length:
            raise ValueError("Value is outside the range representable by the specified bit length")

        return [int(bit) for bit in binary_representation]

    def decodeIntegerbyGivenBits(self, bits: list) -> int:
        """Decode a signed two's-complement bit list into an integer."""
        bit_string = ''.join(map(str, bits))
        value = int(bit_string, 2)

        # If the sign bit is one, recover the negative two's-complement value.
        if bits[0] == 1:
            value -= (1 << len(bits))

        return value

    def encode_auxiliary_information(self, overflow_map_o, min_v_o, max_v_o, stop_coor_o,
                                     overflow_map_x, min_v_x, max_v_x, stop_coor_x):
        """Serialize information needed to reverse the embedding process.

        Encode compressed overflow maps, minimum/maximum histogram bins, and
        stopping coordinates for both checkerboard groups into a bit list.
        """
        # ==================== Encode overflow-map lengths ====================
        l_mask_o = int(self.c_len + self.h_len + self.w_len)
        l_mask_o_bits = [int(b) for b in format(len(overflow_map_o), f'0{l_mask_o}b')]

        # Combine the O-group and X-group overflow maps.
        overflow_map_bitstream = l_mask_o_bits + overflow_map_o + overflow_map_x

        # Compress location maps with arithmetic coding.
        compressed_overflow_map_bitstream = self.customAC.compress(np.asarray(overflow_map_bitstream), ["1", "0"])

        # ==================== Encode minimum and maximum bins ====================
        min_v_o_bits = self.encodeIntegerbyGivenLength(min_v_o, self.grayscale_bit)
        max_v_o_bits = self.encodeIntegerbyGivenLength(max_v_o, self.grayscale_bit)
        min_v_x_bits = self.encodeIntegerbyGivenLength(min_v_x, self.grayscale_bit)
        max_v_x_bits = self.encodeIntegerbyGivenLength(max_v_x, self.grayscale_bit)

        # ==================== Encode stopping coordinates ====================
        h_bits = self.encodeIntegerbyGivenLength(stop_coor_o[0], self.h_len)
        w_bits = self.encodeIntegerbyGivenLength(stop_coor_o[1], self.w_len)
        c_bits = self.encodeIntegerbyGivenLength(stop_coor_o[2], self.c_len)
        coor_o_bits = h_bits + w_bits + c_bits

        h_bits = self.encodeIntegerbyGivenLength(stop_coor_x[0], self.h_len)
        w_bits = self.encodeIntegerbyGivenLength(stop_coor_x[1], self.w_len)
        c_bits = self.encodeIntegerbyGivenLength(stop_coor_x[2], self.c_len)
        coor_x_bits = h_bits + w_bits + c_bits

        # ==================== Combine auxiliary information ====================
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
                    # Convert pixel values to Python int to avoid NumPy uint8 overflow.

                    pixel_value = int(image[h, w, c])
                    # Clear low bits using & (255 - mask) instead of & ~mask.
                    # Python's unbounded two's-complement ~mask is negative and can cause overflow errors.
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


# ==================== Tensor-compatible RDH wrapper ====================
class CustomRDH:
    """PyTorch tensor wrapper for the NumPy-based reversible data hiding core."""

    def __init__(self, img_size: Tuple[int, int, int], device: str = "cpu"):
        """Initialize the tensor wrapper for (H, W, C) images on the selected device."""
        self.device = device
        self.rdh = RDH(img_size)

    def embed(self, cover_img: Tensor, watermark_list: list):
        """Embed a bit list into a cover tensor shaped (1, C, H, W).

        Return (issuccessful, rdh_stego_img), with the stego image as a NumPy array.
        """
        N, C, H, W = cover_img.shape
        assert N == 1, "Only single-image processing is supported"

        # Convert the PyTorch tensor to a NumPy array shaped (H, W, C).
        cover_img_numpy = cover_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # Embed bits using the RDH implementation.
        issuccessful, rdh_stego_img = self.rdh.embed(cover_img_numpy, watermark_list)

        return issuccessful, np.uint8(rdh_stego_img)

    def extract(self, rdh_stego_img: ndarray):
        """Extract a payload and reconstruct the cover from a NumPy stego image.

        Return (issuccessful, rec_stego_img_tensor, wm_list), placing the recovered
        tensor on the configured device.
        """
        # Extract bits using the RDH implementation.
        issuccessful, rec_stego_img, wm_list = self.rdh.extract(rdh_stego_img)

        # Convert the NumPy array back to a PyTorch tensor.
        if rdh_stego_img.ndim == 3:  # RGB image.
            rec_stego_img_tensor = torch.as_tensor(rec_stego_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        else:  # Grayscale image.
            rec_stego_img_tensor = torch.as_tensor(rec_stego_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return issuccessful, rec_stego_img_tensor.to(self.device), wm_list


# ==================== Utilities ====================
def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Return PSNR in dB for two images; identical images produce infinity."""
    # Verify that both images have the same dimensions.
    assert original.shape == compressed.shape, "Input images must have identical dimensions"

    # Compute mean squared error.
    mse = np.mean((original - compressed) ** 2)

    # Identical images have zero MSE and infinite PSNR.
    if mse == 0:
        return float('inf')

    # Maximum pixel value for an 8-bit image.
    max_value = 255.0

    # Compute PSNR.
    psnr = 10 * np.log10(max_value ** 2 / mse)
    return psnr


def test4grayimage(img_path):
    """Exercise grayscale embedding and extraction and verify reversibility."""
    # Load a grayscale image.
    cover_img = Image.open(img_path).convert("L")
    cover_img = np.float32(cover_img)
    cover_img_np = cover_img.reshape(cover_img.shape[0], cover_img.shape[1], 1)

    # Initialize RDH.
    rdh = RDH(img_size=cover_img_np.shape, height_end=10)

    # Generate random watermark bits.
    watermark_length = 50000
    watermark_list = [random.randint(0, 1) for _ in range(watermark_length)]

    # Embed the watermark.
    issuccessful, stego_img = rdh.embed(cover_img_np, watermark_list)

    # Save the stego image.
    Image.fromarray(np.uint8(stego_img)).save('images/stego_img_gray.png')
    reload_stego_img = np.float32(Image.open('images/stego_img_gray.png'))

    # Extract the watermark.
    recovered_cover_img, wm_list = rdh.extract(reload_stego_img)

    # Verify reversibility.
    print("Cover image recovered exactly:", np.array_equal(recovered_cover_img, cover_img_np))
    print("Watermark recovered exactly:", np.array_equal(watermark_list, wm_list))
    print(f"PSNR: {calculate_psnr(cover_img, stego_img):.2f} dB")


def test4rgbimage(img_path):
    """Exercise RGB embedding and extraction with an additional pixel attack."""
    # Load an RGB image.
    cover_img = Image.open(img_path)
    cover_img_np = np.float32(cover_img)

    # Initialize RDH.
    rdh = RDH(img_size=cover_img_np.shape)

    # Generate random watermark bits.
    watermark_length = 10000
    watermark_list = [random.randint(0, 1) for _ in range(watermark_length)]

    # Embed the watermark.
    issuccessful, stego_img = rdh.embed(cover_img_np, watermark_list)

    # Save the stego image.
    Image.fromarray(np.uint8(stego_img)).save('stego_img_rgb.tif')
    reload_stego_img = np.float32(Image.open('stego_img_rgb.tif'))

    # Probe robustness to a small attack by modifying one pixel.
    reload_stego_img_attacked = reload_stego_img.copy()
    reload_stego_img_attacked[100, 12, 0] += 1

    # Extract from both the original and attacked stego images.
    issuccessful, recovered_cover_img_attacked, wm_list_attacked = rdh.extract(reload_stego_img_attacked)
    issuccessful, recovered_cover_img, wm_list = rdh.extract(reload_stego_img)

    # Verify the results.
    print("Recovered images match before and after the attack:", np.array_equal(recovered_cover_img_attacked, recovered_cover_img))
    print("Watermarks match before and after the attack:", np.array_equal(wm_list_attacked, wm_list))
    print("Cover image recovered exactly:", np.array_equal(recovered_cover_img, cover_img_np))
    print("Watermark recovered exactly:", np.array_equal(watermark_list, wm_list))
    print(f"PSNR: {calculate_psnr(cover_img_np, stego_img):.2f} dB")


# ==================== Standalone entry point ====================
if __name__ == "__main__":
    # Run the RGB image example.
    test4rgbimage('images/BaboonRGB.tif')
    # Run the grayscale image example.
    # test4grayimage('compressor/images/BaboonRGB.tif')
