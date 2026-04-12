#!/usr/bin/env python3
"""Reference checks for Q4_0 unpack + integer Q16 dequant semantics."""

from __future__ import annotations

import random
import struct

Q4_0_VALUES_PER_BLOCK = 32
Q4_0_PACKED_BYTES = 16

Q4_0_OK = 0
Q4_0_ERR_NULL_PTR = 1
Q4_0_ERR_BAD_DST_LEN = 2


def round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    round_bias = 1 << (shift - 1)
    return (value + round_bias) >> shift


def f16_to_q16(fp16_bits: int) -> int:
    sign_bit = (fp16_bits >> 15) & 1
    exponent_bits = (fp16_bits >> 10) & 0x1F
    fraction_bits = fp16_bits & 0x03FF

    if exponent_bits == 0:
        if fraction_bits == 0:
            return 0
        magnitude_q16 = round_shift_right_unsigned(fraction_bits, 8)
        return -magnitude_q16 if sign_bit else magnitude_q16

    if exponent_bits == 0x1F:
        return -(0x3FFFFFFFFFFFFFFF) if sign_bit else 0x3FFFFFFFFFFFFFFF

    mantissa = 1024 + fraction_bits
    shift_amount = exponent_bits - 9

    if shift_amount >= 0:
        magnitude_q16 = mantissa << shift_amount
    else:
        magnitude_q16 = round_shift_right_unsigned(mantissa, -shift_amount)

    return -magnitude_q16 if sign_bit else magnitude_q16


def nibble_to_signed(packed: int, upper_nibble: bool) -> int:
    if upper_nibble:
        q_unsigned = (packed >> 4) & 0x0F
    else:
        q_unsigned = packed & 0x0F
    return q_unsigned - 8


def unpack_block_signed(qs: bytes):
    if len(qs) != Q4_0_PACKED_BYTES:
        return Q4_0_ERR_BAD_DST_LEN, []

    out = [0] * Q4_0_VALUES_PER_BLOCK
    for byte_index, packed in enumerate(qs):
        out_index = byte_index << 1
        out[out_index] = nibble_to_signed(packed, False)
        out[out_index + 1] = nibble_to_signed(packed, True)
    return Q4_0_OK, out


def dequantize_block_q16(scale_fp16: int, qs: bytes):
    if len(qs) != Q4_0_PACKED_BYTES:
        return Q4_0_ERR_BAD_DST_LEN, []

    out = [0] * Q4_0_VALUES_PER_BLOCK
    scale_q16 = f16_to_q16(scale_fp16)

    for byte_index, packed in enumerate(qs):
        out_index = byte_index << 1

        q_signed = nibble_to_signed(packed, False)
        out[out_index] = scale_q16 * q_signed

        q_signed = nibble_to_signed(packed, True)
        out[out_index + 1] = scale_q16 * q_signed

    return Q4_0_OK, out


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def test_unpack_known_pattern() -> None:
    qs = bytes(((2 * i) & 0x0F) | (((2 * i + 1) & 0x0F) << 4) for i in range(Q4_0_PACKED_BYTES))
    err, out = unpack_block_signed(qs)
    assert err == Q4_0_OK
    assert out == [i - 8 for i in range(16)] * 2


def test_dequant_known_scale() -> None:
    scale_fp16 = half_bits(0.5)
    qs = bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2)

    err, out_q16 = dequantize_block_q16(scale_fp16, qs)
    assert err == Q4_0_OK

    expected_signed = []
    for packed in qs:
        expected_signed.append((packed & 0x0F) - 8)
        expected_signed.append(((packed >> 4) & 0x0F) - 8)

    expected_q16 = [round(0.5 * q * (1 << 16)) for q in expected_signed]
    assert out_q16 == expected_q16


def test_fp16_to_q16_matches_float_rounding_for_finite_values() -> None:
    finite_values = [
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        1.5,
        -3.25,
        0.125,
        -0.03125,
    ]

    for value in finite_values:
        bits = half_bits(value)
        got = f16_to_q16(bits)
        expected = round(value * (1 << 16))
        assert got == expected


def test_random_blocks_match_reference_float_path() -> None:
    rng = random.Random(1337)
    for _ in range(200):
        # Keep finite, normal scales in a stable range.
        scale = rng.uniform(-4.0, 4.0)
        scale_fp16 = half_bits(scale)
        scale_q16 = f16_to_q16(scale_fp16)

        qs = bytes(rng.randrange(256) for _ in range(Q4_0_PACKED_BYTES))
        err, out_q16 = dequantize_block_q16(scale_fp16, qs)
        assert err == Q4_0_OK

        scale_from_half = struct.unpack("<e", struct.pack("<H", scale_fp16))[0]
        expected_q16 = []
        expected_float_q16 = []
        for packed in qs:
            q_low = (packed & 0x0F) - 8
            q_high = ((packed >> 4) & 0x0F) - 8
            expected_q16.append(scale_q16 * q_low)
            expected_q16.append(scale_q16 * q_high)
            expected_float_q16.append(round(scale_from_half * q_low * (1 << 16)))
            expected_float_q16.append(round(scale_from_half * q_high * (1 << 16)))

        for got, want in zip(out_q16, expected_q16):
            assert got == want

        for got, want in zip(out_q16, expected_float_q16):
            # Integer path rounds scale once per block; float path rounds per value.
            # For Q4_0 q in [-8,7], delta stays bounded and deterministic.
            assert abs(got - want) <= 8


def run() -> None:
    test_unpack_known_pattern()
    test_dequant_known_scale()
    test_fp16_to_q16_matches_float_rounding_for_finite_values()
    test_random_blocks_match_reference_float_path()
    print("q4_0_dequant_reference_checks=ok")


if __name__ == "__main__":
    run()
