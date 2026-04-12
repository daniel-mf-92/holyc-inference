#!/usr/bin/env python3
"""Reference checks for Q8_0 unpack + integer Q16 dequant semantics."""

from __future__ import annotations

import random
import struct

Q8_0_VALUES_PER_BLOCK = 32
Q8_0_PACKED_BYTES = 32

Q8_0_OK = 0
Q8_0_ERR_NULL_PTR = 1
Q8_0_ERR_BAD_DST_LEN = 2


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


def unpack_block_signed(qs: bytes):
    if len(qs) != Q8_0_PACKED_BYTES:
        return Q8_0_ERR_BAD_DST_LEN, []
    return Q8_0_OK, [struct.unpack("<b", bytes([x]))[0] for x in qs]


def dequantize_block_q16(scale_fp16: int, qs: bytes):
    if len(qs) != Q8_0_PACKED_BYTES:
        return Q8_0_ERR_BAD_DST_LEN, []

    out = [0] * Q8_0_VALUES_PER_BLOCK
    scale_q16 = f16_to_q16(scale_fp16)
    signed = [struct.unpack("<b", bytes([x]))[0] for x in qs]

    for idx, q in enumerate(signed):
        out[idx] = scale_q16 * q

    return Q8_0_OK, out


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def test_unpack_signed_bytes() -> None:
    signed = list(range(-16, 16))
    qs = bytes((v + 256) % 256 for v in signed)
    err, out = unpack_block_signed(qs)
    assert err == Q8_0_OK
    assert out == signed


def test_dequant_known_scale() -> None:
    scale_fp16 = half_bits(0.25)
    signed = [
        -128,
        -96,
        -64,
        -32,
        -16,
        -8,
        -4,
        -2,
        -1,
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        121,
        122,
        123,
        124,
        125,
        127,
    ]
    qs = bytes((v + 256) % 256 for v in signed)

    err, out_q16 = dequantize_block_q16(scale_fp16, qs)
    assert err == Q8_0_OK

    expected_q16 = [round(0.25 * q * (1 << 16)) for q in signed]
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
    rng = random.Random(4242)

    for _ in range(200):
        scale = rng.uniform(-4.0, 4.0)
        scale_fp16 = half_bits(scale)
        scale_q16 = f16_to_q16(scale_fp16)

        signed_vals = [rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)]
        qs = bytes((v + 256) % 256 for v in signed_vals)

        err, out_q16 = dequantize_block_q16(scale_fp16, qs)
        assert err == Q8_0_OK

        scale_from_half = struct.unpack("<e", struct.pack("<H", scale_fp16))[0]

        expected_int = [scale_q16 * q for q in signed_vals]
        expected_float = [round(scale_from_half * q * (1 << 16)) for q in signed_vals]

        assert out_q16 == expected_int

        # Integer path rounds scale once per block; float path rounds per element.
        for got, want in zip(out_q16, expected_float):
            assert abs(got - want) <= 128


def test_error_on_bad_input_length() -> None:
    err, _ = unpack_block_signed(b"\x00" * 31)
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = dequantize_block_q16(half_bits(1.0), b"\x00" * 31)
    assert err == Q8_0_ERR_BAD_DST_LEN


def run() -> None:
    test_unpack_signed_bytes()
    test_dequant_known_scale()
    test_fp16_to_q16_matches_float_rounding_for_finite_values()
    test_random_blocks_match_reference_float_path()
    test_error_on_bad_input_length()
    print("q8_0_dequant_reference_checks=ok")


if __name__ == "__main__":
    run()
