#!/usr/bin/env python3
"""Reference checks for integer Q4_0 dot-product semantics."""

from __future__ import annotations

import random
import struct

Q4_0_PACKED_BYTES = 16
Q4_0_VALUES_PER_BLOCK = 32

Q4_0_OK = 0
Q4_0_ERR_NULL_PTR = 1
Q4_0_ERR_BAD_DST_LEN = 2


def round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    round_bias = 1 << (shift - 1)
    return (value + round_bias) >> shift


def round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return round_shift_right_unsigned(value, shift)
    return -round_shift_right_unsigned(-value, shift)


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


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def dot_product_block_q32(lhs_scale_fp16: int, lhs_qs: bytes, rhs_scale_fp16: int, rhs_qs: bytes):
    if len(lhs_qs) != Q4_0_PACKED_BYTES or len(rhs_qs) != Q4_0_PACKED_BYTES:
        return Q4_0_ERR_BAD_DST_LEN, 0

    lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
    rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
    scale_prod_q32 = lhs_scale_q16 * rhs_scale_q16

    nibble_dot_q0 = 0
    for l_packed, r_packed in zip(lhs_qs, rhs_qs):
        nibble_dot_q0 += nibble_to_signed(l_packed, False) * nibble_to_signed(r_packed, False)
        nibble_dot_q0 += nibble_to_signed(l_packed, True) * nibble_to_signed(r_packed, True)

    return Q4_0_OK, scale_prod_q32 * nibble_dot_q0


def dot_q32_to_q16(dot_q32: int) -> int:
    return round_shift_right_signed(dot_q32, 16)


def unpack_signed(qs: bytes) -> list[int]:
    out: list[int] = []
    for packed in qs:
        out.append((packed & 0x0F) - 8)
        out.append(((packed >> 4) & 0x0F) - 8)
    return out


def test_identity_block() -> None:
    lhs_scale_fp16 = half_bits(1.0)
    rhs_scale_fp16 = half_bits(1.0)

    vals = list(range(16)) * 2
    lhs_qs = bytes((vals[i] & 0x0F) | ((vals[i + 1] & 0x0F) << 4) for i in range(0, 32, 2))
    rhs_qs = lhs_qs

    err, got_q32 = dot_product_block_q32(lhs_scale_fp16, lhs_qs, rhs_scale_fp16, rhs_qs)
    assert err == Q4_0_OK

    signed = [v - 8 for v in vals]
    expected_dot_q0 = sum(x * x for x in signed)
    expected_q32 = (1 << 16) * (1 << 16) * expected_dot_q0
    assert got_q32 == expected_q32
    assert dot_q32_to_q16(got_q32) == expected_dot_q0 * (1 << 16)


def test_opposite_sign_scales() -> None:
    lhs_scale_fp16 = half_bits(0.5)
    rhs_scale_fp16 = half_bits(-2.0)
    lhs_qs = bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2)
    rhs_qs = bytes(reversed(lhs_qs))

    err, got_q32 = dot_product_block_q32(lhs_scale_fp16, lhs_qs, rhs_scale_fp16, rhs_qs)
    assert err == Q4_0_OK

    lhs_signed = unpack_signed(lhs_qs)
    rhs_signed = unpack_signed(rhs_qs)
    dot_q0 = sum(a * b for a, b in zip(lhs_signed, rhs_signed))

    lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
    rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
    expected_q32 = lhs_scale_q16 * rhs_scale_q16 * dot_q0
    assert got_q32 == expected_q32


def test_random_blocks_match_float_reference_bounds() -> None:
    rng = random.Random(20260412)

    for _ in range(400):
        lhs_scale = rng.uniform(-4.0, 4.0)
        rhs_scale = rng.uniform(-4.0, 4.0)
        lhs_scale_fp16 = half_bits(lhs_scale)
        rhs_scale_fp16 = half_bits(rhs_scale)

        lhs_qs = bytes(rng.randrange(256) for _ in range(Q4_0_PACKED_BYTES))
        rhs_qs = bytes(rng.randrange(256) for _ in range(Q4_0_PACKED_BYTES))

        err, got_q32 = dot_product_block_q32(lhs_scale_fp16, lhs_qs, rhs_scale_fp16, rhs_qs)
        assert err == Q4_0_OK

        lhs_signed = unpack_signed(lhs_qs)
        rhs_signed = unpack_signed(rhs_qs)
        dot_q0 = sum(a * b for a, b in zip(lhs_signed, rhs_signed))

        lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
        rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
        expected_int_path_q32 = lhs_scale_q16 * rhs_scale_q16 * dot_q0
        assert got_q32 == expected_int_path_q32

        lhs_scale_half = struct.unpack("<e", struct.pack("<H", lhs_scale_fp16))[0]
        rhs_scale_half = struct.unpack("<e", struct.pack("<H", rhs_scale_fp16))[0]
        expected_float_q32 = round((lhs_scale_half * rhs_scale_half) * dot_q0 * (1 << 32))

        # Integer path rounds each scale once to Q16 then multiplies.
        # Bound error on resulting Q32 by small deterministic tolerance.
        assert abs(got_q32 - expected_float_q32) <= 35_000_000


def test_error_on_bad_input_length() -> None:
    err, _ = dot_product_block_q32(half_bits(1.0), b"\x00" * 15, half_bits(1.0), b"\x00" * 16)
    assert err == Q4_0_ERR_BAD_DST_LEN


def run() -> None:
    test_identity_block()
    test_opposite_sign_scales()
    test_random_blocks_match_float_reference_bounds()
    test_error_on_bad_input_length()
    print("q4_0_dot_reference_checks=ok")


if __name__ == "__main__":
    run()
