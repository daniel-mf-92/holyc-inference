#!/usr/bin/env python3
"""Reference checks for integer mixed Q4_0 x Q8_0 dot-product semantics."""

from __future__ import annotations

import random
import struct

Q4_0_PACKED_BYTES = 16
Q8_0_PACKED_BYTES = 32

Q4_0_Q8_0_OK = 0
Q4_0_Q8_0_ERR_NULL_PTR = 1
Q4_0_Q8_0_ERR_BAD_DST_LEN = 2


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


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def nibble_to_signed(packed: int, upper_nibble: bool) -> int:
    if upper_nibble:
        q_unsigned = (packed >> 4) & 0x0F
    else:
        q_unsigned = packed & 0x0F
    return q_unsigned - 8


def unpack_q4_signed(qs: bytes) -> list[int]:
    out: list[int] = []
    for packed in qs:
        out.append(nibble_to_signed(packed, False))
        out.append(nibble_to_signed(packed, True))
    return out


def unpack_q8_signed(qs: bytes) -> list[int]:
    return [struct.unpack("<b", bytes([byte]))[0] for byte in qs]


def pack_q4_from_signed(vals: list[int]) -> bytes:
    assert len(vals) == 32
    out = bytearray()
    for idx in range(0, 32, 2):
        lo = vals[idx] + 8
        hi = vals[idx + 1] + 8
        assert 0 <= lo <= 15 and 0 <= hi <= 15
        out.append((lo & 0x0F) | ((hi & 0x0F) << 4))
    return bytes(out)


def pack_q8_signed(vals: list[int]) -> bytes:
    assert len(vals) == 32
    return bytes((v + 256) % 256 for v in vals)


def dot_product_block_q32(lhs_scale_fp16: int, lhs_q4: bytes, rhs_scale_fp16: int, rhs_q8: bytes) -> tuple[int, int]:
    if len(lhs_q4) != Q4_0_PACKED_BYTES or len(rhs_q8) != Q8_0_PACKED_BYTES:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, 0

    lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
    rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
    scale_prod_q32 = lhs_scale_q16 * rhs_scale_q16

    lhs_signed = unpack_q4_signed(lhs_q4)
    rhs_signed = unpack_q8_signed(rhs_q8)
    q_dot_q0 = sum(a * b for a, b in zip(lhs_signed, rhs_signed))
    return Q4_0_Q8_0_OK, scale_prod_q32 * q_dot_q0


def dot_product_blocks_q32(lhs_blocks, rhs_blocks) -> tuple[int, int]:
    if len(lhs_blocks) != len(rhs_blocks):
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, 0

    total = 0
    for (l_scale, l_q4), (r_scale, r_q8) in zip(lhs_blocks, rhs_blocks):
        err, block_dot = dot_product_block_q32(l_scale, l_q4, r_scale, r_q8)
        if err != Q4_0_Q8_0_OK:
            return err, 0
        total += block_dot

    return Q4_0_Q8_0_OK, total


def dot_q32_to_q16(dot_q32: int) -> int:
    return round_shift_right_signed(dot_q32, 16)


def test_identity_mixed_block() -> None:
    lhs_scale_fp16 = half_bits(1.0)
    rhs_scale_fp16 = half_bits(1.0)

    q4_signed = [((idx % 16) - 8) for idx in range(32)]
    q8_signed = q4_signed[:]

    lhs_q4 = pack_q4_from_signed(q4_signed)
    rhs_q8 = pack_q8_signed(q8_signed)

    err, got_q32 = dot_product_block_q32(lhs_scale_fp16, lhs_q4, rhs_scale_fp16, rhs_q8)
    assert err == Q4_0_Q8_0_OK

    expected_q0 = sum(v * v for v in q4_signed)
    expected_q32 = (1 << 16) * (1 << 16) * expected_q0
    assert got_q32 == expected_q32
    assert dot_q32_to_q16(got_q32) == expected_q0 * (1 << 16)


def test_random_blocks_match_integer_reference() -> None:
    rng = random.Random(20260412)

    for _ in range(500):
        lhs_scale = rng.uniform(-4.0, 4.0)
        rhs_scale = rng.uniform(-4.0, 4.0)

        lhs_scale_fp16 = half_bits(lhs_scale)
        rhs_scale_fp16 = half_bits(rhs_scale)

        q4_signed = [rng.randrange(-8, 8) for _ in range(32)]
        q8_signed = [rng.randrange(-128, 128) for _ in range(32)]

        lhs_q4 = pack_q4_from_signed(q4_signed)
        rhs_q8 = pack_q8_signed(q8_signed)

        err, got_q32 = dot_product_block_q32(lhs_scale_fp16, lhs_q4, rhs_scale_fp16, rhs_q8)
        assert err == Q4_0_Q8_0_OK

        expected_q0 = sum(a * b for a, b in zip(q4_signed, q8_signed))
        lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
        rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
        expected_q32 = lhs_scale_q16 * rhs_scale_q16 * expected_q0
        assert got_q32 == expected_q32

        lhs_scale_half = struct.unpack("<e", struct.pack("<H", lhs_scale_fp16))[0]
        rhs_scale_half = struct.unpack("<e", struct.pack("<H", rhs_scale_fp16))[0]
        expected_float_q32 = round((lhs_scale_half * rhs_scale_half) * expected_q0 * (1 << 32))

        # Integer path rounds scales once to Q16 then multiplies.
        assert abs(got_q32 - expected_float_q32) <= 70_000_000


def test_multiblock_accumulation() -> None:
    b0 = (half_bits(0.75), pack_q4_from_signed([((i % 8) - 4) for i in range(32)]))
    b1 = (half_bits(-0.5), pack_q4_from_signed([4 - (i % 8) for i in range(32)]))

    r0 = (half_bits(1.25), pack_q8_signed([((i % 11) - 5) * 3 for i in range(32)]))
    r1 = (half_bits(-1.5), pack_q8_signed([7 - (i % 13) for i in range(32)]))

    err, total = dot_product_blocks_q32([b0, b1], [r0, r1])
    assert err == Q4_0_Q8_0_OK

    _, part0 = dot_product_block_q32(*b0, *r0)
    _, part1 = dot_product_block_q32(*b1, *r1)
    assert total == part0 + part1


def test_error_on_bad_lengths() -> None:
    err, _ = dot_product_block_q32(half_bits(1.0), b"\x00" * 15, half_bits(1.0), b"\x00" * 32)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_product_block_q32(half_bits(1.0), b"\x00" * 16, half_bits(1.0), b"\x00" * 31)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN


def run() -> None:
    test_identity_mixed_block()
    test_random_blocks_match_integer_reference()
    test_multiblock_accumulation()
    test_error_on_bad_lengths()
    print("q4_0_q8_0_dot_kernel_reference_checks=ok")


if __name__ == "__main__":
    run()
