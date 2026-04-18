#!/usr/bin/env python3
"""Parity + throughput harness for Q4_0DotProductBlocksAVX2Checked."""

from __future__ import annotations

import random
import struct
import time

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)

Q4_0_PACKED_BYTES = 16
Q4_0_VALUES_PER_BLOCK = 32

Q4_0_DOT_AVX2_OK = 0
Q4_0_DOT_AVX2_ERR_NULL_PTR = 1
Q4_0_DOT_AVX2_ERR_BAD_LEN = 2
Q4_0_DOT_AVX2_ERR_OVERFLOW = 3
Q4_0_DOT_AVX2_ERR_PARITY = 4


def _try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if (rhs > 0 and lhs > I64_MAX - rhs) or (rhs < 0 and lhs < I64_MIN - rhs):
        return False, 0
    return True, lhs + rhs


def _try_mul_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    prod = lhs * rhs
    if prod < I64_MIN or prod > I64_MAX:
        return False, 0
    return True, prod


def _round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    return (value + (1 << (shift - 1))) >> shift


def _f16_to_q16(bits: int) -> int:
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    frac = bits & 0x03FF

    if exponent == 0:
        if frac == 0:
            return 0
        mag = _round_shift_right_unsigned(frac, 8)
        return -mag if sign else mag

    if exponent == 0x1F:
        return -0x3FFFFFFFFFFFFFFF if sign else 0x3FFFFFFFFFFFFFFF

    mantissa = 1024 + frac
    shift = exponent - 9
    if shift >= 0:
        mag = mantissa << shift
    else:
        mag = _round_shift_right_unsigned(mantissa, -shift)
    return -mag if sign else mag


def _nibble_to_signed(packed: int, upper: bool) -> int:
    return (((packed >> 4) & 0x0F) if upper else (packed & 0x0F)) - 8


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def q4_0_dot_product_blocks_q32_checked(lhs_blocks, rhs_blocks, block_count: int):
    if lhs_blocks is None or rhs_blocks is None:
        return Q4_0_DOT_AVX2_ERR_NULL_PTR, 0
    if block_count < 0:
        return Q4_0_DOT_AVX2_ERR_BAD_LEN, 0
    if block_count > len(lhs_blocks) or block_count > len(rhs_blocks):
        return Q4_0_DOT_AVX2_ERR_BAD_LEN, 0

    total = 0
    for index in range(block_count):
        lhs_scale_fp16, lhs_qs = lhs_blocks[index]
        rhs_scale_fp16, rhs_qs = rhs_blocks[index]

        if len(lhs_qs) != Q4_0_PACKED_BYTES or len(rhs_qs) != Q4_0_PACKED_BYTES:
            return Q4_0_DOT_AVX2_ERR_BAD_LEN, 0

        lhs_scale_q16 = _f16_to_q16(lhs_scale_fp16)
        rhs_scale_q16 = _f16_to_q16(rhs_scale_fp16)
        ok, scale_prod_q32 = _try_mul_i64(lhs_scale_q16, rhs_scale_q16)
        if not ok:
            return Q4_0_DOT_AVX2_ERR_OVERFLOW, 0

        nibble_dot_q0 = 0
        for lhs_packed, rhs_packed in zip(lhs_qs, rhs_qs):
            ok, nibble_dot_q0 = _try_add_i64(
                nibble_dot_q0,
                _nibble_to_signed(lhs_packed, False) * _nibble_to_signed(rhs_packed, False),
            )
            if not ok:
                return Q4_0_DOT_AVX2_ERR_OVERFLOW, 0

            ok, nibble_dot_q0 = _try_add_i64(
                nibble_dot_q0,
                _nibble_to_signed(lhs_packed, True) * _nibble_to_signed(rhs_packed, True),
            )
            if not ok:
                return Q4_0_DOT_AVX2_ERR_OVERFLOW, 0

        ok, block_q32 = _try_mul_i64(scale_prod_q32, nibble_dot_q0)
        if not ok:
            return Q4_0_DOT_AVX2_ERR_OVERFLOW, 0

        ok, total = _try_add_i64(total, block_q32)
        if not ok:
            return Q4_0_DOT_AVX2_ERR_OVERFLOW, 0

    return Q4_0_DOT_AVX2_OK, total


def _unpack_block_to_i16_lanes(packed_qs: bytes):
    if len(packed_qs) < Q4_0_PACKED_BYTES:
        return Q4_0_DOT_AVX2_ERR_BAD_LEN, []

    lanes = [0] * Q4_0_VALUES_PER_BLOCK
    for byte_index, packed in enumerate(packed_qs[:Q4_0_PACKED_BYTES]):
        lane_base = byte_index << 1
        lanes[lane_base] = _nibble_to_signed(packed, False)
        lanes[lane_base + 1] = _nibble_to_signed(packed, True)
    return Q4_0_DOT_AVX2_OK, lanes


def _dot_i16_lanes_to_q0(lhs_lanes: list[int], rhs_lanes: list[int]):
    if len(lhs_lanes) < Q4_0_VALUES_PER_BLOCK or len(rhs_lanes) < Q4_0_VALUES_PER_BLOCK:
        return Q4_0_DOT_AVX2_ERR_BAD_LEN, 0

    pair_partials = [0] * 16
    for pair_index in range(16):
        lane_base = pair_index << 1
        pair_partials[pair_index] = (
            lhs_lanes[lane_base] * rhs_lanes[lane_base]
            + lhs_lanes[lane_base + 1] * rhs_lanes[lane_base + 1]
        )

    q0_accum = 0
    for term in pair_partials:
        ok, q0_accum = _try_add_i64(q0_accum, term)
        if not ok:
            return Q4_0_DOT_AVX2_ERR_OVERFLOW, 0

    return Q4_0_DOT_AVX2_OK, q0_accum


def q4_0_dot_product_block_q32_avx2_core_checked(lhs_block, rhs_block):
    lhs_scale_fp16, lhs_qs = lhs_block
    rhs_scale_fp16, rhs_qs = rhs_block

    err, lhs_lanes = _unpack_block_to_i16_lanes(lhs_qs)
    if err != Q4_0_DOT_AVX2_OK:
        return err, 0
    err, rhs_lanes = _unpack_block_to_i16_lanes(rhs_qs)
    if err != Q4_0_DOT_AVX2_OK:
        return err, 0

    err, dot_q0 = _dot_i16_lanes_to_q0(lhs_lanes, rhs_lanes)
    if err != Q4_0_DOT_AVX2_OK:
        return err, 0

    lhs_scale_q16 = _f16_to_q16(lhs_scale_fp16)
    rhs_scale_q16 = _f16_to_q16(rhs_scale_fp16)

    ok, scale_prod_q32 = _try_mul_i64(lhs_scale_q16, rhs_scale_q16)
    if not ok:
        return Q4_0_DOT_AVX2_ERR_OVERFLOW, 0
    ok, block_q32 = _try_mul_i64(scale_prod_q32, dot_q0)
    if not ok:
        return Q4_0_DOT_AVX2_ERR_OVERFLOW, 0

    return Q4_0_DOT_AVX2_OK, block_q32


def q4_0_dot_product_blocks_q32_avx2_core_checked(lhs_blocks, rhs_blocks, block_count: int):
    if lhs_blocks is None or rhs_blocks is None:
        return Q4_0_DOT_AVX2_ERR_NULL_PTR, 0
    if block_count < 0:
        return Q4_0_DOT_AVX2_ERR_BAD_LEN, 0
    if block_count > len(lhs_blocks) or block_count > len(rhs_blocks):
        return Q4_0_DOT_AVX2_ERR_BAD_LEN, 0

    total = 0
    for index in range(block_count):
        err, block_q32 = q4_0_dot_product_block_q32_avx2_core_checked(
            lhs_blocks[index], rhs_blocks[index]
        )
        if err != Q4_0_DOT_AVX2_OK:
            return err, 0

        ok, total = _try_add_i64(total, block_q32)
        if not ok:
            return Q4_0_DOT_AVX2_ERR_OVERFLOW, 0

    return Q4_0_DOT_AVX2_OK, total


def q4_0_dot_product_blocks_avx2_checked(lhs_blocks, rhs_blocks, block_count: int):
    err_ref, ref_q32 = q4_0_dot_product_blocks_q32_checked(lhs_blocks, rhs_blocks, block_count)
    err_avx2, avx2_q32 = q4_0_dot_product_blocks_q32_avx2_core_checked(lhs_blocks, rhs_blocks, block_count)
    if err_avx2 != err_ref:
        return Q4_0_DOT_AVX2_ERR_PARITY, 0
    if err_ref != Q4_0_DOT_AVX2_OK:
        return err_ref, 0
    if avx2_q32 != ref_q32:
        return Q4_0_DOT_AVX2_ERR_PARITY, 0

    return Q4_0_DOT_AVX2_OK, avx2_q32


def _make_random_blocks(rng: random.Random, count: int):
    blocks = []
    for _ in range(count):
        scale = rng.uniform(-4.0, 4.0)
        bits = half_bits(scale)
        qs = bytes(rng.randrange(256) for _ in range(Q4_0_PACKED_BYTES))
        blocks.append((bits, qs))
    return blocks


def test_known_identity() -> None:
    vals = list(range(16)) * 2
    packed = bytes((vals[i] & 0x0F) | ((vals[i + 1] & 0x0F) << 4) for i in range(0, 32, 2))
    block = (half_bits(1.0), packed)

    err, got = q4_0_dot_product_blocks_avx2_checked([block], [block], 1)
    assert err == Q4_0_DOT_AVX2_OK

    signed = [v - 8 for v in vals]
    expected_dot_q0 = sum(v * v for v in signed)
    expected_q32 = (1 << 16) * (1 << 16) * expected_dot_q0
    assert got == expected_q32


def test_randomized_parity_and_adversarial_scales() -> None:
    rng = random.Random(20260419_449)

    # Includes tiny, huge, and sign-flipped scales to stress fp16->q16 conversion.
    fixed_scales = [0.0, 1e-5, -1e-5, 0.5, -0.5, 32.0, -32.0, 65504.0, -65504.0]

    for _ in range(2000):
        block_count = rng.randint(0, 64)
        lhs = _make_random_blocks(rng, block_count)
        rhs = _make_random_blocks(rng, block_count)

        if block_count > 0:
            lhs_index = rng.randrange(block_count)
            rhs_index = rng.randrange(block_count)
            lhs_scale = fixed_scales[rng.randrange(len(fixed_scales))]
            rhs_scale = fixed_scales[rng.randrange(len(fixed_scales))]
            lhs[lhs_index] = (half_bits(lhs_scale), lhs[lhs_index][1])
            rhs[rhs_index] = (half_bits(rhs_scale), rhs[rhs_index][1])

        err_avx2, got_avx2 = q4_0_dot_product_blocks_avx2_checked(lhs, rhs, block_count)
        err_ref, got_ref = q4_0_dot_product_blocks_q32_checked(lhs, rhs, block_count)

        assert err_avx2 == err_ref
        assert got_avx2 == got_ref


def test_error_surface() -> None:
    one = [(half_bits(1.0), bytes([0] * Q4_0_PACKED_BYTES))]

    err, _ = q4_0_dot_product_blocks_avx2_checked(None, one, 1)
    assert err == Q4_0_DOT_AVX2_ERR_NULL_PTR

    err, _ = q4_0_dot_product_blocks_avx2_checked(one, one, -1)
    assert err == Q4_0_DOT_AVX2_ERR_BAD_LEN

    err, _ = q4_0_dot_product_blocks_avx2_checked(one, one, 2)
    assert err == Q4_0_DOT_AVX2_ERR_BAD_LEN

    bad = [(half_bits(1.0), bytes([0] * (Q4_0_PACKED_BYTES - 1)))]
    err, _ = q4_0_dot_product_blocks_avx2_checked(bad, bad, 1)
    assert err == Q4_0_DOT_AVX2_ERR_BAD_LEN


def test_throughput_smoke() -> None:
    rng = random.Random(9449)
    block_count = 256
    lhs = _make_random_blocks(rng, block_count)
    rhs = _make_random_blocks(rng, block_count)

    loops = 300

    t0 = time.perf_counter()
    ref_sum = 0
    for _ in range(loops):
        err, dot_q32 = q4_0_dot_product_blocks_q32_checked(lhs, rhs, block_count)
        assert err == Q4_0_DOT_AVX2_OK
        ref_sum ^= dot_q32
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    avx2_sum = 0
    for _ in range(loops):
        err, dot_q32 = q4_0_dot_product_blocks_avx2_checked(lhs, rhs, block_count)
        assert err == Q4_0_DOT_AVX2_OK
        avx2_sum ^= dot_q32
    t3 = time.perf_counter()

    assert ref_sum == avx2_sum

    ref_dt = t1 - t0
    avx2_dt = t3 - t2
    ref_bps = (block_count * loops) / ref_dt if ref_dt > 0 else 0.0
    avx2_bps = (block_count * loops) / avx2_dt if avx2_dt > 0 else 0.0

    # Throughput is an informative smoke metric, not a hard gate.
    print(
        f"throughput blocks/sec scalar={ref_bps:.2f} avx2_shape={avx2_bps:.2f} speedup={(avx2_dt and (ref_dt / avx2_dt) or 0.0):.3f}"
    )


if __name__ == "__main__":
    test_known_identity()
    test_randomized_parity_and_adversarial_scales()
    test_error_surface()
    test_throughput_smoke()
    print("ok")
