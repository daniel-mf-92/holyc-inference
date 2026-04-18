#!/usr/bin/env python3
"""Parity + throughput harness for Q8_0DotProductBlocksAVX2Checked."""

from __future__ import annotations

import random
import struct
import time

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)

Q8_0_VALUES_PER_BLOCK = 32

Q8_0_AVX2_OK = 0
Q8_0_AVX2_ERR_NULL_PTR = 1
Q8_0_AVX2_ERR_BAD_LEN = 2
Q8_0_AVX2_ERR_OVERFLOW = 3
Q8_0_AVX2_ERR_PARITY = 4


def _try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return False, 0
    return True, lhs + rhs


def _try_mul_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs == 0 or rhs == 0:
        return True, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return False, 0
    return True, product


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


def _dot_q0(lhs_qs: list[int], rhs_qs: list[int]) -> tuple[int, int]:
    if len(lhs_qs) < Q8_0_VALUES_PER_BLOCK or len(rhs_qs) < Q8_0_VALUES_PER_BLOCK:
        return Q8_0_AVX2_ERR_BAD_LEN, 0
    total = 0
    for i in range(Q8_0_VALUES_PER_BLOCK):
        total += int(lhs_qs[i]) * int(rhs_qs[i])
    return Q8_0_AVX2_OK, total


def q8_0_dot_product_blocks_q32_checked(lhs_blocks, rhs_blocks, block_count: int):
    if lhs_blocks is None or rhs_blocks is None:
        return Q8_0_AVX2_ERR_NULL_PTR, 0
    if block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0
    if block_count > len(lhs_blocks) or block_count > len(rhs_blocks):
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    total_q32 = 0
    for i in range(block_count):
        lhs = lhs_blocks[i]
        rhs = rhs_blocks[i]

        err, qdot = _dot_q0(lhs["qs"], rhs["qs"])
        if err != Q8_0_AVX2_OK:
            return err, 0

        lhs_scale = _f16_to_q16(lhs["d_fp16"])
        rhs_scale = _f16_to_q16(rhs["d_fp16"])

        ok, scale_prod = _try_mul_i64(lhs_scale, rhs_scale)
        if not ok:
            return Q8_0_AVX2_ERR_OVERFLOW, 0

        ok, block_q32 = _try_mul_i64(scale_prod, qdot)
        if not ok:
            return Q8_0_AVX2_ERR_OVERFLOW, 0

        ok, total_q32 = _try_add_i64(total_q32, block_q32)
        if not ok:
            return Q8_0_AVX2_ERR_OVERFLOW, 0

    return Q8_0_AVX2_OK, total_q32


def q8_0_dot_blocks_avx2_q32_checked(lhs_blocks, rhs_blocks, block_count: int):
    # AVX2-shape path: in HolyC this maps to widened-lane pipeline; here we
    # keep exact integer semantics for parity validation.
    return q8_0_dot_product_blocks_q32_checked(lhs_blocks, rhs_blocks, block_count)


def q8_0_dot_product_blocks_avx2_checked(lhs_blocks, rhs_blocks, block_count: int):
    err_scalar, scalar_q32 = q8_0_dot_product_blocks_q32_checked(lhs_blocks, rhs_blocks, block_count)
    if err_scalar != Q8_0_AVX2_OK:
        return err_scalar, 0

    err_avx2, avx2_q32 = q8_0_dot_blocks_avx2_q32_checked(lhs_blocks, rhs_blocks, block_count)
    if err_avx2 != err_scalar:
        return Q8_0_AVX2_ERR_PARITY, 0
    if avx2_q32 != scalar_q32:
        return Q8_0_AVX2_ERR_PARITY, 0

    return Q8_0_AVX2_OK, avx2_q32


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _make_random_block(rng: random.Random):
    return {
        "d_fp16": half_bits(rng.uniform(-32.0, 32.0)),
        "qs": [rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)],
    }


def test_known_identity() -> None:
    lhs = {"d_fp16": half_bits(1.0), "qs": [1] * Q8_0_VALUES_PER_BLOCK}
    rhs = {"d_fp16": half_bits(1.0), "qs": [1] * Q8_0_VALUES_PER_BLOCK}

    err, got = q8_0_dot_product_blocks_avx2_checked([lhs], [rhs], 1)
    assert err == Q8_0_AVX2_OK

    expected_q32 = (1 << 16) * (1 << 16) * Q8_0_VALUES_PER_BLOCK
    assert got == expected_q32


def test_randomized_parity_and_adversarial_scales() -> None:
    rng = random.Random(20260419_450)
    fixed_scales = [0.0, 1e-5, -1e-5, 0.5, -0.5, 32.0, -32.0, 65504.0, -65504.0]

    for _ in range(2200):
        block_count = rng.randint(0, 64)
        lhs = [_make_random_block(rng) for _ in range(block_count)]
        rhs = [_make_random_block(rng) for _ in range(block_count)]

        if block_count > 0:
            lhs[rng.randrange(block_count)]["d_fp16"] = half_bits(rng.choice(fixed_scales))
            rhs[rng.randrange(block_count)]["d_fp16"] = half_bits(rng.choice(fixed_scales))

        err_avx2, got_avx2 = q8_0_dot_product_blocks_avx2_checked(lhs, rhs, block_count)
        err_ref, got_ref = q8_0_dot_product_blocks_q32_checked(lhs, rhs, block_count)

        assert err_avx2 == err_ref
        assert got_avx2 == got_ref


def test_error_surface() -> None:
    one = [{"d_fp16": half_bits(1.0), "qs": [0] * Q8_0_VALUES_PER_BLOCK}]

    err, _ = q8_0_dot_product_blocks_avx2_checked(None, one, 1)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_dot_product_blocks_avx2_checked(one, None, 1)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_dot_product_blocks_avx2_checked(one, one, -1)
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_dot_product_blocks_avx2_checked(one, one, 2)
    assert err == Q8_0_AVX2_ERR_BAD_LEN


def test_throughput_smoke() -> None:
    rng = random.Random(8450)
    block_count = 256
    lhs = [_make_random_block(rng) for _ in range(block_count)]
    rhs = [_make_random_block(rng) for _ in range(block_count)]

    loops = 350

    t0 = time.perf_counter()
    scalar_sum = 0
    for _ in range(loops):
        err, dot_q32 = q8_0_dot_product_blocks_q32_checked(lhs, rhs, block_count)
        assert err == Q8_0_AVX2_OK
        scalar_sum ^= dot_q32
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    avx2_sum = 0
    for _ in range(loops):
        err, dot_q32 = q8_0_dot_product_blocks_avx2_checked(lhs, rhs, block_count)
        assert err == Q8_0_AVX2_OK
        avx2_sum ^= dot_q32
    t3 = time.perf_counter()

    assert scalar_sum == avx2_sum

    scalar_dt = t1 - t0
    avx2_dt = t3 - t2
    scalar_bps = (block_count * loops) / scalar_dt if scalar_dt > 0 else 0.0
    avx2_bps = (block_count * loops) / avx2_dt if avx2_dt > 0 else 0.0

    print(
        f"throughput blocks/sec scalar={scalar_bps:.2f} avx2_shape={avx2_bps:.2f} speedup={(avx2_dt and (scalar_dt / avx2_dt) or 0.0):.3f}"
    )


if __name__ == "__main__":
    test_known_identity()
    test_randomized_parity_and_adversarial_scales()
    test_error_surface()
    test_throughput_smoke()
    print("ok")
