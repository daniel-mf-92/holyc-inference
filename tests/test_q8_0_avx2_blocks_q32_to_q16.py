#!/usr/bin/env python3
"""Reference checks for Q8_0DotBlocksAVX2Q32ToQ16Checked semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_avx2_blocks_q32 import (
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_ERR_NULL_PTR,
    Q8_0_AVX2_OK,
    make_block,
    q8_0_dot_blocks_avx2_q32_checked,
)


def round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return (value + (1 << (shift - 1))) >> shift
    return -(((-value) + (1 << (shift - 1))) >> shift)


def q8_0_dot_q32_to_q16(dot_q32: int) -> int:
    return round_shift_right_signed(dot_q32, 16)


def q8_0_dot_blocks_avx2_q32_to_q16_checked(lhs_blocks, rhs_blocks, block_count: int):
    if lhs_blocks is None or rhs_blocks is None:
        return Q8_0_AVX2_ERR_NULL_PTR, 0
    if block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    err, dot_q32 = q8_0_dot_blocks_avx2_q32_checked(lhs_blocks, rhs_blocks, block_count)
    if err != Q8_0_AVX2_OK:
        return err, 0

    return Q8_0_AVX2_OK, q8_0_dot_q32_to_q16(dot_q32)


def test_known_blocks_match_single_rounding_reference() -> None:
    lhs = [
        make_block(0x3C00, [i - 16 for i in range(32)]),
        make_block(0x4000, [16 - i for i in range(32)]),
        make_block(0x3800, [(-1) ** i * (i % 17) for i in range(32)]),
    ]
    rhs = [
        make_block(0x3C00, [2 * (i - 16) for i in range(32)]),
        make_block(0x3555, [i - 8 for i in range(32)]),
        make_block(0x3A00, [(-1) ** (i + 1) * (i % 11) for i in range(32)]),
    ]

    err, got_q16 = q8_0_dot_blocks_avx2_q32_to_q16_checked(lhs, rhs, 3)
    assert err == Q8_0_AVX2_OK

    err, dot_q32 = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs, 3)
    assert err == Q8_0_AVX2_OK
    assert got_q16 == q8_0_dot_q32_to_q16(dot_q32)


def test_randomized_q32_then_single_q16_rounding() -> None:
    rng = random.Random(2026041602)
    fp16_scales = [0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00, 0x4000, 0x4400, 0xB800, 0xBC00]

    for _ in range(300):
        block_count = rng.randint(1, 24)
        lhs = []
        rhs = []
        for _ in range(block_count):
            lhs.append(make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]))
            rhs.append(make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]))

        err, got_q16 = q8_0_dot_blocks_avx2_q32_to_q16_checked(lhs, rhs, block_count)
        assert err == Q8_0_AVX2_OK

        err, dot_q32 = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs, block_count)
        assert err == Q8_0_AVX2_OK

        expected_q16 = q8_0_dot_q32_to_q16(dot_q32)
        assert got_q16 == expected_q16


def test_rounding_contract_is_not_per_block() -> None:
    # Build a deterministic case where per-block Q16 rounding differs from
    # full-dot single-rounding, proving the API contract explicitly.
    # For fp16 scale 0x1800 => q16 scale 128, per-block scale product is 16384.
    # Using q_dot_q0=2 gives block_dot_q32=32768 (exactly 0.5 in Q16 units).
    # Two blocks then demonstrate the difference:
    # - per-block rounding: round(0.5)+round(0.5) = 1 + 1 = 2
    # - single rounding: round(1.0) = 1
    lhs_block = make_block(0x1800, [1] + [0] * 31)
    rhs_block = make_block(0x1800, [2] + [0] * 31)
    lhs = [lhs_block, lhs_block]
    rhs = [rhs_block, rhs_block]

    err, got_q16 = q8_0_dot_blocks_avx2_q32_to_q16_checked(lhs, rhs, 2)
    assert err == Q8_0_AVX2_OK

    per_block_q16_sum = 0
    for i in range(2):
        err, block_q32 = q8_0_dot_blocks_avx2_q32_checked([lhs[i]], [rhs[i]], 1)
        assert err == Q8_0_AVX2_OK
        per_block_q16_sum += q8_0_dot_q32_to_q16(block_q32)

    err, full_q32 = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs, 2)
    assert err == Q8_0_AVX2_OK
    expected_single_round = q8_0_dot_q32_to_q16(full_q32)

    assert got_q16 == expected_single_round
    assert got_q16 != per_block_q16_sum


def test_error_paths() -> None:
    err, _ = q8_0_dot_blocks_avx2_q32_to_q16_checked(None, [], 0)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_dot_blocks_avx2_q32_to_q16_checked([], None, 0)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_dot_blocks_avx2_q32_to_q16_checked([], [], -1)
    assert err == Q8_0_AVX2_ERR_BAD_LEN


def run() -> None:
    test_known_blocks_match_single_rounding_reference()
    test_randomized_q32_then_single_q16_rounding()
    test_rounding_contract_is_not_per_block()
    test_error_paths()
    print("q8_0_avx2_blocks_q32_to_q16_reference_checks=ok")


if __name__ == "__main__":
    run()
