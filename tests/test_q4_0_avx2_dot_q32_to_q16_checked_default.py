#!/usr/bin/env python3
"""Parity checks for Q4_0DotBlocksAVX2Q32ToQ16CheckedDefault semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_avx2_dot_q32_checked import (
    Q4_0_AVX2_ERR_BAD_LEN,
    Q4_0_AVX2_ERR_NULL_PTR,
    Q4_0_AVX2_OK,
    dot_product_blocks_q32_avx2_checked,
    half_bits,
    pack_q4_signed,
)


def q4_0_round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    return (value + (1 << (shift - 1))) >> shift


def q4_0_round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return q4_0_round_shift_right_unsigned(value, shift)
    return -q4_0_round_shift_right_unsigned(-value, shift)


def q4_0_dot_q32_to_q16(dot_q32: int) -> int:
    return q4_0_round_shift_right_signed(dot_q32, 16)


def q4_0_dot_blocks_avx2_q32_to_q16_checked_ptr(
    lhs_blocks,
    rhs_blocks,
    block_count: int,
    out_holder,
):
    if lhs_blocks is None or rhs_blocks is None or out_holder is None:
        return Q4_0_AVX2_ERR_NULL_PTR

    dot_q32_holder = {"value": 0}
    err = dot_product_blocks_q32_avx2_checked(
        lhs_blocks,
        block_count,
        rhs_blocks,
        block_count,
        block_count,
        dot_q32_holder,
    )
    if err != Q4_0_AVX2_OK:
        return err

    out_holder["value"] = q4_0_dot_q32_to_q16(dot_q32_holder["value"])
    return Q4_0_AVX2_OK


def q4_0_dot_blocks_avx2_q32_to_q16_checked_default(
    lhs_blocks,
    rhs_blocks,
    block_count: int,
    out_holder,
):
    # Canonical default wrapper is a pass-through to checked core contract.
    return q4_0_dot_blocks_avx2_q32_to_q16_checked_ptr(
        lhs_blocks,
        rhs_blocks,
        block_count,
        out_holder,
    )


def make_random_block(rng: random.Random):
    scale = rng.choice([0.0, 0.125, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0])
    lanes = [rng.randrange(-8, 8) for _ in range(32)]
    return (half_bits(scale), pack_q4_signed(lanes))


def test_default_wrapper_matches_checked_core_success_and_errors() -> None:
    rng = random.Random(2026041813)

    for _ in range(360):
        block_count = rng.randint(0, 24)
        lhs = [make_random_block(rng) for _ in range(block_count)]
        rhs = [make_random_block(rng) for _ in range(block_count)]

        out_default = {"value": 111}
        out_core = {"value": 222}

        err_default = q4_0_dot_blocks_avx2_q32_to_q16_checked_default(
            lhs,
            rhs,
            block_count,
            out_default,
        )
        err_core = q4_0_dot_blocks_avx2_q32_to_q16_checked_ptr(
            lhs,
            rhs,
            block_count,
            out_core,
        )

        assert err_default == err_core
        if err_default == Q4_0_AVX2_OK:
            assert out_default["value"] == out_core["value"]
        else:
            assert out_default["value"] == 111


def test_default_wrapper_negative_count_no_partial_write() -> None:
    block = (half_bits(1.0), pack_q4_signed([0] * 32))
    out_holder = {"value": 987654321}

    err = q4_0_dot_blocks_avx2_q32_to_q16_checked_default([block], [block], -1, out_holder)
    assert err == Q4_0_AVX2_ERR_BAD_LEN
    assert out_holder["value"] == 987654321


def test_default_wrapper_extent_shortfall_no_partial_write() -> None:
    block = (half_bits(0.5), pack_q4_signed([1] * 32))
    out_holder = {"value": -77}

    err = q4_0_dot_blocks_avx2_q32_to_q16_checked_default([block], [block], 2, out_holder)
    assert err == Q4_0_AVX2_ERR_BAD_LEN
    assert out_holder["value"] == -77


def test_default_wrapper_null_ptr_paths() -> None:
    block = (half_bits(1.0), pack_q4_signed([0] * 32))
    out_holder = {"value": 5}

    err = q4_0_dot_blocks_avx2_q32_to_q16_checked_default(None, [block], 0, out_holder)
    assert err == Q4_0_AVX2_ERR_NULL_PTR
    assert out_holder["value"] == 5

    err = q4_0_dot_blocks_avx2_q32_to_q16_checked_default([block], None, 0, out_holder)
    assert err == Q4_0_AVX2_ERR_NULL_PTR
    assert out_holder["value"] == 5

    err = q4_0_dot_blocks_avx2_q32_to_q16_checked_default([block], [block], 0, None)
    assert err == Q4_0_AVX2_ERR_NULL_PTR


if __name__ == "__main__":
    test_default_wrapper_matches_checked_core_success_and_errors()
    test_default_wrapper_negative_count_no_partial_write()
    test_default_wrapper_extent_shortfall_no_partial_write()
    test_default_wrapper_null_ptr_paths()
    print("q4_0_avx2_dot_q32_to_q16_checked_default_reference_checks=ok")
