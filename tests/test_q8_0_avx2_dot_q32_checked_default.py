#!/usr/bin/env python3
"""Parity checks for Q8_0DotBlocksAVX2Q32CheckedDefault semantics."""

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


def q8_0_dot_blocks_avx2_q32_checked_ptr(lhs_blocks, rhs_blocks, block_count: int, out_holder):
    if lhs_blocks is None or rhs_blocks is None or out_holder is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    err, dot_q32 = q8_0_dot_blocks_avx2_q32_checked(lhs_blocks, rhs_blocks, block_count)
    if err != Q8_0_AVX2_OK:
        return err

    out_holder["value"] = dot_q32
    return Q8_0_AVX2_OK


def q8_0_dot_blocks_avx2_q32_checked_default(lhs_blocks, rhs_blocks, block_count: int, out_holder):
    # Canonical default wrapper is a pass-through to checked core contract.
    return q8_0_dot_blocks_avx2_q32_checked_ptr(lhs_blocks, rhs_blocks, block_count, out_holder)


def test_default_wrapper_matches_checked_core_success_and_errors() -> None:
    rng = random.Random(2026041807)
    fp16_scales = [0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4000, 0x4400, 0xBC00]

    for _ in range(320):
        block_count = rng.randint(0, 18)
        lhs = [make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]) for _ in range(block_count)]
        rhs = [make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]) for _ in range(block_count)]

        out_default = {"value": 111}
        out_core = {"value": 222}

        err_default = q8_0_dot_blocks_avx2_q32_checked_default(lhs, rhs, block_count, out_default)
        err_core = q8_0_dot_blocks_avx2_q32_checked_ptr(lhs, rhs, block_count, out_core)

        assert err_default == err_core
        if err_default == Q8_0_AVX2_OK:
            assert out_default["value"] == out_core["value"]
        else:
            assert out_default["value"] == 111


def test_default_wrapper_negative_count_no_partial_write() -> None:
    block = make_block(0x3C00, [1] * 32)
    out_holder = {"value": -77}

    err = q8_0_dot_blocks_avx2_q32_checked_default([block], [block], -1, out_holder)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out_holder["value"] == -77


def test_default_wrapper_extent_shortfall_no_partial_write() -> None:
    block = make_block(0x3C00, [1] * 32)
    out_holder = {"value": 999}

    err = q8_0_dot_blocks_avx2_q32_checked_default([], [], 1, out_holder)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out_holder["value"] == 999


def test_default_wrapper_null_ptr_paths() -> None:
    block = make_block(0x3C00, [0] * 32)
    out_holder = {"value": 5}

    err = q8_0_dot_blocks_avx2_q32_checked_default(None, [block], 0, out_holder)
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out_holder["value"] == 5

    err = q8_0_dot_blocks_avx2_q32_checked_default([block], None, 0, out_holder)
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out_holder["value"] == 5

    err = q8_0_dot_blocks_avx2_q32_checked_default([block], [block], 0, None)
    assert err == Q8_0_AVX2_ERR_NULL_PTR


if __name__ == "__main__":
    test_default_wrapper_matches_checked_core_success_and_errors()
    test_default_wrapper_negative_count_no_partial_write()
    test_default_wrapper_extent_shortfall_no_partial_write()
    test_default_wrapper_null_ptr_paths()
    print("q8_0_avx2_dot_q32_checked_default_reference_checks=ok")
