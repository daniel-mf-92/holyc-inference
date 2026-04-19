#!/usr/bin/env python3
"""Parity checks for Q4_0DotProductBlocksAVX2CheckedDefault semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_dot_avx2 import (
    Q4_0_DOT_AVX2_ERR_BAD_LEN,
    Q4_0_DOT_AVX2_ERR_NULL_PTR,
    Q4_0_DOT_AVX2_OK,
    _make_random_blocks,
    q4_0_dot_product_blocks_avx2_checked,
)


def q4_0_dot_product_blocks_avx2_checked_ptr(lhs_blocks, rhs_blocks, block_count: int, out_holder):
    if lhs_blocks is None or rhs_blocks is None or out_holder is None:
        return Q4_0_DOT_AVX2_ERR_NULL_PTR

    err, dot_q32 = q4_0_dot_product_blocks_avx2_checked(lhs_blocks, rhs_blocks, block_count)
    if err != Q4_0_DOT_AVX2_OK:
        return err

    out_holder["value"] = dot_q32
    return Q4_0_DOT_AVX2_OK


def q4_0_dot_product_blocks_avx2_checked_default(lhs_blocks, rhs_blocks, block_count: int, out_holder):
    # Canonical default wrapper is a pass-through to checked core contract.
    return q4_0_dot_product_blocks_avx2_checked_ptr(lhs_blocks, rhs_blocks, block_count, out_holder)


def test_default_wrapper_matches_checked_core_success_and_errors() -> None:
    rng = random.Random(202604190464)

    for _ in range(320):
        block_count = rng.randint(0, 24)
        lhs = _make_random_blocks(rng, block_count)
        rhs = _make_random_blocks(rng, block_count)

        out_default = {"value": 111}
        out_core = {"value": 222}

        err_default = q4_0_dot_product_blocks_avx2_checked_default(lhs, rhs, block_count, out_default)
        err_core = q4_0_dot_product_blocks_avx2_checked_ptr(lhs, rhs, block_count, out_core)

        assert err_default == err_core
        if err_default == Q4_0_DOT_AVX2_OK:
            assert out_default["value"] == out_core["value"]
        else:
            assert out_default["value"] == 111


def test_default_wrapper_negative_count_no_partial_write() -> None:
    lhs = _make_random_blocks(random.Random(7), 1)
    rhs = _make_random_blocks(random.Random(8), 1)
    out_holder = {"value": -77}

    err = q4_0_dot_product_blocks_avx2_checked_default(lhs, rhs, -1, out_holder)
    assert err == Q4_0_DOT_AVX2_ERR_BAD_LEN
    assert out_holder["value"] == -77


def test_default_wrapper_extent_shortfall_no_partial_write() -> None:
    out_holder = {"value": 999}
    err = q4_0_dot_product_blocks_avx2_checked_default([], [], 1, out_holder)
    assert err == Q4_0_DOT_AVX2_ERR_BAD_LEN
    assert out_holder["value"] == 999


def test_default_wrapper_null_ptr_paths() -> None:
    lhs = _make_random_blocks(random.Random(9), 1)
    rhs = _make_random_blocks(random.Random(10), 1)
    out_holder = {"value": 5}

    err = q4_0_dot_product_blocks_avx2_checked_default(None, rhs, 0, out_holder)
    assert err == Q4_0_DOT_AVX2_ERR_NULL_PTR
    assert out_holder["value"] == 5

    err = q4_0_dot_product_blocks_avx2_checked_default(lhs, None, 0, out_holder)
    assert err == Q4_0_DOT_AVX2_ERR_NULL_PTR
    assert out_holder["value"] == 5

    err = q4_0_dot_product_blocks_avx2_checked_default(lhs, rhs, 0, None)
    assert err == Q4_0_DOT_AVX2_ERR_NULL_PTR


def test_source_contains_default_wrapper() -> None:
    source = open("src/quant/q4_0_dot_avx2.HC", "r", encoding="utf-8").read()
    assert "I32 Q4_0DotProductBlocksAVX2CheckedDefault(" in source
    assert "return Q4_0DotProductBlocksAVX2Checked(" in source


if __name__ == "__main__":
    test_default_wrapper_matches_checked_core_success_and_errors()
    test_default_wrapper_negative_count_no_partial_write()
    test_default_wrapper_extent_shortfall_no_partial_write()
    test_default_wrapper_null_ptr_paths()
    test_source_contains_default_wrapper()
    print("q4_0_dot_avx2_checked_default_reference_checks=ok")
