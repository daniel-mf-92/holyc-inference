#!/usr/bin/env python3
"""Parity checks for Q8_0DotProductBlocksAVX2CheckedDefault semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_dot_avx2 import (
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_ERR_NULL_PTR,
    Q8_0_AVX2_OK,
    _make_random_block,
    q8_0_dot_product_blocks_avx2_checked,
)


def _make_random_blocks(rng: random.Random, count: int):
    return [_make_random_block(rng) for _ in range(count)]


def q8_0_dot_product_blocks_avx2_checked_ptr(lhs_blocks, rhs_blocks, block_count: int, out_holder):
    if lhs_blocks is None or rhs_blocks is None or out_holder is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    err, dot_q32 = q8_0_dot_product_blocks_avx2_checked(lhs_blocks, rhs_blocks, block_count)
    if err != Q8_0_AVX2_OK:
        return err

    out_holder["value"] = dot_q32
    return Q8_0_AVX2_OK


def q8_0_dot_product_blocks_avx2_checked_default(lhs_blocks, rhs_blocks, block_count: int, out_holder):
    # Canonical default wrapper is strict pass-through to checked core.
    return q8_0_dot_product_blocks_avx2_checked_ptr(lhs_blocks, rhs_blocks, block_count, out_holder)


def test_default_wrapper_matches_checked_core_success_and_errors() -> None:
    rng = random.Random(202604190471)

    for _ in range(340):
        block_count = rng.randint(0, 36)
        lhs = _make_random_blocks(rng, block_count)
        rhs = _make_random_blocks(rng, block_count)

        out_default = {"value": 111}
        out_core = {"value": 222}

        err_default = q8_0_dot_product_blocks_avx2_checked_default(lhs, rhs, block_count, out_default)
        err_core = q8_0_dot_product_blocks_avx2_checked_ptr(lhs, rhs, block_count, out_core)

        assert err_default == err_core
        if err_default == Q8_0_AVX2_OK:
            assert out_default["value"] == out_core["value"]
        else:
            assert out_default["value"] == 111


def test_default_wrapper_negative_count_no_partial_write() -> None:
    lhs = _make_random_blocks(random.Random(7), 1)
    rhs = _make_random_blocks(random.Random(8), 1)
    out_holder = {"value": -77}

    err = q8_0_dot_product_blocks_avx2_checked_default(lhs, rhs, -1, out_holder)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out_holder["value"] == -77


def test_default_wrapper_extent_shortfall_no_partial_write() -> None:
    out_holder = {"value": 999}
    err = q8_0_dot_product_blocks_avx2_checked_default([], [], 1, out_holder)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out_holder["value"] == 999


def test_default_wrapper_null_ptr_paths() -> None:
    lhs = _make_random_blocks(random.Random(9), 1)
    rhs = _make_random_blocks(random.Random(10), 1)
    out_holder = {"value": 5}

    err = q8_0_dot_product_blocks_avx2_checked_default(None, rhs, 0, out_holder)
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out_holder["value"] == 5

    err = q8_0_dot_product_blocks_avx2_checked_default(lhs, None, 0, out_holder)
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out_holder["value"] == 5

    err = q8_0_dot_product_blocks_avx2_checked_default(lhs, rhs, 0, None)
    assert err == Q8_0_AVX2_ERR_NULL_PTR


def test_default_wrapper_malformed_block_payload_matches_core_error() -> None:
    # Deliberately malformed payload lengths must surface BAD_LEN identically
    # for default wrapper and checked core path.
    lhs = [{"d_fp16": 0x3C00, "qs": [1] * 31}]
    rhs = [{"d_fp16": 0x3C00, "qs": [2] * 32}]
    out_default = {"value": 123}
    out_core = {"value": 456}

    err_default = q8_0_dot_product_blocks_avx2_checked_default(lhs, rhs, 1, out_default)
    err_core = q8_0_dot_product_blocks_avx2_checked_ptr(lhs, rhs, 1, out_core)

    assert err_default == Q8_0_AVX2_ERR_BAD_LEN
    assert err_default == err_core
    assert out_default["value"] == 123
    assert out_core["value"] == 456


def test_source_contains_default_wrapper() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0DotProductBlocksAVX2CheckedDefault(" in source
    assert "return Q8_0DotProductBlocksAVX2Checked(" in source


if __name__ == "__main__":
    test_default_wrapper_matches_checked_core_success_and_errors()
    test_default_wrapper_negative_count_no_partial_write()
    test_default_wrapper_extent_shortfall_no_partial_write()
    test_default_wrapper_null_ptr_paths()
    test_default_wrapper_malformed_block_payload_matches_core_error()
    test_source_contains_default_wrapper()
    print("q8_0_dot_avx2_checked_default_reference_checks=ok")
