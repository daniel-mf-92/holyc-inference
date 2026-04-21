#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ComputeTotalCellsChecked (IQ-874)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_indexing_checked import (
    I64_MAX,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_ERR_OVERFLOW,
    KV_Q16_OK,
    kv_cache_q16_compute_layer_token_span_cells_checked,
    try_mul_i64_checked,
)


def kv_cache_q16_compute_total_cells_checked(
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    out_total_cells: list[int] | None,
) -> int:
    if out_total_cells is None:
        return KV_Q16_ERR_NULL_PTR

    if layer_count < 0 or token_capacity < 0 or kv_heads < 0 or head_dim < 0:
        return KV_Q16_ERR_BAD_PARAM

    span_cells = [0]
    err = kv_cache_q16_compute_layer_token_span_cells_checked(kv_heads, head_dim, span_cells)
    if err != KV_Q16_OK:
        return err

    err, layer_span_cells = try_mul_i64_checked(token_capacity, span_cells[0])
    if err != KV_Q16_OK:
        return err

    err, total_cells = try_mul_i64_checked(layer_count, layer_span_cells)
    if err != KV_Q16_OK:
        return err

    out_total_cells[0] = total_cells
    return KV_Q16_OK


def explicit_total_cells_formula(layer_count: int, token_capacity: int, kv_heads: int, head_dim: int) -> tuple[int, int]:
    span_cells = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span_cells
    if span_cells > I64_MAX or total_cells > I64_MAX:
        return KV_Q16_ERR_OVERFLOW, 0
    return KV_Q16_OK, total_cells


def test_source_contains_total_cells_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    signature = "I32 KVCacheQ16ComputeTotalCellsChecked("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "KVCacheQ16ComputeLayerTokenSpanCellsChecked" in body
    assert "KVTryMulI64Checked(token_capacity" in body
    assert "KVTryMulI64Checked(layer_count" in body


def test_known_vector_total_cells() -> None:
    out_total = [0]
    err = kv_cache_q16_compute_total_cells_checked(32, 2048, 32, 128, out_total)
    assert err == KV_Q16_OK
    assert out_total == [32 * 2048 * 32 * 128]


def test_null_and_bad_param_rejections() -> None:
    out_total = [111]
    assert kv_cache_q16_compute_total_cells_checked(1, 1, 1, 1, None) == KV_Q16_ERR_NULL_PTR
    assert kv_cache_q16_compute_total_cells_checked(-1, 1, 1, 1, out_total) == KV_Q16_ERR_BAD_PARAM
    assert kv_cache_q16_compute_total_cells_checked(1, -1, 1, 1, out_total) == KV_Q16_ERR_BAD_PARAM
    assert kv_cache_q16_compute_total_cells_checked(1, 1, -1, 1, out_total) == KV_Q16_ERR_BAD_PARAM
    assert kv_cache_q16_compute_total_cells_checked(1, 1, 1, -1, out_total) == KV_Q16_ERR_BAD_PARAM


def test_overflow_detection() -> None:
    out_total = [333]
    assert kv_cache_q16_compute_total_cells_checked(I64_MAX, I64_MAX, 2, 2, out_total) == KV_Q16_ERR_OVERFLOW
    assert kv_cache_q16_compute_total_cells_checked(2, I64_MAX, 2, 2, out_total) == KV_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_formula() -> None:
    rng = random.Random(874)
    for _ in range(1200):
        layer_count = rng.randint(0, 256)
        token_capacity = rng.randint(0, 8192)
        kv_heads = rng.randint(0, 256)
        head_dim = rng.randint(0, 512)
        got_total = [0]
        err = kv_cache_q16_compute_total_cells_checked(layer_count, token_capacity, kv_heads, head_dim, got_total)
        exp_err, exp_total = explicit_total_cells_formula(layer_count, token_capacity, kv_heads, head_dim)
        assert err == exp_err
        if err == KV_Q16_OK:
            assert got_total == [exp_total]


def test_composition_parity_with_span_helper() -> None:
    rng = random.Random(875)
    for _ in range(900):
        layer_count = rng.randint(0, 96)
        token_capacity = rng.randint(0, 4096)
        kv_heads = rng.randint(0, 96)
        head_dim = rng.randint(0, 256)

        got_total = [0]
        err_total = kv_cache_q16_compute_total_cells_checked(layer_count, token_capacity, kv_heads, head_dim, got_total)

        span = [0]
        err_span = kv_cache_q16_compute_layer_token_span_cells_checked(kv_heads, head_dim, span)
        if err_span != KV_Q16_OK:
            assert err_total == err_span
            continue

        err_layer_span, layer_span = try_mul_i64_checked(token_capacity, span[0])
        if err_layer_span != KV_Q16_OK:
            assert err_total == err_layer_span
            continue

        err_expected, expected_total = try_mul_i64_checked(layer_count, layer_span)
        assert err_total == err_expected
        if err_total == KV_Q16_OK:
            assert got_total == [expected_total]


if __name__ == "__main__":
    test_source_contains_total_cells_helper()
    test_known_vector_total_cells()
    test_null_and_bad_param_rejections()
    test_overflow_detection()
    test_randomized_parity_vs_explicit_formula()
    test_composition_parity_with_span_helper()
    print("ok")
