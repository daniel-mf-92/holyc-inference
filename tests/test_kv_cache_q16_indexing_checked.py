#!/usr/bin/env python3
"""Parity harness for KVCacheQ16*Checked helpers (IQ-871)."""

from __future__ import annotations

import random
from pathlib import Path

KV_Q16_OK = 0
KV_Q16_ERR_NULL_PTR = 1
KV_Q16_ERR_BAD_PARAM = 2
KV_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if lhs == 0 or rhs == 0:
        return KV_Q16_OK, 0

    if lhs > 0:
        if rhs > 0:
            if lhs > I64_MAX // rhs:
                return KV_Q16_ERR_OVERFLOW, 0
        else:
            if rhs < I64_MIN // lhs:
                return KV_Q16_ERR_OVERFLOW, 0
    else:
        if rhs > 0:
            if lhs < I64_MIN // rhs:
                return KV_Q16_ERR_OVERFLOW, 0
        else:
            if lhs != 0 and rhs < I64_MAX // lhs:
                return KV_Q16_ERR_OVERFLOW, 0

    return KV_Q16_OK, lhs * rhs


def try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return KV_Q16_ERR_OVERFLOW, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return KV_Q16_ERR_OVERFLOW, 0
    return KV_Q16_OK, lhs + rhs


def kv_cache_q16_compute_layer_token_span_cells_checked(
    kv_heads: int,
    head_dim: int,
    out_span_cells: list[int] | None,
) -> int:
    if out_span_cells is None:
        return KV_Q16_ERR_NULL_PTR
    if kv_heads < 0 or head_dim < 0:
        return KV_Q16_ERR_BAD_PARAM

    err, span = try_mul_i64_checked(kv_heads, head_dim)
    if err != KV_Q16_OK:
        return err

    out_span_cells[0] = span
    return KV_Q16_OK


def kv_cache_q16_compute_layer_token_base_index_checked(
    layer_idx: int,
    token_idx: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    out_base_index: list[int] | None,
    out_span_cells: list[int] | None,
) -> int:
    if out_base_index is None or out_span_cells is None:
        return KV_Q16_ERR_NULL_PTR
    if out_base_index is out_span_cells:
        return KV_Q16_ERR_BAD_PARAM

    if layer_idx < 0 or token_idx < 0:
        return KV_Q16_ERR_BAD_PARAM
    if layer_count < 0 or token_capacity < 0:
        return KV_Q16_ERR_BAD_PARAM

    span = [0]
    err = kv_cache_q16_compute_layer_token_span_cells_checked(kv_heads, head_dim, span)
    if err != KV_Q16_OK:
        return err

    if layer_idx >= layer_count:
        return KV_Q16_ERR_BAD_PARAM
    if token_idx >= token_capacity:
        return KV_Q16_ERR_BAD_PARAM

    err, layer_span_cells = try_mul_i64_checked(token_capacity, span[0])
    if err != KV_Q16_OK:
        return err

    err, layer_base_cells = try_mul_i64_checked(layer_idx, layer_span_cells)
    if err != KV_Q16_OK:
        return err

    err, token_base_cells = try_mul_i64_checked(token_idx, span[0])
    if err != KV_Q16_OK:
        return err

    err, base = try_add_i64_checked(layer_base_cells, token_base_cells)
    if err != KV_Q16_OK:
        return err

    out_base_index[0] = base
    out_span_cells[0] = span[0]
    return KV_Q16_OK


def explicit_base_formula(
    layer_idx: int,
    token_idx: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[int, int]:
    span = kv_heads * head_dim
    return ((layer_idx * token_capacity + token_idx) * span), span


def test_source_contains_kv_cache_index_helpers() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    assert "I32 KVCacheQ16ComputeLayerTokenSpanCellsChecked(" in source
    assert "I32 KVCacheQ16ComputeLayerTokenBaseIndexChecked(" in source
    assert "status = KVCacheQ16ComputeLayerTokenSpanCellsChecked" in source


def test_known_vector_base_and_span() -> None:
    out_base = [0]
    out_span = [0]

    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx=2,
        token_idx=17,
        layer_count=4,
        token_capacity=64,
        kv_heads=8,
        head_dim=128,
        out_base_index=out_base,
        out_span_cells=out_span,
    )
    assert err == KV_Q16_OK

    expected_base, expected_span = explicit_base_formula(2, 17, 64, 8, 128)
    assert out_span == [expected_span]
    assert out_base == [expected_base]


def test_null_alias_and_bounds() -> None:
    out_base = [0]
    out_span = [0]

    assert (
        kv_cache_q16_compute_layer_token_base_index_checked(
            0, 0, 1, 1, 1, 1, None, out_span
        )
        == KV_Q16_ERR_NULL_PTR
    )
    assert (
        kv_cache_q16_compute_layer_token_base_index_checked(
            0, 0, 1, 1, 1, 1, out_base, out_base
        )
        == KV_Q16_ERR_BAD_PARAM
    )
    assert (
        kv_cache_q16_compute_layer_token_base_index_checked(
            -1, 0, 1, 1, 1, 1, out_base, out_span
        )
        == KV_Q16_ERR_BAD_PARAM
    )
    assert (
        kv_cache_q16_compute_layer_token_base_index_checked(
            0, 1, 1, 1, 1, 1, out_base, out_span
        )
        == KV_Q16_ERR_BAD_PARAM
    )
    assert (
        kv_cache_q16_compute_layer_token_base_index_checked(
            1, 0, 1, 1, 1, 1, out_base, out_span
        )
        == KV_Q16_ERR_BAD_PARAM
    )


def test_overflow_surfaces() -> None:
    out_base = [111]
    out_span = [222]

    assert (
        kv_cache_q16_compute_layer_token_span_cells_checked(I64_MAX, 2, out_span)
        == KV_Q16_ERR_OVERFLOW
    )

    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx=1,
        token_idx=0,
        layer_count=2,
        token_capacity=I64_MAX,
        kv_heads=2,
        head_dim=2,
        out_base_index=out_base,
        out_span_cells=out_span,
    )
    assert err == KV_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_formula() -> None:
    rng = random.Random(871)

    for _ in range(800):
        layer_count = rng.randint(1, 32)
        token_capacity = rng.randint(1, 256)
        kv_heads = rng.randint(1, 64)
        head_dim = rng.randint(1, 256)

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        got_base = [0]
        got_span = [0]
        err = kv_cache_q16_compute_layer_token_base_index_checked(
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            got_base,
            got_span,
        )

        exp_base, exp_span = explicit_base_formula(
            layer_idx, token_idx, token_capacity, kv_heads, head_dim
        )

        if exp_span > I64_MAX or exp_base > I64_MAX:
            assert err == KV_Q16_ERR_OVERFLOW
            continue

        assert err == KV_Q16_OK
        assert got_span == [exp_span]
        assert got_base == [exp_base]


if __name__ == "__main__":
    test_source_contains_kv_cache_index_helpers()
    test_known_vector_base_and_span()
    test_null_alias_and_bounds()
    test_overflow_surfaces()
    test_randomized_parity_vs_explicit_formula()
