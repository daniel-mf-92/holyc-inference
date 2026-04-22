#!/usr/bin/env python3
"""Parity harness for KVCacheInitQ16CheckedNoAlloc (IQ-1146)."""

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
from test_kv_cache_q16_total_cells_checked import (
    kv_cache_q16_compute_total_cells_checked,
)


def kv_cache_q16_init_checked_noalloc(
    k_cache_q16: list[int] | None,
    k_cache_capacity: int,
    v_cache_q16: list[int] | None,
    v_cache_capacity: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    out_used_tokens: list[int] | None,
    out_layer_span_cells: list[int] | None,
    out_total_cells: list[int] | None,
) -> int:
    if (
        k_cache_q16 is None
        or v_cache_q16 is None
        or out_used_tokens is None
        or out_layer_span_cells is None
        or out_total_cells is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if k_cache_q16 is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_used_tokens is out_layer_span_cells
        or out_used_tokens is out_total_cells
        or out_layer_span_cells is out_total_cells
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_used_tokens is k_cache_q16
        or out_used_tokens is v_cache_q16
        or out_layer_span_cells is k_cache_q16
        or out_layer_span_cells is v_cache_q16
        or out_total_cells is k_cache_q16
        or out_total_cells is v_cache_q16
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_cache_capacity < 0
        or v_cache_capacity < 0
        or layer_count < 0
        or token_capacity < 0
        or kv_heads < 0
        or head_dim < 0
    ):
        return KV_Q16_ERR_BAD_PARAM

    snapshot = (
        k_cache_capacity,
        v_cache_capacity,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    )

    staged_span = [0]
    err = kv_cache_q16_compute_layer_token_span_cells_checked(
        kv_heads,
        head_dim,
        staged_span,
    )
    if err != KV_Q16_OK:
        return err

    err, staged_layer_span = try_mul_i64_checked(token_capacity, staged_span[0])
    if err != KV_Q16_OK:
        return err

    staged_total = [0]
    err = kv_cache_q16_compute_total_cells_checked(
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        staged_total,
    )
    if err != KV_Q16_OK:
        return err

    if staged_total[0] > k_cache_capacity or staged_total[0] > v_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM

    if snapshot != (
        k_cache_capacity,
        v_cache_capacity,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    ):
        return KV_Q16_ERR_BAD_PARAM

    out_used_tokens[0] = 0
    out_layer_span_cells[0] = staged_layer_span
    out_total_cells[0] = staged_total[0]
    return KV_Q16_OK


def explicit_init_formula(
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[int, int]:
    span = token_capacity * kv_heads * head_dim
    total = layer_count * span
    return span, total


def test_source_contains_init_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheInitQ16CheckedNoAlloc("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheQ16ComputeLayerTokenSpanCellsChecked" in body
    assert "KVCacheQ16ComputeTotalCellsChecked" in body
    assert "*out_used_tokens = 0;" in body
    assert "*out_layer_span_cells = staged_layer_span_cells;" in body
    assert "*out_total_cells = staged_total_cells;" in body


def test_known_vector_success() -> None:
    layer_count = 4
    token_capacity = 16
    kv_heads = 8
    head_dim = 64

    expected_layer_span, expected_total = explicit_init_formula(
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    )

    k_cache = [0] * expected_total
    v_cache = [0] * expected_total

    out_used_tokens = [999]
    out_layer_span = [999]
    out_total = [999]

    err = kv_cache_q16_init_checked_noalloc(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        out_used_tokens,
        out_layer_span,
        out_total,
    )
    assert err == KV_Q16_OK
    assert out_used_tokens == [0]
    assert out_layer_span == [expected_layer_span]
    assert out_total == [expected_total]


def test_null_alias_and_capacity_errors() -> None:
    k_cache = [0] * 64
    v_cache = [0] * 64
    out_a = [7]
    out_b = [8]
    out_c = [9]

    assert (
        kv_cache_q16_init_checked_noalloc(
            None,
            64,
            v_cache,
            64,
            1,
            1,
            1,
            1,
            out_a,
            out_b,
            out_c,
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_init_checked_noalloc(
            k_cache,
            64,
            k_cache,
            64,
            1,
            1,
            1,
            1,
            out_a,
            out_b,
            out_c,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    assert (
        kv_cache_q16_init_checked_noalloc(
            k_cache,
            -1,
            v_cache,
            64,
            1,
            1,
            1,
            1,
            out_a,
            out_b,
            out_c,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    assert (
        kv_cache_q16_init_checked_noalloc(
            k_cache,
            64,
            v_cache,
            64,
            1,
            65,
            1,
            1,
            out_a,
            out_b,
            out_c,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    assert (
        kv_cache_q16_init_checked_noalloc(
            k_cache,
            64,
            v_cache,
            64,
            1,
            1,
            1,
            1,
            out_a,
            out_a,
            out_c,
        )
        == KV_Q16_ERR_BAD_PARAM
    )


def test_overflow_and_randomized_parity() -> None:
    dummy_k = [0] * 8
    dummy_v = [0] * 8

    out_used = [0]
    out_layer_span = [0]
    out_total = [0]

    assert (
        kv_cache_q16_init_checked_noalloc(
            dummy_k,
            8,
            dummy_v,
            8,
            1,
            I64_MAX,
            2,
            1,
            out_used,
            out_layer_span,
            out_total,
        )
        == KV_Q16_ERR_OVERFLOW
    )

    rng = random.Random(1146)
    for _ in range(1000):
        layer_count = rng.randint(0, 24)
        token_capacity = rng.randint(0, 64)
        kv_heads = rng.randint(0, 32)
        head_dim = rng.randint(0, 128)

        exp_layer_span = token_capacity * kv_heads * head_dim
        exp_total = layer_count * exp_layer_span

        if exp_total > I64_MAX:
            k_cap = 10
            v_cap = 10
            expect = KV_Q16_ERR_OVERFLOW
        else:
            slack = rng.randint(0, 64)
            k_cap = max(0, exp_total + rng.randint(-2, slack))
            v_cap = max(0, exp_total + rng.randint(-2, slack))
            expect = KV_Q16_OK if (k_cap >= exp_total and v_cap >= exp_total) else KV_Q16_ERR_BAD_PARAM

        k_cache = [0] * max(1, k_cap)
        v_cache = [0] * max(1, v_cap)
        out_used = [111]
        out_layer_span = [222]
        out_total = [333]

        err = kv_cache_q16_init_checked_noalloc(
            k_cache,
            k_cap,
            v_cache,
            v_cap,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            out_used,
            out_layer_span,
            out_total,
        )

        assert err == expect
        if err == KV_Q16_OK:
            assert out_used == [0]
            assert out_layer_span == [exp_layer_span]
            assert out_total == [exp_total]


if __name__ == "__main__":
    test_source_contains_init_helper()
    test_known_vector_success()
    test_null_alias_and_capacity_errors()
    test_overflow_and_randomized_parity()
