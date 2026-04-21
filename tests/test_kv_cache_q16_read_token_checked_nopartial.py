#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ReadTokenCheckedNoPartial (IQ-873)."""

from __future__ import annotations

import random
from pathlib import Path

from test_kv_cache_q16_indexing_checked import (
    I64_MAX,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_ERR_OVERFLOW,
    KV_Q16_OK,
    kv_cache_q16_compute_layer_token_base_index_checked,
    try_add_i64_checked,
)


def kv_cache_q16_read_token_checked_nopartial(
    k_cache_q16: list[int] | None,
    k_cache_capacity: int,
    v_cache_q16: list[int] | None,
    v_cache_capacity: int,
    layer_idx: int,
    token_idx: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    k_token_out_q16: list[int] | None,
    k_token_out_capacity: int,
    v_token_out_q16: list[int] | None,
    v_token_out_capacity: int,
) -> int:
    if (
        k_cache_q16 is None
        or v_cache_q16 is None
        or k_token_out_q16 is None
        or v_token_out_q16 is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if (
        k_cache_capacity < 0
        or v_cache_capacity < 0
        or k_token_out_capacity < 0
        or v_token_out_capacity < 0
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        layer_idx < 0
        or token_idx < 0
        or layer_count < 0
        or token_capacity < 0
        or kv_heads < 0
        or head_dim < 0
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_token_out_q16 is k_cache_q16
        or k_token_out_q16 is v_cache_q16
        or v_token_out_q16 is k_cache_q16
        or v_token_out_q16 is v_cache_q16
        or k_token_out_q16 is v_token_out_q16
    ):
        return KV_Q16_ERR_BAD_PARAM

    k_base_index = [0]
    k_span_cells = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_base_index,
        k_span_cells,
    )
    if err != KV_Q16_OK:
        return err

    v_base_index = [0]
    v_span_cells = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        v_base_index,
        v_span_cells,
    )
    if err != KV_Q16_OK:
        return err

    if k_base_index[0] != v_base_index[0]:
        return KV_Q16_ERR_BAD_PARAM
    if k_span_cells[0] != v_span_cells[0]:
        return KV_Q16_ERR_BAD_PARAM

    err, k_end_index = try_add_i64_checked(k_base_index[0], k_span_cells[0])
    if err != KV_Q16_OK:
        return err

    err, v_end_index = try_add_i64_checked(v_base_index[0], v_span_cells[0])
    if err != KV_Q16_OK:
        return err

    if k_end_index > k_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if v_end_index > v_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM

    if k_span_cells[0] > k_token_out_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if v_span_cells[0] > v_token_out_capacity:
        return KV_Q16_ERR_BAD_PARAM

    for idx in range(k_span_cells[0]):
        k_token_out_q16[idx] = k_cache_q16[k_base_index[0] + idx]
        v_token_out_q16[idx] = v_cache_q16[v_base_index[0] + idx]

    return KV_Q16_OK


def explicit_read_composition(
    k_cache_q16: list[int] | None,
    k_cache_capacity: int,
    v_cache_q16: list[int] | None,
    v_cache_capacity: int,
    layer_idx: int,
    token_idx: int,
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    k_token_out_q16: list[int] | None,
    k_token_out_capacity: int,
    v_token_out_q16: list[int] | None,
    v_token_out_capacity: int,
) -> int:
    return kv_cache_q16_read_token_checked_nopartial(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity,
    )


def test_source_contains_read_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    signature = "I32 KVCacheQ16ReadTokenCheckedNoPartial("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "KVCacheQ16ComputeLayerTokenBaseIndexChecked" in body
    assert "if (k_end_index > k_cache_capacity)" in body
    assert "if (v_end_index > v_cache_capacity)" in body
    assert "while (cell_idx < k_span_cells)" in body


def test_known_vector_success_and_layout() -> None:
    layer_count = 2
    token_capacity = 4
    kv_heads = 3
    head_dim = 2
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [-(1000 + idx) for idx in range(total_cells)]
    v_cache = [2000 + idx for idx in range(total_cells)]

    k_out = [777] * span
    v_out = [888] * span

    layer_idx = 1
    token_idx = 2
    base = ((layer_idx * token_capacity) + token_idx) * span

    err = kv_cache_q16_read_token_checked_nopartial(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_out,
        len(k_out),
        v_out,
        len(v_out),
    )
    assert err == KV_Q16_OK

    assert k_out == k_cache[base : base + span]
    assert v_out == v_cache[base : base + span]


def test_null_alias_and_no_partial_failure() -> None:
    k_cache = [77] * 24
    v_cache = [88] * 24
    k_out = [0] * 6
    v_out = [0] * 6

    assert (
        kv_cache_q16_read_token_checked_nopartial(
            None,
            24,
            v_cache,
            24,
            0,
            0,
            1,
            4,
            2,
            3,
            k_out,
            6,
            v_out,
            6,
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_read_token_checked_nopartial(
            k_cache,
            24,
            v_cache,
            24,
            0,
            0,
            1,
            4,
            2,
            3,
            k_cache,
            24,
            v_out,
            6,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    before_k = k_out.copy()
    before_v = v_out.copy()
    err = kv_cache_q16_read_token_checked_nopartial(
        k_cache,
        24,
        v_cache,
        24,
        0,
        0,
        1,
        4,
        2,
        3,
        k_out,
        5,
        v_out,
        6,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert k_out == before_k
    assert v_out == before_v


def test_overflow_passthrough_from_index_helpers() -> None:
    k_cache = [1, 2, 3, 4]
    v_cache = [5, 6, 7, 8]
    k_out = [9, 10]
    v_out = [11, 12]

    err = kv_cache_q16_read_token_checked_nopartial(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=1,
        token_idx=0,
        layer_count=2,
        token_capacity=I64_MAX,
        kv_heads=2,
        head_dim=2,
        k_token_out_q16=k_out,
        k_token_out_capacity=len(k_out),
        v_token_out_q16=v_out,
        v_token_out_capacity=len(v_out),
    )
    assert err == KV_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(873)

    for _ in range(900):
        layer_count = rng.randint(1, 6)
        token_capacity = rng.randint(1, 12)
        kv_heads = rng.randint(1, 8)
        head_dim = rng.randint(1, 16)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        k_cache = [rng.randint(-32768, 32767) for _ in range(total_cells)]
        v_cache = [rng.randint(-32768, 32767) for _ in range(total_cells)]

        k_cache_ref = k_cache.copy()
        v_cache_ref = v_cache.copy()

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        k_out_new = [12345] * span
        v_out_new = [23456] * span
        k_out_ref = k_out_new.copy()
        v_out_ref = v_out_new.copy()

        err_new = kv_cache_q16_read_token_checked_nopartial(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_out_new,
            len(k_out_new),
            v_out_new,
            len(v_out_new),
        )

        err_ref = explicit_read_composition(
            k_cache_ref,
            len(k_cache_ref),
            v_cache_ref,
            len(v_cache_ref),
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_out_ref,
            len(k_out_ref),
            v_out_ref,
            len(v_out_ref),
        )

        assert err_new == err_ref
        assert k_cache == k_cache_ref
        assert v_cache == v_cache_ref
        assert k_out_new == k_out_ref
        assert v_out_new == v_out_ref
