#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartial (IQ-877)."""

from __future__ import annotations

import random
from pathlib import Path

from test_kv_cache_q16_indexing_checked import (
    I64_MAX,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_ERR_OVERFLOW,
    KV_Q16_OK,
    kv_cache_q16_compute_layer_token_span_cells_checked,
)
from test_kv_cache_q16_read_token_checked_nopartial import (
    kv_cache_q16_read_token_checked_nopartial,
)
from test_kv_cache_q16_write_token_checked_nopartial import (
    kv_cache_q16_write_token_checked_nopartial,
)


def kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
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
    k_token_src_q16: list[int] | None,
    k_token_src_capacity: int,
    v_token_src_q16: list[int] | None,
    v_token_src_capacity: int,
    k_token_out_q16: list[int] | None,
    k_token_out_capacity: int,
    v_token_out_q16: list[int] | None,
    v_token_out_capacity: int,
) -> int:
    if (
        k_cache_q16 is None
        or v_cache_q16 is None
        or k_token_src_q16 is None
        or v_token_src_q16 is None
        or k_token_out_q16 is None
        or v_token_out_q16 is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if k_cache_q16 is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM

    if k_token_src_q16 is v_token_src_q16:
        return KV_Q16_ERR_BAD_PARAM
    if k_token_out_q16 is v_token_out_q16:
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_token_src_q16 is k_token_out_q16
        or k_token_src_q16 is v_token_out_q16
        or v_token_src_q16 is k_token_out_q16
        or v_token_src_q16 is v_token_out_q16
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_cache_capacity < 0
        or v_cache_capacity < 0
        or k_token_src_capacity < 0
        or v_token_src_capacity < 0
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

    err = kv_cache_q16_write_token_checked_nopartial(
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
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
    )
    if err != KV_Q16_OK:
        return err

    err = kv_cache_q16_read_token_checked_nopartial(
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
    if err != KV_Q16_OK:
        return err

    span_cells = [0]
    err = kv_cache_q16_compute_layer_token_span_cells_checked(
        kv_heads,
        head_dim,
        span_cells,
    )
    if err != KV_Q16_OK:
        return err

    if span_cells[0] > k_token_src_capacity or span_cells[0] > v_token_src_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if span_cells[0] > k_token_out_capacity or span_cells[0] > v_token_out_capacity:
        return KV_Q16_ERR_BAD_PARAM

    for idx in range(span_cells[0]):
        if k_token_out_q16[idx] != k_token_src_q16[idx]:
            return KV_Q16_ERR_BAD_PARAM
        if v_token_out_q16[idx] != v_token_src_q16[idx]:
            return KV_Q16_ERR_BAD_PARAM

    return KV_Q16_OK


def explicit_roundtrip_composition(
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
    k_token_src_q16: list[int] | None,
    k_token_src_capacity: int,
    v_token_src_q16: list[int] | None,
    v_token_src_capacity: int,
    k_token_out_q16: list[int] | None,
    k_token_out_capacity: int,
    v_token_out_q16: list[int] | None,
    v_token_out_capacity: int,
) -> int:
    return kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
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
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity,
    )


def test_source_contains_roundtrip_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    signature = "I32 KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartial("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "KVCacheQ16WriteTokenCheckedNoPartial" in body
    assert "KVCacheQ16ReadTokenCheckedNoPartial" in body
    assert "while (cell_idx < span_cells)" in body
    assert "if (k_token_out_q16[cell_idx] != k_token_src_q16[cell_idx])" in body


def test_known_vector_success_and_roundtrip() -> None:
    layer_count = 3
    token_capacity = 4
    kv_heads = 2
    head_dim = 3
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [-111] * total_cells
    v_cache = [-222] * total_cells

    layer_idx = 2
    token_idx = 1
    base = ((layer_idx * token_capacity) + token_idx) * span

    k_src = [500 + idx for idx in range(span)]
    v_src = [900 + idx for idx in range(span)]
    k_out = [77] * span
    v_out = [88] * span

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
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
        k_src,
        len(k_src),
        v_src,
        len(v_src),
        k_out,
        len(k_out),
        v_out,
        len(v_out),
    )
    assert err == KV_Q16_OK

    assert k_cache[base : base + span] == k_src
    assert v_cache[base : base + span] == v_src
    assert k_out == k_src
    assert v_out == v_src


def test_alias_and_null_guards() -> None:
    k_cache = [0] * 16
    v_cache = [0] * 16
    k_src = [1, 2, 3, 4]
    v_src = [5, 6, 7, 8]
    k_out = [0, 0, 0, 0]
    v_out = [0, 0, 0, 0]

    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
            None,
            16,
            v_cache,
            16,
            0,
            0,
            1,
            2,
            2,
            2,
            k_src,
            4,
            v_src,
            4,
            k_out,
            4,
            v_out,
            4,
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
            k_cache,
            16,
            k_cache,
            16,
            0,
            0,
            1,
            2,
            2,
            2,
            k_src,
            4,
            v_src,
            4,
            k_out,
            4,
            v_out,
            4,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
            k_cache,
            16,
            v_cache,
            16,
            0,
            0,
            1,
            2,
            2,
            2,
            k_src,
            4,
            v_src,
            4,
            k_src,
            4,
            v_out,
            4,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
            k_cache,
            16,
            v_cache,
            16,
            0,
            0,
            1,
            2,
            2,
            2,
            k_src,
            4,
            v_src,
            4,
            k_out,
            4,
            k_out,
            4,
        )
        == KV_Q16_ERR_BAD_PARAM
    )


def test_overflow_and_capacity_propagation() -> None:
    k_cache = [7] * 64
    v_cache = [9] * 64
    k_src = [11] * 8
    v_src = [13] * 8
    k_out = [0] * 8
    v_out = [0] * 8

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=1,
        token_idx=0,
        layer_count=2,
        token_capacity=I64_MAX,
        kv_heads=2,
        head_dim=4,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        k_token_out_q16=k_out,
        k_token_out_capacity=len(k_out),
        v_token_out_q16=v_out,
        v_token_out_capacity=len(v_out),
    )
    assert err == KV_Q16_ERR_OVERFLOW

    k_before = k_cache.copy()
    v_before = v_cache.copy()
    k_out_before = k_out.copy()
    v_out_before = v_out.copy()

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=0,
        token_idx=0,
        layer_count=2,
        token_capacity=4,
        kv_heads=2,
        head_dim=4,
        k_token_src_q16=k_src,
        k_token_src_capacity=3,
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        k_token_out_q16=k_out,
        k_token_out_capacity=len(k_out),
        v_token_out_q16=v_out,
        v_token_out_capacity=len(v_out),
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert k_cache == k_before
    assert v_cache == v_before
    assert k_out == k_out_before
    assert v_out == v_out_before


def test_randomized_parity(seed: int = 877, trials: int = 1600) -> None:
    rng = random.Random(seed)

    for _ in range(trials):
        layer_count = rng.randint(0, 5)
        token_capacity = rng.randint(0, 8)
        kv_heads = rng.randint(0, 6)
        head_dim = rng.randint(0, 6)

        layer_idx = rng.randint(0, max(layer_count, 1))
        token_idx = rng.randint(0, max(token_capacity, 1))

        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        k_cache_capacity = total_cells + rng.randint(0, 3)
        v_cache_capacity = total_cells + rng.randint(0, 3)

        if rng.random() < 0.18 and k_cache_capacity > 0:
            k_cache_capacity -= 1
        if rng.random() < 0.18 and v_cache_capacity > 0:
            v_cache_capacity -= 1

        k_cache_a = [0x1010] * max(1, total_cells + 3)
        v_cache_a = [0x2020] * max(1, total_cells + 3)
        k_cache_b = k_cache_a.copy()
        v_cache_b = v_cache_a.copy()

        src_pad = rng.randint(0, 3)
        out_pad = rng.randint(0, 3)
        k_src_capacity = max(0, span + src_pad - (1 if rng.random() < 0.2 else 0))
        v_src_capacity = max(0, span + src_pad - (1 if rng.random() < 0.2 else 0))
        k_out_capacity = max(0, span + out_pad - (1 if rng.random() < 0.2 else 0))
        v_out_capacity = max(0, span + out_pad - (1 if rng.random() < 0.2 else 0))

        k_src = [3000 + i for i in range(max(span, k_src_capacity, 1) + 2)]
        v_src = [4000 + i for i in range(max(span, v_src_capacity, 1) + 2)]
        k_out_a = [0x3030] * max(k_out_capacity, 1)
        v_out_a = [0x4040] * max(v_out_capacity, 1)
        k_out_b = k_out_a.copy()
        v_out_b = v_out_a.copy()

        if rng.random() < 0.08:
            k_src = v_src
        if rng.random() < 0.08:
            k_out_a = v_out_a
            k_out_b = v_out_b
        if rng.random() < 0.08:
            k_src = k_out_a

        err_a = kv_cache_q16_read_write_token_roundtrip_checked_nopartial(
            k_cache_a,
            k_cache_capacity,
            v_cache_a,
            v_cache_capacity,
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            k_src_capacity,
            v_src,
            v_src_capacity,
            k_out_a,
            k_out_capacity,
            v_out_a,
            v_out_capacity,
        )

        err_b = explicit_roundtrip_composition(
            k_cache_b,
            k_cache_capacity,
            v_cache_b,
            v_cache_capacity,
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_src,
            k_src_capacity,
            v_src,
            v_src_capacity,
            k_out_b,
            k_out_capacity,
            v_out_b,
            v_out_capacity,
        )

        assert err_a == err_b
        assert k_cache_a == k_cache_b
        assert v_cache_a == v_cache_b
        assert k_out_a == k_out_b
        assert v_out_a == v_out_b
