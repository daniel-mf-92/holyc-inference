#!/usr/bin/env python3
"""Parity harness for KVCacheQ16WriteTokenCheckedNoPartial (IQ-872)."""

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
    kv_cache_q16_compute_layer_token_base_index_checked,
    try_add_i64_checked,
)


def kv_cache_q16_write_token_checked_nopartial(
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
) -> int:
    if (
        k_cache_q16 is None
        or v_cache_q16 is None
        or k_token_src_q16 is None
        or v_token_src_q16 is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if k_cache_q16 is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_cache_capacity < 0
        or v_cache_capacity < 0
        or k_token_src_capacity < 0
        or v_token_src_capacity < 0
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
        k_token_src_q16 is k_cache_q16
        or k_token_src_q16 is v_cache_q16
        or v_token_src_q16 is k_cache_q16
        or v_token_src_q16 is v_cache_q16
        or k_token_src_q16 is v_token_src_q16
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

    if k_span_cells[0] > k_token_src_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if v_span_cells[0] > v_token_src_capacity:
        return KV_Q16_ERR_BAD_PARAM

    for idx in range(k_span_cells[0]):
        k_cache_q16[k_base_index[0] + idx] = k_token_src_q16[idx]
        v_cache_q16[v_base_index[0] + idx] = v_token_src_q16[idx]

    return KV_Q16_OK


def explicit_write_composition(
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
) -> int:
    return kv_cache_q16_write_token_checked_nopartial(
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


def test_source_contains_write_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    signature = "I32 KVCacheQ16WriteTokenCheckedNoPartial("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "KVCacheQ16ComputeLayerTokenBaseIndexChecked" in body
    assert "if (k_end_index > k_cache_capacity)" in body
    assert "if (v_end_index > v_cache_capacity)" in body
    assert "while (cell_idx < k_span_cells)" in body


def test_known_vector_success_and_layout() -> None:
    layer_count = 3
    token_capacity = 5
    kv_heads = 2
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [-1] * total_cells
    v_cache = [-2] * total_cells

    k_src = [101 + idx for idx in range(span)]
    v_src = [201 + idx for idx in range(span)]

    err = kv_cache_q16_write_token_checked_nopartial(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=1,
        token_idx=3,
        layer_count=layer_count,
        token_capacity=token_capacity,
        kv_heads=kv_heads,
        head_dim=head_dim,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
    )
    assert err == KV_Q16_OK

    base = ((1 * token_capacity) + 3) * span
    assert k_cache[base : base + span] == k_src
    assert v_cache[base : base + span] == v_src

    untouched_prefix = list(range(0, base))
    assert all(k_cache[idx] == -1 for idx in untouched_prefix)
    assert all(v_cache[idx] == -2 for idx in untouched_prefix)


def test_null_alias_and_no_partial_failure() -> None:
    k_cache = [77] * 16
    v_cache = [88] * 16
    k_src = [11] * 4
    v_src = [22] * 4

    assert (
        kv_cache_q16_write_token_checked_nopartial(
            None,
            16,
            v_cache,
            16,
            0,
            0,
            1,
            4,
            1,
            4,
            k_src,
            4,
            v_src,
            4,
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_write_token_checked_nopartial(
            k_cache,
            16,
            v_cache,
            16,
            0,
            0,
            1,
            4,
            1,
            4,
            k_cache,
            16,
            v_src,
            4,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    assert (
        kv_cache_q16_write_token_checked_nopartial(
            k_cache,
            16,
            k_cache,
            16,
            0,
            0,
            1,
            4,
            1,
            4,
            k_src,
            4,
            v_src,
            4,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    k_before = k_cache.copy()
    v_before = v_cache.copy()

    err = kv_cache_q16_write_token_checked_nopartial(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=0,
        token_idx=3,
        layer_count=1,
        token_capacity=4,
        kv_heads=1,
        head_dim=4,
        k_token_src_q16=k_src,
        k_token_src_capacity=3,
        v_token_src_q16=v_src,
        v_token_src_capacity=4,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert k_cache == k_before
    assert v_cache == v_before


def test_overflow_and_bounds() -> None:
    small = [0] * 8

    err = kv_cache_q16_write_token_checked_nopartial(
        small,
        len(small),
        [0] * 8,
        8,
        layer_idx=1,
        token_idx=0,
        layer_count=2,
        token_capacity=I64_MAX,
        kv_heads=2,
        head_dim=2,
        k_token_src_q16=[1, 2, 3, 4],
        k_token_src_capacity=4,
        v_token_src_q16=[5, 6, 7, 8],
        v_token_src_capacity=4,
    )
    assert err == KV_Q16_ERR_OVERFLOW

    err = kv_cache_q16_write_token_checked_nopartial(
        [0] * 16,
        15,
        [0] * 16,
        16,
        layer_idx=0,
        token_idx=3,
        layer_count=1,
        token_capacity=4,
        kv_heads=1,
        head_dim=4,
        k_token_src_q16=[1, 2, 3, 4],
        k_token_src_capacity=4,
        v_token_src_q16=[5, 6, 7, 8],
        v_token_src_capacity=4,
    )
    assert err == KV_Q16_ERR_BAD_PARAM


def test_randomized_parity(seed: int = 872, trials: int = 1200) -> None:
    rng = random.Random(seed)

    for _ in range(trials):
        layer_count = rng.randint(0, 6)
        token_capacity = rng.randint(0, 8)
        kv_heads = rng.randint(0, 6)
        head_dim = rng.randint(0, 8)

        layer_idx = rng.randint(0, max(layer_count, 1))
        token_idx = rng.randint(0, max(token_capacity, 1))

        span = kv_heads * head_dim
        total = layer_count * token_capacity * span

        cap_pad = rng.randint(0, 4)
        k_cache_capacity = total + cap_pad
        v_cache_capacity = total + cap_pad

        k_cache_a = [0x1111] * max(1, k_cache_capacity)
        k_cache_b = [0x1111] * max(1, k_cache_capacity)
        v_cache_a = [0x2222] * max(1, v_cache_capacity)
        v_cache_b = [0x2222] * max(1, v_cache_capacity)

        src_len = span + rng.randint(0, 3)
        k_src = [rng.randint(-32000, 32000) for _ in range(max(src_len, 1))]
        v_src = [rng.randint(-32000, 32000) for _ in range(max(src_len, 1))]

        k_src_capacity = src_len
        v_src_capacity = src_len
        if rng.random() < 0.2 and span > 0:
            k_src_capacity = span - 1
        if rng.random() < 0.2 and span > 0:
            v_src_capacity = span - 1

        err_a = kv_cache_q16_write_token_checked_nopartial(
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
        )
        err_b = explicit_write_composition(
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
        )

        assert err_a == err_b
        assert k_cache_a == k_cache_b
        assert v_cache_a == v_cache_b


if __name__ == "__main__":
    test_source_contains_write_helper()
    test_known_vector_success_and_layout()
    test_null_alias_and_no_partial_failure()
    test_overflow_and_bounds()
    test_randomized_parity()
    print("ok")
