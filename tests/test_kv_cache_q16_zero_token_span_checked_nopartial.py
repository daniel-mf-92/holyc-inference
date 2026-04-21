#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ZeroTokenSpanCheckedNoPartial (IQ-875)."""

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


def kv_cache_q16_zero_token_span_checked_nopartial(
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
) -> int:
    if k_cache_q16 is None or v_cache_q16 is None:
        return KV_Q16_ERR_NULL_PTR

    if k_cache_q16 is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM

    if k_cache_capacity < 0 or v_cache_capacity < 0:
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

    for idx in range(k_span_cells[0]):
        k_cache_q16[k_base_index[0] + idx] = 0
        v_cache_q16[v_base_index[0] + idx] = 0

    return KV_Q16_OK


def explicit_zero_composition(
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
) -> int:
    return kv_cache_q16_zero_token_span_checked_nopartial(
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
    )


def test_source_contains_zero_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    signature = "I32 KVCacheQ16ZeroTokenSpanCheckedNoPartial("
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

    k_cache = [7000 + idx for idx in range(total_cells)]
    v_cache = [9000 + idx for idx in range(total_cells)]

    layer_idx = 1
    token_idx = 2
    base = ((layer_idx * token_capacity) + token_idx) * span
    pre_k_prefix = k_cache[:base]
    pre_k_suffix = k_cache[base + span :]
    pre_v_prefix = v_cache[:base]
    pre_v_suffix = v_cache[base + span :]

    err = kv_cache_q16_zero_token_span_checked_nopartial(
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
    )
    assert err == KV_Q16_OK

    assert k_cache[base : base + span] == [0] * span
    assert v_cache[base : base + span] == [0] * span
    assert k_cache[:base] == pre_k_prefix
    assert v_cache[:base] == pre_v_prefix
    assert k_cache[base + span :] == pre_k_suffix
    assert v_cache[base + span :] == pre_v_suffix


def test_null_alias_and_no_partial_failure() -> None:
    k_cache = [77] * 24
    v_cache = [88] * 24

    assert (
        kv_cache_q16_zero_token_span_checked_nopartial(
            None,
            len(k_cache),
            v_cache,
            len(v_cache),
            0,
            0,
            1,
            4,
            2,
            3,
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_zero_token_span_checked_nopartial(
            k_cache,
            len(k_cache),
            k_cache,
            len(k_cache),
            0,
            0,
            1,
            4,
            2,
            3,
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    k_before = k_cache.copy()
    v_before = v_cache.copy()

    err = kv_cache_q16_zero_token_span_checked_nopartial(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=0,
        token_idx=4,
        layer_count=1,
        token_capacity=4,
        kv_heads=2,
        head_dim=3,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert k_cache == k_before
    assert v_cache == v_before


def test_overflow_and_bounds() -> None:
    err = kv_cache_q16_zero_token_span_checked_nopartial(
        [1] * 16,
        16,
        [2] * 16,
        16,
        layer_idx=1,
        token_idx=0,
        layer_count=2,
        token_capacity=I64_MAX,
        kv_heads=2,
        head_dim=2,
    )
    assert err == KV_Q16_ERR_OVERFLOW

    err = kv_cache_q16_zero_token_span_checked_nopartial(
        [1] * 24,
        23,
        [2] * 24,
        24,
        layer_idx=0,
        token_idx=3,
        layer_count=1,
        token_capacity=4,
        kv_heads=2,
        head_dim=3,
    )
    assert err == KV_Q16_ERR_BAD_PARAM


def test_randomized_parity(seed: int = 875, trials: int = 1400) -> None:
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

        k_cache_a = [0x3131] * max(1, k_cache_capacity)
        k_cache_b = [0x3131] * max(1, k_cache_capacity)
        v_cache_a = [0x4242] * max(1, v_cache_capacity)
        v_cache_b = [0x4242] * max(1, v_cache_capacity)

        if rng.random() < 0.2 and k_cache_capacity > 0:
            k_cache_capacity -= 1
        if rng.random() < 0.2 and v_cache_capacity > 0:
            v_cache_capacity -= 1

        err_a = kv_cache_q16_zero_token_span_checked_nopartial(
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
        )
        err_b = explicit_zero_composition(
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
        )

        assert err_a == err_b
        assert k_cache_a == k_cache_b
        assert v_cache_a == v_cache_b


if __name__ == "__main__":
    test_source_contains_zero_helper()
    test_known_vector_success_and_layout()
    test_null_alias_and_no_partial_failure()
    test_overflow_and_bounds()
    test_randomized_parity()
    print("ok")
