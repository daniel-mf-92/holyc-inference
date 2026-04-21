#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnly (IQ-878)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_indexing_checked import (
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_ERR_OVERFLOW,
    KV_Q16_OK,
    kv_cache_q16_compute_layer_token_base_index_checked,
    try_add_i64_checked,
)
from test_kv_cache_q16_zero_token_span_checked_nopartial import (
    kv_cache_q16_zero_token_span_checked_nopartial,
)


def kv_cache_q16_zero_token_span_checked_nopartial_commit_only(
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
    out_base_index: list[int] | None,
    out_span_cells: list[int] | None,
    out_end_index: list[int] | None,
) -> int:
    if out_base_index is None or out_span_cells is None or out_end_index is None:
        return KV_Q16_ERR_NULL_PTR

    if out_base_index is out_span_cells or out_base_index is out_end_index or out_span_cells is out_end_index:
        return KV_Q16_ERR_BAD_PARAM

    if k_cache_q16 is None or v_cache_q16 is None:
        return KV_Q16_ERR_NULL_PTR
    if k_cache_q16 is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_cache_capacity < 0
        or v_cache_capacity < 0
        or layer_idx < 0
        or token_idx < 0
        or layer_count < 0
        or token_capacity < 0
        or kv_heads < 0
        or head_dim < 0
    ):
        return KV_Q16_ERR_BAD_PARAM

    if out_base_index is k_cache_q16 or out_base_index is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM
    if out_span_cells is k_cache_q16 or out_span_cells is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM
    if out_end_index is k_cache_q16 or out_end_index is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM

    snapshot = (
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    )

    k_base = [0]
    k_span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_base,
        k_span,
    )
    if err != KV_Q16_OK:
        return err

    v_base = [0]
    v_span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        v_base,
        v_span,
    )
    if err != KV_Q16_OK:
        return err

    if k_base[0] != v_base[0] or k_span[0] != v_span[0]:
        return KV_Q16_ERR_BAD_PARAM

    err, staged_end_index = try_add_i64_checked(k_base[0], k_span[0])
    if err != KV_Q16_OK:
        return err

    if staged_end_index > k_cache_capacity or staged_end_index > v_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM

    if snapshot != (
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    ):
        return KV_Q16_ERR_BAD_PARAM

    err = kv_cache_q16_zero_token_span_checked_nopartial(
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
    if err != KV_Q16_OK:
        return err

    out_base_index[0] = k_base[0]
    out_span_cells[0] = k_span[0]
    out_end_index[0] = staged_end_index
    return KV_Q16_OK


def explicit_commit_composition(
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
    out_base_index: list[int] | None,
    out_span_cells: list[int] | None,
    out_end_index: list[int] | None,
) -> int:
    return kv_cache_q16_zero_token_span_checked_nopartial_commit_only(
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
        out_base_index,
        out_span_cells,
        out_end_index,
    )


def test_source_contains_commit_only_zero_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    signature = "I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "KVCacheQ16ComputeLayerTokenBaseIndexChecked" in body
    assert "KVTryAddI64Checked" in body
    assert "KVCacheQ16ZeroTokenSpanCheckedNoPartial(" in body
    assert "if (staged_end_index > k_cache_capacity)" in body
    assert "if (staged_end_index > v_cache_capacity)" in body


def test_known_vector_commit_and_diagnostics() -> None:
    layer_count = 3
    token_capacity = 5
    kv_heads = 2
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [5000 + idx for idx in range(total_cells)]
    v_cache = [9000 + idx for idx in range(total_cells)]

    layer_idx = 1
    token_idx = 3
    expected_base = ((layer_idx * token_capacity) + token_idx) * span

    k_prefix = k_cache[:expected_base]
    k_suffix = k_cache[expected_base + span :]
    v_prefix = v_cache[:expected_base]
    v_suffix = v_cache[expected_base + span :]

    out_base = [0]
    out_span = [0]
    out_end = [0]

    err = kv_cache_q16_zero_token_span_checked_nopartial_commit_only(
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
        out_base,
        out_span,
        out_end,
    )
    assert err == KV_Q16_OK

    assert out_base == [expected_base]
    assert out_span == [span]
    assert out_end == [expected_base + span]

    assert k_cache[expected_base : expected_base + span] == [0] * span
    assert v_cache[expected_base : expected_base + span] == [0] * span
    assert k_cache[:expected_base] == k_prefix
    assert v_cache[:expected_base] == v_prefix
    assert k_cache[expected_base + span :] == k_suffix
    assert v_cache[expected_base + span :] == v_suffix


def test_null_alias_and_no_partial_failure() -> None:
    k_cache = [11] * 32
    v_cache = [22] * 32

    assert (
        kv_cache_q16_zero_token_span_checked_nopartial_commit_only(
            None,
            len(k_cache),
            v_cache,
            len(v_cache),
            0,
            0,
            1,
            4,
            2,
            4,
            [0],
            [0],
            [0],
        )
        == KV_Q16_ERR_NULL_PTR
    )

    shared = [0]
    assert (
        kv_cache_q16_zero_token_span_checked_nopartial_commit_only(
            k_cache,
            len(k_cache),
            v_cache,
            len(v_cache),
            0,
            0,
            1,
            4,
            2,
            4,
            shared,
            shared,
            [0],
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    k_before = k_cache.copy()
    v_before = v_cache.copy()
    out_base = [123]
    out_span = [456]
    out_end = [789]

    err = kv_cache_q16_zero_token_span_checked_nopartial_commit_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=0,
        token_idx=99,
        layer_count=1,
        token_capacity=4,
        kv_heads=2,
        head_dim=4,
        out_base_index=out_base,
        out_span_cells=out_span,
        out_end_index=out_end,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert k_cache == k_before
    assert v_cache == v_before
    assert out_base == [123]
    assert out_span == [456]
    assert out_end == [789]


def test_overflow_passthrough_from_checked_add() -> None:
    # Large dimensions force checked multiplication/addition overflow in base/span math.
    out_base = [0]
    out_span = [0]
    out_end = [0]

    err = kv_cache_q16_zero_token_span_checked_nopartial_commit_only(
        [0] * 8,
        8,
        [0] * 8,
        8,
        layer_idx=1,
        token_idx=1,
        layer_count=2,
        token_capacity=2**62,
        kv_heads=2**30,
        head_dim=2**30,
        out_base_index=out_base,
        out_span_cells=out_span,
        out_end_index=out_end,
    )
    assert err == KV_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(878)

    for _ in range(700):
        layer_count = rng.randint(1, 6)
        token_capacity = rng.randint(1, 12)
        kv_heads = rng.randint(1, 6)
        head_dim = rng.randint(1, 12)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        k_cache_a = [rng.randint(-2000, 2000) for _ in range(total_cells)]
        v_cache_a = [rng.randint(-2000, 2000) for _ in range(total_cells)]
        k_cache_b = k_cache_a.copy()
        v_cache_b = v_cache_a.copy()

        out_base_a = [0]
        out_span_a = [0]
        out_end_a = [0]
        err_a = kv_cache_q16_zero_token_span_checked_nopartial_commit_only(
            k_cache_a,
            len(k_cache_a),
            v_cache_a,
            len(v_cache_a),
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            out_base_a,
            out_span_a,
            out_end_a,
        )

        out_base_b = [0]
        out_span_b = [0]
        out_end_b = [0]
        err_b = explicit_commit_composition(
            k_cache_b,
            len(k_cache_b),
            v_cache_b,
            len(v_cache_b),
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            out_base_b,
            out_span_b,
            out_end_b,
        )

        assert err_a == err_b
        assert k_cache_a == k_cache_b
        assert v_cache_a == v_cache_b

        if err_a == KV_Q16_OK:
            assert out_base_a == out_base_b
            assert out_span_a == out_span_b
            assert out_end_a == out_end_b


if __name__ == "__main__":
    test_source_contains_commit_only_zero_helper()
    test_known_vector_commit_and_diagnostics()
    test_null_alias_and_no_partial_failure()
    test_overflow_passthrough_from_checked_add()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
