#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ReadTokenCheckedNoPartialCommitOnly (IQ-879)."""

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
from test_kv_cache_q16_read_token_checked_nopartial import (
    kv_cache_q16_read_token_checked_nopartial,
)


def kv_cache_q16_read_token_checked_nopartial_commit_only(
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
    out_base_index: list[int] | None,
    out_span_cells: list[int] | None,
    out_end_index: list[int] | None,
) -> int:
    if out_base_index is None or out_span_cells is None or out_end_index is None:
        return KV_Q16_ERR_NULL_PTR
    if (
        out_base_index is out_span_cells
        or out_base_index is out_end_index
        or out_span_cells is out_end_index
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        k_cache_q16 is None
        or v_cache_q16 is None
        or k_token_out_q16 is None
        or v_token_out_q16 is None
    ):
        return KV_Q16_ERR_NULL_PTR
    if k_cache_q16 is v_cache_q16:
        return KV_Q16_ERR_BAD_PARAM
    if k_token_out_q16 is v_token_out_q16:
        return KV_Q16_ERR_BAD_PARAM

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
    ):
        return KV_Q16_ERR_BAD_PARAM

    snapshot_layer_idx = layer_idx
    snapshot_token_idx = token_idx
    snapshot_layer_count = layer_count
    snapshot_token_capacity = token_capacity
    snapshot_kv_heads = kv_heads
    snapshot_head_dim = head_dim
    snapshot_k_out_capacity = k_token_out_capacity
    snapshot_v_out_capacity = v_token_out_capacity

    staged_base = [0]
    staged_span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        staged_base,
        staged_span,
    )
    if err != KV_Q16_OK:
        return err

    err, staged_end = try_add_i64_checked(staged_base[0], staged_span[0])
    if err != KV_Q16_OK:
        return err

    if staged_end > k_cache_capacity or staged_end > v_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if staged_span[0] > k_token_out_capacity or staged_span[0] > v_token_out_capacity:
        return KV_Q16_ERR_BAD_PARAM

    if (
        snapshot_layer_idx != layer_idx
        or snapshot_token_idx != token_idx
        or snapshot_layer_count != layer_count
        or snapshot_token_capacity != token_capacity
        or snapshot_kv_heads != kv_heads
        or snapshot_head_dim != head_dim
        or snapshot_k_out_capacity != k_token_out_capacity
        or snapshot_v_out_capacity != v_token_out_capacity
    ):
        return KV_Q16_ERR_BAD_PARAM

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

    out_base_index[0] = staged_base[0]
    out_span_cells[0] = staged_span[0]
    out_end_index[0] = staged_end
    return KV_Q16_OK


def explicit_commit_only_composition(
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
    out_base_index: list[int] | None,
    out_span_cells: list[int] | None,
    out_end_index: list[int] | None,
) -> int:
    if out_base_index is None or out_span_cells is None or out_end_index is None:
        return KV_Q16_ERR_NULL_PTR
    if (
        out_base_index is out_span_cells
        or out_base_index is out_end_index
        or out_span_cells is out_end_index
    ):
        return KV_Q16_ERR_BAD_PARAM

    staged_base = [0]
    staged_span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        staged_base,
        staged_span,
    )
    if err != KV_Q16_OK:
        return err

    err, staged_end = try_add_i64_checked(staged_base[0], staged_span[0])
    if err != KV_Q16_OK:
        return err

    if staged_end > k_cache_capacity or staged_end > v_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if staged_span[0] > k_token_out_capacity or staged_span[0] > v_token_out_capacity:
        return KV_Q16_ERR_BAD_PARAM

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

    out_base_index[0] = staged_base[0]
    out_span_cells[0] = staged_span[0]
    out_end_index[0] = staged_end
    return KV_Q16_OK


def test_source_contains_commit_only_wrapper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    assert "I32 KVCacheQ16ReadTokenCheckedNoPartialCommitOnly(" in source
    assert "KVCacheQ16ReadTokenCheckedNoPartial(" in source
    assert "KVTryAddI64Checked(staged_k_base_index" in source


def test_known_vector_outputs() -> None:
    layer_count = 3
    token_capacity = 5
    kv_heads = 2
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [1000 + idx for idx in range(total_cells)]
    v_cache = [2000 + idx for idx in range(total_cells)]
    k_out = [-1] * span
    v_out = [-2] * span

    layer_idx = 2
    token_idx = 1
    base = ((layer_idx * token_capacity) + token_idx) * span

    out_base = [0]
    out_span = [0]
    out_end = [0]

    err = kv_cache_q16_read_token_checked_nopartial_commit_only(
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
        out_base,
        out_span,
        out_end,
    )
    assert err == KV_Q16_OK
    assert out_base == [base]
    assert out_span == [span]
    assert out_end == [base + span]
    assert k_out == k_cache[base : base + span]
    assert v_out == v_cache[base : base + span]


def test_null_alias_and_bounds() -> None:
    k_cache = [1] * 16
    v_cache = [2] * 16
    k_out = [0] * 4
    v_out = [0] * 4
    out_base = [0]
    out_span = [0]
    out_end = [0]

    err = kv_cache_q16_read_token_checked_nopartial_commit_only(
        None,
        16,
        v_cache,
        16,
        0,
        0,
        1,
        4,
        2,
        2,
        k_out,
        len(k_out),
        v_out,
        len(v_out),
        out_base,
        out_span,
        out_end,
    )
    assert err == KV_Q16_ERR_NULL_PTR

    err = kv_cache_q16_read_token_checked_nopartial_commit_only(
        k_cache,
        16,
        v_cache,
        16,
        0,
        0,
        1,
        4,
        2,
        2,
        k_cache,
        16,
        v_out,
        len(v_out),
        out_base,
        out_span,
        out_end,
    )
    assert err == KV_Q16_ERR_BAD_PARAM

    err = kv_cache_q16_read_token_checked_nopartial_commit_only(
        k_cache,
        16,
        v_cache,
        16,
        0,
        1,
        1,
        4,
        2,
        2,
        k_out,
        3,
        v_out,
        len(v_out),
        out_base,
        out_span,
        out_end,
    )
    assert err == KV_Q16_ERR_BAD_PARAM


def test_overflow_path() -> None:
    k_cache = [0] * 8
    v_cache = [0] * 8
    k_out = [0] * 8
    v_out = [0] * 8
    out_base = [0]
    out_span = [0]
    out_end = [0]

    err = kv_cache_q16_read_token_checked_nopartial_commit_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        0,
        0,
        1,
        I64_MAX,
        I64_MAX,
        2,
        k_out,
        len(k_out),
        v_out,
        len(v_out),
        out_base,
        out_span,
        out_end,
    )
    assert err == KV_Q16_ERR_OVERFLOW


def test_randomized_parity(seed: int = 879, trials: int = 250) -> None:
    rng = random.Random(seed)

    for _ in range(trials):
        layer_count = rng.randint(1, 5)
        token_capacity = rng.randint(1, 6)
        kv_heads = rng.randint(1, 4)
        head_dim = rng.randint(1, 4)
        span = kv_heads * head_dim

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        total = layer_count * token_capacity * span
        pad = rng.randint(0, 6)

        k_cache = [rng.randint(-2000, 2000) for _ in range(total + pad)]
        v_cache = [rng.randint(-2000, 2000) for _ in range(total + pad)]

        k_out_a = [0] * (span + pad)
        v_out_a = [0] * (span + pad)
        k_out_b = [0] * (span + pad)
        v_out_b = [0] * (span + pad)

        out_base_a = [0]
        out_span_a = [0]
        out_end_a = [0]
        out_base_b = [0]
        out_span_b = [0]
        out_end_b = [0]

        if rng.random() < 0.2 and span > 0:
            k_out_capacity = span - 1
        else:
            k_out_capacity = span + pad
        v_out_capacity = k_out_capacity

        err_a = kv_cache_q16_read_token_checked_nopartial_commit_only(
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
            k_out_a,
            k_out_capacity,
            v_out_a,
            v_out_capacity,
            out_base_a,
            out_span_a,
            out_end_a,
        )

        err_b = explicit_commit_only_composition(
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
            k_out_b,
            k_out_capacity,
            v_out_b,
            v_out_capacity,
            out_base_b,
            out_span_b,
            out_end_b,
        )

        assert err_a == err_b
        if err_a == KV_Q16_OK:
            assert out_base_a == out_base_b
            assert out_span_a == out_span_b
            assert out_end_a == out_end_b
            assert k_out_a[:span] == k_out_b[:span]
            assert v_out_a[:span] == v_out_b[:span]


def main() -> None:
    test_source_contains_commit_only_wrapper()
    test_known_vector_outputs()
    test_null_alias_and_bounds()
    test_overflow_path()
    test_randomized_parity()
    print("ok")


if __name__ == "__main__":
    main()
