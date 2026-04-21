#!/usr/bin/env python3
"""Parity harness for KVCacheQ16WriteTokenCheckedNoPartialCommitOnlyPreflightOnly (IQ-878)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_indexing_checked import (
    I64_MAX,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_OK,
)
from test_kv_cache_q16_write_token_checked_nopartial_commit_only import (
    kv_cache_q16_write_token_checked_nopartial_preflight,
)


def kv_cache_q16_write_token_checked_nopartial_commit_only_preflight_only(
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
    out_required_span_cells: list[int] | None,
    out_k_base_index: list[int] | None,
    out_v_base_index: list[int] | None,
) -> int:
    if out_required_span_cells is None or out_k_base_index is None or out_v_base_index is None:
        return KV_Q16_ERR_NULL_PTR

    if (
        out_required_span_cells is out_k_base_index
        or out_required_span_cells is out_v_base_index
        or out_k_base_index is out_v_base_index
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_required_span_cells is k_cache_q16
        or out_required_span_cells is v_cache_q16
        or out_required_span_cells is k_token_src_q16
        or out_required_span_cells is v_token_src_q16
        or out_k_base_index is k_cache_q16
        or out_k_base_index is v_cache_q16
        or out_k_base_index is k_token_src_q16
        or out_k_base_index is v_token_src_q16
        or out_v_base_index is k_cache_q16
        or out_v_base_index is v_cache_q16
        or out_v_base_index is k_token_src_q16
        or out_v_base_index is v_token_src_q16
    ):
        return KV_Q16_ERR_BAD_PARAM

    snapshot_layer_idx = layer_idx
    snapshot_token_idx = token_idx
    snapshot_layer_count = layer_count
    snapshot_token_capacity = token_capacity
    snapshot_kv_heads = kv_heads
    snapshot_head_dim = head_dim
    snapshot_k_src_capacity = k_token_src_capacity
    snapshot_v_src_capacity = v_token_src_capacity

    staged_k_base = [0]
    staged_v_base = [0]
    staged_span = [0]
    err = kv_cache_q16_write_token_checked_nopartial_preflight(
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
        staged_k_base,
        staged_v_base,
        staged_span,
    )
    if err != KV_Q16_OK:
        return err

    if (
        snapshot_layer_idx != layer_idx
        or snapshot_token_idx != token_idx
        or snapshot_layer_count != layer_count
        or snapshot_token_capacity != token_capacity
        or snapshot_kv_heads != kv_heads
        or snapshot_head_dim != head_dim
        or snapshot_k_src_capacity != k_token_src_capacity
        or snapshot_v_src_capacity != v_token_src_capacity
    ):
        return KV_Q16_ERR_BAD_PARAM

    out_required_span_cells[0] = staged_span[0]
    out_k_base_index[0] = staged_k_base[0]
    out_v_base_index[0] = staged_v_base[0]
    return KV_Q16_OK


def test_source_contains_commit_only_preflight_only_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    assert "I32 KVCacheQ16WriteTokenCheckedNoPartialCommitOnlyPreflightOnly(" in source
    assert "KVCacheQ16WriteTokenCheckedNoPartialPreflight(" in source


def test_known_vector_preflight_only_publishes_geometry_and_does_not_write_cache() -> None:
    layer_count = 3
    token_capacity = 4
    kv_heads = 2
    head_dim = 5
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [1000 + idx for idx in range(total_cells)]
    v_cache = [2000 + idx for idx in range(total_cells)]
    k_before = k_cache.copy()
    v_before = v_cache.copy()
    k_src = [300 + idx for idx in range(span)]
    v_src = [700 + idx for idx in range(span)]

    out_span = [0]
    out_k_base = [0]
    out_v_base = [0]
    err = kv_cache_q16_write_token_checked_nopartial_commit_only_preflight_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        layer_idx=2,
        token_idx=1,
        layer_count=layer_count,
        token_capacity=token_capacity,
        kv_heads=kv_heads,
        head_dim=head_dim,
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        out_required_span_cells=out_span,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
    )
    assert err == KV_Q16_OK

    expected_base = ((2 * token_capacity) + 1) * span
    assert out_span == [span]
    assert out_k_base == [expected_base]
    assert out_v_base == [expected_base]
    assert k_cache == k_before
    assert v_cache == v_before


def test_no_partial_outputs_and_alias_guards() -> None:
    k_cache = [11] * 64
    v_cache = [22] * 64
    k_src = [33] * 8
    v_src = [44] * 8

    assert (
        kv_cache_q16_write_token_checked_nopartial_commit_only_preflight_only(
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
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            None,
            [0],
            [0],
        )
        == KV_Q16_ERR_NULL_PTR
    )

    out_shared = [0]
    assert (
        kv_cache_q16_write_token_checked_nopartial_commit_only_preflight_only(
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
            k_src,
            len(k_src),
            v_src,
            len(v_src),
            out_shared,
            out_shared,
            [0],
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    out_span = [123]
    out_k_base = [456]
    out_v_base = [789]
    k_before = k_cache.copy()
    v_before = v_cache.copy()

    err = kv_cache_q16_write_token_checked_nopartial_commit_only_preflight_only(
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
        k_token_src_q16=k_src,
        k_token_src_capacity=len(k_src),
        v_token_src_q16=v_src,
        v_token_src_capacity=len(v_src),
        out_required_span_cells=out_span,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert out_span == [123]
    assert out_k_base == [456]
    assert out_v_base == [789]
    assert k_cache == k_before
    assert v_cache == v_before


def test_randomized_parity_vs_canonical_preflight_adversarial() -> None:
    rng = random.Random(878)

    for _ in range(700):
        layer_count = rng.randint(1, 8)
        token_capacity = rng.randint(1, 12)
        kv_heads = rng.randint(1, 8)
        head_dim = rng.randint(1, 16)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        k_cache = [rng.randint(-300, 300) for _ in range(total_cells)]
        v_cache = [rng.randint(-300, 300) for _ in range(total_cells)]
        k_before = k_cache.copy()
        v_before = v_cache.copy()

        k_src = [rng.randint(-50, 50) for _ in range(span)]
        v_src = [rng.randint(-50, 50) for _ in range(span)]

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        # Inject adversarial overflow/bounds/capacity and alias cases.
        if rng.random() < 0.2:
            token_idx = token_capacity + rng.randint(1, 4)
        if rng.random() < 0.2:
            k_src_capacity = span - rng.randint(1, span)
            v_src_capacity = span - rng.randint(1, span)
        else:
            k_src_capacity = len(k_src)
            v_src_capacity = len(v_src)
        if rng.random() < 0.05:
            layer_count = I64_MAX

        out_span = [0]
        out_k_base = [0]
        out_v_base = [0]

        err_a = kv_cache_q16_write_token_checked_nopartial_commit_only_preflight_only(
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
            k_src_capacity,
            v_src,
            v_src_capacity,
            out_span,
            out_k_base,
            out_v_base,
        )

        exp_k_base = [0]
        exp_v_base = [0]
        exp_span = [0]
        err_b = kv_cache_q16_write_token_checked_nopartial_preflight(
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
            k_src_capacity,
            v_src,
            v_src_capacity,
            exp_k_base,
            exp_v_base,
            exp_span,
        )

        assert err_a == err_b
        assert k_cache == k_before
        assert v_cache == v_before
        if err_a == KV_Q16_OK:
            assert out_span == exp_span
            assert out_k_base == exp_k_base
            assert out_v_base == exp_v_base


if __name__ == "__main__":
    test_source_contains_commit_only_preflight_only_helper()
    test_known_vector_preflight_only_publishes_geometry_and_does_not_write_cache()
    test_no_partial_outputs_and_alias_guards()
    test_randomized_parity_vs_canonical_preflight_adversarial()
    print("ok")
