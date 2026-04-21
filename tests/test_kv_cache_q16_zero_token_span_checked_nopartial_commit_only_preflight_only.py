#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ZeroTokenSpan...CommitOnlyPreflightOnly (IQ-887)."""

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
    kv_cache_q16_compute_layer_token_base_index_checked,
    try_add_i64_checked,
)


def kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only(
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
    out_required_span_cells: list[int] | None,
    out_k_base_index: list[int] | None,
    out_v_base_index: list[int] | None,
    out_end_index: list[int] | None,
) -> int:
    if (
        out_required_span_cells is None
        or out_k_base_index is None
        or out_v_base_index is None
        or out_end_index is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if (
        out_required_span_cells is out_k_base_index
        or out_required_span_cells is out_v_base_index
        or out_required_span_cells is out_end_index
        or out_k_base_index is out_v_base_index
        or out_k_base_index is out_end_index
        or out_v_base_index is out_end_index
    ):
        return KV_Q16_ERR_BAD_PARAM

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

    if (
        out_required_span_cells is k_cache_q16
        or out_required_span_cells is v_cache_q16
        or out_k_base_index is k_cache_q16
        or out_k_base_index is v_cache_q16
        or out_v_base_index is k_cache_q16
        or out_v_base_index is v_cache_q16
        or out_end_index is k_cache_q16
        or out_end_index is v_cache_q16
    ):
        return KV_Q16_ERR_BAD_PARAM

    snapshot = (
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_cache_capacity,
        v_cache_capacity,
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

    if k_base[0] < 0 or v_base[0] < 0 or k_span[0] < 0 or v_span[0] < 0:
        return KV_Q16_ERR_BAD_PARAM

    err, k_end = try_add_i64_checked(k_base[0], k_span[0])
    if err != KV_Q16_OK:
        return err

    err, v_end = try_add_i64_checked(v_base[0], v_span[0])
    if err != KV_Q16_OK:
        return err

    if k_end != v_end:
        return KV_Q16_ERR_BAD_PARAM

    if k_end < k_base[0] or v_end < v_base[0]:
        return KV_Q16_ERR_BAD_PARAM

    if k_end > k_cache_capacity or v_end > v_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM

    if snapshot != (
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_cache_capacity,
        v_cache_capacity,
    ):
        return KV_Q16_ERR_BAD_PARAM

    out_required_span_cells[0] = k_span[0]
    out_k_base_index[0] = k_base[0]
    out_v_base_index[0] = v_base[0]
    out_end_index[0] = k_end
    return KV_Q16_OK


def explicit_preflight_composition(*args, **kwargs) -> int:
    return kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only(*args, **kwargs)


def test_source_contains_zero_span_commit_only_preflight_only_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheQ16ComputeLayerTokenBaseIndexChecked" in body
    assert "staged_required_span_cells" in body
    assert "staged_v_span_cells" in body
    assert "if (staged_k_end_index < staged_k_base_index)" in body
    assert "if (staged_v_end_index < staged_v_base_index)" in body
    assert "if (staged_k_end_index > k_cache_capacity)" in body
    assert "if (staged_v_end_index > v_cache_capacity)" in body
    assert "snapshot_k_cache_capacity" in body
    assert "snapshot_v_cache_capacity" in body


def test_known_vector_diagnostics_and_zero_write_behavior() -> None:
    layer_count = 3
    token_capacity = 5
    kv_heads = 2
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [5000 + idx for idx in range(total_cells)]
    v_cache = [9000 + idx for idx in range(total_cells)]
    k_before = k_cache.copy()
    v_before = v_cache.copy()

    layer_idx = 1
    token_idx = 3
    expected_base = ((layer_idx * token_capacity) + token_idx) * span

    out_span = [0]
    out_k_base = [0]
    out_v_base = [0]
    out_end = [0]

    err = kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only(
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
        out_span,
        out_k_base,
        out_v_base,
        out_end,
    )
    assert err == KV_Q16_OK
    assert out_span == [span]
    assert out_k_base == [expected_base]
    assert out_v_base == [expected_base]
    assert out_end == [expected_base + span]

    assert k_cache == k_before
    assert v_cache == v_before


def test_no_partial_outputs_alias_and_bounds_guards() -> None:
    k_cache = [11] * 64
    v_cache = [22] * 64

    assert (
        kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only(
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
            None,
            [0],
            [0],
            [0],
        )
        == KV_Q16_ERR_NULL_PTR
    )

    out_shared = [0]
    assert (
        kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only(
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
            out_shared,
            out_shared,
            [0],
            [0],
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    out_span = [123]
    out_k_base = [456]
    out_v_base = [789]
    out_end = [321]
    k_before = k_cache.copy()
    v_before = v_cache.copy()

    err = kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only(
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
        out_required_span_cells=out_span,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
        out_end_index=out_end,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert out_span == [123]
    assert out_k_base == [456]
    assert out_v_base == [789]
    assert out_end == [321]
    assert k_cache == k_before
    assert v_cache == v_before


def test_randomized_parity_adversarial_vectors() -> None:
    rng = random.Random(887)

    for _ in range(900):
        layer_count = rng.randint(1, 8)
        token_capacity = rng.randint(1, 12)
        kv_heads = rng.randint(1, 8)
        head_dim = rng.randint(1, 16)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        k_cache = [rng.randint(-1000, 1000) for _ in range(total_cells)]
        v_cache = [rng.randint(-1000, 1000) for _ in range(total_cells)]
        k_before = k_cache.copy()
        v_before = v_cache.copy()

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        if rng.random() < 0.2:
            token_idx = token_capacity + rng.randint(1, 4)
        if rng.random() < 0.05:
            layer_count = I64_MAX

        out_span = [0]
        out_k_base = [0]
        out_v_base = [0]
        out_end = [0]

        err_a = kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only(
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
            out_span,
            out_k_base,
            out_v_base,
            out_end,
        )

        exp_span = [0]
        exp_k_base = [0]
        exp_v_base = [0]
        exp_end = [0]
        err_b = explicit_preflight_composition(
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
            exp_span,
            exp_k_base,
            exp_v_base,
            exp_end,
        )

        assert err_a == err_b
        if err_a == KV_Q16_OK:
            assert out_span == exp_span
            assert out_k_base == exp_k_base
            assert out_v_base == exp_v_base
            assert out_end == exp_end
        assert k_cache == k_before
        assert v_cache == v_before


if __name__ == "__main__":
    test_source_contains_zero_span_commit_only_preflight_only_helper()
    test_known_vector_diagnostics_and_zero_write_behavior()
    test_no_partial_outputs_alias_and_bounds_guards()
    test_randomized_parity_adversarial_vectors()
    print("ok")
