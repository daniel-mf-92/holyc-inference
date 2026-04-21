#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ReadTokenCheckedNoPartialCommitOnlyPreflightOnly (IQ-881)."""

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


def kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only(
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

    if k_cache_q16 is None or v_cache_q16 is None or k_token_out_q16 is None or v_token_out_q16 is None:
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

    if (
        out_required_span_cells is k_cache_q16
        or out_required_span_cells is v_cache_q16
        or out_required_span_cells is k_token_out_q16
        or out_required_span_cells is v_token_out_q16
        or out_k_base_index is k_cache_q16
        or out_k_base_index is v_cache_q16
        or out_k_base_index is k_token_out_q16
        or out_k_base_index is v_token_out_q16
        or out_v_base_index is k_cache_q16
        or out_v_base_index is v_cache_q16
        or out_v_base_index is k_token_out_q16
        or out_v_base_index is v_token_out_q16
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

    staged_k_base = [0]
    staged_k_span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        staged_k_base,
        staged_k_span,
    )
    if err != KV_Q16_OK:
        return err

    staged_v_base = [0]
    staged_v_span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        staged_v_base,
        staged_v_span,
    )
    if err != KV_Q16_OK:
        return err

    if staged_k_base[0] != staged_v_base[0] or staged_k_span[0] != staged_v_span[0]:
        return KV_Q16_ERR_BAD_PARAM

    err, staged_k_end = try_add_i64_checked(staged_k_base[0], staged_k_span[0])
    if err != KV_Q16_OK:
        return err

    err, staged_v_end = try_add_i64_checked(staged_v_base[0], staged_v_span[0])
    if err != KV_Q16_OK:
        return err

    if staged_k_end != staged_v_end:
        return KV_Q16_ERR_BAD_PARAM

    if staged_k_end > k_cache_capacity or staged_v_end > v_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM

    if staged_k_span[0] > k_token_out_capacity or staged_v_span[0] > v_token_out_capacity:
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

    out_required_span_cells[0] = staged_k_span[0]
    out_k_base_index[0] = staged_k_base[0]
    out_v_base_index[0] = staged_v_base[0]
    return KV_Q16_OK


def explicit_preflight_composition(*args, **kwargs) -> int:
    return kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only(*args, **kwargs)


def test_source_contains_preflight_only_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ReadTokenCheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheQ16ComputeLayerTokenBaseIndexChecked" in body
    assert "KVTryAddI64Checked(staged_v_base_index" in body
    assert "if (staged_k_span_cells > k_token_out_capacity)" in body
    assert "if (staged_v_span_cells > v_token_out_capacity)" in body
    assert "if (staged_k_base_index < 0 || staged_v_base_index < 0 ||" in body
    assert "if (staged_k_end_index < staged_k_base_index)" in body
    assert "if (staged_v_end_index < staged_v_base_index)" in body
    assert "snapshot_k_out_capacity" in body
    assert "snapshot_v_out_capacity" in body


def test_known_vector_preflight_only_outputs_and_no_writes() -> None:
    layer_count = 3
    token_capacity = 5
    kv_heads = 2
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [1000 + idx for idx in range(total_cells)]
    v_cache = [2000 + idx for idx in range(total_cells)]
    k_out = [-1] * (span + 3)
    v_out = [-2] * (span + 3)

    k_cache_before = k_cache.copy()
    v_cache_before = v_cache.copy()
    k_out_before = k_out.copy()
    v_out_before = v_out.copy()

    out_span = [0]
    out_k_base = [0]
    out_v_base = [0]

    layer_idx = 2
    token_idx = 1
    expected_base = ((layer_idx * token_capacity) + token_idx) * span

    err = kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only(
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
        out_span,
        out_k_base,
        out_v_base,
    )
    assert err == KV_Q16_OK

    assert out_span == [span]
    assert out_k_base == [expected_base]
    assert out_v_base == [expected_base]
    assert k_cache == k_cache_before
    assert v_cache == v_cache_before
    assert k_out == k_out_before
    assert v_out == v_out_before


def test_null_alias_and_no_partial_output_contracts() -> None:
    k_cache = [11] * 64
    v_cache = [22] * 64
    k_out = [33] * 8
    v_out = [44] * 8

    assert (
        kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only(
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
            k_out,
            len(k_out),
            v_out,
            len(v_out),
            [0],
            [0],
            [0],
        )
        == KV_Q16_ERR_NULL_PTR
    )

    shared = [0]
    assert (
        kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only(
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
            k_out,
            len(k_out),
            v_out,
            len(v_out),
            shared,
            shared,
            [0],
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    out_span = [123]
    out_k_base = [456]
    out_v_base = [789]
    k_cache_before = k_cache.copy()
    v_cache_before = v_cache.copy()
    k_out_before = k_out.copy()
    v_out_before = v_out.copy()

    err = kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only(
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
        k_token_out_q16=k_out,
        k_token_out_capacity=len(k_out),
        v_token_out_q16=v_out,
        v_token_out_capacity=len(v_out),
        out_required_span_cells=out_span,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert out_span == [123]
    assert out_k_base == [456]
    assert out_v_base == [789]
    assert k_cache == k_cache_before
    assert v_cache == v_cache_before
    assert k_out == k_out_before
    assert v_out == v_out_before


def test_overflow_passthrough_from_checked_index_math() -> None:
    out_span = [0]
    out_k_base = [0]
    out_v_base = [0]

    err = kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only(
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
        k_token_out_q16=[0] * 4,
        k_token_out_capacity=4,
        v_token_out_q16=[0] * 4,
        v_token_out_capacity=4,
        out_required_span_cells=out_span,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
    )
    assert err == KV_Q16_ERR_OVERFLOW


def test_randomized_parity_adversarial() -> None:
    rng = random.Random(881)

    for _ in range(700):
        layer_count = rng.randint(1, 8)
        token_capacity = rng.randint(1, 12)
        kv_heads = rng.randint(1, 8)
        head_dim = rng.randint(1, 16)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        k_cache_a = [rng.randint(-300, 300) for _ in range(total_cells)]
        v_cache_a = [rng.randint(-300, 300) for _ in range(total_cells)]
        k_cache_b = k_cache_a.copy()
        v_cache_b = v_cache_a.copy()

        extra = rng.randint(0, 4)
        k_out_a = [rng.randint(-50, 50) for _ in range(span + extra)]
        v_out_a = [rng.randint(-50, 50) for _ in range(span + extra)]
        k_out_b = k_out_a.copy()
        v_out_b = v_out_a.copy()

        out_span_a = [0]
        out_k_base_a = [0]
        out_v_base_a = [0]

        err_a = kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only(
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
            k_out_a,
            len(k_out_a),
            v_out_a,
            len(v_out_a),
            out_span_a,
            out_k_base_a,
            out_v_base_a,
        )

        out_span_b = [0]
        out_k_base_b = [0]
        out_v_base_b = [0]

        err_b = explicit_preflight_composition(
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
            k_out_b,
            len(k_out_b),
            v_out_b,
            len(v_out_b),
            out_span_b,
            out_k_base_b,
            out_v_base_b,
        )

        assert err_a == err_b
        assert k_cache_a == k_cache_b
        assert v_cache_a == v_cache_b
        assert k_out_a == k_out_b
        assert v_out_a == v_out_b

        if err_a == KV_Q16_OK:
            assert out_span_a == out_span_b
            assert out_k_base_a == out_k_base_b
            assert out_v_base_a == out_v_base_b


if __name__ == "__main__":
    test_source_contains_preflight_only_helper()
    test_known_vector_preflight_only_outputs_and_no_writes()
    test_null_alias_and_no_partial_output_contracts()
    test_overflow_passthrough_from_checked_index_math()
    test_randomized_parity_adversarial()
    print("ok")
