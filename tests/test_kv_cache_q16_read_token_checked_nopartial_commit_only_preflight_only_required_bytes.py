#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ReadToken...PreflightOnlyRequiredBytes (IQ-888)."""

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
    try_mul_i64_checked,
)
from test_kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only import (
    kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only,
)


def kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only_required_bytes(
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
    out_required_token_bytes: list[int] | None,
    out_k_base_index: list[int] | None,
    out_v_base_index: list[int] | None,
) -> int:
    if (
        out_required_span_cells is None
        or out_required_token_bytes is None
        or out_k_base_index is None
        or out_v_base_index is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if (
        out_required_span_cells is out_required_token_bytes
        or out_required_span_cells is out_k_base_index
        or out_required_span_cells is out_v_base_index
        or out_required_token_bytes is out_k_base_index
        or out_required_token_bytes is out_v_base_index
        or out_k_base_index is out_v_base_index
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_required_span_cells is k_cache_q16
        or out_required_span_cells is v_cache_q16
        or out_required_span_cells is k_token_out_q16
        or out_required_span_cells is v_token_out_q16
        or out_required_token_bytes is k_cache_q16
        or out_required_token_bytes is v_cache_q16
        or out_required_token_bytes is k_token_out_q16
        or out_required_token_bytes is v_token_out_q16
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
    snapshot_k_cache_capacity = k_cache_capacity
    snapshot_v_cache_capacity = v_cache_capacity
    snapshot_k_out_capacity = k_token_out_capacity
    snapshot_v_out_capacity = v_token_out_capacity

    staged_required_span_cells = [0]
    staged_k_base_index = [0]
    staged_v_base_index = [0]
    err = kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only(
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
        staged_required_span_cells,
        staged_k_base_index,
        staged_v_base_index,
    )
    if err != KV_Q16_OK:
        return err

    err, staged_required_token_bytes = try_mul_i64_checked(staged_required_span_cells[0], 8)
    if err != KV_Q16_OK:
        return err

    if (
        snapshot_layer_idx != layer_idx
        or snapshot_token_idx != token_idx
        or snapshot_layer_count != layer_count
        or snapshot_token_capacity != token_capacity
        or snapshot_kv_heads != kv_heads
        or snapshot_head_dim != head_dim
        or snapshot_k_cache_capacity != k_cache_capacity
        or snapshot_v_cache_capacity != v_cache_capacity
        or snapshot_k_out_capacity != k_token_out_capacity
        or snapshot_v_out_capacity != v_token_out_capacity
    ):
        return KV_Q16_ERR_BAD_PARAM

    out_required_span_cells[0] = staged_required_span_cells[0]
    out_required_token_bytes[0] = staged_required_token_bytes
    out_k_base_index[0] = staged_k_base_index[0]
    out_v_base_index[0] = staged_v_base_index[0]
    return KV_Q16_OK


def explicit_required_bytes_composition(*args, **kwargs) -> tuple[int, tuple[int, int, int, int]]:
    span_out = [0]
    k_base_out = [0]
    v_base_out = [0]
    err = kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only(
        *args,
        out_required_span_cells=span_out,
        out_k_base_index=k_base_out,
        out_v_base_index=v_base_out,
        **kwargs,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0)

    err, bytes_out = try_mul_i64_checked(span_out[0], 8)
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0)

    return KV_Q16_OK, (span_out[0], bytes_out, k_base_out[0], v_base_out[0])


def test_source_contains_required_bytes_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ReadTokenCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheQ16ReadTokenCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "KVTryMulI64Checked(staged_required_span_cells" in body
    assert "snapshot_k_cache_capacity" in body
    assert "snapshot_v_cache_capacity" in body
    assert "snapshot_k_out_capacity" in body
    assert "snapshot_v_out_capacity" in body


def test_known_vector_required_bytes_and_zero_write_semantics() -> None:
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
    out_bytes = [0]
    out_k_base = [0]
    out_v_base = [0]

    layer_idx = 2
    token_idx = 1
    expected_base = ((layer_idx * token_capacity) + token_idx) * span

    err = kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only_required_bytes(
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
        out_bytes,
        out_k_base,
        out_v_base,
    )
    assert err == KV_Q16_OK

    assert out_span == [span]
    assert out_bytes == [span * 8]
    assert out_k_base == [expected_base]
    assert out_v_base == [expected_base]
    assert k_cache == k_cache_before
    assert v_cache == v_cache_before
    assert k_out == k_out_before
    assert v_out == v_out_before


def test_alias_null_and_no_partial_output_contracts() -> None:
    k_cache = [11] * 64
    v_cache = [22] * 64
    k_out = [33] * 8
    v_out = [44] * 8

    assert (
        kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only_required_bytes(
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
            None,
            [0],
            [0],
            [0],
        )
        == KV_Q16_ERR_NULL_PTR
    )

    shared = [0]
    assert (
        kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only_required_bytes(
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
            [0],
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    out_span = [123]
    out_bytes = [456]
    out_k_base = [789]
    out_v_base = [987]

    k_cache_before = k_cache.copy()
    v_cache_before = v_cache.copy()
    k_out_before = k_out.copy()
    v_out_before = v_out.copy()

    err = kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only_required_bytes(
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
        out_required_token_bytes=out_bytes,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert out_span == [123]
    assert out_bytes == [456]
    assert out_k_base == [789]
    assert out_v_base == [987]
    assert k_cache == k_cache_before
    assert v_cache == v_cache_before
    assert k_out == k_out_before
    assert v_out == v_out_before


def test_required_bytes_overflow_passthrough() -> None:
    out_span = [0]
    out_bytes = [0]
    out_k_base = [0]
    out_v_base = [0]

    err = kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only_required_bytes(
        [0],
        (I64_MAX // 8) + 1,
        [0],
        (I64_MAX // 8) + 1,
        layer_idx=0,
        token_idx=0,
        layer_count=1,
        token_capacity=1,
        kv_heads=(I64_MAX // 8) + 1,
        head_dim=1,
        k_token_out_q16=[0],
        k_token_out_capacity=I64_MAX,
        v_token_out_q16=[0],
        v_token_out_capacity=I64_MAX,
        out_required_span_cells=out_span,
        out_required_token_bytes=out_bytes,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
    )
    assert err == KV_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_composition_adversarial() -> None:
    rng = random.Random(888)

    for _ in range(800):
        layer_count = rng.randint(1, 8)
        token_capacity = rng.randint(1, 12)
        kv_heads = rng.randint(1, 8)
        head_dim = rng.randint(1, 16)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        k_cache = [rng.randint(-500, 500) for _ in range(total_cells)]
        v_cache = [rng.randint(-500, 500) for _ in range(total_cells)]
        k_cache_before = k_cache.copy()
        v_cache_before = v_cache.copy()

        extra = rng.randint(0, 4)
        k_out = [rng.randint(-50, 50) for _ in range(span + extra)]
        v_out = [rng.randint(-50, 50) for _ in range(span + extra)]
        k_out_before = k_out.copy()
        v_out_before = v_out.copy()

        k_cache_capacity = len(k_cache)
        v_cache_capacity = len(v_cache)
        k_out_capacity = len(k_out)
        v_out_capacity = len(v_out)

        if rng.random() < 0.2:
            token_idx = token_capacity + rng.randint(1, 4)

        if rng.random() < 0.05:
            kv_heads = (I64_MAX // 8) + 1
            head_dim = 1
            layer_count = 1
            token_capacity = 1
            layer_idx = 0
            token_idx = 0
            k_out_capacity = I64_MAX
            v_out_capacity = I64_MAX
            k_cache_capacity = 1
            v_cache_capacity = 1
            k_cache = [0]
            v_cache = [0]
            k_cache_before = k_cache.copy()
            v_cache_before = v_cache.copy()

        out_span = [0]
        out_bytes = [0]
        out_k_base = [0]
        out_v_base = [0]

        err_a = kv_cache_q16_read_token_checked_nopartial_commit_only_preflight_only_required_bytes(
            k_cache,
            k_cache_capacity,
            v_cache,
            v_cache_capacity,
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_out,
            k_out_capacity,
            v_out,
            v_out_capacity,
            out_span,
            out_bytes,
            out_k_base,
            out_v_base,
        )

        err_b, expected = explicit_required_bytes_composition(
            k_cache,
            k_cache_capacity,
            v_cache,
            v_cache_capacity,
            layer_idx,
            token_idx,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            k_out,
            k_out_capacity,
            v_out,
            v_out_capacity,
        )

        assert err_a == err_b
        assert k_cache == k_cache_before
        assert v_cache == v_cache_before
        assert k_out == k_out_before
        assert v_out == v_out_before

        if err_a == KV_Q16_OK:
            assert (out_span[0], out_bytes[0], out_k_base[0], out_v_base[0]) == expected


if __name__ == "__main__":
    test_source_contains_required_bytes_helper()
    test_known_vector_required_bytes_and_zero_write_semantics()
    test_alias_null_and_no_partial_output_contracts()
    test_required_bytes_overflow_passthrough()
    test_randomized_parity_vs_explicit_composition_adversarial()
    print("ok")
