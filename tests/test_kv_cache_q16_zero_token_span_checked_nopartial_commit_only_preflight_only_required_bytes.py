#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ZeroTokenSpan...PreflightOnlyRequiredBytes (IQ-893)."""

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
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only import (
    kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only,
)


def kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only_required_bytes(
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
    out_required_token_bytes: list[int] | None,
    out_k_base_index: list[int] | None,
    out_v_base_index: list[int] | None,
    out_end_index: list[int] | None,
) -> int:
    if (
        out_required_span_cells is None
        or out_required_token_bytes is None
        or out_k_base_index is None
        or out_v_base_index is None
        or out_end_index is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if (
        out_required_span_cells is out_required_token_bytes
        or out_required_span_cells is out_k_base_index
        or out_required_span_cells is out_v_base_index
        or out_required_span_cells is out_end_index
        or out_required_token_bytes is out_k_base_index
        or out_required_token_bytes is out_v_base_index
        or out_required_token_bytes is out_end_index
        or out_k_base_index is out_v_base_index
        or out_k_base_index is out_end_index
        or out_v_base_index is out_end_index
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_required_span_cells is k_cache_q16
        or out_required_span_cells is v_cache_q16
        or out_required_token_bytes is k_cache_q16
        or out_required_token_bytes is v_cache_q16
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

    staged_required_span_cells = [0]
    staged_k_base_index = [0]
    staged_v_base_index = [0]
    staged_end_index = [0]
    err = kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only(
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
        staged_required_span_cells,
        staged_k_base_index,
        staged_v_base_index,
        staged_end_index,
    )
    if err != KV_Q16_OK:
        return err

    err, staged_required_token_bytes = try_mul_i64_checked(staged_required_span_cells[0], 8)
    if err != KV_Q16_OK:
        return err

    err, staged_k_end = try_add_i64_checked(staged_k_base_index[0], staged_required_span_cells[0])
    if err != KV_Q16_OK:
        return err

    err, staged_v_end = try_add_i64_checked(staged_v_base_index[0], staged_required_span_cells[0])
    if err != KV_Q16_OK:
        return err

    if staged_k_end != staged_end_index[0]:
        return KV_Q16_ERR_BAD_PARAM
    if staged_v_end != staged_end_index[0]:
        return KV_Q16_ERR_BAD_PARAM
    if staged_end_index[0] < staged_k_base_index[0]:
        return KV_Q16_ERR_BAD_PARAM
    if staged_end_index[0] < staged_v_base_index[0]:
        return KV_Q16_ERR_BAD_PARAM
    if staged_end_index[0] > k_cache_capacity:
        return KV_Q16_ERR_BAD_PARAM
    if staged_end_index[0] > v_cache_capacity:
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

    out_required_span_cells[0] = staged_required_span_cells[0]
    out_required_token_bytes[0] = staged_required_token_bytes
    out_k_base_index[0] = staged_k_base_index[0]
    out_v_base_index[0] = staged_v_base_index[0]
    out_end_index[0] = staged_end_index[0]
    return KV_Q16_OK


def explicit_required_bytes_composition(*args, **kwargs) -> tuple[int, tuple[int, int, int, int, int]]:
    span_out = [0]
    k_base_out = [0]
    v_base_out = [0]
    end_out = [0]
    err = kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only(
        *args,
        out_required_span_cells=span_out,
        out_k_base_index=k_base_out,
        out_v_base_index=v_base_out,
        out_end_index=end_out,
        **kwargs,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0, 0)

    err, bytes_out = try_mul_i64_checked(span_out[0], 8)
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0, 0)

    return KV_Q16_OK, (span_out[0], bytes_out, k_base_out[0], v_base_out[0], end_out[0])


def test_source_contains_zero_span_required_bytes_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "KVTryMulI64Checked(staged_required_span_cells" in body
    assert "KVTryAddI64Checked(staged_k_base_index" in body
    assert "KVTryAddI64Checked(staged_v_base_index" in body
    assert "if (staged_k_end_index != staged_end_index)" in body
    assert "if (staged_end_index > k_cache_capacity)" in body
    assert "if (staged_end_index > v_cache_capacity)" in body
    assert "snapshot_k_cache_capacity" in body
    assert "snapshot_v_cache_capacity" in body
    assert "*out_end_index = staged_end_index;" in body


def test_known_vector_required_bytes_and_zero_write_semantics() -> None:
    layer_count = 2
    token_capacity = 5
    kv_heads = 3
    head_dim = 4
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [1000 + idx for idx in range(total_cells)]
    v_cache = [2000 + idx for idx in range(total_cells)]
    k_before = k_cache.copy()
    v_before = v_cache.copy()

    layer_idx = 1
    token_idx = 2
    expected_base = ((layer_idx * token_capacity) + token_idx) * span
    expected_end = expected_base + span

    out_span = [0]
    out_bytes = [0]
    out_k_base = [0]
    out_v_base = [0]
    out_end = [0]

    err = kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only_required_bytes(
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
        out_bytes,
        out_k_base,
        out_v_base,
        out_end,
    )
    assert err == KV_Q16_OK
    assert out_span == [span]
    assert out_bytes == [span * 8]
    assert out_k_base == [expected_base]
    assert out_v_base == [expected_base]
    assert out_end == [expected_end]

    assert k_cache == k_before
    assert v_cache == v_before


def test_alias_null_and_no_partial_contracts() -> None:
    k_cache = [31] * 64
    v_cache = [47] * 64

    assert (
        kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only_required_bytes(
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
            [0],
        )
        == KV_Q16_ERR_NULL_PTR
    )

    out_shared = [0]
    assert (
        kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only_required_bytes(
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
            [0],
        )
        == KV_Q16_ERR_BAD_PARAM
    )

    out_span = [123]
    out_bytes = [456]
    out_k_base = [789]
    out_v_base = [987]
    out_end = [654]
    k_before = k_cache.copy()
    v_before = v_cache.copy()

    err = kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only_required_bytes(
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
        out_required_token_bytes=out_bytes,
        out_k_base_index=out_k_base,
        out_v_base_index=out_v_base,
        out_end_index=out_end,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert out_span == [123]
    assert out_bytes == [456]
    assert out_k_base == [789]
    assert out_v_base == [987]
    assert out_end == [654]
    assert k_cache == k_before
    assert v_cache == v_before


def test_randomized_parity_vs_explicit_composition_adversarial() -> None:
    rng = random.Random(893)

    for _ in range(900):
        layer_count = rng.randint(1, 8)
        token_capacity = rng.randint(1, 12)
        kv_heads = rng.randint(1, 16)
        head_dim = rng.randint(1, 16)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        k_cache = [rng.randint(-500, 500) for _ in range(total_cells)]
        v_cache = [rng.randint(-500, 500) for _ in range(total_cells)]
        k_before = k_cache.copy()
        v_before = v_cache.copy()

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)
        k_cache_capacity = len(k_cache)
        v_cache_capacity = len(v_cache)

        if rng.random() < 0.2:
            token_idx = token_capacity + rng.randint(1, 4)

        # Force required-bytes overflow while keeping tiny host allocations.
        if rng.random() < 0.06:
            layer_count = 1
            token_capacity = 1
            layer_idx = 0
            token_idx = 0
            kv_heads = (I64_MAX // 8) + 1
            head_dim = 1
            k_cache_capacity = kv_heads
            v_cache_capacity = kv_heads
            k_cache = [0]
            v_cache = [0]
            k_before = k_cache.copy()
            v_before = v_cache.copy()

        out_span = [0]
        out_bytes = [0]
        out_k_base = [0]
        out_v_base = [0]
        out_end = [0]

        err_a = kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only_required_bytes(
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
            out_span,
            out_bytes,
            out_k_base,
            out_v_base,
            out_end,
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
        )

        assert err_a == err_b
        assert k_cache == k_before
        assert v_cache == v_before
        if err_a == KV_Q16_OK:
            assert (out_span[0], out_bytes[0], out_k_base[0], out_v_base[0], out_end[0]) == expected


def test_required_bytes_overflow_preserves_no_partial_outputs() -> None:
    layer_count = 1
    token_capacity = 1
    layer_idx = 0
    token_idx = 0
    kv_heads = (I64_MAX // 8) + 1
    head_dim = 1

    # Keep host allocations tiny; this is diagnostics-only preflight.
    k_cache = [17]
    v_cache = [23]

    out_span = [111]
    out_bytes = [222]
    out_k_base = [333]
    out_v_base = [444]
    out_end = [555]

    err = kv_cache_q16_zero_token_span_checked_nopartial_commit_only_preflight_only_required_bytes(
        k_cache,
        kv_heads,
        v_cache,
        kv_heads,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        out_span,
        out_bytes,
        out_k_base,
        out_v_base,
        out_end,
    )

    assert err == KV_Q16_ERR_OVERFLOW
    assert out_span == [111]
    assert out_bytes == [222]
    assert out_k_base == [333]
    assert out_v_base == [444]
    assert out_end == [555]


if __name__ == "__main__":
    test_source_contains_zero_span_required_bytes_helper()
    test_known_vector_required_bytes_and_zero_write_semantics()
    test_alias_null_and_no_partial_contracts()
    test_randomized_parity_vs_explicit_composition_adversarial()
    test_required_bytes_overflow_preserves_no_partial_outputs()
    print("ok")
