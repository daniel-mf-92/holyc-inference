#!/usr/bin/env python3
"""Parity harness for ...RoundTrip...PreflightOnlyRequiredBytesCommitOnly (IQ-917)."""

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
from test_kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes import (
    kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes,
)


def kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only(
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
        or out_required_span_cells is k_token_src_q16
        or out_required_span_cells is v_token_src_q16
        or out_required_span_cells is k_token_out_q16
        or out_required_span_cells is v_token_out_q16
        or out_required_token_bytes is k_cache_q16
        or out_required_token_bytes is v_cache_q16
        or out_required_token_bytes is k_token_src_q16
        or out_required_token_bytes is v_token_src_q16
        or out_required_token_bytes is k_token_out_q16
        or out_required_token_bytes is v_token_out_q16
        or out_k_base_index is k_cache_q16
        or out_k_base_index is v_cache_q16
        or out_k_base_index is k_token_src_q16
        or out_k_base_index is v_token_src_q16
        or out_k_base_index is k_token_out_q16
        or out_k_base_index is v_token_out_q16
        or out_v_base_index is k_cache_q16
        or out_v_base_index is v_cache_q16
        or out_v_base_index is k_token_src_q16
        or out_v_base_index is v_token_src_q16
        or out_v_base_index is k_token_out_q16
        or out_v_base_index is v_token_out_q16
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
        k_token_src_capacity,
        v_token_src_capacity,
        k_token_out_capacity,
        v_token_out_capacity,
    )

    staged_span = [0]
    staged_bytes = [0]
    staged_k_base = [0]
    staged_v_base = [0]

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes(
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
        staged_span,
        staged_bytes,
        staged_k_base,
        staged_v_base,
    )
    if err != KV_Q16_OK:
        return err

    if (
        staged_span[0] < 0
        or staged_bytes[0] < 0
        or staged_k_base[0] < 0
        or staged_v_base[0] < 0
    ):
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
        k_token_src_capacity,
        v_token_src_capacity,
        k_token_out_capacity,
        v_token_out_capacity,
    ):
        return KV_Q16_ERR_BAD_PARAM

    out_required_span_cells[0] = staged_span[0]
    out_required_token_bytes[0] = staged_bytes[0]
    out_k_base_index[0] = staged_k_base[0]
    out_v_base_index[0] = staged_v_base[0]
    return KV_Q16_OK


def explicit_commit_only_composition(*args, **kwargs) -> tuple[int, tuple[int, int, int, int]]:
    span = [0]
    token_bytes = [0]
    k_base = [0]
    v_base = [0]

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes(
        *args,
        out_required_span_cells=span,
        out_required_token_bytes=token_bytes,
        out_k_base_index=k_base,
        out_v_base_index=v_base,
        **kwargs,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0)

    if span[0] < 0 or token_bytes[0] < 0 or k_base[0] < 0 or v_base[0] < 0:
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0)

    return KV_Q16_OK, (span[0], token_bytes[0], k_base[0], v_base[0])


def test_source_contains_iq917_function() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes(" in body
    assert "snapshot_layer_idx = layer_idx;" in body
    assert "snapshot_k_cache_capacity = k_cache_capacity;" in body
    assert "if (staged_required_span_cells < 0 ||" in body
    assert "*out_required_span_cells = staged_required_span_cells;" in body
    assert "*out_required_token_bytes = staged_required_token_bytes;" in body
    assert "*out_k_base_index = staged_k_base_index;" in body
    assert "*out_v_base_index = staged_v_base_index;" in body


def test_null_and_alias_guards() -> None:
    k_cache = [1] * 64
    v_cache = [2] * 64
    k_src = [3] * 8
    v_src = [4] * 8
    k_out = [5] * 8
    v_out = [6] * 8

    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only(
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
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only(
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


def test_known_vector_success_and_no_partial_failure_publish() -> None:
    layer_count = 3
    token_capacity = 4
    kv_heads = 2
    head_dim = 5
    span = kv_heads * head_dim
    total_cells = layer_count * token_capacity * span

    k_cache = [1000 + idx for idx in range(total_cells)]
    v_cache = [2000 + idx for idx in range(total_cells)]
    k_src = [300 + idx for idx in range(span)]
    v_src = [700 + idx for idx in range(span)]
    k_out = [-11] * span
    v_out = [-17] * span

    out_span = [0]
    out_bytes = [0]
    out_k_base = [0]
    out_v_base = [0]

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        2,
        1,
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
        out_span,
        out_bytes,
        out_k_base,
        out_v_base,
    )
    assert err == KV_Q16_OK
    expected_base = ((2 * token_capacity) + 1) * span
    assert out_span == [span]
    assert out_bytes == [span * 8]
    assert out_k_base == [expected_base]
    assert out_v_base == [expected_base]

    req_span = [111]
    req_bytes = [222]
    req_k_base = [333]
    req_v_base = [444]
    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        0,
        token_capacity + 1,
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
        req_span,
        req_bytes,
        req_k_base,
        req_v_base,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert req_span == [111]
    assert req_bytes == [222]
    assert req_k_base == [333]
    assert req_v_base == [444]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(917)

    for _ in range(1200):
        layer_count = rng.randint(1, 6)
        token_capacity = rng.randint(1, 10)
        kv_heads = rng.randint(1, 16)
        head_dim = rng.randint(1, 16)
        span = kv_heads * head_dim
        total_cells = layer_count * token_capacity * span

        k_cache = [rng.randint(-1000, 1000) for _ in range(total_cells)]
        v_cache = [rng.randint(-1000, 1000) for _ in range(total_cells)]
        k_src = [rng.randint(-1000, 1000) for _ in range(span)]
        v_src = [rng.randint(-1000, 1000) for _ in range(span)]
        k_out = [rng.randint(-1000, 1000) for _ in range(span)]
        v_out = [rng.randint(-1000, 1000) for _ in range(span)]

        layer_idx = rng.randint(0, layer_count - 1)
        token_idx = rng.randint(0, token_capacity - 1)

        if rng.random() < 0.2:
            token_idx = token_capacity + rng.randint(1, 5)

        k_src_capacity = len(k_src)
        v_src_capacity = len(v_src)
        k_out_capacity = len(k_out)
        v_out_capacity = len(v_out)

        if rng.random() < 0.15:
            k_src_capacity = max(0, len(k_src) - rng.randint(1, len(k_src)))
        if rng.random() < 0.15:
            v_src_capacity = max(0, len(v_src) - rng.randint(1, len(v_src)))
        if rng.random() < 0.15:
            k_out_capacity = max(0, len(k_out) - rng.randint(1, len(k_out)))
        if rng.random() < 0.15:
            v_out_capacity = max(0, len(v_out) - rng.randint(1, len(v_out)))

        if rng.random() < 0.06:
            layer_count = 1
            token_capacity = 1
            layer_idx = 0
            token_idx = 0
            kv_heads = (I64_MAX // 8) + 1
            head_dim = 1
            k_cache = [0]
            v_cache = [0]
            k_src = [0]
            v_src = [0]
            k_out = [0]
            v_out = [0]
            k_src_capacity = 1
            v_src_capacity = 1
            k_out_capacity = 1
            v_out_capacity = 1

        out_span = [0]
        out_bytes = [0]
        out_k_base = [0]
        out_v_base = [0]

        err_a = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only(
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
            k_out,
            k_out_capacity,
            v_out,
            v_out_capacity,
            out_span,
            out_bytes,
            out_k_base,
            out_v_base,
        )

        err_b, expected = explicit_commit_only_composition(
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
            k_out,
            k_out_capacity,
            v_out,
            v_out_capacity,
        )

        assert err_a == err_b
        if err_a == KV_Q16_OK:
            assert (out_span[0], out_bytes[0], out_k_base[0], out_v_base[0]) == expected


if __name__ == "__main__":
    test_source_contains_iq917_function()
    test_null_and_alias_guards()
    test_known_vector_success_and_no_partial_failure_publish()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
