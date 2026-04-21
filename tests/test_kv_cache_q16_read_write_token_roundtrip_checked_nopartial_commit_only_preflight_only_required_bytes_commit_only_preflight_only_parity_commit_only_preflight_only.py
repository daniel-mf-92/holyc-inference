#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly (IQ-927)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_indexing_checked import (
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_OK,
    kv_cache_q16_compute_layer_token_base_index_checked,
    try_mul_i64_checked,
)
from test_kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only_preflight_only_parity_commit_only import (
    kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only_preflight_only_parity_commit_only,
)


def kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    staged_required_span_cells = [0]
    staged_required_token_bytes = [0]
    staged_k_base_index = [0]
    staged_v_base_index = [0]
    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only_preflight_only_parity_commit_only(
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
        staged_required_span_cells,
        staged_required_token_bytes,
        staged_k_base_index,
        staged_v_base_index,
    )
    if err != KV_Q16_OK:
        return err

    recomputed_k_base = [0]
    recomputed_k_span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        recomputed_k_base,
        recomputed_k_span,
    )
    if err != KV_Q16_OK:
        return err

    recomputed_v_base = [0]
    recomputed_v_span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        recomputed_v_base,
        recomputed_v_span,
    )
    if err != KV_Q16_OK:
        return err

    if recomputed_k_span[0] != recomputed_v_span[0]:
        return KV_Q16_ERR_BAD_PARAM

    err, recomputed_token_bytes = try_mul_i64_checked(recomputed_k_span[0], 8)
    if err != KV_Q16_OK:
        return err

    if (
        staged_required_span_cells[0] < 0
        or staged_required_token_bytes[0] < 0
        or staged_k_base_index[0] < 0
        or staged_v_base_index[0] < 0
        or recomputed_k_span[0] < 0
        or recomputed_token_bytes < 0
        or recomputed_k_base[0] < 0
        or recomputed_v_base[0] < 0
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        staged_required_span_cells[0] != recomputed_k_span[0]
        or staged_required_token_bytes[0] != recomputed_token_bytes
        or staged_k_base_index[0] != recomputed_k_base[0]
        or staged_v_base_index[0] != recomputed_v_base[0]
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

    out_required_span_cells[0] = staged_required_span_cells[0]
    out_required_token_bytes[0] = staged_required_token_bytes[0]
    out_k_base_index[0] = staged_k_base_index[0]
    out_v_base_index[0] = staged_v_base_index[0]
    return KV_Q16_OK


def explicit_preflight_only_composition(*args, **kwargs) -> tuple[int, tuple[int, int, int, int]]:
    staged_span = [0]
    staged_bytes = [0]
    staged_k_base = [0]
    staged_v_base = [0]
    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only_preflight_only_parity_commit_only(
        *args,
        out_required_span_cells=staged_span,
        out_required_token_bytes=staged_bytes,
        out_k_base_index=staged_k_base,
        out_v_base_index=staged_v_base,
        **kwargs,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0)

    recomputed_k_base = [0]
    recomputed_k_span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        recomputed_k_base,
        recomputed_k_span,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0)

    recomputed_v_base = [0]
    recomputed_v_span = [0]
    err = kv_cache_q16_compute_layer_token_base_index_checked(
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        recomputed_v_base,
        recomputed_v_span,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0)

    if recomputed_k_span[0] != recomputed_v_span[0]:
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0)

    err, recomputed_token_bytes = try_mul_i64_checked(recomputed_k_span[0], 8)
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0)

    if (
        staged_span[0] != recomputed_k_span[0]
        or staged_bytes[0] != recomputed_token_bytes
        or staged_k_base[0] != recomputed_k_base[0]
        or staged_v_base[0] != recomputed_v_base[0]
    ):
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0)

    return KV_Q16_OK, (staged_span[0], staged_bytes[0], staged_k_base[0], staged_v_base[0])


def test_source_contains_iq927_function() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "status = KVTryMulI64Checked(staged_span_cells_recomputed_k," in body
    assert "snapshot_k_cache_capacity = k_cache_capacity;" in body
    assert "if (staged_required_span_cells != staged_span_cells_recomputed_k ||" in body
    assert "*out_required_token_bytes = staged_required_token_bytes;" in body


def _make_valid_case(rng: random.Random) -> tuple:
    layer_count = rng.randint(1, 4)
    token_capacity = rng.randint(1, 6)
    kv_heads = rng.randint(1, 3)
    head_dim = rng.randint(1, 8)
    span = kv_heads * head_dim
    total = layer_count * token_capacity * span

    k_cache = [rng.randint(-20000, 20000) for _ in range(total)]
    v_cache = [rng.randint(-20000, 20000) for _ in range(total)]

    k_src = [rng.randint(-20000, 20000) for _ in range(span)]
    v_src = [rng.randint(-20000, 20000) for _ in range(span)]
    k_out = [rng.randint(-20000, 20000) for _ in range(span)]
    v_out = [rng.randint(-20000, 20000) for _ in range(span)]

    return (
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        rng.randint(0, layer_count - 1),
        rng.randint(0, token_capacity - 1),
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
    )


def test_null_and_alias_guards() -> None:
    rng = random.Random(927001)
    args = _make_valid_case(rng)
    out_span = [0]
    out_bytes = [0]
    out_k = [0]
    out_v = [0]

    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
            *args,
            None,
            out_bytes,
            out_k,
            out_v,
        )
        == KV_Q16_ERR_NULL_PTR
    )

    shared = [0]
    assert (
        kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
            *args,
            shared,
            shared,
            out_k,
            out_v,
        )
        == KV_Q16_ERR_BAD_PARAM
    )


def test_known_vector_success_and_failure_no_partial() -> None:
    layer_count = 3
    token_capacity = 4
    kv_heads = 2
    head_dim = 5
    span = kv_heads * head_dim
    total = layer_count * token_capacity * span

    k_cache = [1000 + i for i in range(total)]
    v_cache = [2000 + i for i in range(total)]
    k_src = [3000 + i for i in range(span)]
    v_src = [4000 + i for i in range(span)]
    k_out = [5000 + i for i in range(span)]
    v_out = [6000 + i for i in range(span)]

    out_span = [111]
    out_bytes = [222]
    out_k = [333]
    out_v = [444]

    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        1,
        2,
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
        out_k,
        out_v,
    )
    assert err == KV_Q16_OK
    assert out_span == [span]
    assert out_bytes == [span * 8]

    fail_span = [77]
    fail_bytes = [88]
    fail_k = [99]
    fail_v = [111]
    err = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
        k_cache,
        len(k_cache),
        v_cache,
        len(v_cache),
        1,
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
        fail_span,
        fail_bytes,
        fail_k,
        fail_v,
    )
    assert err != KV_Q16_OK
    assert fail_span == [77]
    assert fail_bytes == [88]
    assert fail_k == [99]
    assert fail_v == [111]


def test_randomized_parity() -> None:
    rng = random.Random(20260421_927)

    for _ in range(2200):
        args = _make_valid_case(rng)

        if rng.random() < 0.16:
            li = list(args)
            li[5] = li[7] + rng.randint(0, 3)
            args = tuple(li)
        if rng.random() < 0.16:
            li = list(args)
            li[1] = rng.randint(0, max(0, li[1] - 1))
            args = tuple(li)

        out_span = [rng.randint(-10000, 10000)]
        out_bytes = [rng.randint(-10000, 10000)]
        out_k = [rng.randint(-10000, 10000)]
        out_v = [rng.randint(-10000, 10000)]

        err_impl = kv_cache_q16_read_write_token_roundtrip_checked_nopartial_commit_only_preflight_only_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
            *args,
            out_span,
            out_bytes,
            out_k,
            out_v,
        )
        err_ref, tup_ref = explicit_preflight_only_composition(*args)

        assert err_impl == err_ref
        if err_impl == KV_Q16_OK:
            assert (out_span[0], out_bytes[0], out_k[0], out_v[0]) == tup_ref


if __name__ == "__main__":
    test_source_contains_iq927_function()
    test_null_and_alias_guards()
    test_known_vector_success_and_failure_no_partial()
    test_randomized_parity()
    print("ok")

