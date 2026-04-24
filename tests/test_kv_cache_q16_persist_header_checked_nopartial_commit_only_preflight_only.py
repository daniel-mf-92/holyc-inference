#!/usr/bin/env python3
"""Parity harness for KVCacheQ16PersistHeader...CommitOnlyPreflightOnly (IQ-1285)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_kv_cache_q16_persist_header_checked_nopartial_commit_only import (
    IDX_HEAD_DIM,
    IDX_HEADS,
    IDX_LAYERS,
    IDX_MAGIC,
    IDX_TOKENS,
    IDX_TOTAL,
    IDX_USED,
    IDX_VERSION,
    KV_CACHE_Q16_PERSIST_HEADER_CELLS,
    KV_CACHE_Q16_PERSIST_MAGIC,
    KV_CACHE_Q16_PERSIST_MODE_READ,
    KV_CACHE_Q16_PERSIST_MODE_WRITE,
    KV_CACHE_Q16_PERSIST_VERSION,
    KV_Q16_ERR_BAD_PARAM,
    KV_Q16_ERR_NULL_PTR,
    KV_Q16_OK,
    kv_cache_q16_compute_total_cells_checked,
    kv_cache_q16_persist_header_checked_nopartial,
    kv_cache_q16_persist_header_checked_nopartial_commit_only,
)


def kv_cache_q16_persist_header_checked_nopartial_commit_only_preflight_only(
    header_cells: list[int] | None,
    header_capacity: int,
    mode: int,
    inout_layer_count: list[int] | None,
    inout_token_capacity: list[int] | None,
    inout_kv_heads: list[int] | None,
    inout_head_dim: list[int] | None,
    inout_used_tokens: list[int] | None,
    inout_total_cells: list[int] | None,
) -> int:
    if (
        header_cells is None
        or inout_layer_count is None
        or inout_token_capacity is None
        or inout_kv_heads is None
        or inout_head_dim is None
        or inout_used_tokens is None
        or inout_total_cells is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if header_capacity < KV_CACHE_Q16_PERSIST_HEADER_CELLS:
        return KV_Q16_ERR_BAD_PARAM

    snapshot_header = list(header_cells[:KV_CACHE_Q16_PERSIST_HEADER_CELLS])
    staged_commit_header = list(snapshot_header)
    staged_canonical_header = list(snapshot_header)

    snapshot_header_capacity = header_capacity
    snapshot_mode = mode
    snap_layer = inout_layer_count[0]
    snap_tokens = inout_token_capacity[0]
    snap_heads = inout_kv_heads[0]
    snap_head_dim = inout_head_dim[0]
    snap_used = inout_used_tokens[0]
    snap_total = inout_total_cells[0]

    staged_commit_layer = [snap_layer]
    staged_commit_tokens = [snap_tokens]
    staged_commit_heads = [snap_heads]
    staged_commit_head_dim = [snap_head_dim]
    staged_commit_used = [snap_used]
    staged_commit_total = [snap_total]

    staged_canonical_layer = [snap_layer]
    staged_canonical_tokens = [snap_tokens]
    staged_canonical_heads = [snap_heads]
    staged_canonical_head_dim = [snap_head_dim]
    staged_canonical_used = [snap_used]
    staged_canonical_total = [snap_total]

    err = kv_cache_q16_persist_header_checked_nopartial_commit_only(
        staged_commit_header,
        header_capacity,
        mode,
        staged_commit_layer,
        staged_commit_tokens,
        staged_commit_heads,
        staged_commit_head_dim,
        staged_commit_used,
        staged_commit_total,
    )
    if err != KV_Q16_OK:
        return err

    err = kv_cache_q16_persist_header_checked_nopartial(
        staged_canonical_header,
        header_capacity,
        mode,
        staged_canonical_layer,
        staged_canonical_tokens,
        staged_canonical_heads,
        staged_canonical_head_dim,
        staged_canonical_used,
        staged_canonical_total,
    )
    if err != KV_Q16_OK:
        return err

    if snapshot_header_capacity != header_capacity or snapshot_mode != mode:
        return KV_Q16_ERR_BAD_PARAM

    if [
        inout_layer_count[0],
        inout_token_capacity[0],
        inout_kv_heads[0],
        inout_head_dim[0],
        inout_used_tokens[0],
        inout_total_cells[0],
    ] != [snap_layer, snap_tokens, snap_heads, snap_head_dim, snap_used, snap_total]:
        return KV_Q16_ERR_BAD_PARAM

    if header_cells[:KV_CACHE_Q16_PERSIST_HEADER_CELLS] != snapshot_header:
        return KV_Q16_ERR_BAD_PARAM

    if staged_commit_header != staged_canonical_header:
        return KV_Q16_ERR_BAD_PARAM

    if [
        staged_commit_layer[0],
        staged_commit_tokens[0],
        staged_commit_heads[0],
        staged_commit_head_dim[0],
        staged_commit_used[0],
        staged_commit_total[0],
    ] != [
        staged_canonical_layer[0],
        staged_canonical_tokens[0],
        staged_canonical_heads[0],
        staged_canonical_head_dim[0],
        staged_canonical_used[0],
        staged_canonical_total[0],
    ]:
        return KV_Q16_ERR_BAD_PARAM

    for idx in range(KV_CACHE_Q16_PERSIST_HEADER_CELLS):
        header_cells[idx] = staged_commit_header[idx]

    inout_layer_count[0] = staged_commit_layer[0]
    inout_token_capacity[0] = staged_commit_tokens[0]
    inout_kv_heads[0] = staged_commit_heads[0]
    inout_head_dim[0] = staged_commit_head_dim[0]
    inout_used_tokens[0] = staged_commit_used[0]
    inout_total_cells[0] = staged_commit_total[0]
    return KV_Q16_OK


def build_valid_header_tuple(
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    used_tokens: int,
) -> tuple[list[int], list[list[int]]]:
    err, total_cells = kv_cache_q16_compute_total_cells_checked(
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    )
    assert err == KV_Q16_OK

    header = [
        KV_CACHE_Q16_PERSIST_MAGIC,
        KV_CACHE_Q16_PERSIST_VERSION,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total_cells,
    ]
    tuple_ptrs = [
        [layer_count],
        [token_capacity],
        [kv_heads],
        [head_dim],
        [used_tokens],
        [total_cells],
    ]
    return header, tuple_ptrs


def test_source_contains_persist_header_commit_only_preflight_only_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16PersistHeaderCheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "snapshot_header" in body
    assert "staged_commit_header" in body
    assert "staged_canonical_header" in body
    assert "KVCacheQ16PersistHeaderCheckedNoPartialCommitOnly" in body
    assert "KVCacheQ16PersistHeaderCheckedNoPartial" in body
    assert "snapshot_header_capacity" in body
    assert "snapshot_mode" in body


def test_write_mode_matches_explicit_header_layout() -> None:
    layer_count = 4
    token_capacity = 24
    kv_heads = 2
    head_dim = 8
    used_tokens = 9

    err, total_cells = kv_cache_q16_compute_total_cells_checked(
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    )
    assert err == KV_Q16_OK

    header = [0x33] * KV_CACHE_Q16_PERSIST_HEADER_CELLS
    l = [layer_count]
    t = [token_capacity]
    h = [kv_heads]
    d = [head_dim]
    u = [used_tokens]
    c = [total_cells]

    err = kv_cache_q16_persist_header_checked_nopartial_commit_only_preflight_only(
        header,
        KV_CACHE_Q16_PERSIST_HEADER_CELLS,
        KV_CACHE_Q16_PERSIST_MODE_WRITE,
        l,
        t,
        h,
        d,
        u,
        c,
    )
    assert err == KV_Q16_OK

    assert header == [
        KV_CACHE_Q16_PERSIST_MAGIC,
        KV_CACHE_Q16_PERSIST_VERSION,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total_cells,
    ]
    assert [l[0], t[0], h[0], d[0], u[0], c[0]] == [
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total_cells,
    ]


def test_read_mode_preserves_header_and_publishes_tuple() -> None:
    header, tuple_ptrs = build_valid_header_tuple(3, 20, 4, 6, 11)
    l, t, h, d, u, c = tuple_ptrs

    # Distinct initial tuple to prove atomic publish comes from staged decode.
    l[0], t[0], h[0], d[0], u[0], c[0] = 99, 98, 97, 96, 95, 94
    header_before = list(header)

    err = kv_cache_q16_persist_header_checked_nopartial_commit_only_preflight_only(
        header,
        KV_CACHE_Q16_PERSIST_HEADER_CELLS,
        KV_CACHE_Q16_PERSIST_MODE_READ,
        l,
        t,
        h,
        d,
        u,
        c,
    )
    assert err == KV_Q16_OK

    assert header == header_before
    assert [l[0], t[0], h[0], d[0], u[0], c[0]] == [
        header[IDX_LAYERS],
        header[IDX_TOKENS],
        header[IDX_HEADS],
        header[IDX_HEAD_DIM],
        header[IDX_USED],
        header[IDX_TOTAL],
    ]


def test_failure_keeps_header_and_tuple_unchanged() -> None:
    # Invalid total cell count should fail in both staged paths and publish nothing.
    header = [
        KV_CACHE_Q16_PERSIST_MAGIC,
        KV_CACHE_Q16_PERSIST_VERSION,
        2,
        10,
        4,
        8,
        5,
        7777,
    ]
    header_before = list(header)

    l = [1]
    t = [2]
    h = [3]
    d = [4]
    u = [5]
    c = [6]
    tuple_before = [l[0], t[0], h[0], d[0], u[0], c[0]]

    err = kv_cache_q16_persist_header_checked_nopartial_commit_only_preflight_only(
        header,
        KV_CACHE_Q16_PERSIST_HEADER_CELLS,
        KV_CACHE_Q16_PERSIST_MODE_READ,
        l,
        t,
        h,
        d,
        u,
        c,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert header == header_before
    assert [l[0], t[0], h[0], d[0], u[0], c[0]] == tuple_before


def test_null_and_capacity_guards() -> None:
    header, tuple_ptrs = build_valid_header_tuple(1, 8, 2, 4, 3)
    l, t, h, d, u, c = tuple_ptrs

    assert (
        kv_cache_q16_persist_header_checked_nopartial_commit_only_preflight_only(
            None,
            KV_CACHE_Q16_PERSIST_HEADER_CELLS,
            KV_CACHE_Q16_PERSIST_MODE_WRITE,
            l,
            t,
            h,
            d,
            u,
            c,
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_persist_header_checked_nopartial_commit_only_preflight_only(
            header,
            KV_CACHE_Q16_PERSIST_HEADER_CELLS - 1,
            KV_CACHE_Q16_PERSIST_MODE_WRITE,
            l,
            t,
            h,
            d,
            u,
            c,
        )
        == KV_Q16_ERR_BAD_PARAM
    )


def test_randomized_read_write_vectors() -> None:
    rng = random.Random(0x1285)

    for _ in range(200):
        layer_count = rng.randint(0, 6)
        token_capacity = rng.randint(0, 24)
        kv_heads = rng.randint(0, 8)
        head_dim = rng.randint(0, 16)
        err, total_cells = kv_cache_q16_compute_total_cells_checked(
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
        )
        if err != KV_Q16_OK:
            continue

        used_tokens = rng.randint(0, token_capacity)
        mode = rng.choice([KV_CACHE_Q16_PERSIST_MODE_WRITE, KV_CACHE_Q16_PERSIST_MODE_READ])

        if mode == KV_CACHE_Q16_PERSIST_MODE_READ:
            header = [
                KV_CACHE_Q16_PERSIST_MAGIC,
                KV_CACHE_Q16_PERSIST_VERSION,
                layer_count,
                token_capacity,
                kv_heads,
                head_dim,
                used_tokens,
                total_cells,
            ]
            l = [rng.randint(0, 127)]
            t = [rng.randint(0, 127)]
            h = [rng.randint(0, 127)]
            d = [rng.randint(0, 127)]
            u = [rng.randint(0, 127)]
            c = [rng.randint(0, 127)]
        else:
            header = [rng.randint(0, 0xFFFF) for _ in range(KV_CACHE_Q16_PERSIST_HEADER_CELLS)]
            l = [layer_count]
            t = [token_capacity]
            h = [kv_heads]
            d = [head_dim]
            u = [used_tokens]
            c = [total_cells]

        err = kv_cache_q16_persist_header_checked_nopartial_commit_only_preflight_only(
            header,
            KV_CACHE_Q16_PERSIST_HEADER_CELLS,
            mode,
            l,
            t,
            h,
            d,
            u,
            c,
        )
        assert err == KV_Q16_OK

        assert [l[0], t[0], h[0], d[0], u[0], c[0]] == [
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            used_tokens,
            total_cells,
        ]

        assert header[IDX_MAGIC] == KV_CACHE_Q16_PERSIST_MAGIC
        assert header[IDX_VERSION] == KV_CACHE_Q16_PERSIST_VERSION
        assert header[IDX_LAYERS] == layer_count
        assert header[IDX_TOKENS] == token_capacity
        assert header[IDX_HEADS] == kv_heads
        assert header[IDX_HEAD_DIM] == head_dim
        assert header[IDX_USED] == used_tokens
        assert header[IDX_TOTAL] == total_cells
