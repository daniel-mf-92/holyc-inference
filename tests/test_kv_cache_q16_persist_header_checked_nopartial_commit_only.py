#!/usr/bin/env python3
"""Parity + no-partial tests for KVCacheQ16PersistHeaderCheckedNoPartialCommitOnly."""

from __future__ import annotations

KV_Q16_OK = 0
KV_Q16_ERR_NULL_PTR = -1
KV_Q16_ERR_BAD_PARAM = -2
KV_Q16_ERR_OVERFLOW = -3

KV_CACHE_Q16_PERSIST_MODE_WRITE = 1
KV_CACHE_Q16_PERSIST_MODE_READ = 2

KV_CACHE_Q16_PERSIST_MAGIC = 0x4B56483136504844
KV_CACHE_Q16_PERSIST_VERSION = 1
KV_CACHE_Q16_PERSIST_HEADER_CELLS = 8

IDX_MAGIC = 0
IDX_VERSION = 1
IDX_LAYERS = 2
IDX_TOKENS = 3
IDX_HEADS = 4
IDX_HEAD_DIM = 5
IDX_USED = 6
IDX_TOTAL = 7

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


def kv_try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if lhs == 0 or rhs == 0:
        return KV_Q16_OK, 0
    if lhs == -1:
        if rhs == I64_MIN:
            return KV_Q16_ERR_OVERFLOW, 0
        return KV_Q16_OK, -rhs
    if rhs == -1:
        if lhs == I64_MIN:
            return KV_Q16_ERR_OVERFLOW, 0
        return KV_Q16_OK, -lhs
    if lhs > 0:
        if rhs > 0:
            if lhs > (I64_MAX // rhs):
                return KV_Q16_ERR_OVERFLOW, 0
        else:
            if rhs < (I64_MIN // lhs):
                return KV_Q16_ERR_OVERFLOW, 0
    else:
        if rhs > 0:
            if lhs < (I64_MIN // rhs):
                return KV_Q16_ERR_OVERFLOW, 0
        else:
            if lhs != 0 and rhs < (I64_MAX // lhs):
                return KV_Q16_ERR_OVERFLOW, 0
    return KV_Q16_OK, lhs * rhs


def kv_cache_q16_compute_total_cells_checked(
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[int, int]:
    if layer_count < 0 or token_capacity < 0 or kv_heads < 0 or head_dim < 0:
        return KV_Q16_ERR_BAD_PARAM, 0
    err, token_span = kv_try_mul_i64_checked(kv_heads, head_dim)
    if err != KV_Q16_OK:
        return err, 0
    err, layer_span = kv_try_mul_i64_checked(token_capacity, token_span)
    if err != KV_Q16_OK:
        return err, 0
    return kv_try_mul_i64_checked(layer_count, layer_span)


def kv_cache_q16_persist_header_checked_nopartial(
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

    if mode == KV_CACHE_Q16_PERSIST_MODE_WRITE:
        layer_count = inout_layer_count[0]
        token_capacity = inout_token_capacity[0]
        kv_heads = inout_kv_heads[0]
        head_dim = inout_head_dim[0]
        used_tokens = inout_used_tokens[0]
        total_cells = inout_total_cells[0]

        if min(layer_count, token_capacity, kv_heads, head_dim, used_tokens, total_cells) < 0:
            return KV_Q16_ERR_BAD_PARAM
        if used_tokens > token_capacity:
            return KV_Q16_ERR_BAD_PARAM

        err, computed_total = kv_cache_q16_compute_total_cells_checked(
            layer_count, token_capacity, kv_heads, head_dim
        )
        if err != KV_Q16_OK:
            return err
        if computed_total != total_cells:
            return KV_Q16_ERR_BAD_PARAM

        staged = [0] * KV_CACHE_Q16_PERSIST_HEADER_CELLS
        staged[IDX_MAGIC] = KV_CACHE_Q16_PERSIST_MAGIC
        staged[IDX_VERSION] = KV_CACHE_Q16_PERSIST_VERSION
        staged[IDX_LAYERS] = layer_count
        staged[IDX_TOKENS] = token_capacity
        staged[IDX_HEADS] = kv_heads
        staged[IDX_HEAD_DIM] = head_dim
        staged[IDX_USED] = used_tokens
        staged[IDX_TOTAL] = total_cells

        for idx in range(KV_CACHE_Q16_PERSIST_HEADER_CELLS):
            header_cells[idx] = staged[idx]
        return KV_Q16_OK

    if mode == KV_CACHE_Q16_PERSIST_MODE_READ:
        staged = [header_cells[idx] for idx in range(KV_CACHE_Q16_PERSIST_HEADER_CELLS)]
        if staged[IDX_MAGIC] != KV_CACHE_Q16_PERSIST_MAGIC:
            return KV_Q16_ERR_BAD_PARAM
        if staged[IDX_VERSION] != KV_CACHE_Q16_PERSIST_VERSION:
            return KV_Q16_ERR_BAD_PARAM

        layer_count = staged[IDX_LAYERS]
        token_capacity = staged[IDX_TOKENS]
        kv_heads = staged[IDX_HEADS]
        head_dim = staged[IDX_HEAD_DIM]
        used_tokens = staged[IDX_USED]
        total_cells = staged[IDX_TOTAL]

        if min(layer_count, token_capacity, kv_heads, head_dim, used_tokens, total_cells) < 0:
            return KV_Q16_ERR_BAD_PARAM
        if used_tokens > token_capacity:
            return KV_Q16_ERR_BAD_PARAM

        err, computed_total = kv_cache_q16_compute_total_cells_checked(
            layer_count, token_capacity, kv_heads, head_dim
        )
        if err != KV_Q16_OK:
            return err
        if computed_total != total_cells:
            return KV_Q16_ERR_BAD_PARAM

        inout_layer_count[0] = layer_count
        inout_token_capacity[0] = token_capacity
        inout_kv_heads[0] = kv_heads
        inout_head_dim[0] = head_dim
        inout_used_tokens[0] = used_tokens
        inout_total_cells[0] = total_cells
        return KV_Q16_OK

    return KV_Q16_ERR_BAD_PARAM


def kv_cache_q16_persist_header_checked_nopartial_commit_only(
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

    snap_layer = inout_layer_count[0]
    snap_tokens = inout_token_capacity[0]
    snap_heads = inout_kv_heads[0]
    snap_head_dim = inout_head_dim[0]
    snap_used = inout_used_tokens[0]
    snap_total = inout_total_cells[0]

    staged_layer = [snap_layer]
    staged_tokens = [snap_tokens]
    staged_heads = [snap_heads]
    staged_head_dim = [snap_head_dim]
    staged_used = [snap_used]
    staged_total = [snap_total]

    err = kv_cache_q16_persist_header_checked_nopartial(
        header_cells,
        header_capacity,
        mode,
        staged_layer,
        staged_tokens,
        staged_heads,
        staged_head_dim,
        staged_used,
        staged_total,
    )
    if err != KV_Q16_OK:
        return err

    inout_layer_count[0] = staged_layer[0]
    inout_token_capacity[0] = staged_tokens[0]
    inout_kv_heads[0] = staged_heads[0]
    inout_head_dim[0] = staged_head_dim[0]
    inout_used_tokens[0] = staged_used[0]
    inout_total_cells[0] = staged_total[0]
    return KV_Q16_OK


def test_write_commit_only_matches_canonical_and_writes_header() -> None:
    layer_count, token_capacity, kv_heads, head_dim, used_tokens = 2, 16, 4, 8, 7
    _, total_cells = kv_cache_q16_compute_total_cells_checked(
        layer_count, token_capacity, kv_heads, head_dim
    )
    header = [0x77] * KV_CACHE_Q16_PERSIST_HEADER_CELLS

    l = [layer_count]
    t = [token_capacity]
    h = [kv_heads]
    d = [head_dim]
    u = [used_tokens]
    c = [total_cells]

    err = kv_cache_q16_persist_header_checked_nopartial_commit_only(
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


def test_read_commit_only_publishes_tuple_atomically() -> None:
    layer_count, token_capacity, kv_heads, head_dim, used_tokens = 3, 20, 2, 16, 9
    _, total_cells = kv_cache_q16_compute_total_cells_checked(
        layer_count, token_capacity, kv_heads, head_dim
    )
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

    l, t, h, d, u, c = [111], [222], [333], [444], [555], [666]
    err = kv_cache_q16_persist_header_checked_nopartial_commit_only(
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
    assert [l[0], t[0], h[0], d[0], u[0], c[0]] == [
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total_cells,
    ]


def test_failure_preserves_tuple_no_partial_publish() -> None:
    layer_count, token_capacity, kv_heads, head_dim, used_tokens = 2, 16, 4, 8, 7
    _, total_cells = kv_cache_q16_compute_total_cells_checked(
        layer_count, token_capacity, kv_heads, head_dim
    )
    header = [
        KV_CACHE_Q16_PERSIST_MAGIC,
        KV_CACHE_Q16_PERSIST_VERSION,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total_cells + 1,
    ]

    l, t, h, d, u, c = [10], [11], [12], [13], [14], [15]
    before = [l[0], t[0], h[0], d[0], u[0], c[0]]
    err = kv_cache_q16_persist_header_checked_nopartial_commit_only(
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
    assert [l[0], t[0], h[0], d[0], u[0], c[0]] == before


def test_null_and_capacity_validation() -> None:
    l, t, h, d, u, c = [1], [2], [3], [4], [0], [24]
    header = [0] * KV_CACHE_Q16_PERSIST_HEADER_CELLS

    assert (
        kv_cache_q16_persist_header_checked_nopartial_commit_only(
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
        kv_cache_q16_persist_header_checked_nopartial_commit_only(
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
