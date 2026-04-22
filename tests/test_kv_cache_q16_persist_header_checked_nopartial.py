#!/usr/bin/env python3
"""Parity harness for KVCacheQ16PersistHeaderCheckedNoPartial (IQ-1169)."""

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
    kv_cache_q16_compute_layer_token_span_cells_checked,
    try_mul_i64_checked,
)

KV_CACHE_Q16_PERSIST_MODE_WRITE = 0
KV_CACHE_Q16_PERSIST_MODE_READ = 1

KV_CACHE_Q16_PERSIST_MAGIC = 0x4B56435131364844
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


def kv_cache_q16_compute_total_cells_checked(
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[int, int]:
    if layer_count < 0 or token_capacity < 0 or kv_heads < 0 or head_dim < 0:
        return KV_Q16_ERR_BAD_PARAM, 0

    span = [0]
    err = kv_cache_q16_compute_layer_token_span_cells_checked(kv_heads, head_dim, span)
    if err != KV_Q16_OK:
        return err, 0

    err, layer_span = try_mul_i64_checked(token_capacity, span[0])
    if err != KV_Q16_OK:
        return err, 0

    err, total = try_mul_i64_checked(layer_count, layer_span)
    if err != KV_Q16_OK:
        return err, 0

    return KV_Q16_OK, total


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
        staged_layer_count = inout_layer_count[0]
        staged_token_capacity = inout_token_capacity[0]
        staged_kv_heads = inout_kv_heads[0]
        staged_head_dim = inout_head_dim[0]
        staged_used_tokens = inout_used_tokens[0]
        staged_total_cells = inout_total_cells[0]

        if (
            staged_layer_count < 0
            or staged_token_capacity < 0
            or staged_kv_heads < 0
            or staged_head_dim < 0
            or staged_used_tokens < 0
            or staged_total_cells < 0
        ):
            return KV_Q16_ERR_BAD_PARAM

        if staged_used_tokens > staged_token_capacity:
            return KV_Q16_ERR_BAD_PARAM

        err, computed_total_cells = kv_cache_q16_compute_total_cells_checked(
            staged_layer_count,
            staged_token_capacity,
            staged_kv_heads,
            staged_head_dim,
        )
        if err != KV_Q16_OK:
            return err

        if computed_total_cells != staged_total_cells:
            return KV_Q16_ERR_BAD_PARAM

        staged_header = [0] * KV_CACHE_Q16_PERSIST_HEADER_CELLS
        staged_header[IDX_MAGIC] = KV_CACHE_Q16_PERSIST_MAGIC
        staged_header[IDX_VERSION] = KV_CACHE_Q16_PERSIST_VERSION
        staged_header[IDX_LAYERS] = staged_layer_count
        staged_header[IDX_TOKENS] = staged_token_capacity
        staged_header[IDX_HEADS] = staged_kv_heads
        staged_header[IDX_HEAD_DIM] = staged_head_dim
        staged_header[IDX_USED] = staged_used_tokens
        staged_header[IDX_TOTAL] = staged_total_cells

        for idx in range(KV_CACHE_Q16_PERSIST_HEADER_CELLS):
            header_cells[idx] = staged_header[idx]
        return KV_Q16_OK

    if mode == KV_CACHE_Q16_PERSIST_MODE_READ:
        staged_header = [header_cells[idx] for idx in range(KV_CACHE_Q16_PERSIST_HEADER_CELLS)]

        if staged_header[IDX_MAGIC] != KV_CACHE_Q16_PERSIST_MAGIC:
            return KV_Q16_ERR_BAD_PARAM
        if staged_header[IDX_VERSION] != KV_CACHE_Q16_PERSIST_VERSION:
            return KV_Q16_ERR_BAD_PARAM

        staged_layer_count = staged_header[IDX_LAYERS]
        staged_token_capacity = staged_header[IDX_TOKENS]
        staged_kv_heads = staged_header[IDX_HEADS]
        staged_head_dim = staged_header[IDX_HEAD_DIM]
        staged_used_tokens = staged_header[IDX_USED]
        staged_total_cells = staged_header[IDX_TOTAL]

        if (
            staged_layer_count < 0
            or staged_token_capacity < 0
            or staged_kv_heads < 0
            or staged_head_dim < 0
            or staged_used_tokens < 0
            or staged_total_cells < 0
        ):
            return KV_Q16_ERR_BAD_PARAM

        if staged_used_tokens > staged_token_capacity:
            return KV_Q16_ERR_BAD_PARAM

        err, computed_total_cells = kv_cache_q16_compute_total_cells_checked(
            staged_layer_count,
            staged_token_capacity,
            staged_kv_heads,
            staged_head_dim,
        )
        if err != KV_Q16_OK:
            return err
        if computed_total_cells != staged_total_cells:
            return KV_Q16_ERR_BAD_PARAM

        inout_layer_count[0] = staged_layer_count
        inout_token_capacity[0] = staged_token_capacity
        inout_kv_heads[0] = staged_kv_heads
        inout_head_dim[0] = staged_head_dim
        inout_used_tokens[0] = staged_used_tokens
        inout_total_cells[0] = staged_total_cells
        return KV_Q16_OK

    return KV_Q16_ERR_BAD_PARAM


def explicit_write_header(
    layer_count: int,
    token_capacity: int,
    kv_heads: int,
    head_dim: int,
    used_tokens: int,
    total_cells: int,
) -> tuple[int, list[int]]:
    err, computed_total = kv_cache_q16_compute_total_cells_checked(
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    )
    if err != KV_Q16_OK:
        return err, []
    if used_tokens < 0 or used_tokens > token_capacity:
        return KV_Q16_ERR_BAD_PARAM, []
    if total_cells != computed_total:
        return KV_Q16_ERR_BAD_PARAM, []

    return (
        KV_Q16_OK,
        [
            KV_CACHE_Q16_PERSIST_MAGIC,
            KV_CACHE_Q16_PERSIST_VERSION,
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
            used_tokens,
            total_cells,
        ],
    )


def explicit_read_header(header_cells: list[int]) -> tuple[int, tuple[int, int, int, int, int, int]]:
    if len(header_cells) < KV_CACHE_Q16_PERSIST_HEADER_CELLS:
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0, 0, 0)

    if header_cells[IDX_MAGIC] != KV_CACHE_Q16_PERSIST_MAGIC:
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0, 0, 0)
    if header_cells[IDX_VERSION] != KV_CACHE_Q16_PERSIST_VERSION:
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0, 0, 0)

    layer_count = header_cells[IDX_LAYERS]
    token_capacity = header_cells[IDX_TOKENS]
    kv_heads = header_cells[IDX_HEADS]
    head_dim = header_cells[IDX_HEAD_DIM]
    used_tokens = header_cells[IDX_USED]
    total_cells = header_cells[IDX_TOTAL]

    if (
        layer_count < 0
        or token_capacity < 0
        or kv_heads < 0
        or head_dim < 0
        or used_tokens < 0
        or total_cells < 0
    ):
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0, 0, 0)

    if used_tokens > token_capacity:
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0, 0, 0)

    err, computed_total = kv_cache_q16_compute_total_cells_checked(
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0, 0, 0, 0)
    if computed_total != total_cells:
        return KV_Q16_ERR_BAD_PARAM, (0, 0, 0, 0, 0, 0)

    return KV_Q16_OK, (
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total_cells,
    )


def test_source_contains_persist_header_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    signature = "I32 KVCacheQ16PersistHeaderCheckedNoPartial("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "KV_CACHE_Q16_PERSIST_MAGIC" in body
    assert "KV_CACHE_Q16_PERSIST_MODE_WRITE" in body
    assert "KV_CACHE_Q16_PERSIST_MODE_READ" in body
    assert "KVCacheQ16ComputeTotalCellsChecked" in body
    assert "header_cells[cell_idx] = staged_header[cell_idx]" in body


def test_write_mode_known_vector() -> None:
    layer_count = [16]
    token_capacity = [512]
    kv_heads = [8]
    head_dim = [128]
    used_tokens = [123]
    err, total_cells = kv_cache_q16_compute_total_cells_checked(16, 512, 8, 128)
    assert err == KV_Q16_OK
    total = [total_cells]

    header = [0xDEADBEEF] * 16

    err = kv_cache_q16_persist_header_checked_nopartial(
        header,
        16,
        KV_CACHE_Q16_PERSIST_MODE_WRITE,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total,
    )
    assert err == KV_Q16_OK

    explicit_err, expected = explicit_write_header(16, 512, 8, 128, 123, total_cells)
    assert explicit_err == KV_Q16_OK
    assert header[:8] == expected


def test_read_mode_roundtrip_known_vector() -> None:
    err, total_cells = kv_cache_q16_compute_total_cells_checked(4, 64, 4, 64)
    assert err == KV_Q16_OK

    header = [
        KV_CACHE_Q16_PERSIST_MAGIC,
        KV_CACHE_Q16_PERSIST_VERSION,
        4,
        64,
        4,
        64,
        31,
        total_cells,
        999,
        1000,
    ]

    layer_count = [-1]
    token_capacity = [-1]
    kv_heads = [-1]
    head_dim = [-1]
    used_tokens = [-1]
    total = [-1]

    got_err = kv_cache_q16_persist_header_checked_nopartial(
        header,
        len(header),
        KV_CACHE_Q16_PERSIST_MODE_READ,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total,
    )
    assert got_err == KV_Q16_OK

    exp_err, expected_tuple = explicit_read_header(header)
    assert exp_err == KV_Q16_OK
    assert (
        layer_count[0],
        token_capacity[0],
        kv_heads[0],
        head_dim[0],
        used_tokens[0],
        total[0],
    ) == expected_tuple


def test_invalid_mode_and_capacity_rejected_without_publish() -> None:
    header = [0] * 8
    layer_count = [7]
    token_capacity = [8]
    kv_heads = [9]
    head_dim = [10]
    used_tokens = [1]
    total = [2]

    snapshot = (
        layer_count[0],
        token_capacity[0],
        kv_heads[0],
        head_dim[0],
        used_tokens[0],
        total[0],
        tuple(header),
    )

    err = kv_cache_q16_persist_header_checked_nopartial(
        header,
        8,
        99,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert (
        layer_count[0],
        token_capacity[0],
        kv_heads[0],
        head_dim[0],
        used_tokens[0],
        total[0],
        tuple(header),
    ) == snapshot

    err = kv_cache_q16_persist_header_checked_nopartial(
        header,
        7,
        KV_CACHE_Q16_PERSIST_MODE_WRITE,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert (
        layer_count[0],
        token_capacity[0],
        kv_heads[0],
        head_dim[0],
        used_tokens[0],
        total[0],
        tuple(header),
    ) == snapshot


def test_no_partial_write_on_invalid_write_tuple() -> None:
    header = [111 + i for i in range(8)]
    layer_count = [2]
    token_capacity = [32]
    kv_heads = [4]
    head_dim = [16]
    used_tokens = [33]
    total = [0]

    snapshot_header = list(header)
    snapshot_tuple = (
        layer_count[0],
        token_capacity[0],
        kv_heads[0],
        head_dim[0],
        used_tokens[0],
        total[0],
    )

    err = kv_cache_q16_persist_header_checked_nopartial(
        header,
        8,
        KV_CACHE_Q16_PERSIST_MODE_WRITE,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert header == snapshot_header
    assert (
        layer_count[0],
        token_capacity[0],
        kv_heads[0],
        head_dim[0],
        used_tokens[0],
        total[0],
    ) == snapshot_tuple


def test_no_partial_publish_on_invalid_read_header() -> None:
    err, total_cells = kv_cache_q16_compute_total_cells_checked(2, 8, 2, 8)
    assert err == KV_Q16_OK

    header = [
        KV_CACHE_Q16_PERSIST_MAGIC,
        KV_CACHE_Q16_PERSIST_VERSION,
        2,
        8,
        2,
        8,
        9,
        total_cells,
    ]

    layer_count = [100]
    token_capacity = [101]
    kv_heads = [102]
    head_dim = [103]
    used_tokens = [104]
    total = [105]

    snapshot = (
        layer_count[0],
        token_capacity[0],
        kv_heads[0],
        head_dim[0],
        used_tokens[0],
        total[0],
    )

    got_err = kv_cache_q16_persist_header_checked_nopartial(
        header,
        8,
        KV_CACHE_Q16_PERSIST_MODE_READ,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total,
    )

    assert got_err == KV_Q16_ERR_BAD_PARAM
    assert (
        layer_count[0],
        token_capacity[0],
        kv_heads[0],
        head_dim[0],
        used_tokens[0],
        total[0],
    ) == snapshot


def test_overflow_surface_matches_total_cell_math() -> None:
    header = [0] * 8

    layer_count = [I64_MAX]
    token_capacity = [2]
    kv_heads = [2]
    head_dim = [2]
    used_tokens = [0]
    total = [0]

    err = kv_cache_q16_persist_header_checked_nopartial(
        header,
        8,
        KV_CACHE_Q16_PERSIST_MODE_WRITE,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        used_tokens,
        total,
    )
    assert err == KV_Q16_ERR_OVERFLOW


def test_randomized_write_read_roundtrip() -> None:
    rng = random.Random(1169)

    for _ in range(400):
        layer_count = rng.randint(0, 48)
        token_capacity = rng.randint(0, 256)
        kv_heads = rng.randint(0, 32)
        head_dim = rng.randint(0, 256)

        err, total_cells = kv_cache_q16_compute_total_cells_checked(
            layer_count,
            token_capacity,
            kv_heads,
            head_dim,
        )
        if err == KV_Q16_ERR_OVERFLOW:
            continue
        assert err == KV_Q16_OK

        used_tokens = 0 if token_capacity == 0 else rng.randint(0, token_capacity)

        header = [0xABCD] * 16
        layer_box = [layer_count]
        token_box = [token_capacity]
        heads_box = [kv_heads]
        dim_box = [head_dim]
        used_box = [used_tokens]
        total_box = [total_cells]

        write_err = kv_cache_q16_persist_header_checked_nopartial(
            header,
            16,
            KV_CACHE_Q16_PERSIST_MODE_WRITE,
            layer_box,
            token_box,
            heads_box,
            dim_box,
            used_box,
            total_box,
        )
        assert write_err == KV_Q16_OK

        read_layer = [-1]
        read_token = [-1]
        read_heads = [-1]
        read_dim = [-1]
        read_used = [-1]
        read_total = [-1]

        read_err = kv_cache_q16_persist_header_checked_nopartial(
            header,
            16,
            KV_CACHE_Q16_PERSIST_MODE_READ,
            read_layer,
            read_token,
            read_heads,
            read_dim,
            read_used,
            read_total,
        )
        assert read_err == KV_Q16_OK
        assert read_layer == [layer_count]
        assert read_token == [token_capacity]
        assert read_heads == [kv_heads]
        assert read_dim == [head_dim]
        assert read_used == [used_tokens]
        assert read_total == [total_cells]


def test_null_ptr_rejections() -> None:
    header = [0] * 8
    layer = [1]
    tokens = [2]
    heads = [3]
    dim = [4]
    used = [0]
    total = [0]

    assert (
        kv_cache_q16_persist_header_checked_nopartial(
            None,
            8,
            KV_CACHE_Q16_PERSIST_MODE_WRITE,
            layer,
            tokens,
            heads,
            dim,
            used,
            total,
        )
        == KV_Q16_ERR_NULL_PTR
    )
    assert (
        kv_cache_q16_persist_header_checked_nopartial(
            header,
            8,
            KV_CACHE_Q16_PERSIST_MODE_WRITE,
            None,
            tokens,
            heads,
            dim,
            used,
            total,
        )
        == KV_Q16_ERR_NULL_PTR
    )


if __name__ == "__main__":
    test_source_contains_persist_header_helper()
    test_write_mode_known_vector()
    test_read_mode_roundtrip_known_vector()
    test_invalid_mode_and_capacity_rejected_without_publish()
    test_no_partial_write_on_invalid_write_tuple()
    test_no_partial_publish_on_invalid_read_header()
    test_overflow_surface_matches_total_cell_math()
    test_randomized_write_read_roundtrip()
    test_null_ptr_rejections()
    print("ok")
