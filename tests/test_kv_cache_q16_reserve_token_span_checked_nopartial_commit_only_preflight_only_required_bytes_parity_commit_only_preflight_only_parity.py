#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ReserveTokenSpan...RequiredBytesParityCommitOnlyPreflightOnlyParity (IQ-1072)."""

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
)
from test_kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only import (
    kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only,
)
from test_kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only import (
    kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only,
)


def kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
    layer_count: int,
    head_count: int,
    head_dim: int,
    token_start: int,
    token_count: int,
    cache_capacity: int,
    out_required_cells: list[int] | None,
    out_required_bytes: list[int] | None,
    out_last_cell_index: list[int] | None,
) -> int:
    if out_required_cells is None or out_required_bytes is None or out_last_cell_index is None:
        return KV_Q16_ERR_NULL_PTR

    if out_required_cells is out_required_bytes or out_required_cells is out_last_cell_index or out_required_bytes is out_last_cell_index:
        return KV_Q16_ERR_BAD_PARAM

    if layer_count < 0 or head_count < 0 or head_dim < 0 or token_start < 0 or token_count < 0 or cache_capacity < 0:
        return KV_Q16_ERR_BAD_PARAM

    snapshot_geom = (layer_count, head_count, head_dim, token_start, token_count, cache_capacity)
    snapshot_ptrs = (out_required_cells, out_required_bytes, out_last_cell_index)

    pre_cells = [0]
    pre_bytes = [0]
    pre_last = [0]
    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
        layer_count, head_count, head_dim, token_start, token_count, cache_capacity, pre_cells, pre_bytes, pre_last
    )
    if err != KV_Q16_OK:
        return err

    com_cells = [0]
    com_bytes = [0]
    com_last = [0]
    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
        layer_count, head_count, head_dim, token_start, token_count, cache_capacity, com_cells, com_bytes, com_last
    )
    if err != KV_Q16_OK:
        return err

    if snapshot_geom != (layer_count, head_count, head_dim, token_start, token_count, cache_capacity):
        return KV_Q16_ERR_BAD_PARAM
    if snapshot_ptrs != (out_required_cells, out_required_bytes, out_last_cell_index):
        return KV_Q16_ERR_BAD_PARAM

    if pre_cells[0] < 0 or pre_bytes[0] < 0 or pre_last[0] < 0 or com_cells[0] < 0 or com_bytes[0] < 0 or com_last[0] < 0:
        return KV_Q16_ERR_BAD_PARAM

    if (pre_bytes[0] % 8) != 0 or (pre_bytes[0] // 8) != pre_cells[0]:
        return KV_Q16_ERR_BAD_PARAM
    if (com_bytes[0] % 8) != 0 or (com_bytes[0] // 8) != com_cells[0]:
        return KV_Q16_ERR_BAD_PARAM

    if pre_cells[0] == 0 and pre_last[0] != 0:
        return KV_Q16_ERR_BAD_PARAM
    if com_cells[0] == 0 and com_last[0] != 0:
        return KV_Q16_ERR_BAD_PARAM

    if pre_cells[0] != com_cells[0] or pre_bytes[0] != com_bytes[0] or pre_last[0] != com_last[0]:
        return KV_Q16_ERR_BAD_PARAM

    out_required_cells[0] = pre_cells[0]
    out_required_bytes[0] = pre_bytes[0]
    out_last_cell_index[0] = pre_last[0]
    return KV_Q16_OK


def explicit_composition(
    layer_count: int,
    head_count: int,
    head_dim: int,
    token_start: int,
    token_count: int,
    cache_capacity: int,
) -> tuple[int, tuple[int, int, int]]:
    out_cells = [0]
    out_bytes = [0]
    out_last = [0]
    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
        layer_count, head_count, head_dim, token_start, token_count, cache_capacity, out_cells, out_bytes, out_last
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0)
    return KV_Q16_OK, (out_cells[0], out_bytes[0], out_last[0])


def test_source_contains_required_bytes_parity_commit_only_preflight_only_parity_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ReserveTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "KVCacheQ16ReserveTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyPreflightOnly(" in body
    assert "KVCacheQ16ReserveTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnly(" in body
    assert "if (staged_preflight_required_cells != staged_commit_required_cells)" in body


def test_known_vector_required_cells_bytes_and_last_index() -> None:
    layer_count = 3
    head_count = 4
    head_dim = 16
    token_start = 5
    token_count = 7

    cells_per_token = layer_count * head_count * head_dim
    required_cells = token_count * cells_per_token
    base_index = token_start * cells_per_token
    end_index = base_index + required_cells

    out_cells = [111]
    out_bytes = [222]
    out_last = [333]

    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
        layer_count, head_count, head_dim, token_start, token_count, end_index, out_cells, out_bytes, out_last
    )

    assert err == KV_Q16_OK
    assert out_cells == [required_cells]
    assert out_bytes == [required_cells * 8]
    assert out_last == [end_index - 1]


def test_zero_token_count_returns_zero_tuple() -> None:
    out_cells = [10]
    out_bytes = [11]
    out_last = [12]

    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
        layer_count=2,
        head_count=8,
        head_dim=32,
        token_start=9,
        token_count=0,
        cache_capacity=10_000,
        out_required_cells=out_cells,
        out_required_bytes=out_bytes,
        out_last_cell_index=out_last,
    )

    assert err == KV_Q16_OK
    assert out_cells == [0]
    assert out_bytes == [0]
    assert out_last == [0]


def test_error_paths_preserve_outputs() -> None:
    out_cells = [91]
    out_bytes = [92]
    out_last = [93]

    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
        layer_count=-1,
        head_count=1,
        head_dim=1,
        token_start=0,
        token_count=1,
        cache_capacity=100,
        out_required_cells=out_cells,
        out_required_bytes=out_bytes,
        out_last_cell_index=out_last,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert out_cells == [91]
    assert out_bytes == [92]
    assert out_last == [93]


def test_null_and_alias_guards() -> None:
    out_cells = [0]
    out_bytes = [0]
    out_last = [0]

    assert (
        kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
            1, 1, 1, 0, 0, 10, None, out_bytes, out_last
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
            1, 1, 1, 0, 0, 10, out_cells, out_cells, out_last
        )
        == KV_Q16_ERR_BAD_PARAM
    )


def test_overflow_surfaces() -> None:
    out_cells = [0]
    out_bytes = [0]
    out_last = [0]

    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
        layer_count=I64_MAX,
        head_count=2,
        head_dim=1,
        token_start=0,
        token_count=1,
        cache_capacity=I64_MAX,
        out_required_cells=out_cells,
        out_required_bytes=out_bytes,
        out_last_cell_index=out_last,
    )
    assert err == KV_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1072)

    for _ in range(2000):
        layer_count = rng.randint(0, 16)
        head_count = rng.randint(0, 32)
        head_dim = rng.randint(0, 128)
        token_start = rng.randint(0, 128)
        token_count = rng.randint(0, 128)

        cells_per_token = layer_count * head_count * head_dim
        base = token_start * cells_per_token
        required = token_count * cells_per_token
        min_capacity = base + required
        capacity = min_capacity + rng.randint(0, 64)

        got_cells = [7001]
        got_bytes = [7002]
        got_last = [7003]

        err_new = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
            layer_count, head_count, head_dim, token_start, token_count, capacity, got_cells, got_bytes, got_last
        )
        err_ref, ref_tuple = explicit_composition(layer_count, head_count, head_dim, token_start, token_count, capacity)

        assert err_new == err_ref
        if err_new == KV_Q16_OK:
            assert got_cells == [ref_tuple[0]]
            assert got_bytes == [ref_tuple[1]]
            assert got_last == [ref_tuple[2]]
        else:
            assert got_cells == [7001]
            assert got_bytes == [7002]
            assert got_last == [7003]


if __name__ == "__main__":
    test_source_contains_required_bytes_parity_commit_only_preflight_only_parity_helper()
    test_known_vector_required_cells_bytes_and_last_index()
    test_zero_token_count_returns_zero_tuple()
    test_error_paths_preserve_outputs()
    test_null_and_alias_guards()
    test_overflow_surfaces()
    test_randomized_parity_vs_explicit_composition()
    print(
        "kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity=ok"
    )
