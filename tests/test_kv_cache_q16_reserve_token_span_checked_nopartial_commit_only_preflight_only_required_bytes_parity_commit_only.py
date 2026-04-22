#!/usr/bin/env python3
"""Parity harness for KVCacheQ16ReserveTokenSpan...RequiredBytesParityCommitOnly (IQ-1070)."""

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
from test_kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes import (
    kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes,
)
from test_kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity import (
    kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity,
)


def kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
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
    if (
        out_required_cells is None
        or out_required_bytes is None
        or out_last_cell_index is None
    ):
        return KV_Q16_ERR_NULL_PTR

    if (
        out_required_cells is out_required_bytes
        or out_required_cells is out_last_cell_index
        or out_required_bytes is out_last_cell_index
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        layer_count < 0
        or head_count < 0
        or head_dim < 0
        or token_start < 0
        or token_count < 0
        or cache_capacity < 0
    ):
        return KV_Q16_ERR_BAD_PARAM

    snapshot_geom = (
        layer_count,
        head_count,
        head_dim,
        token_start,
        token_count,
        cache_capacity,
    )
    snapshot_ptrs = (
        out_required_cells,
        out_required_bytes,
        out_last_cell_index,
    )

    staged_required_cells = [0]
    staged_required_bytes = [0]
    staged_last_cell_index = [0]
    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity(
        layer_count,
        head_count,
        head_dim,
        token_start,
        token_count,
        cache_capacity,
        staged_required_cells,
        staged_required_bytes,
        staged_last_cell_index,
    )
    if err != KV_Q16_OK:
        return err

    canonical_required_cells = [0]
    canonical_required_bytes = [0]
    canonical_last_cell_index = [0]
    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes(
        layer_count,
        head_count,
        head_dim,
        token_start,
        token_count,
        cache_capacity,
        canonical_required_cells,
        canonical_required_bytes,
        canonical_last_cell_index,
    )
    if err != KV_Q16_OK:
        return err

    if snapshot_geom != (
        layer_count,
        head_count,
        head_dim,
        token_start,
        token_count,
        cache_capacity,
    ):
        return KV_Q16_ERR_BAD_PARAM

    if snapshot_ptrs != (
        out_required_cells,
        out_required_bytes,
        out_last_cell_index,
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (
        out_required_cells is out_required_bytes
        or out_required_cells is out_last_cell_index
        or out_required_bytes is out_last_cell_index
    ):
        return KV_Q16_ERR_BAD_PARAM

    staged_cells = staged_required_cells[0]
    staged_bytes = staged_required_bytes[0]
    staged_last = staged_last_cell_index[0]

    canonical_cells = canonical_required_cells[0]
    canonical_bytes = canonical_required_bytes[0]
    canonical_last = canonical_last_cell_index[0]

    if (
        staged_cells < 0
        or staged_bytes < 0
        or staged_last < 0
        or canonical_cells < 0
        or canonical_bytes < 0
        or canonical_last < 0
    ):
        return KV_Q16_ERR_BAD_PARAM

    if (staged_bytes % 8) != 0:
        return KV_Q16_ERR_BAD_PARAM
    if (staged_bytes // 8) != staged_cells:
        return KV_Q16_ERR_BAD_PARAM

    if (canonical_bytes % 8) != 0:
        return KV_Q16_ERR_BAD_PARAM
    if (canonical_bytes // 8) != canonical_cells:
        return KV_Q16_ERR_BAD_PARAM

    if staged_cells == 0 and staged_last != 0:
        return KV_Q16_ERR_BAD_PARAM
    if canonical_cells == 0 and canonical_last != 0:
        return KV_Q16_ERR_BAD_PARAM

    if staged_cells != canonical_cells:
        return KV_Q16_ERR_BAD_PARAM
    if staged_bytes != canonical_bytes:
        return KV_Q16_ERR_BAD_PARAM
    if staged_last != canonical_last:
        return KV_Q16_ERR_BAD_PARAM

    out_required_cells[0] = staged_cells
    out_required_bytes[0] = staged_bytes
    out_last_cell_index[0] = staged_last
    return KV_Q16_OK


def explicit_composition(
    layer_count: int,
    head_count: int,
    head_dim: int,
    token_start: int,
    token_count: int,
    cache_capacity: int,
) -> tuple[int, tuple[int, int, int]]:
    cells = [0]
    bytes_ = [0]
    last = [0]
    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes(
        layer_count,
        head_count,
        head_dim,
        token_start,
        token_count,
        cache_capacity,
        cells,
        bytes_,
        last,
    )
    if err != KV_Q16_OK:
        return err, (0, 0, 0)
    return KV_Q16_OK, (cells[0], bytes_[0], last[0])


def test_source_contains_required_bytes_parity_commit_only_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ReserveTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "KVCacheQ16ReserveTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesParity(" in body
    assert "KVCacheQ16ReserveTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes(" in body
    assert "snapshot_out_required_cells" in body
    assert "if ((canonical_required_bytes % sizeof(I64)) != 0)" in body
    assert "if (!staged_required_cells && staged_last_cell_index)" in body
    assert "if (!canonical_required_cells && canonical_last_cell_index)" in body


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

    required_cells_out = [111]
    required_bytes_out = [222]
    last_cell_index_out = [333]

    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
        layer_count,
        head_count,
        head_dim,
        token_start,
        token_count,
        end_index,
        required_cells_out,
        required_bytes_out,
        last_cell_index_out,
    )

    assert err == KV_Q16_OK
    assert required_cells_out == [required_cells]
    assert required_bytes_out == [required_cells * 8]
    assert last_cell_index_out == [end_index - 1]


def test_zero_token_count_returns_zero_tuple() -> None:
    required_cells_out = [10]
    required_bytes_out = [11]
    last_cell_index_out = [12]

    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
        layer_count=2,
        head_count=8,
        head_dim=32,
        token_start=9,
        token_count=0,
        cache_capacity=10_000,
        out_required_cells=required_cells_out,
        out_required_bytes=required_bytes_out,
        out_last_cell_index=last_cell_index_out,
    )

    assert err == KV_Q16_OK
    assert required_cells_out == [0]
    assert required_bytes_out == [0]
    assert last_cell_index_out == [0]


def test_error_paths_preserve_outputs() -> None:
    required_cells_out = [91]
    required_bytes_out = [92]
    last_cell_index_out = [93]

    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
        layer_count=-1,
        head_count=1,
        head_dim=1,
        token_start=0,
        token_count=1,
        cache_capacity=100,
        out_required_cells=required_cells_out,
        out_required_bytes=required_bytes_out,
        out_last_cell_index=last_cell_index_out,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert required_cells_out == [91]
    assert required_bytes_out == [92]
    assert last_cell_index_out == [93]

    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
        layer_count=1,
        head_count=1,
        head_dim=1,
        token_start=101,
        token_count=0,
        cache_capacity=100,
        out_required_cells=required_cells_out,
        out_required_bytes=required_bytes_out,
        out_last_cell_index=last_cell_index_out,
    )
    assert err == KV_Q16_ERR_BAD_PARAM
    assert required_cells_out == [91]
    assert required_bytes_out == [92]
    assert last_cell_index_out == [93]


def test_null_and_alias_guards() -> None:
    out_cells = [0]
    out_bytes = [0]
    out_last = [0]

    assert (
        kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
            1, 1, 1, 0, 0, 10, None, out_bytes, out_last
        )
        == KV_Q16_ERR_NULL_PTR
    )

    assert (
        kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
            1, 1, 1, 0, 0, 10, out_cells, out_cells, out_last
        )
        == KV_Q16_ERR_BAD_PARAM
    )


def test_overflow_surfaces() -> None:
    out_cells = [0]
    out_bytes = [0]
    out_last = [0]

    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
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

    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
        layer_count=1,
        head_count=1,
        head_dim=1,
        token_start=I64_MAX,
        token_count=2,
        cache_capacity=I64_MAX,
        out_required_cells=out_cells,
        out_required_bytes=out_bytes,
        out_last_cell_index=out_last,
    )
    assert err == KV_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1070)

    for _ in range(1800):
        layer_count = rng.randint(0, 16)
        head_count = rng.randint(0, 32)
        head_dim = rng.randint(0, 128)
        token_start = rng.randint(0, 128)
        token_count = rng.randint(0, 128)

        cells_per_token = layer_count * head_count * head_dim
        base = token_start * cells_per_token
        required = token_count * cells_per_token
        minimum_capacity = base + required
        capacity = minimum_capacity + rng.randint(0, 64)

        got_cells = [7001]
        got_bytes = [7002]
        got_last = [7003]

        err_new = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
            layer_count,
            head_count,
            head_dim,
            token_start,
            token_count,
            capacity,
            got_cells,
            got_bytes,
            got_last,
        )
        err_ref, ref_tuple = explicit_composition(
            layer_count,
            head_count,
            head_dim,
            token_start,
            token_count,
            capacity,
        )

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
    test_source_contains_required_bytes_parity_commit_only_helper()
    test_known_vector_required_cells_bytes_and_last_index()
    test_zero_token_count_returns_zero_tuple()
    test_error_paths_preserve_outputs()
    test_null_and_alias_guards()
    test_overflow_surfaces()
    test_randomized_parity_vs_explicit_composition()
    print(
        "kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only=ok"
    )
