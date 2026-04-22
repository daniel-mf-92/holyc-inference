#!/usr/bin/env python3
"""Harness for ...RequiredBytesParity...ParityCommitOnly (IQ-1074)."""

from __future__ import annotations

import random
import re
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
from test_kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity import (
    kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity,
)


def kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
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

    snapshot_geom = (layer_count, head_count, head_dim, token_start, token_count, cache_capacity)
    snapshot_ptrs = (out_required_cells, out_required_bytes, out_last_cell_index)

    staged_cells = [0]
    staged_bytes = [0]
    staged_last = [0]
    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
        layer_count,
        head_count,
        head_dim,
        token_start,
        token_count,
        cache_capacity,
        staged_cells,
        staged_bytes,
        staged_last,
    )
    if err != KV_Q16_OK:
        return err

    canonical_cells = [0]
    canonical_bytes = [0]
    canonical_last = [0]
    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes(
        layer_count,
        head_count,
        head_dim,
        token_start,
        token_count,
        cache_capacity,
        canonical_cells,
        canonical_bytes,
        canonical_last,
    )
    if err != KV_Q16_OK:
        return err

    # Integer-only independent recompute from immutable snapshot.
    cells_per_token = layer_count * head_count
    if head_count and layer_count and (cells_per_token // head_count) != layer_count:
        return KV_Q16_ERR_OVERFLOW
    cells_per_token = cells_per_token * head_dim
    if head_dim and (cells_per_token // head_dim) != (layer_count * head_count):
        return KV_Q16_ERR_OVERFLOW

    base_index = token_start * cells_per_token
    if cells_per_token and token_start and (base_index // cells_per_token) != token_start:
        return KV_Q16_ERR_OVERFLOW
    if base_index > cache_capacity:
        return KV_Q16_ERR_BAD_PARAM

    recomputed_cells = token_count * cells_per_token
    if cells_per_token and token_count and (recomputed_cells // cells_per_token) != token_count:
        return KV_Q16_ERR_OVERFLOW

    end_index = base_index + recomputed_cells
    if end_index < base_index:
        return KV_Q16_ERR_OVERFLOW
    if end_index > cache_capacity:
        return KV_Q16_ERR_BAD_PARAM

    recomputed_bytes = recomputed_cells * 8
    if recomputed_cells and (recomputed_bytes // 8) != recomputed_cells:
        return KV_Q16_ERR_OVERFLOW

    recomputed_last = 0 if recomputed_cells == 0 else end_index - 1

    if snapshot_geom != (layer_count, head_count, head_dim, token_start, token_count, cache_capacity):
        return KV_Q16_ERR_BAD_PARAM
    if snapshot_ptrs != (out_required_cells, out_required_bytes, out_last_cell_index):
        return KV_Q16_ERR_BAD_PARAM

    if staged_cells[0] < 0 or staged_bytes[0] < 0 or staged_last[0] < 0:
        return KV_Q16_ERR_BAD_PARAM
    if canonical_cells[0] < 0 or canonical_bytes[0] < 0 or canonical_last[0] < 0:
        return KV_Q16_ERR_BAD_PARAM
    if recomputed_cells < 0 or recomputed_bytes < 0 or recomputed_last < 0:
        return KV_Q16_ERR_BAD_PARAM

    if (staged_bytes[0] % 8) != 0 or (staged_bytes[0] // 8) != staged_cells[0]:
        return KV_Q16_ERR_BAD_PARAM
    if (canonical_bytes[0] % 8) != 0 or (canonical_bytes[0] // 8) != canonical_cells[0]:
        return KV_Q16_ERR_BAD_PARAM
    if (recomputed_bytes % 8) != 0 or (recomputed_bytes // 8) != recomputed_cells:
        return KV_Q16_ERR_BAD_PARAM

    if staged_cells[0] == 0 and staged_last[0] != 0:
        return KV_Q16_ERR_BAD_PARAM
    if canonical_cells[0] == 0 and canonical_last[0] != 0:
        return KV_Q16_ERR_BAD_PARAM
    if recomputed_cells == 0 and recomputed_last != 0:
        return KV_Q16_ERR_BAD_PARAM

    if staged_cells[0] != canonical_cells[0] or staged_bytes[0] != canonical_bytes[0] or staged_last[0] != canonical_last[0]:
        return KV_Q16_ERR_BAD_PARAM
    if staged_cells[0] != recomputed_cells or staged_bytes[0] != recomputed_bytes or staged_last[0] != recomputed_last:
        return KV_Q16_ERR_BAD_PARAM

    out_required_cells[0] = staged_cells[0]
    out_required_bytes[0] = staged_bytes[0]
    out_last_cell_index[0] = staged_last[0]
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
    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity(
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


def test_source_contains_required_bytes_parity_commit_only_preflight_only_parity_commit_only_helper() -> None:
    source = Path("src/model/kv_cache.HC").read_text(encoding="utf-8")
    sig = "I32 KVCacheQ16ReserveTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source
    assert len(re.findall(re.escape(sig), source)) == 1
    body = source.split(sig, 1)[1]
    assert "KVCacheQ16ReserveTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyPreflightOnlyParity(" in body
    assert "KVCacheQ16ReserveTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes(" in body
    assert "if (staged_required_cells != recomputed_required_cells)" in body


def test_known_vector_required_cells_bytes_and_last_index() -> None:
    out_cells = [111]
    out_bytes = [222]
    out_last = [333]
    err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
        3, 4, 16, 5, 7, 2304, out_cells, out_bytes, out_last
    )
    assert err == KV_Q16_OK
    assert out_cells == [1344]
    assert out_bytes == [10752]
    assert out_last == [2303]


def test_capacity_edges_and_no_partial_publish() -> None:
    vectors = [
        dict(layer_count=1, head_count=1, head_dim=1, token_start=0, token_count=0),
        dict(layer_count=2, head_count=4, head_dim=8, token_start=9, token_count=3),
        dict(layer_count=3, head_count=2, head_dim=5, token_start=17, token_count=1),
    ]

    for vector in vectors:
        cells_per_token = vector["layer_count"] * vector["head_count"] * vector["head_dim"]
        base_index = vector["token_start"] * cells_per_token
        required_cells = vector["token_count"] * cells_per_token
        exact_capacity = base_index + required_cells

        out_cells = [444]
        out_bytes = [555]
        out_last = [666]
        err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
            vector["layer_count"],
            vector["head_count"],
            vector["head_dim"],
            vector["token_start"],
            vector["token_count"],
            exact_capacity,
            out_cells,
            out_bytes,
            out_last,
        )
        assert err == KV_Q16_OK
        assert out_cells == [required_cells]
        assert out_bytes == [required_cells * 8]
        assert out_last == ([0] if required_cells == 0 else [base_index + required_cells - 1])

        out_cells = [777]
        out_bytes = [888]
        out_last = [999]
        err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
            vector["layer_count"],
            vector["head_count"],
            vector["head_dim"],
            vector["token_start"],
            vector["token_count"],
            max(0, exact_capacity - 1),
            out_cells,
            out_bytes,
            out_last,
        )
        if required_cells == 0 and exact_capacity == 0:
            assert err == KV_Q16_OK
            assert out_cells == [0]
            assert out_bytes == [0]
            assert out_last == [0]
        else:
            assert err == KV_Q16_ERR_BAD_PARAM
            assert out_cells == [777]
            assert out_bytes == [888]
            assert out_last == [999]


def test_error_and_guard_paths() -> None:
    out_cells = [91]
    out_bytes = [92]
    out_last = [93]
    assert (
        kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
            -1, 1, 1, 0, 0, 10, out_cells, out_bytes, out_last
        )
        == KV_Q16_ERR_BAD_PARAM
    )
    assert out_cells == [91] and out_bytes == [92] and out_last == [93]
    assert (
        kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
            1, 1, 1, 0, 0, 10, None, out_bytes, out_last
        )
        == KV_Q16_ERR_NULL_PTR
    )


def test_i64_limit_overflow_vectors_preserve_outputs() -> None:
    vectors = [
        dict(layer_count=I64_MAX, head_count=2, head_dim=1, token_start=0, token_count=1),
        dict(layer_count=I64_MAX // 4, head_count=2, head_dim=2, token_start=3, token_count=1),
        dict(layer_count=I64_MAX // 8 + 1, head_count=1, head_dim=1, token_start=0, token_count=1),
    ]

    for vector in vectors:
        out_cells = [9001]
        out_bytes = [9002]
        out_last = [9003]
        err = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
            vector["layer_count"],
            vector["head_count"],
            vector["head_dim"],
            vector["token_start"],
            vector["token_count"],
            I64_MAX,
            out_cells,
            out_bytes,
            out_last,
        )
        assert err == KV_Q16_ERR_OVERFLOW
        assert out_cells == [9001]
        assert out_bytes == [9002]
        assert out_last == [9003]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1074)
    for _ in range(2400):
        layer_count = rng.randint(0, 16)
        head_count = rng.randint(0, 32)
        head_dim = rng.randint(0, 128)
        token_start = rng.randint(0, 128)
        token_count = rng.randint(0, 128)

        cells_per_token = layer_count * head_count * head_dim
        minimum_capacity = token_start * cells_per_token + token_count * cells_per_token
        capacity = minimum_capacity + rng.randint(0, 64)

        got_cells = [7001]
        got_bytes = [7002]
        got_last = [7003]
        err_new = kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only(
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
            assert got_cells == [7001] and got_bytes == [7002] and got_last == [7003]


if __name__ == "__main__":
    test_source_contains_required_bytes_parity_commit_only_preflight_only_parity_commit_only_helper()
    test_known_vector_required_cells_bytes_and_last_index()
    test_capacity_edges_and_no_partial_publish()
    test_error_and_guard_paths()
    test_i64_limit_overflow_vectors_preserve_outputs()
    test_randomized_parity_vs_explicit_composition()
    print(
        "kv_cache_q16_reserve_token_span_checked_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only_parity_commit_only=ok"
    )
