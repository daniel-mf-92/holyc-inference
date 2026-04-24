#!/usr/bin/env python3
"""Reference checks for Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1202)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_q8_0_matmul_tile_scheduler_checked import (
    I64_MAX,
    Q8_0_MATMUL_ERR_BAD_DST_LEN,
    Q8_0_MATMUL_ERR_NULL_PTR,
    Q8_0_MATMUL_ERR_OVERFLOW,
    Q8_0_MATMUL_OK,
)
from test_q8_0_matmul_tile_scheduler_checked_nopartial_commit_only import (
    q8_0_matmul_tile_scheduler_checked_nopartial_commit_only,
)
from test_q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only import (
    q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only,
)


# Mirrors HolyC parity gate in src/matmul/q8_0_matmul.HC.
def q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only_parity(
    row_count: int,
    col_count: int,
    tile_rows: int,
    tile_cols: int,
    span_capacity: int,
    out_tile_count: list[int] | None,
    out_last_row_end: list[int] | None,
    out_last_col_end: list[int] | None,
) -> int:
    if out_tile_count is None or out_last_row_end is None or out_last_col_end is None:
        return Q8_0_MATMUL_ERR_NULL_PTR

    snap = (row_count, col_count, tile_rows, tile_cols, span_capacity)

    pre_tile_count = [0]
    pre_last_row_end = [0]
    pre_last_col_end = [0]
    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only(
        row_count,
        col_count,
        tile_rows,
        tile_cols,
        span_capacity,
        pre_tile_count,
        pre_last_row_end,
        pre_last_col_end,
    )
    if err != Q8_0_MATMUL_OK:
        return err

    commit_tile_count = [0]
    commit_last_row_end = [0]
    commit_last_col_end = [0]
    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only(
        row_count,
        col_count,
        tile_rows,
        tile_cols,
        span_capacity,
        commit_tile_count,
        commit_last_row_end,
        commit_last_col_end,
    )
    if err != Q8_0_MATMUL_OK:
        return err

    if snap != (row_count, col_count, tile_rows, tile_cols, span_capacity):
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if (
        pre_tile_count[0] != commit_tile_count[0]
        or pre_last_row_end[0] != commit_last_row_end[0]
        or pre_last_col_end[0] != commit_last_col_end[0]
    ):
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    out_tile_count[0] = commit_tile_count[0]
    out_last_row_end[0] = commit_last_row_end[0]
    out_last_col_end[0] = commit_last_col_end[0]
    return Q8_0_MATMUL_OK


def test_source_contains_parity_symbol() -> None:
    source = Path("src/matmul/q8_0_matmul.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnlyPreflightOnlyParity(" in source


def test_tail_tile_parity_outputs() -> None:
    out_tile_count = [17]
    out_last_row_end = [18]
    out_last_col_end = [19]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only_parity(
        5,
        7,
        4,
        3,
        6,
        out_tile_count,
        out_last_row_end,
        out_last_col_end,
    )
    assert err == Q8_0_MATMUL_OK
    assert out_tile_count[0] == 6
    assert out_last_row_end[0] == 5
    assert out_last_col_end[0] == 7


def test_null_and_overflow_contracts_keep_outputs() -> None:
    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only_parity(
        1,
        1,
        1,
        1,
        1,
        None,
        [0],
        [0],
    )
    assert err == Q8_0_MATMUL_ERR_NULL_PTR

    out_tile_count = [111]
    out_last_row_end = [222]
    out_last_col_end = [333]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only_parity(
        0,
        0,
        1,
        1,
        I64_MAX,
        out_tile_count,
        out_last_row_end,
        out_last_col_end,
    )
    assert err == Q8_0_MATMUL_ERR_OVERFLOW
    assert out_tile_count == [111]
    assert out_last_row_end == [222]
    assert out_last_col_end == [333]


def test_bad_capacity_and_bad_tile_shape_keep_outputs() -> None:
    out_tile_count = [41]
    out_last_row_end = [42]
    out_last_col_end = [43]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only_parity(
        5,
        7,
        4,
        3,
        0,
        out_tile_count,
        out_last_row_end,
        out_last_col_end,
    )
    assert err == Q8_0_MATMUL_ERR_BAD_DST_LEN
    assert out_tile_count == [41]
    assert out_last_row_end == [42]
    assert out_last_col_end == [43]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only_parity(
        5,
        7,
        0,
        3,
        6,
        out_tile_count,
        out_last_row_end,
        out_last_col_end,
    )
    assert err == Q8_0_MATMUL_ERR_BAD_DST_LEN
    assert out_tile_count == [41]
    assert out_last_row_end == [42]
    assert out_last_col_end == [43]


def run() -> None:
    test_source_contains_parity_symbol()
    test_tail_tile_parity_outputs()
    test_null_and_overflow_contracts_keep_outputs()
    test_bad_capacity_and_bad_tile_shape_keep_outputs()
    print("q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only_parity_reference_checks=ok")


if __name__ == "__main__":
    run()
