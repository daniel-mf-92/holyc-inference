#!/usr/bin/env python3
"""Reference checks for Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnlyPreflightOnly (IQ-1201)."""

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


# Mirrors HolyC wrapper in src/matmul/q8_0_matmul.HC.
def q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only(
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

    if row_count < 0 or col_count < 0 or span_capacity < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    snap = (row_count, col_count, tile_rows, tile_cols, span_capacity)

    staged_tile_count = [0]
    staged_last_row_end = [0]
    staged_last_col_end = [0]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only(
        row_count,
        col_count,
        tile_rows,
        tile_cols,
        span_capacity,
        staged_tile_count,
        staged_last_row_end,
        staged_last_col_end,
    )
    if err != Q8_0_MATMUL_OK:
        return err

    if snap != (row_count, col_count, tile_rows, tile_cols, span_capacity):
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    out_tile_count[0] = staged_tile_count[0]
    out_last_row_end[0] = staged_last_row_end[0]
    out_last_col_end[0] = staged_last_col_end[0]
    return Q8_0_MATMUL_OK


def test_source_contains_preflight_only_symbol() -> None:
    source = Path("src/matmul/q8_0_matmul.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnlyPreflightOnly(" in source


def test_tail_tile_parity_outputs() -> None:
    out_tile_count = [7]
    out_last_row_end = [8]
    out_last_col_end = [9]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only(
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


def test_zero_capacity_empty_axes_publish_zero_tuple() -> None:
    out_tile_count = [11]
    out_last_row_end = [12]
    out_last_col_end = [13]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only(
        0,
        0,
        2,
        2,
        0,
        out_tile_count,
        out_last_row_end,
        out_last_col_end,
    )
    assert err == Q8_0_MATMUL_OK
    assert out_tile_count[0] == 0
    assert out_last_row_end[0] == 0
    assert out_last_col_end[0] == 0


def test_bad_capacity_and_overflow_keep_outputs_unchanged() -> None:
    out_tile_count = [101]
    out_last_row_end = [202]
    out_last_col_end = [303]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only(
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
    assert out_tile_count == [101]
    assert out_last_row_end == [202]
    assert out_last_col_end == [303]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only(
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
    assert out_tile_count == [101]
    assert out_last_row_end == [202]
    assert out_last_col_end == [303]


def test_null_and_bad_input_contracts() -> None:
    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only(
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

    ok_tuple = [1], [2], [3]
    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only(
        -1,
        1,
        1,
        1,
        1,
        ok_tuple[0],
        ok_tuple[1],
        ok_tuple[2],
    )
    assert err == Q8_0_MATMUL_ERR_BAD_DST_LEN


def run() -> None:
    test_source_contains_preflight_only_symbol()
    test_tail_tile_parity_outputs()
    test_zero_capacity_empty_axes_publish_zero_tuple()
    test_bad_capacity_and_overflow_keep_outputs_unchanged()
    test_null_and_bad_input_contracts()
    print("q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only_reference_checks=ok")


if __name__ == "__main__":
    run()
