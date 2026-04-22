#!/usr/bin/env python3
"""Reference checks for Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnly (IQ-1187)."""

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
from test_q8_0_matmul_tile_scheduler_checked_nopartial import (
    q8_0_matmul_tile_scheduler_checked_nopartial,
    try_mul_i64_nonneg,
)


# Mirrors HolyC wrapper in src/matmul/q8_0_matmul.HC.
def q8_0_matmul_tile_scheduler_checked_nopartial_commit_only(
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

    if tile_rows <= 0 or tile_cols <= 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if span_capacity == 0:
        staged_tile_count = [0]
        err = q8_0_matmul_tile_scheduler_checked_nopartial(
            row_count,
            col_count,
            tile_rows,
            tile_cols,
            None,
            None,
            None,
            None,
            0,
            staged_tile_count,
        )
        if err != Q8_0_MATMUL_OK:
            return err

        out_tile_count[0] = staged_tile_count[0]
        out_last_row_end[0] = 0
        out_last_col_end[0] = 0
        return Q8_0_MATMUL_OK

    ok, _stage_bytes = try_mul_i64_nonneg(span_capacity, 8)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW

    staged_row_starts = [0] * span_capacity
    staged_row_ends = [0] * span_capacity
    staged_col_starts = [0] * span_capacity
    staged_col_ends = [0] * span_capacity
    staged_tile_count = [0]

    err = q8_0_matmul_tile_scheduler_checked_nopartial(
        row_count,
        col_count,
        tile_rows,
        tile_cols,
        staged_row_starts,
        staged_row_ends,
        staged_col_starts,
        staged_col_ends,
        span_capacity,
        staged_tile_count,
    )
    if err != Q8_0_MATMUL_OK:
        return err

    staged_last_row_end = 0
    staged_last_col_end = 0
    if staged_tile_count[0] > 0:
        staged_last_row_end = staged_row_ends[staged_tile_count[0] - 1]
        staged_last_col_end = staged_col_ends[staged_tile_count[0] - 1]

    out_tile_count[0] = staged_tile_count[0]
    out_last_row_end[0] = staged_last_row_end
    out_last_col_end[0] = staged_last_col_end
    return Q8_0_MATMUL_OK


def test_source_contains_commit_only_symbol() -> None:
    source = Path("src/matmul/q8_0_matmul.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnly(" in source


def test_tail_tile_parity_outputs() -> None:
    out_tile_count = [77]
    out_last_row_end = [88]
    out_last_col_end = [99]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only(
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
    out_tile_count = [31]
    out_last_row_end = [32]
    out_last_col_end = [33]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only(
        0,
        10,
        4,
        3,
        0,
        out_tile_count,
        out_last_row_end,
        out_last_col_end,
    )
    assert err == Q8_0_MATMUL_OK
    assert out_tile_count[0] == 0
    assert out_last_row_end[0] == 0
    assert out_last_col_end[0] == 0


def test_rejects_nonempty_with_zero_capacity_without_writes() -> None:
    out_tile_count = [101]
    out_last_row_end = [202]
    out_last_col_end = [303]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only(
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


def test_overflow_stage_bytes_without_writes() -> None:
    out_tile_count = [13]
    out_last_row_end = [14]
    out_last_col_end = [15]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only(
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
    assert out_tile_count == [13]
    assert out_last_row_end == [14]
    assert out_last_col_end == [15]


def test_null_output_contracts() -> None:
    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only(
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


def run() -> None:
    test_source_contains_commit_only_symbol()
    test_tail_tile_parity_outputs()
    test_zero_capacity_empty_axes_publish_zero_tuple()
    test_rejects_nonempty_with_zero_capacity_without_writes()
    test_overflow_stage_bytes_without_writes()
    test_null_output_contracts()
    print("q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_reference_checks=ok")


if __name__ == "__main__":
    run()
