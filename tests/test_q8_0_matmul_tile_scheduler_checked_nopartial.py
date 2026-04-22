#!/usr/bin/env python3
"""Reference checks for Q8_0MatMulTileSchedulerCheckedNoPartial (IQ-1186)."""

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
    q8_0_matmul_tile_scheduler_checked,
    try_add_i64_nonneg,
)


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def q8_0_matmul_tile_scheduler_checked_nopartial(
    row_count: int,
    col_count: int,
    tile_rows: int,
    tile_cols: int,
    out_row_tile_starts: list[int] | None,
    out_row_tile_ends: list[int] | None,
    out_col_tile_starts: list[int] | None,
    out_col_tile_ends: list[int] | None,
    span_capacity: int,
    out_span_count: list[int] | None,
) -> int:
    if out_span_count is None:
        return Q8_0_MATMUL_ERR_NULL_PTR

    if span_capacity < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if span_capacity > 0 and (
        out_row_tile_starts is None
        or out_row_tile_ends is None
        or out_col_tile_starts is None
        or out_col_tile_ends is None
    ):
        return Q8_0_MATMUL_ERR_NULL_PTR

    if row_count < 0 or col_count < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if tile_rows <= 0 or tile_cols <= 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    ok, row_numer = try_add_i64_nonneg(row_count, tile_rows - 1)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW
    ok, col_numer = try_add_i64_nonneg(col_count, tile_cols - 1)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW

    row_tile_count = row_numer // tile_rows
    col_tile_count = col_numer // tile_cols

    ok, tile_total_count = try_mul_i64_nonneg(row_tile_count, col_tile_count)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW

    if tile_total_count > span_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if tile_total_count == 0:
        out_span_count[0] = 0
        return Q8_0_MATMUL_OK

    snap = (row_count, col_count, tile_rows, tile_cols, span_capacity)

    staged_row_starts = [0] * tile_total_count
    staged_row_ends = [0] * tile_total_count
    staged_col_starts = [0] * tile_total_count
    staged_col_ends = [0] * tile_total_count

    span_index = 0
    row_tile_start = 0
    while row_tile_start < row_count:
        col_tile_start = 0
        while col_tile_start < col_count:
            err, row_tile_end, col_tile_end = q8_0_matmul_tile_scheduler_checked(
                row_tile_start,
                col_tile_start,
                row_count,
                col_count,
                tile_rows,
                tile_cols,
            )
            if err != Q8_0_MATMUL_OK:
                return err
            if span_index >= tile_total_count:
                return Q8_0_MATMUL_ERR_BAD_DST_LEN

            staged_row_starts[span_index] = row_tile_start
            staged_row_ends[span_index] = row_tile_end
            staged_col_starts[span_index] = col_tile_start
            staged_col_ends[span_index] = col_tile_end
            span_index += 1

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    if span_index != tile_total_count:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if snap != (row_count, col_count, tile_rows, tile_cols, span_capacity):
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    for i in range(tile_total_count):
        out_row_tile_starts[i] = staged_row_starts[i]  # type: ignore[index]
        out_row_tile_ends[i] = staged_row_ends[i]  # type: ignore[index]
        out_col_tile_starts[i] = staged_col_starts[i]  # type: ignore[index]
        out_col_tile_ends[i] = staged_col_ends[i]  # type: ignore[index]
    out_span_count[0] = tile_total_count
    return Q8_0_MATMUL_OK


def test_source_contains_nopartial_symbol() -> None:
    source = Path("src/matmul/q8_0_matmul.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0MatMulTileSchedulerCheckedNoPartial(" in source


def test_tail_tile_schedule_rows_cols() -> None:
    row_count = 5
    col_count = 7
    tile_rows = 4
    tile_cols = 3
    span_capacity = 6

    row_starts = [999] * span_capacity
    row_ends = [999] * span_capacity
    col_starts = [999] * span_capacity
    col_ends = [999] * span_capacity
    span_count = [123]

    err = q8_0_matmul_tile_scheduler_checked_nopartial(
        row_count,
        col_count,
        tile_rows,
        tile_cols,
        row_starts,
        row_ends,
        col_starts,
        col_ends,
        span_capacity,
        span_count,
    )
    assert err == Q8_0_MATMUL_OK
    assert span_count[0] == 6

    assert row_starts[:6] == [0, 0, 0, 4, 4, 4]
    assert row_ends[:6] == [4, 4, 4, 5, 5, 5]
    assert col_starts[:6] == [0, 3, 6, 0, 3, 6]
    assert col_ends[:6] == [3, 6, 7, 3, 6, 7]


def test_zero_axes_publish_zero_span_count() -> None:
    row_starts = [7, 7]
    row_ends = [7, 7]
    col_starts = [7, 7]
    col_ends = [7, 7]
    span_count = [99]

    err = q8_0_matmul_tile_scheduler_checked_nopartial(
        0,
        7,
        4,
        3,
        row_starts,
        row_ends,
        col_starts,
        col_ends,
        2,
        span_count,
    )
    assert err == Q8_0_MATMUL_OK
    assert span_count[0] == 0
    assert row_starts == [7, 7]
    assert row_ends == [7, 7]
    assert col_starts == [7, 7]
    assert col_ends == [7, 7]


def test_rejects_insufficient_span_capacity_without_writes() -> None:
    row_starts = [11, 11, 11, 11]
    row_ends = [12, 12, 12, 12]
    col_starts = [13, 13, 13, 13]
    col_ends = [14, 14, 14, 14]
    span_count = [22]

    err = q8_0_matmul_tile_scheduler_checked_nopartial(
        5,
        7,
        4,
        3,
        row_starts,
        row_ends,
        col_starts,
        col_ends,
        4,
        span_count,
    )
    assert err == Q8_0_MATMUL_ERR_BAD_DST_LEN
    assert row_starts == [11, 11, 11, 11]
    assert row_ends == [12, 12, 12, 12]
    assert col_starts == [13, 13, 13, 13]
    assert col_ends == [14, 14, 14, 14]
    assert span_count == [22]


def test_overflow_in_tile_count_ceiling_add_without_writes() -> None:
    row_starts = [1]
    row_ends = [2]
    col_starts = [3]
    col_ends = [4]
    span_count = [5]

    err = q8_0_matmul_tile_scheduler_checked_nopartial(
        I64_MAX,
        8,
        2,
        2,
        row_starts,
        row_ends,
        col_starts,
        col_ends,
        1,
        span_count,
    )
    assert err == Q8_0_MATMUL_ERR_OVERFLOW
    assert row_starts == [1]
    assert row_ends == [2]
    assert col_starts == [3]
    assert col_ends == [4]
    assert span_count == [5]


def test_null_pointer_contracts() -> None:
    span_count = [0]

    err = q8_0_matmul_tile_scheduler_checked_nopartial(
        3,
        3,
        2,
        2,
        None,
        [0],
        [0],
        [0],
        1,
        span_count,
    )
    assert err == Q8_0_MATMUL_ERR_NULL_PTR

    err = q8_0_matmul_tile_scheduler_checked_nopartial(
        3,
        3,
        2,
        2,
        [0],
        [0],
        [0],
        [0],
        1,
        None,
    )
    assert err == Q8_0_MATMUL_ERR_NULL_PTR


def run() -> None:
    test_source_contains_nopartial_symbol()
    test_tail_tile_schedule_rows_cols()
    test_zero_axes_publish_zero_span_count()
    test_rejects_insufficient_span_capacity_without_writes()
    test_overflow_in_tile_count_ceiling_add_without_writes()
    test_null_pointer_contracts()
    print("q8_0_matmul_tile_scheduler_checked_nopartial_reference_checks=ok")


if __name__ == "__main__":
    run()
