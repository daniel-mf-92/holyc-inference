#!/usr/bin/env python3
"""Reference checks for Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1202)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_q8_0_matmul_tile_scheduler_checked import (
    Q8_0_MATMUL_ERR_BAD_DST_LEN,
    Q8_0_MATMUL_ERR_NULL_PTR,
    Q8_0_MATMUL_OK,
)
from test_q8_0_matmul_tile_scheduler_checked_nopartial_commit_only import (
    q8_0_matmul_tile_scheduler_checked_nopartial_commit_only,
)
from test_q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only import (
    q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only,
)


# Mirrors HolyC wrapper in src/matmul/q8_0_matmul.HC.
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

    com_tile_count = [0]
    com_last_row_end = [0]
    com_last_col_end = [0]
    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only(
        row_count,
        col_count,
        tile_rows,
        tile_cols,
        span_capacity,
        com_tile_count,
        com_last_row_end,
        com_last_col_end,
    )
    if err != Q8_0_MATMUL_OK:
        return err

    if snap != (row_count, col_count, tile_rows, tile_cols, span_capacity):
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if pre_tile_count[0] != com_tile_count[0]:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if pre_last_row_end[0] != com_last_row_end[0]:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if pre_last_col_end[0] != com_last_col_end[0]:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    out_tile_count[0] = com_tile_count[0]
    out_last_row_end[0] = com_last_row_end[0]
    out_last_col_end[0] = com_last_col_end[0]
    return Q8_0_MATMUL_OK


def test_source_contains_parity_symbol() -> None:
    source = Path("src/matmul/q8_0_matmul.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnlyPreflightOnlyParity(" in source


def test_tail_tile_parity_outputs() -> None:
    out_tile_count = [71]
    out_last_row_end = [72]
    out_last_col_end = [73]

    err = q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only_parity(
        9,
        7,
        4,
        3,
        9,
        out_tile_count,
        out_last_row_end,
        out_last_col_end,
    )
    assert err == Q8_0_MATMUL_OK
    assert out_tile_count[0] == 9
    assert out_last_row_end[0] == 9
    assert out_last_col_end[0] == 7


def test_null_outputs_and_bad_capacity_keep_outputs_unchanged() -> None:
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

    out_tile_count = [501]
    out_last_row_end = [502]
    out_last_col_end = [503]

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
    assert out_tile_count == [501]
    assert out_last_row_end == [502]
    assert out_last_col_end == [503]


def run() -> None:
    test_source_contains_parity_symbol()
    test_tail_tile_parity_outputs()
    test_null_outputs_and_bad_capacity_keep_outputs_unchanged()
    print("q8_0_matmul_tile_scheduler_checked_nopartial_commit_only_preflight_only_parity_reference_checks=ok")


if __name__ == "__main__":
    run()
