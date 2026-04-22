#!/usr/bin/env python3
"""Reference checks for Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_q8_0_dot import Q8_0_ERR_BAD_DST_LEN, Q8_0_ERR_OVERFLOW, Q8_0_OK
from test_q8_0_matmul_q16_naive_checked_nopartial import (
    q8_0_matmul_q16_naive_checked_nopartial,
)
from test_q8_0_matmul_tiled_checked import make_block


def q8_0_matmul_q16_naive_checked_nopartial_commit_only(
    lhs_blocks: list[tuple[int, bytes]] | None,
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks: list[tuple[int, bytes]] | None,
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells_q16: list[int] | None,
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> int:
    if lhs_blocks is None or rhs_col_blocks is None or out_cells_q16 is None:
        return 1

    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_cell_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN

    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q8_0_ERR_BAD_DST_LEN

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return Q8_0_ERR_BAD_DST_LEN
    if row_count > 0 and out_row_stride_cells < col_count:
        return Q8_0_ERR_BAD_DST_LEN

    lhs_required = row_count * lhs_row_stride_blocks
    rhs_required = col_count * rhs_col_stride_blocks
    out_required = row_count * out_row_stride_cells

    if lhs_required > lhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if rhs_required > rhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if out_required > out_cell_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    if out_required == 0:
        return Q8_0_OK

    # Immutable geometry snapshots.
    snap = (
        row_count,
        col_count,
        lhs_row_stride_blocks,
        rhs_col_stride_blocks,
        k_block_count,
        out_row_stride_cells,
    )

    # Commit-only staging: initialize from caller output so padding semantics stay identical.
    staged = out_cells_q16[:out_required]

    err = q8_0_matmul_q16_naive_checked_nopartial(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        staged,
        out_required,
        out_row_stride_cells,
    )
    if err != Q8_0_OK:
        return err

    if snap != (
        row_count,
        col_count,
        lhs_row_stride_blocks,
        rhs_col_stride_blocks,
        k_block_count,
        out_row_stride_cells,
    ):
        return Q8_0_ERR_BAD_DST_LEN

    for index in range(out_required):
        out_cells_q16[index] = staged[index]

    return Q8_0_OK


def test_source_contains_commit_only_symbol() -> None:
    source = Path("src/matmul/q8_0_matmul.HC").read_text(encoding="utf-8")
    assert "Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnly" in source


def test_matches_nopartial_reference_randomized() -> None:
    rng = random.Random(2026042301)

    for _ in range(220):
        row_count = rng.randint(1, 7)
        col_count = rng.randint(1, 7)
        k_block_count = rng.randint(1, 6)

        lhs_row_stride_blocks = k_block_count + rng.randint(0, 3)
        rhs_col_stride_blocks = k_block_count + rng.randint(0, 3)
        out_row_stride_cells = col_count + rng.randint(0, 3)

        lhs_capacity = row_count * lhs_row_stride_blocks
        rhs_capacity = col_count * rhs_col_stride_blocks
        out_capacity = row_count * out_row_stride_cells

        lhs_blocks = [make_block(rng) for _ in range(lhs_capacity)]
        rhs_blocks = [make_block(rng) for _ in range(rhs_capacity)]

        out_a = [rng.randint(-1234, 1234) for _ in range(out_capacity)]
        out_b = list(out_a)

        err = q8_0_matmul_q16_naive_checked_nopartial_commit_only(
            lhs_blocks,
            lhs_capacity,
            row_count,
            lhs_row_stride_blocks,
            rhs_blocks,
            rhs_capacity,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            out_a,
            out_capacity,
            out_row_stride_cells,
        )
        assert err == Q8_0_OK

        err_ref = q8_0_matmul_q16_naive_checked_nopartial(
            lhs_blocks,
            lhs_capacity,
            row_count,
            lhs_row_stride_blocks,
            rhs_blocks,
            rhs_capacity,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            out_b,
            out_capacity,
            out_row_stride_cells,
        )
        assert err_ref == Q8_0_OK
        assert out_a == out_b


def test_no_partial_commit_on_overflow_error() -> None:
    # Force checked Q16 accumulation overflow in row 1 after row 0 would succeed.
    zero_block = (0, bytes([0] * 32))
    inf_hi = (0x7C00, bytes([0x7F] * 32))

    lhs_blocks = [zero_block, inf_hi]
    rhs_blocks = [inf_hi, zero_block]

    out = [0x1111, 0x2222]
    expected = list(out)

    err = q8_0_matmul_q16_naive_checked_nopartial_commit_only(
        lhs_blocks,
        2,
        2,
        1,
        rhs_blocks,
        2,
        1,
        1,
        1,
        out,
        2,
        1,
    )
    assert err == Q8_0_ERR_OVERFLOW
    assert out == expected


def test_rejects_bad_capacity_without_writing() -> None:
    rng = random.Random(2026042302)
    lhs_blocks = [make_block(rng) for _ in range(8)]
    rhs_blocks = [make_block(rng) for _ in range(8)]

    out = [9, 9, 9, 9]
    expected = list(out)

    err = q8_0_matmul_q16_naive_checked_nopartial_commit_only(
        lhs_blocks,
        8,
        2,
        2,
        rhs_blocks,
        8,
        2,
        2,
        3,
        out,
        4,
        2,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert out == expected


def run() -> None:
    test_source_contains_commit_only_symbol()
    test_matches_nopartial_reference_randomized()
    test_no_partial_commit_on_overflow_error()
    test_rejects_bad_capacity_without_writing()
    print("q8_0_matmul_q16_naive_checked_nopartial_commit_only_reference_checks=ok")


if __name__ == "__main__":
    run()
