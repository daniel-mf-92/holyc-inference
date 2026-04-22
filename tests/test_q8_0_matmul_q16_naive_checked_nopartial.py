#!/usr/bin/env python3
"""Reference checks for Q8_0MatMulQ16NaiveCheckedNoPartial semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_q8_0_dot import (
    Q8_0_ERR_BAD_DST_LEN,
    Q8_0_ERR_OVERFLOW,
    Q8_0_I64_MAX,
    Q8_0_OK,
    dot_product_blocks_q16_accumulate_checked,
)
from test_q8_0_matmul_tiled_checked import (
    compute_out_index_checked,
    compute_rhs_col_base_checked,
    make_block,
    q8_0_matmul_q16_reference_untiled,
    try_mul_i64_nonneg,
    validate_row_slice_checked,
)


def q8_0_matmul_q16_naive_checked_nopartial(
    lhs_blocks: list[tuple[int, bytes]],
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks: list[tuple[int, bytes]],
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells_q16: list[int],
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

    ok, lhs_required = try_mul_i64_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_ERR_OVERFLOW
    ok, rhs_required = try_mul_i64_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_ERR_OVERFLOW
    ok, out_required = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q8_0_ERR_OVERFLOW

    if lhs_required > lhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if rhs_required > rhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if out_required > out_cell_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    if out_required == 0:
        return Q8_0_OK

    staged_out = out_cells_q16[:out_required]

    for row_index in range(row_count):
        ok, lhs_row_base = try_mul_i64_nonneg(row_index, lhs_row_stride_blocks)
        if not ok:
            return Q8_0_ERR_OVERFLOW
        ok, out_row_base = try_mul_i64_nonneg(row_index, out_row_stride_cells)
        if not ok:
            return Q8_0_ERR_OVERFLOW

        err, _ = validate_row_slice_checked(lhs_row_base, k_block_count, lhs_block_capacity)
        if err != Q8_0_OK:
            return err

        lhs_row_slice = lhs_blocks[lhs_row_base : lhs_row_base + k_block_count]
        if len(lhs_row_slice) != k_block_count:
            return Q8_0_ERR_BAD_DST_LEN

        for col_index in range(col_count):
            err, rhs_col_base = compute_rhs_col_base_checked(col_index, rhs_col_stride_blocks)
            if err != Q8_0_OK:
                return err
            err, out_index = compute_out_index_checked(out_row_base, col_index)
            if err != Q8_0_OK:
                return err

            err, _ = validate_row_slice_checked(rhs_col_base, k_block_count, rhs_block_capacity)
            if err != Q8_0_OK:
                return err

            rhs_col_slice = rhs_col_blocks[rhs_col_base : rhs_col_base + k_block_count]
            if len(rhs_col_slice) != k_block_count:
                return Q8_0_ERR_BAD_DST_LEN

            err, cell_dot_q16 = dot_product_blocks_q16_accumulate_checked(
                lhs_row_slice,
                rhs_col_slice,
                0,
            )
            if err != Q8_0_OK:
                return err

            staged_out[out_index] = cell_dot_q16

    for index in range(out_required):
        out_cells_q16[index] = staged_out[index]

    return Q8_0_OK


def test_matches_reference_randomized() -> None:
    rng = random.Random(2026042201)

    for _ in range(240):
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
        rhs_col_blocks = [make_block(rng) for _ in range(rhs_capacity)]

        out_actual = [0 for _ in range(out_capacity)]

        err = q8_0_matmul_q16_naive_checked_nopartial(
            lhs_blocks,
            lhs_capacity,
            row_count,
            lhs_row_stride_blocks,
            rhs_col_blocks,
            rhs_capacity,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            out_actual,
            out_capacity,
            out_row_stride_cells,
        )
        assert err == Q8_0_OK

        err_ref, out_ref = q8_0_matmul_q16_reference_untiled(
            lhs_blocks,
            row_count,
            lhs_row_stride_blocks,
            rhs_col_blocks,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            out_row_stride_cells,
        )
        assert err_ref == Q8_0_OK
        assert out_actual == out_ref


def test_rejects_bad_lengths() -> None:
    rng = random.Random(2026042202)
    lhs_blocks = [make_block(rng) for _ in range(8)]
    rhs_blocks = [make_block(rng) for _ in range(8)]

    out = [7] * 4
    err = q8_0_matmul_q16_naive_checked_nopartial(
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

    err = q8_0_matmul_q16_naive_checked_nopartial(
        lhs_blocks,
        8,
        2,
        2,
        rhs_blocks,
        8,
        1,
        2,
        1,
        out,
        3,
        2,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN


def test_reports_extent_overflow() -> None:
    rng = random.Random(2026042203)
    lhs_blocks = [make_block(rng)]
    rhs_blocks = [make_block(rng)]
    out = [0]

    err = q8_0_matmul_q16_naive_checked_nopartial(
        lhs_blocks,
        Q8_0_I64_MAX,
        Q8_0_I64_MAX,
        2,
        rhs_blocks,
        1,
        1,
        1,
        1,
        out,
        1,
        1,
    )
    assert err == Q8_0_ERR_OVERFLOW


def test_no_partial_commit_on_row_overflow() -> None:
    # Row0 uses zero blocks (safe), row1 uses Inf-scale blocks to force checked
    # Q16 accumulator overflow. Output must remain unchanged.
    zero_block = (0, bytes([0] * 32))
    inf_hi = (0x7C00, bytes([0x7F] * 32))

    lhs_blocks = [
        zero_block,
        inf_hi,
    ]
    rhs_blocks = [
        inf_hi,
        zero_block,
    ]

    out = [0x77, 0x88]
    expected = list(out)

    err = q8_0_matmul_q16_naive_checked_nopartial(
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


def test_padding_lanes_preserved_on_success() -> None:
    rng = random.Random(2026042204)

    row_count = 3
    col_count = 2
    k_block_count = 2
    lhs_row_stride_blocks = 2
    rhs_col_stride_blocks = 2
    out_row_stride_cells = 4

    lhs_capacity = row_count * lhs_row_stride_blocks
    rhs_capacity = col_count * rhs_col_stride_blocks
    out_capacity = row_count * out_row_stride_cells

    lhs_blocks = [make_block(rng) for _ in range(lhs_capacity)]
    rhs_blocks = [make_block(rng) for _ in range(rhs_capacity)]

    out = [0x55AA for _ in range(out_capacity)]
    err = q8_0_matmul_q16_naive_checked_nopartial(
        lhs_blocks,
        lhs_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_blocks,
        rhs_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out,
        out_capacity,
        out_row_stride_cells,
    )
    assert err == Q8_0_OK

    err_ref, out_ref = q8_0_matmul_q16_reference_untiled(
        lhs_blocks,
        row_count,
        lhs_row_stride_blocks,
        rhs_blocks,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_row_stride_cells,
    )
    assert err_ref == Q8_0_OK

    for row in range(row_count):
        base = row * out_row_stride_cells
        for col in range(col_count):
            assert out[base + col] == out_ref[base + col]
        for col in range(col_count, out_row_stride_cells):
            assert out[base + col] == 0x55AA


def run() -> None:
    test_matches_reference_randomized()
    test_rejects_bad_lengths()
    test_reports_extent_overflow()
    test_no_partial_commit_on_row_overflow()
    test_padding_lanes_preserved_on_success()
    print("q8_0_matmul_q16_naive_checked_nopartial_reference_checks=ok")


if __name__ == "__main__":
    run()
