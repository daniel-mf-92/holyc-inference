#!/usr/bin/env python3
"""Parity harness for MatMulQ4_0Q8_0Q32CheckedTiledNoPartial."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    I64_MAX,
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    dot_blocks_avx2_q32_checked,
    make_q4_block,
    make_q8_block,
    q4_0_q8_0_matmul_tiled_avx2_q32_checked,
)


def try_add_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
    lhs_q4_blocks,
    lhs_q4_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_q8_col_blocks,
    rhs_q8_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    tile_rows: int,
    tile_cols: int,
    out_cells_q32,
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> int:
    if lhs_q4_blocks is None or rhs_q8_col_blocks is None or out_cells_q32 is None:
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    if lhs_q4_block_capacity < 0 or rhs_q8_block_capacity < 0 or out_cell_capacity < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if tile_rows <= 0 or tile_cols <= 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if row_count > 0 and out_row_stride_cells < col_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, lhs_required = try_mul_i64_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, rhs_required = try_mul_i64_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, out_required = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    if lhs_required > lhs_q4_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if rhs_required > rhs_q8_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if out_required > out_cell_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, row_plus_tile_minus_one = try_add_i64_nonneg(row_count, tile_rows - 1)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, col_plus_tile_minus_one = try_add_i64_nonneg(col_count, tile_cols - 1)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    tile_row_count = row_plus_tile_minus_one // tile_rows
    tile_col_count = col_plus_tile_minus_one // tile_cols
    ok, tile_total_count = try_mul_i64_nonneg(tile_row_count, tile_col_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    if row_count > 0 and col_count > 0 and tile_total_count == 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    # Preflight pass guarantees no-partial output writes.
    row_tile_start = 0
    while row_tile_start < row_count:
        ok, row_tile_end = try_add_i64_nonneg(row_tile_start, tile_rows)
        if not ok:
            return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
        row_tile_end = min(row_tile_end, row_count)

        col_tile_start = 0
        while col_tile_start < col_count:
            ok, col_tile_end = try_add_i64_nonneg(col_tile_start, tile_cols)
            if not ok:
                return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
            col_tile_end = min(col_tile_end, col_count)

            for row_index in range(row_tile_start, row_tile_end):
                ok, lhs_row_base = try_mul_i64_nonneg(row_index, lhs_row_stride_blocks)
                if not ok:
                    return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
                lhs_row_slice = lhs_q4_blocks[lhs_row_base : lhs_row_base + k_block_count]
                if len(lhs_row_slice) != k_block_count:
                    return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

                for col_index in range(col_tile_start, col_tile_end):
                    ok, rhs_col_base = try_mul_i64_nonneg(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
                    rhs_col_slice = rhs_q8_col_blocks[rhs_col_base : rhs_col_base + k_block_count]
                    if len(rhs_col_slice) != k_block_count:
                        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

                    err, _ = dot_blocks_avx2_q32_checked(lhs_row_slice, rhs_col_slice, k_block_count)
                    if err != Q4_0_Q8_0_AVX2_OK:
                        return err

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    # Commit-only after successful preflight.
    err, staged = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
        lhs_q4_blocks=lhs_q4_blocks,
        lhs_q4_block_capacity=lhs_q4_block_capacity,
        row_count=row_count,
        lhs_row_stride_blocks=lhs_row_stride_blocks,
        rhs_q8_col_blocks=rhs_q8_col_blocks,
        rhs_q8_block_capacity=rhs_q8_block_capacity,
        col_count=col_count,
        rhs_col_stride_blocks=rhs_col_stride_blocks,
        k_block_count=k_block_count,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        out_cell_capacity=out_cell_capacity,
        out_row_stride_cells=out_row_stride_cells,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    for index in range(out_required):
        out_cells_q32[index] = staged[index]

    return Q4_0_Q8_0_AVX2_OK


def test_source_contains_iq_990_function() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    signature = "I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartial("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "tile_row_count" in body
    assert "tile_col_count" in body
    assert "tile_total_count" in body
    assert "if (row_count && col_count && !tile_total_count)" in body


def test_known_vector_matches_reference_and_capacity_error_no_partial() -> None:
    rng = random.Random(2026042201)

    row_count = 4
    col_count = 3
    k_block_count = 2
    lhs_stride = 3
    rhs_stride = 3
    out_stride = 5
    tile_rows = 3
    tile_cols = 2

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]

    out_capacity = row_count * out_stride
    out = [1234567] * out_capacity

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        tile_rows,
        tile_cols,
        out,
        out_capacity,
        out_stride,
    )
    assert err == Q4_0_Q8_0_AVX2_OK

    err_ref, ref = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
        lhs_q4_blocks=lhs,
        lhs_q4_block_capacity=len(lhs),
        row_count=row_count,
        lhs_row_stride_blocks=lhs_stride,
        rhs_q8_col_blocks=rhs,
        rhs_q8_block_capacity=len(rhs),
        col_count=col_count,
        rhs_col_stride_blocks=rhs_stride,
        k_block_count=k_block_count,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        out_cell_capacity=out_capacity,
        out_row_stride_cells=out_stride,
    )
    assert err_ref == Q4_0_Q8_0_AVX2_OK
    assert out == ref

    sentinel = [91] * out_capacity
    out_bad = sentinel.copy()
    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        tile_rows,
        tile_cols,
        out_bad,
        out_capacity - 1,
        out_stride,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert out_bad == sentinel


def test_randomized_parity_and_tile_geometry_edges() -> None:
    rng = random.Random(2026042202)

    for _ in range(120):
        row_count = rng.randint(0, 7)
        col_count = rng.randint(0, 7)
        k_block_count = rng.randint(0, 4)

        lhs_stride = k_block_count + rng.randint(0, 3)
        rhs_stride = k_block_count + rng.randint(0, 3)
        out_stride = col_count + rng.randint(0, 3)

        tile_rows = rng.randint(1, 5)
        tile_cols = rng.randint(1, 5)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        lhs = [make_q4_block(rng) for _ in range(lhs_capacity)]
        rhs = [make_q8_block(rng) for _ in range(rhs_capacity)]

        out = [rng.randint(-1000, 1000) for _ in range(out_capacity)]
        out_explicit = out.copy()

        err_a = matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            tile_rows,
            tile_cols,
            out,
            out_capacity,
            out_stride,
        )

        err_b, ref = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
            lhs_q4_blocks=lhs,
            lhs_q4_block_capacity=lhs_capacity,
            row_count=row_count,
            lhs_row_stride_blocks=lhs_stride,
            rhs_q8_col_blocks=rhs,
            rhs_q8_block_capacity=rhs_capacity,
            col_count=col_count,
            rhs_col_stride_blocks=rhs_stride,
            k_block_count=k_block_count,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            out_cell_capacity=out_capacity,
            out_row_stride_cells=out_stride,
        )

        assert err_a == err_b
        if err_a == Q4_0_Q8_0_AVX2_OK:
            assert out == ref
        else:
            assert out == out_explicit


def test_input_validation_errors() -> None:
    dummy = []
    out = [0]

    assert (
        matmul_q4_0_q8_0_q32_checked_tiled_nopartial(None, 0, 0, 0, dummy, 0, 0, 0, 0, 1, 1, out, 1, 0)
        == Q4_0_Q8_0_AVX2_ERR_NULL_PTR
    )
    assert (
        matmul_q4_0_q8_0_q32_checked_tiled_nopartial(dummy, 0, 1, 0, dummy, 0, 0, 0, 0, 1, 1, out, 1, 0)
        == Q4_0_Q8_0_AVX2_OK
    )
    assert (
        matmul_q4_0_q8_0_q32_checked_tiled_nopartial(dummy, 0, 0, 0, dummy, 0, 0, 0, 0, 0, 1, out, 1, 0)
        == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    )
    assert (
        matmul_q4_0_q8_0_q32_checked_tiled_nopartial(dummy, 0, 0, 0, dummy, 0, 0, 0, 0, 1, 0, out, 1, 0)
        == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    )
