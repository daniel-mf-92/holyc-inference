#!/usr/bin/env python3
"""Parity checks for Q4_0MatMulTiledAVX2Q32CheckedDefaultTiles semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_avx2_dot_q32_checked import (
    Q4_0_AVX2_ERR_BAD_LEN,
    Q4_0_AVX2_ERR_NULL_PTR,
)
from test_q4_0_avx2_dot_rows_q32_checked import (
    I64_MAX,
    Q4_0_AVX2_ERR_OVERFLOW,
)
from test_q4_0_matmul_tiled_avx2_q32_checked import (
    Q4_0_AVX2_OK,
    build_matrix_cols_as_blocks,
    build_matrix_rows_as_blocks,
    q4_0_matmul_tiled_avx2_q32_checked,
)


def q4_0_matmul_tiled_avx2_q32_checked_default_tiles(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_row_stride_cols: int,
):
    if lhs_matrix_blocks is None or rhs_col_blocks is None:
        return Q4_0_AVX2_ERR_NULL_PTR, []

    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    i64_min = -(1 << 63)

    lhs_block_capacity = lhs_rows * lhs_row_stride_blocks
    if lhs_block_capacity < i64_min or lhs_block_capacity > I64_MAX:
        return Q4_0_AVX2_ERR_OVERFLOW, []

    rhs_block_capacity = rhs_cols * rhs_col_stride_blocks
    if rhs_block_capacity < i64_min or rhs_block_capacity > I64_MAX:
        return Q4_0_AVX2_ERR_OVERFLOW, []

    out_capacity = lhs_rows * out_row_stride_cols
    if out_capacity < i64_min or out_capacity > I64_MAX:
        return Q4_0_AVX2_ERR_OVERFLOW, []

    return q4_0_matmul_tiled_avx2_q32_checked(
        lhs_matrix_blocks=lhs_matrix_blocks,
        lhs_block_capacity=lhs_block_capacity,
        lhs_rows=lhs_rows,
        lhs_row_stride_blocks=lhs_row_stride_blocks,
        rhs_col_blocks=rhs_col_blocks,
        rhs_block_capacity=rhs_block_capacity,
        rhs_cols=rhs_cols,
        rhs_col_stride_blocks=rhs_col_stride_blocks,
        k_block_count=k_block_count,
        tile_rows=1,
        tile_cols=1,
        out_capacity=out_capacity,
        out_row_stride_cols=out_row_stride_cols,
    )


def test_default_tiles_wrapper_matches_checked_core_randomized() -> None:
    rng = random.Random(2026041810)

    for _ in range(180):
        lhs_rows = rng.randint(0, 6)
        rhs_cols = rng.randint(0, 6)
        k_block_count = rng.randint(0, 5)
        lhs_row_stride_blocks = k_block_count + rng.randint(0, 3)
        rhs_col_stride_blocks = k_block_count + rng.randint(0, 3)
        out_row_stride_cols = rhs_cols + rng.randint(0, 3)

        lhs = build_matrix_rows_as_blocks(lhs_rows, lhs_row_stride_blocks, k_block_count, rng)
        rhs = build_matrix_cols_as_blocks(rhs_cols, rhs_col_stride_blocks, k_block_count, rng)

        err_default, out_default = q4_0_matmul_tiled_avx2_q32_checked_default_tiles(
            lhs_matrix_blocks=lhs,
            lhs_rows=lhs_rows,
            lhs_row_stride_blocks=lhs_row_stride_blocks,
            rhs_col_blocks=rhs,
            rhs_cols=rhs_cols,
            rhs_col_stride_blocks=rhs_col_stride_blocks,
            k_block_count=k_block_count,
            out_row_stride_cols=out_row_stride_cols,
        )

        err_core, out_core = q4_0_matmul_tiled_avx2_q32_checked(
            lhs_matrix_blocks=lhs,
            lhs_block_capacity=lhs_rows * lhs_row_stride_blocks,
            lhs_rows=lhs_rows,
            lhs_row_stride_blocks=lhs_row_stride_blocks,
            rhs_col_blocks=rhs,
            rhs_block_capacity=rhs_cols * rhs_col_stride_blocks,
            rhs_cols=rhs_cols,
            rhs_col_stride_blocks=rhs_col_stride_blocks,
            k_block_count=k_block_count,
            tile_rows=1,
            tile_cols=1,
            out_capacity=lhs_rows * out_row_stride_cols,
            out_row_stride_cols=out_row_stride_cols,
        )

        assert err_default == err_core
        assert out_default == out_core


def test_default_tiles_wrapper_bad_len_and_null_parity() -> None:
    rng = random.Random(2026041811)
    lhs = build_matrix_rows_as_blocks(1, 1, 1, rng)
    rhs = build_matrix_cols_as_blocks(1, 1, 1, rng)

    err, out = q4_0_matmul_tiled_avx2_q32_checked_default_tiles(
        lhs_matrix_blocks=None,
        lhs_rows=1,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=rhs,
        rhs_cols=1,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        out_row_stride_cols=1,
    )
    assert err == Q4_0_AVX2_ERR_NULL_PTR
    assert out == []

    err, out = q4_0_matmul_tiled_avx2_q32_checked_default_tiles(
        lhs_matrix_blocks=lhs,
        lhs_rows=-1,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=rhs,
        rhs_cols=1,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        out_row_stride_cols=1,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN
    assert out == []


def test_default_tiles_wrapper_shortfall_parity() -> None:
    rng = random.Random(2026041812)
    lhs = build_matrix_rows_as_blocks(2, 1, 1, rng)
    rhs = build_matrix_cols_as_blocks(1, 1, 1, rng)

    err_default, out_default = q4_0_matmul_tiled_avx2_q32_checked_default_tiles(
        lhs_matrix_blocks=lhs,
        lhs_rows=2,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=rhs,
        rhs_cols=2,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        out_row_stride_cols=2,
    )
    err_core, out_core = q4_0_matmul_tiled_avx2_q32_checked(
        lhs_matrix_blocks=lhs,
        lhs_block_capacity=2,
        lhs_rows=2,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=rhs,
        rhs_block_capacity=2,
        rhs_cols=2,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        tile_rows=1,
        tile_cols=1,
        out_capacity=4,
        out_row_stride_cols=2,
    )

    assert err_default == err_core == Q4_0_AVX2_ERR_BAD_LEN
    assert out_default == out_core == []


def test_default_tiles_wrapper_capacity_overflow_rejected() -> None:
    err, out = q4_0_matmul_tiled_avx2_q32_checked_default_tiles(
        lhs_matrix_blocks=[],
        lhs_rows=I64_MAX,
        lhs_row_stride_blocks=2,
        rhs_col_blocks=[],
        rhs_cols=0,
        rhs_col_stride_blocks=0,
        k_block_count=0,
        out_row_stride_cols=1,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert out == []

    err, out = q4_0_matmul_tiled_avx2_q32_checked_default_tiles(
        lhs_matrix_blocks=[],
        lhs_rows=1,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=[],
        rhs_cols=I64_MAX,
        rhs_col_stride_blocks=2,
        k_block_count=0,
        out_row_stride_cols=1,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert out == []

    err, out = q4_0_matmul_tiled_avx2_q32_checked_default_tiles(
        lhs_matrix_blocks=[],
        lhs_rows=I64_MAX,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=[],
        rhs_cols=0,
        rhs_col_stride_blocks=0,
        k_block_count=0,
        out_row_stride_cols=2,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert out == []


def run() -> None:
    test_default_tiles_wrapper_matches_checked_core_randomized()
    test_default_tiles_wrapper_bad_len_and_null_parity()
    test_default_tiles_wrapper_shortfall_parity()
    test_default_tiles_wrapper_capacity_overflow_rejected()
    print("q4_0_matmul_tiled_avx2_q32_checked_default_tiles_reference_checks=ok")


if __name__ == "__main__":
    run()
