#!/usr/bin/env python3
"""Parity harness for Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartial."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_avx2_blocks_q32 import I64_MAX, Q8_0_AVX2_ERR_BAD_LEN, Q8_0_AVX2_ERR_NULL_PTR, Q8_0_AVX2_ERR_OVERFLOW
from test_q8_0_matmul_tiled_avx2_q32 import (
    Q8_0_AVX2_OK,
    build_matrix_cols_as_blocks,
    build_matrix_rows_as_blocks,
    q8_0_matmul_tiled_avx2_q32_checked,
)


def q8_0_matmul_tiled_avx2_q32_checked_default(
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
        return Q8_0_AVX2_ERR_NULL_PTR, []

    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    i64_min = -(1 << 63)

    lhs_block_capacity = lhs_rows * lhs_row_stride_blocks
    if lhs_block_capacity < i64_min or lhs_block_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW, []

    rhs_block_capacity = rhs_cols * rhs_col_stride_blocks
    if rhs_block_capacity < i64_min or rhs_block_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW, []

    out_capacity = lhs_rows * out_row_stride_cols
    if out_capacity < i64_min or out_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW, []

    return q8_0_matmul_tiled_avx2_q32_checked(
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


def q8_0_matmul_tiled_avx2_q32_checked_default_no_partial(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_mat_q32: list[int] | None,
    out_row_stride_cols: int,
):
    if out_mat_q32 is None:
        return Q8_0_AVX2_ERR_NULL_PTR
    if lhs_rows < 0 or out_row_stride_cols < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    if lhs_rows == 0 or rhs_cols == 0:
        err, _ = q8_0_matmul_tiled_avx2_q32_checked_default(
            lhs_matrix_blocks,
            lhs_rows,
            lhs_row_stride_blocks,
            rhs_col_blocks,
            rhs_cols,
            rhs_col_stride_blocks,
            k_block_count,
            out_row_stride_cols,
        )
        return err

    i64_min = -(1 << 63)
    staged_out_capacity = lhs_rows * out_row_stride_cols
    if staged_out_capacity < i64_min or staged_out_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW
    if staged_out_capacity <= 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    staged_out_bytes = staged_out_capacity * 8
    if staged_out_bytes < i64_min or staged_out_bytes > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW
    if staged_out_bytes <= 0:
        return Q8_0_AVX2_ERR_OVERFLOW

    err, staged = q8_0_matmul_tiled_avx2_q32_checked_default(
        lhs_matrix_blocks,
        lhs_rows,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_cols,
        rhs_col_stride_blocks,
        k_block_count,
        out_row_stride_cols,
    )
    if err != Q8_0_AVX2_OK:
        return err

    for i in range(staged_out_capacity):
        out_mat_q32[i] = staged[i]

    return Q8_0_AVX2_OK


def explicit_staged_composition(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_mat_q32: list[int] | None,
    out_row_stride_cols: int,
):
    if out_mat_q32 is None:
        return Q8_0_AVX2_ERR_NULL_PTR
    if lhs_rows < 0 or out_row_stride_cols < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    sentinel = list(out_mat_q32)

    if lhs_rows == 0 or rhs_cols == 0:
        err, _ = q8_0_matmul_tiled_avx2_q32_checked_default(
            lhs_matrix_blocks,
            lhs_rows,
            lhs_row_stride_blocks,
            rhs_col_blocks,
            rhs_cols,
            rhs_col_stride_blocks,
            k_block_count,
            out_row_stride_cols,
        )
        if err != Q8_0_AVX2_OK:
            out_mat_q32[:] = sentinel
            return err
        return Q8_0_AVX2_OK

    staged_out_capacity = lhs_rows * out_row_stride_cols
    if staged_out_capacity < 0 or staged_out_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW
    if staged_out_capacity <= 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    err, staged = q8_0_matmul_tiled_avx2_q32_checked_default(
        lhs_matrix_blocks,
        lhs_rows,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_cols,
        rhs_col_stride_blocks,
        k_block_count,
        out_row_stride_cols,
    )
    if err != Q8_0_AVX2_OK:
        out_mat_q32[:] = sentinel
        return err

    for i in range(staged_out_capacity):
        out_mat_q32[i] = staged[i]
    return Q8_0_AVX2_OK


def test_source_contains_default_and_no_partial_shapes() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0MatMulTiledAVX2Q32CheckedDefault(" in source
    assert "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartial(" in source
    assert "return Q8_0MatMulTiledAVX2Q32CheckedDefaultTiles(" in source
    assert "status = Q8_0MatMulTiledAVX2Q32CheckedDefault(lhs_matrix_blocks," in source


def test_null_and_bad_len_surfaces() -> None:
    rng = random.Random(20260419_503_1)
    lhs = build_matrix_rows_as_blocks(2, 2, 2, rng)
    rhs = build_matrix_cols_as_blocks(2, 2, 2, rng)
    out = [9] * 8

    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial(
            lhs,
            2,
            2,
            rhs,
            2,
            2,
            2,
            None,
            4,
        )
        == Q8_0_AVX2_ERR_NULL_PTR
    )

    sentinel = out.copy()
    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial(
        lhs,
        -1,
        2,
        rhs,
        2,
        2,
        2,
        out,
        4,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out == sentinel


def test_no_partial_on_underprovisioned_rhs_failure() -> None:
    rng = random.Random(20260419_503_2)
    lhs_rows = 3
    lhs_row_stride = 2
    rhs_cols = 3
    rhs_col_stride = 2
    k_blocks = 2
    out_row_stride = rhs_cols

    lhs = build_matrix_rows_as_blocks(lhs_rows, lhs_row_stride, k_blocks, rng)
    rhs_full = build_matrix_cols_as_blocks(rhs_cols, rhs_col_stride, k_blocks, rng)
    rhs_short = rhs_full[: rhs_col_stride]  # one column only: guaranteed BAD_LEN

    out = [777] * (lhs_rows * out_row_stride)
    sentinel = out.copy()

    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial(
        lhs,
        lhs_rows,
        lhs_row_stride,
        rhs_short,
        rhs_cols,
        rhs_col_stride,
        k_blocks,
        out,
        out_row_stride,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out == sentinel


def test_overflow_passthrough_and_randomized_parity() -> None:
    out = [1, 2, 3]
    sentinel = out.copy()

    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial(
        [],
        I64_MAX,
        1,
        [],
        0,
        0,
        0,
        out,
        2,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert out == sentinel

    rng = random.Random(20260419_503_3)
    for _ in range(280):
        lhs_rows = rng.randint(1, 6)
        rhs_cols = rng.randint(1, 6)
        k_blocks = rng.randint(0, 5)
        lhs_row_stride = k_blocks + rng.randint(0, 3)
        rhs_col_stride = k_blocks + rng.randint(0, 3)
        out_row_stride = rhs_cols + rng.randint(0, 3)

        lhs = build_matrix_rows_as_blocks(lhs_rows, lhs_row_stride, k_blocks, rng)
        rhs = build_matrix_cols_as_blocks(rhs_cols, rhs_col_stride, k_blocks, rng)

        out_wrap = [0x1234] * (lhs_rows * out_row_stride)
        out_ref = out_wrap.copy()

        err_wrap = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial(
            lhs,
            lhs_rows,
            lhs_row_stride,
            rhs,
            rhs_cols,
            rhs_col_stride,
            k_blocks,
            out_wrap,
            out_row_stride,
        )
        err_ref = explicit_staged_composition(
            lhs,
            lhs_rows,
            lhs_row_stride,
            rhs,
            rhs_cols,
            rhs_col_stride,
            k_blocks,
            out_ref,
            out_row_stride,
        )

        assert err_wrap == err_ref
        assert out_wrap == out_ref


def run() -> None:
    test_source_contains_default_and_no_partial_shapes()
    test_null_and_bad_len_surfaces()
    test_no_partial_on_underprovisioned_rhs_failure()
    test_overflow_passthrough_and_randomized_parity()
    print("q8_0_avx2_matmul_tiled_q32_checked_default_no_partial=ok")


if __name__ == "__main__":
    run()
