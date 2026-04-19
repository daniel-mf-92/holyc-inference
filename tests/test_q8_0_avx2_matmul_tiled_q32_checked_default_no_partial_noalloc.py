#!/usr/bin/env python3
"""Parity harness for Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialNoAlloc."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_avx2_blocks_q32 import (
    I64_MAX,
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_ERR_NULL_PTR,
    Q8_0_AVX2_ERR_OVERFLOW,
)
from test_q8_0_avx2_matmul_tiled_q32_checked_default_no_partial_preflight import (
    q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight,
)
from test_q8_0_avx2_matmul_tiled_q32_checked_default_no_partial import (
    q8_0_matmul_tiled_avx2_q32_checked_default_tiles,
)
from test_q8_0_matmul_tiled_avx2_q32 import (
    Q8_0_AVX2_OK,
    build_matrix_cols_as_blocks,
    build_matrix_rows_as_blocks,
)


def q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_mat_q32,
    out_row_stride_cols: int,
    staged_out_q32,
    staged_out_capacity: int,
):
    staged_cap = [0]
    staged_bytes = [0]
    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight(
        lhs_matrix_blocks,
        lhs_rows,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_cols,
        rhs_col_stride_blocks,
        k_block_count,
        out_mat_q32,
        out_row_stride_cols,
        staged_cap,
        staged_bytes,
    )
    if err != Q8_0_AVX2_OK:
        return err

    if lhs_rows == 0 or rhs_cols == 0:
        err, _ = q8_0_matmul_tiled_avx2_q32_checked_default_tiles(
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

    if staged_out_q32 is None:
        return Q8_0_AVX2_ERR_NULL_PTR
    if staged_out_capacity < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if staged_out_capacity < staged_cap[0]:
        return Q8_0_AVX2_ERR_BAD_LEN
    staged_bytes_provided = staged_out_capacity * 8
    if staged_bytes_provided < -(1 << 63) or staged_bytes_provided > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW
    if staged_bytes_provided < staged_bytes[0]:
        return Q8_0_AVX2_ERR_BAD_LEN
    if staged_out_q32 is out_mat_q32:
        return Q8_0_AVX2_ERR_BAD_LEN

    err, staged_ref = q8_0_matmul_tiled_avx2_q32_checked_default_tiles(
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

    for idx in range(staged_cap[0]):
        staged_out_q32[idx] = staged_ref[idx]
        out_mat_q32[idx] = staged_out_q32[idx]

    return Q8_0_AVX2_OK


def explicit_staged_composition_reference(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_mat_q32,
    out_row_stride_cols: int,
    staged_out_q32,
    staged_out_capacity: int,
):
    if out_mat_q32 is None:
        return Q8_0_AVX2_ERR_NULL_PTR
    if lhs_matrix_blocks is None or rhs_col_blocks is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    i64_min = -(1 << 63)

    lhs_block_capacity = lhs_rows * lhs_row_stride_blocks
    if lhs_block_capacity < i64_min or lhs_block_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW

    rhs_block_capacity = rhs_cols * rhs_col_stride_blocks
    if rhs_block_capacity < i64_min or rhs_block_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW

    staged_cap = lhs_rows * out_row_stride_cols
    if staged_cap < i64_min or staged_cap > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW

    if lhs_rows == 0 or rhs_cols == 0:
        err, _ = q8_0_matmul_tiled_avx2_q32_checked_default_tiles(
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

    if staged_cap <= 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    staged_bytes = staged_cap * 8
    if staged_bytes < i64_min or staged_bytes > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW
    if staged_bytes <= 0:
        return Q8_0_AVX2_ERR_OVERFLOW

    if staged_out_q32 is None:
        return Q8_0_AVX2_ERR_NULL_PTR
    if staged_out_capacity < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if staged_out_capacity < staged_cap:
        return Q8_0_AVX2_ERR_BAD_LEN
    staged_out_bytes = staged_out_capacity * 8
    if staged_out_bytes < i64_min or staged_out_bytes > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW
    if staged_out_bytes < staged_bytes:
        return Q8_0_AVX2_ERR_BAD_LEN
    if staged_out_q32 is out_mat_q32:
        return Q8_0_AVX2_ERR_BAD_LEN

    err, staged = q8_0_matmul_tiled_avx2_q32_checked_default_tiles(
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

    for idx in range(staged_cap):
        staged_out_q32[idx] = staged[idx]
        out_mat_q32[idx] = staged_out_q32[idx]

    return Q8_0_AVX2_OK


def test_source_contains_noalloc_wrapper_shape() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialNoAlloc(" in source

    body = source.split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialNoAlloc(",
        1,
    )[1].split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialPreflight(",
        1,
    )[0]
    assert "Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialPreflight(" in body
    assert "if (!staged_out_q32)" in body
    assert "if (staged_out_capacity < 0)" in body
    assert "if (staged_out_capacity < required_stage_capacity)" in body
    assert "if (!Q8_0AVX2TryMulI64(staged_out_capacity, sizeof(I64), &provided_stage_bytes))" in body
    assert "if (provided_stage_bytes < required_stage_bytes)" in body
    assert "if (staged_out_q32 == out_mat_q32)" in body


def test_error_surface_and_no_partial_behavior() -> None:
    rng = random.Random(20260419_523_1)
    lhs = build_matrix_rows_as_blocks(2, 2, 2, rng)
    rhs = build_matrix_cols_as_blocks(2, 2, 2, rng)

    out = [11] * 8
    staged = [22] * 8

    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc(
            lhs,
            2,
            2,
            rhs,
            2,
            2,
            2,
            None,
            4,
            staged,
            8,
        )
        == Q8_0_AVX2_ERR_NULL_PTR
    )

    baseline = list(out)
    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc(
            lhs,
            2,
            2,
            rhs,
            2,
            2,
            2,
            out,
            4,
            None,
            8,
        )
        == Q8_0_AVX2_ERR_NULL_PTR
    )
    assert out == baseline

    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc(
            lhs,
            2,
            2,
            rhs,
            2,
            2,
            2,
            out,
            4,
            staged,
            -1,
        )
        == Q8_0_AVX2_ERR_BAD_LEN
    )

    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc(
            lhs,
            2,
            2,
            rhs,
            2,
            2,
            2,
            out,
            4,
            staged,
            (I64_MAX // 8) + 1,
        )
        == Q8_0_AVX2_ERR_OVERFLOW
    )

    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc(
            lhs,
            2,
            2,
            rhs,
            2,
            2,
            2,
            out,
            4,
            staged,
            3,
        )
        == Q8_0_AVX2_ERR_BAD_LEN
    )

    alias_buf = [33] * 16
    alias_baseline = list(alias_buf)
    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc(
            lhs,
            2,
            2,
            rhs,
            2,
            2,
            2,
            alias_buf,
            4,
            alias_buf,
            16,
        )
        == Q8_0_AVX2_ERR_BAD_LEN
    )
    assert alias_buf == alias_baseline


def test_randomized_parity_vs_explicit_staged_composition() -> None:
    rng = random.Random(20260419_523_2)

    for _ in range(2000):
        lhs_rows = rng.randint(-2, 24)
        lhs_stride = rng.randint(-2, 12)
        rhs_cols = rng.randint(-2, 24)
        rhs_stride = rng.randint(-2, 12)
        k_blocks = rng.randint(-2, 12)
        out_stride = rng.randint(-2, 20)

        null_mode = rng.randint(0, 12)

        lhs_rows_build = max(lhs_rows, 0)
        lhs_stride_build = max(lhs_stride, 0)
        rhs_cols_build = max(rhs_cols, 0)
        rhs_stride_build = max(rhs_stride, 0)
        k_build = max(k_blocks, 0)

        lhs_k_for_build = min(k_build, lhs_stride_build)
        rhs_k_for_build = min(k_build, rhs_stride_build)

        lhs = (
            None
            if null_mode == 0
            else build_matrix_rows_as_blocks(
                lhs_rows_build,
                lhs_stride_build,
                lhs_k_for_build,
                rng,
            )
        )
        rhs = (
            None
            if null_mode == 1
            else build_matrix_cols_as_blocks(
                rhs_cols_build,
                rhs_stride_build,
                rhs_k_for_build,
                rng,
            )
        )
        out_got = None if null_mode == 2 else [rng.randint(-20, 20) for _ in range(1024)]
        out_want = None if out_got is None else list(out_got)

        staged_got = None if null_mode == 3 else [rng.randint(-20, 20) for _ in range(1024)]
        staged_want = None if staged_got is None else list(staged_got)

        if out_got is not None and staged_got is not None and rng.randint(0, 19) == 0:
            staged_got = out_got
            staged_want = out_want

        if rng.randint(0, 14) == 0:
            staged_cap = -rng.randint(1, 32)
        else:
            staged_cap = rng.randint(0, 1024)

        got = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc(
            lhs,
            lhs_rows,
            lhs_stride,
            rhs,
            rhs_cols,
            rhs_stride,
            k_blocks,
            out_got,
            out_stride,
            staged_got,
            staged_cap,
        )
        want = explicit_staged_composition_reference(
            lhs,
            lhs_rows,
            lhs_stride,
            rhs,
            rhs_cols,
            rhs_stride,
            k_blocks,
            out_want,
            out_stride,
            staged_want,
            staged_cap,
        )

        assert got == want
        assert out_got == out_want
        assert staged_got == staged_want


def run() -> None:
    test_source_contains_noalloc_wrapper_shape()
    test_error_surface_and_no_partial_behavior()
    test_randomized_parity_vs_explicit_staged_composition()
    print("q8_0_avx2_matmul_tiled_q32_checked_default_no_partial_noalloc=ok")


if __name__ == "__main__":
    run()
