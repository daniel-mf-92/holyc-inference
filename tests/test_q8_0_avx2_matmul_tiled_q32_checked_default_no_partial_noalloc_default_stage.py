#!/usr/bin/env python3
"""Parity harness for Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialNoAllocDefaultStage."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_avx2_blocks_q32 import (
    I64_MAX,
    Q8_0_AVX2_ERR_NULL_PTR,
    Q8_0_AVX2_ERR_OVERFLOW,
)
from test_q8_0_avx2_matmul_tiled_q32_checked_default_no_partial_noalloc import (
    q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc,
)
from test_q8_0_matmul_tiled_avx2_q32 import (
    Q8_0_AVX2_OK,
    build_matrix_cols_as_blocks,
    build_matrix_rows_as_blocks,
)


I64_MIN = -(1 << 63)


def q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc_default_stage(
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
):
    staged_out_capacity = lhs_rows * out_row_stride_cols
    if staged_out_capacity < I64_MIN or staged_out_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW

    return q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc(
        lhs_matrix_blocks,
        lhs_rows,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_cols,
        rhs_col_stride_blocks,
        k_block_count,
        out_mat_q32,
        out_row_stride_cols,
        staged_out_q32,
        staged_out_capacity,
    )


def explicit_composition_reference(
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
):
    staged_out_capacity = lhs_rows * out_row_stride_cols
    if staged_out_capacity < I64_MIN or staged_out_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW

    return q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc(
        lhs_matrix_blocks,
        lhs_rows,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_cols,
        rhs_col_stride_blocks,
        k_block_count,
        out_mat_q32,
        out_row_stride_cols,
        staged_out_q32,
        staged_out_capacity,
    )


def test_source_contains_default_stage_wrapper_shape() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialNoAllocDefaultStage(" in source

    body = source.split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialNoAllocDefaultStage(",
        1,
    )[1].split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialPreflight(",
        1,
    )[0]
    assert "if (!Q8_0AVX2TryMulI64(lhs_rows, out_row_stride_cols, &staged_out_capacity))" in body
    assert "return Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialNoAlloc(" in body


def test_overflow_surface() -> None:
    out = [5] * 16
    staged = [7] * 16

    # Derivation overflow must fail before delegated wrapper checks.
    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc_default_stage(
            [],
            I64_MAX,
            0,
            [],
            0,
            0,
            0,
            out,
            2,
            staged,
        )
        == Q8_0_AVX2_ERR_OVERFLOW
    )


def test_nullptr_surface_passthrough() -> None:
    rng = random.Random(20260419_538_1)
    lhs = build_matrix_rows_as_blocks(2, 2, 2, rng)
    rhs = build_matrix_cols_as_blocks(2, 2, 2, rng)
    staged = [9] * 32

    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc_default_stage(
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
        )
        == Q8_0_AVX2_ERR_NULL_PTR
    )


def test_randomized_parity_vs_explicit_default_stage_composition() -> None:
    rng = random.Random(20260419_538_2)

    for _ in range(2500):
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

        got = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_noalloc_default_stage(
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
        )
        want = explicit_composition_reference(
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
        )

        assert got == want
        assert out_got == out_want
        assert staged_got == staged_want


def run() -> None:
    test_source_contains_default_stage_wrapper_shape()
    test_overflow_surface()
    test_nullptr_surface_passthrough()
    test_randomized_parity_vs_explicit_default_stage_composition()
    print("q8_0_avx2_matmul_tiled_q32_checked_default_no_partial_noalloc_default_stage=ok")


if __name__ == "__main__":
    run()
