#!/usr/bin/env python3
"""Parity harness for Q8_0MatMulTiledAVX2Q32CheckedDefaultPreflight."""

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
from test_q8_0_matmul_tiled_avx2_q32 import Q8_0_AVX2_OK, build_matrix_cols_as_blocks, build_matrix_rows_as_blocks


I64_MIN = -(1 << 63)


def try_mul_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    value = lhs * rhs
    if value < I64_MIN or value > I64_MAX:
        return False, 0
    return True, value


def q8_0_matmul_tiled_avx2_q32_checked_default_preflight(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_mat_q32,
    out_row_stride_cols: int,
    out_lhs_block_capacity: list[int] | None,
    out_rhs_block_capacity: list[int] | None,
    out_out_capacity: list[int] | None,
) -> int:
    if (
        lhs_matrix_blocks is None
        or rhs_col_blocks is None
        or out_mat_q32 is None
        or out_lhs_block_capacity is None
        or out_rhs_block_capacity is None
        or out_out_capacity is None
    ):
        return Q8_0_AVX2_ERR_NULL_PTR

    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, lhs_capacity = try_mul_i64(lhs_rows, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    ok, rhs_capacity = try_mul_i64(rhs_cols, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    ok, out_capacity = try_mul_i64(lhs_rows, out_row_stride_cols)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    out_lhs_block_capacity[0] = lhs_capacity
    out_rhs_block_capacity[0] = rhs_capacity
    out_out_capacity[0] = out_capacity
    return Q8_0_AVX2_OK


def explicit_inline_preflight(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_mat_q32,
    out_row_stride_cols: int,
    out_lhs_block_capacity: list[int] | None,
    out_rhs_block_capacity: list[int] | None,
    out_out_capacity: list[int] | None,
) -> int:
    if (
        lhs_matrix_blocks is None
        or rhs_col_blocks is None
        or out_mat_q32 is None
        or out_lhs_block_capacity is None
        or out_rhs_block_capacity is None
        or out_out_capacity is None
    ):
        return Q8_0_AVX2_ERR_NULL_PTR

    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, lhs_capacity = try_mul_i64(lhs_rows, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW
    ok, rhs_capacity = try_mul_i64(rhs_cols, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW
    ok, out_capacity = try_mul_i64(lhs_rows, out_row_stride_cols)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    out_lhs_block_capacity[0] = lhs_capacity
    out_rhs_block_capacity[0] = rhs_capacity
    out_out_capacity[0] = out_capacity
    return Q8_0_AVX2_OK


def test_source_contains_default_preflight_and_wrapper_use() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")

    assert "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultPreflight(" in source
    preflight_body = source.split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultPreflight(",
        1,
    )[1].split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefault(",
        1,
    )[0]
    assert "!out_lhs_block_capacity || !out_rhs_block_capacity || !out_out_capacity" in preflight_body
    assert "Q8_0AVX2TryMulI64(lhs_rows, lhs_row_stride_blocks, out_lhs_block_capacity)" in preflight_body
    assert "Q8_0AVX2TryMulI64(rhs_cols, rhs_col_stride_blocks, out_rhs_block_capacity)" in preflight_body
    assert "Q8_0AVX2TryMulI64(lhs_rows, out_row_stride_cols, out_out_capacity)" in preflight_body

    default_tiles_body = source.split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultTiles(",
        1,
    )[1].split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefault(",
        1,
    )[0]
    assert "status = Q8_0MatMulTiledAVX2Q32CheckedDefaultPreflight(" in default_tiles_body

    no_partial_body = source.split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartial(",
        1,
    )[1].split(
        "I32 Q8_0DotBlocksAVX2Q32ToQ16Checked(",
        1,
    )[0]
    assert "status = Q8_0MatMulTiledAVX2Q32CheckedDefaultPreflight(" in no_partial_body


def test_null_and_bad_len_surfaces() -> None:
    rng = random.Random(20260419_521_1)
    lhs = build_matrix_rows_as_blocks(2, 2, 2, rng)
    rhs = build_matrix_cols_as_blocks(2, 2, 2, rng)
    out = [0] * 8

    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_preflight(
            None,
            2,
            2,
            rhs,
            2,
            2,
            2,
            out,
            2,
            [0],
            [0],
            [0],
        )
        == Q8_0_AVX2_ERR_NULL_PTR
    )
    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_preflight(
            lhs,
            2,
            2,
            rhs,
            2,
            2,
            2,
            out,
            2,
            None,
            [0],
            [0],
        )
        == Q8_0_AVX2_ERR_NULL_PTR
    )
    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_preflight(
            lhs,
            -1,
            2,
            rhs,
            2,
            2,
            2,
            out,
            2,
            [0],
            [0],
            [0],
        )
        == Q8_0_AVX2_ERR_BAD_LEN
    )
    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_preflight(
            lhs,
            1,
            2,
            rhs,
            -1,
            2,
            2,
            out,
            2,
            [0],
            [0],
            [0],
        )
        == Q8_0_AVX2_ERR_BAD_LEN
    )
    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_preflight(
            lhs,
            1,
            2,
            rhs,
            1,
            2,
            -1,
            out,
            2,
            [0],
            [0],
            [0],
        )
        == Q8_0_AVX2_ERR_BAD_LEN
    )


def test_overflow_surfaces_do_not_mutate_outputs() -> None:
    rng = random.Random(20260419_521_2)
    lhs = build_matrix_rows_as_blocks(1, 1, 1, rng)
    rhs = build_matrix_cols_as_blocks(1, 1, 1, rng)
    out = [0]

    lhs_cap = [111]
    rhs_cap = [222]
    out_cap = [333]

    err = q8_0_matmul_tiled_avx2_q32_checked_default_preflight(
        lhs,
        I64_MAX,
        2,
        rhs,
        1,
        1,
        1,
        out,
        1,
        lhs_cap,
        rhs_cap,
        out_cap,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert lhs_cap[0] == 111
    assert rhs_cap[0] == 222
    assert out_cap[0] == 333


def test_randomized_inline_parity() -> None:
    rng = random.Random(20260419_521_3)

    for _ in range(6000):
        lhs_rows = rng.randint(0, 32)
        rhs_cols = rng.randint(0, 32)
        k_blocks = rng.randint(0, 16)
        lhs_stride = rng.randint(0, 32)
        rhs_stride = rng.randint(0, 32)
        out_stride = rng.randint(0, 32)

        lhs = [None]
        rhs = [None]
        out = [0]

        lhs_new = [0x1111]
        rhs_new = [0x2222]
        out_new = [0x3333]

        lhs_ref = [0xAAAA]
        rhs_ref = [0xBBBB]
        out_ref = [0xCCCC]

        err_new = q8_0_matmul_tiled_avx2_q32_checked_default_preflight(
            lhs,
            lhs_rows,
            lhs_stride,
            rhs,
            rhs_cols,
            rhs_stride,
            k_blocks,
            out,
            out_stride,
            lhs_new,
            rhs_new,
            out_new,
        )
        err_ref = explicit_inline_preflight(
            lhs,
            lhs_rows,
            lhs_stride,
            rhs,
            rhs_cols,
            rhs_stride,
            k_blocks,
            out,
            out_stride,
            lhs_ref,
            rhs_ref,
            out_ref,
        )

        assert err_new == err_ref
        if err_new == Q8_0_AVX2_OK:
            assert lhs_new[0] == lhs_ref[0]
            assert rhs_new[0] == rhs_ref[0]
            assert out_new[0] == out_ref[0]


def run() -> None:
    test_source_contains_default_preflight_and_wrapper_use()
    test_null_and_bad_len_surfaces()
    test_overflow_surfaces_do_not_mutate_outputs()
    test_randomized_inline_parity()
    print("q8_0_avx2_matmul_tiled_q32_checked_default_preflight=ok")


if __name__ == "__main__":
    run()
