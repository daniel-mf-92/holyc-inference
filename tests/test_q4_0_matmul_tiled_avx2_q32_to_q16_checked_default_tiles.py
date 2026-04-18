#!/usr/bin/env python3
"""Parity checks for Q4_0MatMulTiledAVX2Q32ToQ16CheckedDefaultTiles semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_avx2_dot_q32_checked import (
    Q4_0_AVX2_ERR_BAD_LEN,
    Q4_0_AVX2_ERR_NULL_PTR,
    Q4_0_AVX2_OK,
)
from test_q4_0_avx2_dot_q32_to_q16_checked_default import (
    q4_0_dot_blocks_avx2_q32_to_q16_checked_ptr,
)
from test_q4_0_avx2_dot_rows_q32_checked import (
    I64_MAX,
    Q4_0_AVX2_ERR_OVERFLOW,
)
from test_q4_0_matmul_tiled_avx2_q32_checked import (
    build_matrix_cols_as_blocks,
    build_matrix_rows_as_blocks,
)


def q4_0_try_mul_i64_nonneg(lhs: int, rhs: int):
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def q4_0_try_add_i64_nonneg(lhs: int, rhs: int):
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def q4_0_matmul_tiled_avx2_q32_to_q16_checked(
    lhs_matrix_blocks,
    lhs_block_capacity: int,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_block_capacity: int,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    tile_rows: int,
    tile_cols: int,
    out_capacity: int,
    out_row_stride_cols: int,
):
    if lhs_matrix_blocks is None or rhs_col_blocks is None:
        return Q4_0_AVX2_ERR_NULL_PTR, []

    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_capacity < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []
    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []
    if tile_rows <= 0 or tile_cols <= 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    if lhs_rows > 0 and lhs_row_stride_blocks < k_block_count:
        return Q4_0_AVX2_ERR_BAD_LEN, []
    if rhs_cols > 0 and rhs_col_stride_blocks < k_block_count:
        return Q4_0_AVX2_ERR_BAD_LEN, []
    if lhs_rows > 0 and out_row_stride_cols < rhs_cols:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    ok, required_lhs_blocks = q4_0_try_mul_i64_nonneg(lhs_rows, lhs_row_stride_blocks)
    if not ok:
        return Q4_0_AVX2_ERR_OVERFLOW, []
    if required_lhs_blocks > lhs_block_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    ok, required_rhs_blocks = q4_0_try_mul_i64_nonneg(rhs_cols, rhs_col_stride_blocks)
    if not ok:
        return Q4_0_AVX2_ERR_OVERFLOW, []
    if required_rhs_blocks > rhs_block_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    ok, required_out_cells = q4_0_try_mul_i64_nonneg(lhs_rows, out_row_stride_cols)
    if not ok:
        return Q4_0_AVX2_ERR_OVERFLOW, []
    if required_out_cells > out_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    out = [0] * out_capacity

    row_tile_start = 0
    while row_tile_start < lhs_rows:
        ok, row_tile_end = q4_0_try_add_i64_nonneg(row_tile_start, tile_rows)
        if not ok:
            return Q4_0_AVX2_ERR_OVERFLOW, []
        row_tile_end = min(row_tile_end, lhs_rows)

        col_tile_start = 0
        while col_tile_start < rhs_cols:
            ok, col_tile_end = q4_0_try_add_i64_nonneg(col_tile_start, tile_cols)
            if not ok:
                return Q4_0_AVX2_ERR_OVERFLOW, []
            col_tile_end = min(col_tile_end, rhs_cols)

            for row_index in range(row_tile_start, row_tile_end):
                ok, lhs_row_base = q4_0_try_mul_i64_nonneg(row_index, lhs_row_stride_blocks)
                if not ok:
                    return Q4_0_AVX2_ERR_OVERFLOW, []
                ok, out_row_base = q4_0_try_mul_i64_nonneg(row_index, out_row_stride_cols)
                if not ok:
                    return Q4_0_AVX2_ERR_OVERFLOW, []

                for col_index in range(col_tile_start, col_tile_end):
                    ok, rhs_col_base = q4_0_try_mul_i64_nonneg(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q4_0_AVX2_ERR_OVERFLOW, []
                    ok, out_index = q4_0_try_add_i64_nonneg(out_row_base, col_index)
                    if not ok:
                        return Q4_0_AVX2_ERR_OVERFLOW, []

                    dot_holder = {"value": 0}
                    status = q4_0_dot_blocks_avx2_q32_to_q16_checked_ptr(
                        lhs_blocks=lhs_matrix_blocks[lhs_row_base : lhs_row_base + k_block_count],
                        rhs_blocks=rhs_col_blocks[rhs_col_base : rhs_col_base + k_block_count],
                        block_count=k_block_count,
                        out_holder=dot_holder,
                    )
                    if status != Q4_0_AVX2_OK:
                        return status, []

                    out[out_index] = dot_holder["value"]

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    return Q4_0_AVX2_OK, out


def q4_0_matmul_tiled_avx2_q32_to_q16_checked_ptr(
    lhs_matrix_blocks,
    lhs_block_capacity: int,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_block_capacity: int,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    tile_rows: int,
    tile_cols: int,
    out_capacity: int,
    out_row_stride_cols: int,
    out_holder,
):
    if out_holder is None:
        return Q4_0_AVX2_ERR_NULL_PTR

    err, out = q4_0_matmul_tiled_avx2_q32_to_q16_checked(
        lhs_matrix_blocks=lhs_matrix_blocks,
        lhs_block_capacity=lhs_block_capacity,
        lhs_rows=lhs_rows,
        lhs_row_stride_blocks=lhs_row_stride_blocks,
        rhs_col_blocks=rhs_col_blocks,
        rhs_block_capacity=rhs_block_capacity,
        rhs_cols=rhs_cols,
        rhs_col_stride_blocks=rhs_col_stride_blocks,
        k_block_count=k_block_count,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        out_capacity=out_capacity,
        out_row_stride_cols=out_row_stride_cols,
    )
    if err != Q4_0_AVX2_OK:
        return err

    out_holder["mat"] = out
    return Q4_0_AVX2_OK


def q4_0_matmul_tiled_avx2_q32_to_q16_checked_default_tiles_ptr(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_row_stride_cols: int,
    out_holder,
):
    if lhs_matrix_blocks is None or rhs_col_blocks is None or out_holder is None:
        return Q4_0_AVX2_ERR_NULL_PTR

    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q4_0_AVX2_ERR_BAD_LEN
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q4_0_AVX2_ERR_BAD_LEN
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q4_0_AVX2_ERR_BAD_LEN

    i64_min = -(1 << 63)

    lhs_block_capacity = lhs_rows * lhs_row_stride_blocks
    if lhs_block_capacity < i64_min or lhs_block_capacity > I64_MAX:
        return Q4_0_AVX2_ERR_OVERFLOW

    rhs_block_capacity = rhs_cols * rhs_col_stride_blocks
    if rhs_block_capacity < i64_min or rhs_block_capacity > I64_MAX:
        return Q4_0_AVX2_ERR_OVERFLOW

    out_capacity = lhs_rows * out_row_stride_cols
    if out_capacity < i64_min or out_capacity > I64_MAX:
        return Q4_0_AVX2_ERR_OVERFLOW

    return q4_0_matmul_tiled_avx2_q32_to_q16_checked_ptr(
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
        out_holder=out_holder,
    )


def test_default_tiles_wrapper_matches_checked_core_randomized() -> None:
    rng = random.Random(2026041815)

    for _ in range(200):
        lhs_rows = rng.randint(0, 6)
        rhs_cols = rng.randint(0, 6)
        k_block_count = rng.randint(0, 5)
        lhs_row_stride_blocks = k_block_count + rng.randint(0, 3)
        rhs_col_stride_blocks = k_block_count + rng.randint(0, 3)
        out_row_stride_cols = rhs_cols + rng.randint(0, 3)

        lhs = build_matrix_rows_as_blocks(lhs_rows, lhs_row_stride_blocks, k_block_count, rng)
        rhs = build_matrix_cols_as_blocks(rhs_cols, rhs_col_stride_blocks, k_block_count, rng)

        out_default = {"mat": [111, 222, 333]}
        out_core = {"mat": [444, 555, 666]}

        err_default = q4_0_matmul_tiled_avx2_q32_to_q16_checked_default_tiles_ptr(
            lhs_matrix_blocks=lhs,
            lhs_rows=lhs_rows,
            lhs_row_stride_blocks=lhs_row_stride_blocks,
            rhs_col_blocks=rhs,
            rhs_cols=rhs_cols,
            rhs_col_stride_blocks=rhs_col_stride_blocks,
            k_block_count=k_block_count,
            out_row_stride_cols=out_row_stride_cols,
            out_holder=out_default,
        )
        err_core = q4_0_matmul_tiled_avx2_q32_to_q16_checked_ptr(
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
            out_holder=out_core,
        )

        assert err_default == err_core
        if err_default == Q4_0_AVX2_OK:
            assert out_default["mat"] == out_core["mat"]
        else:
            assert out_default["mat"] == [111, 222, 333]


def test_default_tiles_wrapper_bad_len_and_null_parity() -> None:
    rng = random.Random(2026041816)
    lhs = build_matrix_rows_as_blocks(1, 1, 1, rng)
    rhs = build_matrix_cols_as_blocks(1, 1, 1, rng)

    out_holder = {"mat": [9, 9]}
    err = q4_0_matmul_tiled_avx2_q32_to_q16_checked_default_tiles_ptr(
        lhs_matrix_blocks=None,
        lhs_rows=1,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=rhs,
        rhs_cols=1,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        out_row_stride_cols=1,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_NULL_PTR
    assert out_holder["mat"] == [9, 9]

    err = q4_0_matmul_tiled_avx2_q32_to_q16_checked_default_tiles_ptr(
        lhs_matrix_blocks=lhs,
        lhs_rows=-1,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=rhs,
        rhs_cols=1,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        out_row_stride_cols=1,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN
    assert out_holder["mat"] == [9, 9]


def test_default_tiles_wrapper_shortfall_parity_and_no_partial_write() -> None:
    rng = random.Random(2026041817)
    lhs = build_matrix_rows_as_blocks(2, 1, 1, rng)
    rhs = build_matrix_cols_as_blocks(1, 1, 1, rng)

    out_default = {"mat": [1234]}
    out_core = {"mat": [5678]}

    err_default = q4_0_matmul_tiled_avx2_q32_to_q16_checked_default_tiles_ptr(
        lhs_matrix_blocks=lhs,
        lhs_rows=2,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=rhs,
        rhs_cols=2,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        out_row_stride_cols=2,
        out_holder=out_default,
    )
    err_core = q4_0_matmul_tiled_avx2_q32_to_q16_checked_ptr(
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
        out_holder=out_core,
    )

    assert err_default == err_core == Q4_0_AVX2_ERR_BAD_LEN
    assert out_default["mat"] == [1234]
    assert out_core["mat"] == [5678]


def test_default_tiles_wrapper_capacity_overflow_rejected() -> None:
    out_holder = {"mat": [42]}

    err = q4_0_matmul_tiled_avx2_q32_to_q16_checked_default_tiles_ptr(
        lhs_matrix_blocks=[],
        lhs_rows=I64_MAX,
        lhs_row_stride_blocks=2,
        rhs_col_blocks=[],
        rhs_cols=0,
        rhs_col_stride_blocks=0,
        k_block_count=0,
        out_row_stride_cols=1,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert out_holder["mat"] == [42]

    err = q4_0_matmul_tiled_avx2_q32_to_q16_checked_default_tiles_ptr(
        lhs_matrix_blocks=[],
        lhs_rows=1,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=[],
        rhs_cols=I64_MAX,
        rhs_col_stride_blocks=2,
        k_block_count=0,
        out_row_stride_cols=1,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert out_holder["mat"] == [42]

    err = q4_0_matmul_tiled_avx2_q32_to_q16_checked_default_tiles_ptr(
        lhs_matrix_blocks=[],
        lhs_rows=I64_MAX,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=[],
        rhs_cols=0,
        rhs_col_stride_blocks=0,
        k_block_count=0,
        out_row_stride_cols=2,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert out_holder["mat"] == [42]


def run() -> None:
    test_default_tiles_wrapper_matches_checked_core_randomized()
    test_default_tiles_wrapper_bad_len_and_null_parity()
    test_default_tiles_wrapper_shortfall_parity_and_no_partial_write()
    test_default_tiles_wrapper_capacity_overflow_rejected()
    print("q4_0_matmul_tiled_avx2_q32_to_q16_checked_default_tiles_reference_checks=ok")


if __name__ == "__main__":
    run()
