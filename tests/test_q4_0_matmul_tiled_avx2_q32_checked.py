#!/usr/bin/env python3
"""Reference checks for Q4_0MatMulTiledAVX2Q32Checked semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_avx2_dot_q32_checked import (
    Q4_0_AVX2_ERR_BAD_LEN,
    Q4_0_AVX2_ERR_NULL_PTR,
    Q4_0_AVX2_OK,
    dot_product_blocks_q32_avx2_checked,
    half_bits,
    pack_q4_signed,
)
from test_q4_0_avx2_dot_rows_q32_checked import (
    I64_MAX,
    Q4_0_AVX2_ERR_OVERFLOW,
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


def q4_0_dot_rows_avx2_q32_checked(
    matrix_blocks,
    matrix_block_capacity: int,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_capacity: int,
    vec_block_count: int,
    out_rows_q32,
):
    if matrix_blocks is None or vec_blocks is None or out_rows_q32 is None:
        return Q4_0_AVX2_ERR_NULL_PTR

    if matrix_block_capacity < 0 or vec_block_capacity < 0:
        return Q4_0_AVX2_ERR_BAD_LEN
    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q4_0_AVX2_ERR_BAD_LEN

    if row_count > 0 and row_stride_blocks < vec_block_count:
        return Q4_0_AVX2_ERR_BAD_LEN
    if vec_block_count > vec_block_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN

    ok, required_matrix_blocks = q4_0_try_mul_i64_nonneg(row_count, row_stride_blocks)
    if not ok:
        return Q4_0_AVX2_ERR_OVERFLOW
    if required_matrix_blocks > matrix_block_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN

    if len(out_rows_q32) < row_count:
        return Q4_0_AVX2_ERR_BAD_LEN

    for row_index in range(row_count):
        ok, row_base = q4_0_try_mul_i64_nonneg(row_index, row_stride_blocks)
        if not ok:
            return Q4_0_AVX2_ERR_OVERFLOW

        row_blocks = matrix_blocks[row_base : row_base + vec_block_count]
        out_holder = {"value": -1}
        err = dot_product_blocks_q32_avx2_checked(
            row_blocks,
            matrix_block_capacity - row_base,
            vec_blocks,
            vec_block_capacity,
            vec_block_count,
            out_holder,
        )
        if err != Q4_0_AVX2_OK:
            return err

        out_rows_q32[row_index] = out_holder["value"]

    return Q4_0_AVX2_OK


def q4_0_matmul_tiled_avx2_q32_checked(
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

                lhs_row_ptr = lhs_matrix_blocks[lhs_row_base:]

                for col_index in range(col_tile_start, col_tile_end):
                    ok, rhs_col_base = q4_0_try_mul_i64_nonneg(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q4_0_AVX2_ERR_OVERFLOW, []
                    ok, out_index = q4_0_try_add_i64_nonneg(out_row_base, col_index)
                    if not ok:
                        return Q4_0_AVX2_ERR_OVERFLOW, []

                    rhs_col_ptr = rhs_col_blocks[rhs_col_base:]
                    row_dot_holder = [0]
                    status = q4_0_dot_rows_avx2_q32_checked(
                        lhs_row_ptr,
                        lhs_block_capacity - lhs_row_base,
                        1,
                        lhs_row_stride_blocks,
                        rhs_col_ptr,
                        rhs_block_capacity - rhs_col_base,
                        k_block_count,
                        row_dot_holder,
                    )
                    if status != Q4_0_AVX2_OK:
                        return status, []

                    out[out_index] = row_dot_holder[0]

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    return Q4_0_AVX2_OK, out


def q4_0_matmul_scalar_reference(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_row_stride_cols: int,
):
    out = [0] * (lhs_rows * out_row_stride_cols)
    for row in range(lhs_rows):
        lhs_base = row * lhs_row_stride_blocks
        for col in range(rhs_cols):
            rhs_base = col * rhs_col_stride_blocks
            out_holder = {"value": 0}
            err = dot_product_blocks_q32_avx2_checked(
                lhs_matrix_blocks[lhs_base : lhs_base + k_block_count],
                k_block_count,
                rhs_col_blocks[rhs_base : rhs_base + k_block_count],
                k_block_count,
                k_block_count,
                out_holder,
            )
            assert err == Q4_0_AVX2_OK
            out[row * out_row_stride_cols + col] = out_holder["value"]
    return out


def make_block(scale: float, vals: list[int]):
    return (half_bits(scale), pack_q4_signed(vals))


def build_matrix_rows_as_blocks(rows: int, row_stride_blocks: int, k_blocks: int, rng: random.Random):
    scales = [0.0, 0.125, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0]
    out = []
    for _ in range(rows):
        for block_index in range(row_stride_blocks):
            if block_index < k_blocks:
                out.append(make_block(rng.choice(scales), [rng.randrange(-8, 8) for _ in range(32)]))
            else:
                out.append(make_block(0.0, [0] * 32))
    return out


def build_matrix_cols_as_blocks(cols: int, col_stride_blocks: int, k_blocks: int, rng: random.Random):
    scales = [0.0, 0.125, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0]
    out = []
    for _ in range(cols):
        for block_index in range(col_stride_blocks):
            if block_index < k_blocks:
                out.append(make_block(rng.choice(scales), [rng.randrange(-8, 8) for _ in range(32)]))
            else:
                out.append(make_block(0.0, [0] * 32))
    return out


def test_known_small_matches_scalar_reference() -> None:
    rng = random.Random(2026041805)
    lhs_rows = 3
    rhs_cols = 4
    k_blocks = 2
    lhs_row_stride = 3
    rhs_col_stride = 4
    out_row_stride = 6

    lhs = build_matrix_rows_as_blocks(lhs_rows, lhs_row_stride, k_blocks, rng)
    rhs = build_matrix_cols_as_blocks(rhs_cols, rhs_col_stride, k_blocks, rng)

    err, out = q4_0_matmul_tiled_avx2_q32_checked(
        lhs,
        len(lhs),
        lhs_rows,
        lhs_row_stride,
        rhs,
        len(rhs),
        rhs_cols,
        rhs_col_stride,
        k_blocks,
        2,
        3,
        lhs_rows * out_row_stride,
        out_row_stride,
    )
    assert err == Q4_0_AVX2_OK

    expected = q4_0_matmul_scalar_reference(
        lhs,
        lhs_rows,
        lhs_row_stride,
        rhs,
        rhs_cols,
        rhs_col_stride,
        k_blocks,
        out_row_stride,
    )
    assert out == expected


def test_randomized_tiling_matches_scalar_many_shapes() -> None:
    rng = random.Random(2026041806)

    for _ in range(180):
        lhs_rows = rng.randint(1, 6)
        rhs_cols = rng.randint(1, 6)
        k_blocks = rng.randint(1, 6)
        lhs_row_stride = k_blocks + rng.randint(0, 3)
        rhs_col_stride = k_blocks + rng.randint(0, 3)
        out_row_stride = rhs_cols + rng.randint(0, 3)
        tile_rows = rng.randint(1, 4)
        tile_cols = rng.randint(1, 4)

        lhs = build_matrix_rows_as_blocks(lhs_rows, lhs_row_stride, k_blocks, rng)
        rhs = build_matrix_cols_as_blocks(rhs_cols, rhs_col_stride, k_blocks, rng)

        err, out = q4_0_matmul_tiled_avx2_q32_checked(
            lhs,
            len(lhs),
            lhs_rows,
            lhs_row_stride,
            rhs,
            len(rhs),
            rhs_cols,
            rhs_col_stride,
            k_blocks,
            tile_rows,
            tile_cols,
            lhs_rows * out_row_stride,
            out_row_stride,
        )
        assert err == Q4_0_AVX2_OK

        expected = q4_0_matmul_scalar_reference(
            lhs,
            lhs_rows,
            lhs_row_stride,
            rhs,
            rhs_cols,
            rhs_col_stride,
            k_blocks,
            out_row_stride,
        )
        assert out == expected


def test_error_paths() -> None:
    rng = random.Random(2026041807)
    lhs = build_matrix_rows_as_blocks(2, 2, 2, rng)
    rhs = build_matrix_cols_as_blocks(2, 2, 2, rng)

    err, _ = q4_0_matmul_tiled_avx2_q32_checked(
        None,
        0,
        0,
        0,
        rhs,
        len(rhs),
        2,
        2,
        2,
        1,
        1,
        0,
        0,
    )
    assert err == Q4_0_AVX2_ERR_NULL_PTR

    err, _ = q4_0_matmul_tiled_avx2_q32_checked(
        lhs,
        len(lhs),
        2,
        1,
        rhs,
        len(rhs),
        2,
        2,
        2,
        1,
        1,
        8,
        4,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN

    err, _ = q4_0_matmul_tiled_avx2_q32_checked(
        lhs,
        len(lhs),
        2,
        2,
        rhs,
        len(rhs),
        2,
        2,
        2,
        0,
        1,
        8,
        4,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN

    err, _ = q4_0_matmul_tiled_avx2_q32_checked(
        lhs,
        len(lhs),
        2,
        2,
        rhs,
        len(rhs),
        2,
        2,
        2,
        1,
        1,
        7,
        4,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN


def test_tile_bound_overflow_rejected() -> None:
    err, out = q4_0_matmul_tiled_avx2_q32_checked(
        lhs_matrix_blocks=[],
        lhs_block_capacity=0,
        lhs_rows=I64_MAX,
        lhs_row_stride_blocks=2,
        rhs_col_blocks=[],
        rhs_block_capacity=0,
        rhs_cols=0,
        rhs_col_stride_blocks=0,
        k_block_count=0,
        tile_rows=I64_MAX,
        tile_cols=1,
        out_capacity=0,
        out_row_stride_cols=0,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert out == []


def run() -> None:
    test_known_small_matches_scalar_reference()
    test_randomized_tiling_matches_scalar_many_shapes()
    test_error_paths()
    test_tile_bound_overflow_rejected()
    print("q4_0_matmul_tiled_avx2_q32_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
