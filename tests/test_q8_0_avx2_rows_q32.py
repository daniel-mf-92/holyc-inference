#!/usr/bin/env python3
"""Reference checks for Q8_0DotRowsAVX2Q32Checked semantics."""

from __future__ import annotations

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_avx2_blocks_q32 import (
    I64_MAX,
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_ERR_NULL_PTR,
    Q8_0_AVX2_ERR_OVERFLOW,
    Q8_0_AVX2_OK,
    make_block,
    q8_0_dot_blocks_avx2_q32_checked,
    q8_0_try_mul_i64,
)


def q8_0_dot_rows_avx2_q32_checked(
    matrix_blocks,
    matrix_block_capacity: int,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_capacity: int,
    vec_block_count: int,
):
    if matrix_blocks is None or vec_blocks is None:
        return Q8_0_AVX2_ERR_NULL_PTR, []

    if matrix_block_capacity < 0 or vec_block_capacity < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    if row_count > 0 and row_stride_blocks < vec_block_count:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if vec_block_count > vec_block_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    ok, required_matrix_blocks = q8_0_try_mul_i64(row_count, row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, []
    if required_matrix_blocks > matrix_block_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    out_rows_q32 = []
    for row_index in range(row_count):
        ok, row_base = q8_0_try_mul_i64(row_index, row_stride_blocks)
        if not ok:
            return Q8_0_AVX2_ERR_OVERFLOW, []

        row_blocks = matrix_blocks[row_base : row_base + vec_block_count]
        err, row_dot_q32 = q8_0_dot_blocks_avx2_q32_checked(row_blocks, vec_blocks, vec_block_count)
        if err != Q8_0_AVX2_OK:
            return err, []

        out_rows_q32.append(row_dot_q32)

    return Q8_0_AVX2_OK, out_rows_q32


def test_known_rows_match_per_row_block_dot() -> None:
    vec = [
        make_block(0x3C00, [i - 16 for i in range(32)]),
        make_block(0x3C00, [16 - i for i in range(32)]),
    ]

    row0 = [
        make_block(0x3C00, [i - 8 for i in range(32)]),
        make_block(0x3800, [(-1) ** i * (i % 9) for i in range(32)]),
    ]
    row1 = [
        make_block(0x4000, [2 * i - 31 for i in range(32)]),
        make_block(0x3555, [7 - (i % 13) for i in range(32)]),
    ]

    matrix = row0 + row1

    err, rows = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=len(matrix),
        row_count=2,
        row_stride_blocks=2,
        vec_blocks=vec,
        vec_block_capacity=len(vec),
        vec_block_count=2,
    )
    assert err == Q8_0_AVX2_OK

    err, exp0 = q8_0_dot_blocks_avx2_q32_checked(row0, vec, 2)
    assert err == Q8_0_AVX2_OK
    err, exp1 = q8_0_dot_blocks_avx2_q32_checked(row1, vec, 2)
    assert err == Q8_0_AVX2_OK

    assert rows == [exp0, exp1]


def test_row_stride_with_padding_blocks() -> None:
    vec = [
        make_block(0x3C00, [1] * 32),
        make_block(0x3C00, [2] * 32),
    ]

    row0_live = [
        make_block(0x3C00, [3] * 32),
        make_block(0x3C00, [4] * 32),
    ]
    row1_live = [
        make_block(0x3C00, [5] * 32),
        make_block(0x3C00, [6] * 32),
    ]

    pad = make_block(0x3C00, [127] * 32)
    matrix = row0_live + [pad] + row1_live + [pad]

    err, rows = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=len(matrix),
        row_count=2,
        row_stride_blocks=3,
        vec_blocks=vec,
        vec_block_capacity=len(vec),
        vec_block_count=2,
    )
    assert err == Q8_0_AVX2_OK

    err, exp0 = q8_0_dot_blocks_avx2_q32_checked(row0_live, vec, 2)
    assert err == Q8_0_AVX2_OK
    err, exp1 = q8_0_dot_blocks_avx2_q32_checked(row1_live, vec, 2)
    assert err == Q8_0_AVX2_OK

    assert rows == [exp0, exp1]


def test_bad_len_guards() -> None:
    vec = [make_block(0x3C00, [1] * 32)]
    matrix = [make_block(0x3C00, [1] * 32)]

    err, _ = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=1,
        row_count=1,
        row_stride_blocks=0,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=1,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=1,
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_capacity=0,
        vec_block_count=1,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=0,
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=1,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN


def test_capacity_multiply_overflow_is_rejected() -> None:
    vec = []
    matrix = []

    err, _ = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=I64_MAX,
        row_count=I64_MAX,
        row_stride_blocks=2,
        vec_blocks=vec,
        vec_block_capacity=0,
        vec_block_count=0,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW


def test_null_ptr_rejected() -> None:
    err, _ = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=None,
        matrix_block_capacity=0,
        row_count=0,
        row_stride_blocks=0,
        vec_blocks=[],
        vec_block_capacity=0,
        vec_block_count=0,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR


if __name__ == "__main__":
    test_known_rows_match_per_row_block_dot()
    test_row_stride_with_padding_blocks()
    test_bad_len_guards()
    test_capacity_multiply_overflow_is_rejected()
    test_null_ptr_rejected()
    print("ok")
