#!/usr/bin/env python3
"""Parity checks for Q8_0DotRowsAVX2Q32Checked semantics."""

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
    Q8_0_AVX2_OK,
    make_block,
    q8_0_dot_blocks_avx2_q32_checked,
    q8_0_try_mul_i64,
)
from test_q8_0_dot import dot_product_blocks_q32_checked


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


def scalar_row_reference_q32(matrix_blocks, row_count, row_stride_blocks, vec_blocks, vec_block_count):
    expected = []
    for row_index in range(row_count):
        row_base = row_index * row_stride_blocks
        row_slice = matrix_blocks[row_base : row_base + vec_block_count]
        err, row_dot_q32 = dot_product_blocks_q32_checked(
            [(blk["d_fp16"], bytes((q + 256) % 256 for q in blk["qs"])) for blk in row_slice],
            [(blk["d_fp16"], bytes((q + 256) % 256 for q in blk["qs"])) for blk in vec_blocks[:vec_block_count]],
        )
        assert err == 0
        expected.append(row_dot_q32)
    return expected


def test_known_rows_match_scalar_row_reference() -> None:
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

    expected = scalar_row_reference_q32(matrix, 2, 2, vec, 2)
    assert rows == expected


def test_randomized_rows_match_scalar_reference() -> None:
    rng = random.Random(2026041802)
    fp16_scales = [0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00, 0x4000, 0x4400, 0xBC00]

    for _ in range(180):
        row_count = rng.randint(1, 6)
        vec_block_count = rng.randint(1, 5)
        row_stride_blocks = vec_block_count + rng.randint(0, 3)

        matrix = []
        for _row in range(row_count):
            for _ in range(row_stride_blocks):
                matrix.append(
                    make_block(
                        rng.choice(fp16_scales),
                        [rng.randint(-128, 127) for _ in range(32)],
                    )
                )

        vec = [
            make_block(
                rng.choice(fp16_scales),
                [rng.randint(-128, 127) for _ in range(32)],
            )
            for _ in range(vec_block_count)
        ]

        err, rows = q8_0_dot_rows_avx2_q32_checked(
            matrix_blocks=matrix,
            matrix_block_capacity=len(matrix),
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_capacity=len(vec),
            vec_block_count=vec_block_count,
        )
        assert err == Q8_0_AVX2_OK

        expected = scalar_row_reference_q32(matrix, row_count, row_stride_blocks, vec, vec_block_count)
        assert rows == expected


def test_row_stride_padding_ignored() -> None:
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

    expected = scalar_row_reference_q32(matrix, 2, 3, vec, 2)
    assert rows == expected


def test_guard_paths_and_no_partial_output() -> None:
    vec = [make_block(0x3C00, [1] * 32)]
    matrix = [make_block(0x3C00, [1] * 32)]

    err, rows = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=1,
        row_count=1,
        row_stride_blocks=0,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=1,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert rows == []

    err, rows = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=1,
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_capacity=0,
        vec_block_count=1,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert rows == []

    err, rows = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=0,
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=1,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert rows == []

    err, rows = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=None,
        matrix_block_capacity=0,
        row_count=0,
        row_stride_blocks=0,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=0,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert rows == []


def test_capacity_multiply_overflow_is_rejected() -> None:
    err, rows = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=[],
        matrix_block_capacity=I64_MAX,
        row_count=I64_MAX,
        row_stride_blocks=2,
        vec_blocks=[],
        vec_block_capacity=0,
        vec_block_count=0,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert rows == []


if __name__ == "__main__":
    test_known_rows_match_scalar_row_reference()
    test_randomized_rows_match_scalar_reference()
    test_row_stride_padding_ignored()
    test_guard_paths_and_no_partial_output()
    test_capacity_multiply_overflow_is_rejected()
    print("ok")
