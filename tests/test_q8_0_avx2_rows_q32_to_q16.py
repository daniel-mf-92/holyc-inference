#!/usr/bin/env python3
"""Reference checks for Q8_0DotRowsAVX2Q32ToQ16Checked semantics."""

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
)
from test_q8_0_avx2_rows_q32 import q8_0_dot_rows_avx2_q32_checked
from test_q8_0_avx2_blocks_q32_to_q16 import q8_0_dot_q32_to_q16


def q8_0_dot_rows_avx2_q32_to_q16_checked(
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

    required_matrix_blocks = row_count * row_stride_blocks
    if row_count and row_stride_blocks and required_matrix_blocks > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW, []
    if required_matrix_blocks > matrix_block_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    err, rows_q32 = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix_blocks,
        matrix_block_capacity=matrix_block_capacity,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_capacity=vec_block_capacity,
        vec_block_count=vec_block_count,
    )
    if err != Q8_0_AVX2_OK:
        return err, []

    return Q8_0_AVX2_OK, [q8_0_dot_q32_to_q16(v) for v in rows_q32]


def test_known_rows_match_q32_then_single_round() -> None:
    vec = [
        make_block(0x3C00, [i - 16 for i in range(32)]),
        make_block(0x3800, [16 - i for i in range(32)]),
    ]

    row0 = [
        make_block(0x3A00, [(-1) ** i * (i % 13) for i in range(32)]),
        make_block(0x3555, [7 - (i % 11) for i in range(32)]),
    ]
    row1 = [
        make_block(0x4000, [2 * i - 31 for i in range(32)]),
        make_block(0x3000, [i % 9 - 4 for i in range(32)]),
    ]

    matrix = row0 + row1

    err, got_rows_q16 = q8_0_dot_rows_avx2_q32_to_q16_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=len(matrix),
        row_count=2,
        row_stride_blocks=2,
        vec_blocks=vec,
        vec_block_capacity=len(vec),
        vec_block_count=2,
    )
    assert err == Q8_0_AVX2_OK

    err, ref_rows_q32 = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=len(matrix),
        row_count=2,
        row_stride_blocks=2,
        vec_blocks=vec,
        vec_block_capacity=len(vec),
        vec_block_count=2,
    )
    assert err == Q8_0_AVX2_OK

    assert got_rows_q16 == [q8_0_dot_q32_to_q16(v) for v in ref_rows_q32]


def test_stride_padding_keeps_single_rounding_contract() -> None:
    vec = [
        make_block(0x1800, [2] + [0] * 31),
        make_block(0x1800, [2] + [0] * 31),
    ]

    live_row = [
        make_block(0x1800, [1] + [0] * 31),
        make_block(0x1800, [1] + [0] * 31),
    ]
    pad = make_block(0x3C00, [127] * 32)

    matrix = live_row + [pad] + live_row + [pad]

    err, got = q8_0_dot_rows_avx2_q32_to_q16_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=len(matrix),
        row_count=2,
        row_stride_blocks=3,
        vec_blocks=vec,
        vec_block_capacity=len(vec),
        vec_block_count=2,
    )
    assert err == Q8_0_AVX2_OK

    # Per-row expected is single rounded full-row Q32 sum.
    err, row_q32 = q8_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=len(matrix),
        row_count=2,
        row_stride_blocks=3,
        vec_blocks=vec,
        vec_block_capacity=len(vec),
        vec_block_count=2,
    )
    assert err == Q8_0_AVX2_OK
    expected = [q8_0_dot_q32_to_q16(v) for v in row_q32]
    assert got == expected


def test_randomized_reference_parity() -> None:
    rng = random.Random(2026041603)
    fp16_scales = [0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00, 0x4000, 0x4400, 0xB800, 0xBC00]

    for _ in range(200):
        row_count = rng.randint(1, 8)
        vec_block_count = rng.randint(1, 6)
        row_stride_blocks = vec_block_count + rng.randint(0, 2)
        matrix_capacity = row_count * row_stride_blocks

        matrix = []
        for _ in range(matrix_capacity):
            matrix.append(make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]))

        vec = []
        for _ in range(vec_block_count):
            vec.append(make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]))

        err, got_q16 = q8_0_dot_rows_avx2_q32_to_q16_checked(
            matrix_blocks=matrix,
            matrix_block_capacity=matrix_capacity,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_capacity=vec_block_count,
            vec_block_count=vec_block_count,
        )
        assert err == Q8_0_AVX2_OK

        err, rows_q32 = q8_0_dot_rows_avx2_q32_checked(
            matrix_blocks=matrix,
            matrix_block_capacity=matrix_capacity,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_capacity=vec_block_count,
            vec_block_count=vec_block_count,
        )
        assert err == Q8_0_AVX2_OK

        expected = [q8_0_dot_q32_to_q16(v) for v in rows_q32]
        assert got_q16 == expected


def test_error_paths() -> None:
    err, _ = q8_0_dot_rows_avx2_q32_to_q16_checked(None, 0, 0, 0, [], 0, 0)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_dot_rows_avx2_q32_to_q16_checked([], 0, 1, 0, [make_block(0x3C00, [0] * 32)], 1, 1)
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_dot_rows_avx2_q32_to_q16_checked([], I64_MAX, I64_MAX, 2, [], 0, 0)
    assert err == Q8_0_AVX2_ERR_OVERFLOW


def run() -> None:
    test_known_rows_match_q32_then_single_round()
    test_stride_padding_keeps_single_rounding_contract()
    test_randomized_reference_parity()
    test_error_paths()
    print("q8_0_avx2_rows_q32_to_q16_reference_checks=ok")


if __name__ == "__main__":
    run()
