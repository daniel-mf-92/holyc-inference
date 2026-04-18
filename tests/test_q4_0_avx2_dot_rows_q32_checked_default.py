#!/usr/bin/env python3
"""Parity checks for Q4_0DotRowsAVX2Q32CheckedDefault wrapper semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_avx2_dot_q32_checked import (
    Q4_0_AVX2_ERR_BAD_LEN,
    Q4_0_AVX2_ERR_NULL_PTR,
    Q4_0_AVX2_OK,
    half_bits,
    pack_q4_signed,
)
from test_q4_0_avx2_dot_rows_q32_checked import (
    I64_MAX,
    Q4_0_AVX2_ERR_OVERFLOW,
    q4_0_dot_rows_avx2_q32_checked,
)


def q4_0_dot_rows_avx2_q32_checked_default(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
):
    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    matrix_block_capacity = row_count * row_stride_blocks
    i64_min = -(1 << 63)
    if matrix_block_capacity < i64_min or matrix_block_capacity > I64_MAX:
        return Q4_0_AVX2_ERR_OVERFLOW, []

    return q4_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix_blocks,
        matrix_block_capacity=matrix_block_capacity,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_capacity=vec_block_count,
        vec_block_count=vec_block_count,
    )


def make_block(scale: float, vals: list[int]):
    return (half_bits(scale), pack_q4_signed(vals))


def make_random_block(rng: random.Random):
    scale = rng.choice([0.0, 0.125, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0])
    vals = [rng.randrange(-8, 8) for _ in range(32)]
    return make_block(scale, vals)


def test_default_wrapper_matches_checked_core_success_and_errors() -> None:
    rng = random.Random(2026041804)

    for _ in range(260):
        row_count = rng.randint(0, 6)
        vec_block_count = rng.randint(0, 5)
        row_stride_blocks = vec_block_count + rng.randint(0, 4)

        matrix = []
        for _row in range(row_count):
            for _ in range(row_stride_blocks):
                matrix.append(make_random_block(rng))

        vec = [make_random_block(rng) for _ in range(vec_block_count)]

        err_default, rows_default = q4_0_dot_rows_avx2_q32_checked_default(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
        )
        err_core, rows_core = q4_0_dot_rows_avx2_q32_checked(
            matrix_blocks=matrix,
            matrix_block_capacity=row_count * row_stride_blocks,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_capacity=vec_block_count,
            vec_block_count=vec_block_count,
        )

        assert err_default == err_core
        assert rows_default == rows_core


def test_default_wrapper_bad_len_and_null_parity() -> None:
    block = make_block(1.0, [0] * 32)

    err, rows = q4_0_dot_rows_avx2_q32_checked_default(
        matrix_blocks=[block],
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[block],
        vec_block_count=1,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN
    assert rows == []

    err, rows = q4_0_dot_rows_avx2_q32_checked_default(
        matrix_blocks=[block],
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=None,
        vec_block_count=1,
    )
    assert err == Q4_0_AVX2_ERR_NULL_PTR
    assert rows == []


def test_default_wrapper_extent_shortfalls_match_checked_core() -> None:
    block = make_block(0.5, [1] * 32)
    matrix = [block]
    vec = [block]

    # Default wrapper maps matrix capacity to row_count*row_stride (=2).
    # With one physical matrix block and two logical rows, row1 produces an
    # empty slice and the checked core must surface BAD_LEN.
    err_default, rows_default = q4_0_dot_rows_avx2_q32_checked_default(
        matrix_blocks=matrix,
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_count=1,
    )
    err_core, rows_core = q4_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=2,
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=1,
    )

    assert err_default == err_core == Q4_0_AVX2_ERR_BAD_LEN
    assert rows_default == rows_core == []


def test_default_wrapper_capacity_multiply_overflow_is_rejected() -> None:
    err, rows = q4_0_dot_rows_avx2_q32_checked_default(
        matrix_blocks=[],
        row_count=I64_MAX,
        row_stride_blocks=2,
        vec_blocks=[],
        vec_block_count=0,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert rows == []


if __name__ == "__main__":
    test_default_wrapper_matches_checked_core_success_and_errors()
    test_default_wrapper_bad_len_and_null_parity()
    test_default_wrapper_extent_shortfalls_match_checked_core()
    test_default_wrapper_capacity_multiply_overflow_is_rejected()
    print("q4_0_avx2_dot_rows_q32_checked_default_reference_checks=ok")
