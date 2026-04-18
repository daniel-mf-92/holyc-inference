#!/usr/bin/env python3
"""Parity checks for Q4_0DotRowsAVX2Q32ToQ16CheckedDefault wrapper semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_avx2_dot_rows_q32_checked import (
    I64_MAX,
    Q4_0_AVX2_ERR_BAD_LEN,
    Q4_0_AVX2_ERR_NULL_PTR,
    Q4_0_AVX2_ERR_OVERFLOW,
    Q4_0_AVX2_OK,
    make_block,
    q4_0_dot_rows_avx2_q32_checked,
)


def q4_0_round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    return (value + (1 << (shift - 1))) >> shift


def q4_0_round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return q4_0_round_shift_right_unsigned(value, shift)
    return -q4_0_round_shift_right_unsigned(-value, shift)


def q4_0_dot_q32_to_q16(dot_q32: int) -> int:
    return q4_0_round_shift_right_signed(dot_q32, 16)


def q4_0_dot_rows_avx2_q32_to_q16_checked(
    matrix_blocks,
    matrix_block_capacity: int,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_capacity: int,
    vec_block_count: int,
):
    err_q32, rows_q32 = q4_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix_blocks,
        matrix_block_capacity=matrix_block_capacity,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_capacity=vec_block_capacity,
        vec_block_count=vec_block_count,
    )
    if err_q32 != Q4_0_AVX2_OK:
        return err_q32, []

    return Q4_0_AVX2_OK, [q4_0_dot_q32_to_q16(v) for v in rows_q32]


def q4_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_holder,
):
    if matrix_blocks is None or vec_blocks is None or out_holder is None:
        return Q4_0_AVX2_ERR_NULL_PTR

    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q4_0_AVX2_ERR_BAD_LEN

    i64_min = -(1 << 63)
    matrix_block_capacity = row_count * row_stride_blocks
    if matrix_block_capacity < i64_min or matrix_block_capacity > I64_MAX:
        return Q4_0_AVX2_ERR_OVERFLOW

    err, rows_q16 = q4_0_dot_rows_avx2_q32_to_q16_checked(
        matrix_blocks=matrix_blocks,
        matrix_block_capacity=matrix_block_capacity,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_capacity=vec_block_count,
        vec_block_count=vec_block_count,
    )
    if err != Q4_0_AVX2_OK:
        return err

    out_holder["rows"] = rows_q16
    return Q4_0_AVX2_OK


def q4_0_dot_rows_avx2_q32_to_q16_checked_ptr(
    matrix_blocks,
    matrix_block_capacity: int,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_capacity: int,
    vec_block_count: int,
    out_holder,
):
    if matrix_blocks is None or vec_blocks is None or out_holder is None:
        return Q4_0_AVX2_ERR_NULL_PTR

    err, rows_q16 = q4_0_dot_rows_avx2_q32_to_q16_checked(
        matrix_blocks=matrix_blocks,
        matrix_block_capacity=matrix_block_capacity,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_capacity=vec_block_capacity,
        vec_block_count=vec_block_count,
    )
    if err != Q4_0_AVX2_OK:
        return err

    out_holder["rows"] = rows_q16
    return Q4_0_AVX2_OK


def test_default_wrapper_matches_checked_core_success_and_errors() -> None:
    rng = random.Random(2026041812)

    for _ in range(260):
        row_count = rng.randint(0, 6)
        vec_block_count = rng.randint(0, 5)
        row_stride_blocks = vec_block_count + rng.randint(0, 4)

        matrix = []
        for _row in range(row_count):
            for _ in range(row_stride_blocks):
                matrix.append(
                    make_block(
                        rng.choice([0.125, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0]),
                        [rng.randint(-8, 7) for _ in range(32)],
                    )
                )

        vec = [
            make_block(
                rng.choice([0.125, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0]),
                [rng.randint(-8, 7) for _ in range(32)],
            )
            for _ in range(vec_block_count)
        ]

        out_default = {"rows": [111, 222, 333]}
        out_core = {"rows": [444, 555, 666]}

        err_default = q4_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_default,
        )
        err_core = q4_0_dot_rows_avx2_q32_to_q16_checked_ptr(
            matrix_blocks=matrix,
            matrix_block_capacity=row_count * row_stride_blocks,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_capacity=vec_block_count,
            vec_block_count=vec_block_count,
            out_holder=out_core,
        )

        assert err_default == err_core
        if err_default == Q4_0_AVX2_OK:
            assert out_default["rows"] == out_core["rows"]
        else:
            assert out_default["rows"] == [111, 222, 333]


def test_default_wrapper_bad_len_and_null_parity() -> None:
    block = make_block(1.0, [0] * 32)

    out_holder = {"rows": [9, 9]}
    err = q4_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
        matrix_blocks=[block],
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[block],
        vec_block_count=1,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN
    assert out_holder["rows"] == [9, 9]

    err = q4_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
        matrix_blocks=[block],
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=None,
        vec_block_count=1,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_NULL_PTR
    assert out_holder["rows"] == [9, 9]


def test_default_wrapper_extent_shortfalls_match_checked_core() -> None:
    block = make_block(1.0, [1] * 32)
    matrix = [block]
    vec = [block]

    out_default = {"rows": [1234]}
    out_core = {"rows": [5678]}

    err_default = q4_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
        matrix_blocks=matrix,
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_count=1,
        out_holder=out_default,
    )
    err_core = q4_0_dot_rows_avx2_q32_to_q16_checked_ptr(
        matrix_blocks=matrix,
        matrix_block_capacity=2,
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=1,
        out_holder=out_core,
    )

    assert err_default == err_core == Q4_0_AVX2_ERR_BAD_LEN
    assert out_default["rows"] == [1234]
    assert out_core["rows"] == [5678]


def test_default_wrapper_capacity_multiply_overflow_is_rejected() -> None:
    out_holder = {"rows": [42]}
    err = q4_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
        matrix_blocks=[],
        row_count=I64_MAX,
        row_stride_blocks=2,
        vec_blocks=[],
        vec_block_count=0,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert out_holder["rows"] == [42]


if __name__ == "__main__":
    test_default_wrapper_matches_checked_core_success_and_errors()
    test_default_wrapper_bad_len_and_null_parity()
    test_default_wrapper_extent_shortfalls_match_checked_core()
    test_default_wrapper_capacity_multiply_overflow_is_rejected()
    print("q4_0_avx2_dot_rows_q32_to_q16_checked_default_reference_checks=ok")
