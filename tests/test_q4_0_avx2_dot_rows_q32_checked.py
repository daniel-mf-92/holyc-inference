#!/usr/bin/env python3
"""Reference checks for Q4_0DotRowsAVX2Q32Checked semantics."""

from __future__ import annotations

import pathlib
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

Q4_0_AVX2_ERR_OVERFLOW = 3
I64_MAX = (1 << 63) - 1


def q4_0_try_mul_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs == 0 or rhs == 0:
        return True, 0

    if lhs == -1 and rhs == -(1 << 63):
        return False, 0
    if rhs == -1 and lhs == -(1 << 63):
        return False, 0

    if lhs > 0:
        if rhs > 0:
            if lhs > I64_MAX // rhs:
                return False, 0
        else:
            if rhs < (-(1 << 63)) // lhs:
                return False, 0
    else:
        if rhs > 0:
            if lhs < (-(1 << 63)) // rhs:
                return False, 0
        else:
            if lhs < I64_MAX // rhs:
                return False, 0

    return True, lhs * rhs


def q4_0_dot_rows_avx2_q32_checked(
    matrix_blocks,
    matrix_block_capacity: int,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_capacity: int,
    vec_block_count: int,
):
    if matrix_blocks is None or vec_blocks is None:
        return Q4_0_AVX2_ERR_NULL_PTR, []

    if matrix_block_capacity < 0 or vec_block_capacity < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []
    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    if row_count > 0 and row_stride_blocks < vec_block_count:
        return Q4_0_AVX2_ERR_BAD_LEN, []
    if vec_block_count > vec_block_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    ok, required_matrix_blocks = q4_0_try_mul_i64(row_count, row_stride_blocks)
    if not ok:
        return Q4_0_AVX2_ERR_OVERFLOW, []
    if required_matrix_blocks > matrix_block_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    out_rows_q32: list[int] = []
    for row_index in range(row_count):
        ok, row_base = q4_0_try_mul_i64(row_index, row_stride_blocks)
        if not ok:
            return Q4_0_AVX2_ERR_OVERFLOW, []

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
            return err, []

        out_rows_q32.append(out_holder["value"])

    return Q4_0_AVX2_OK, out_rows_q32


def make_block(scale: float, vals: list[int]):
    return (half_bits(scale), pack_q4_signed(vals))


def test_known_rows_match_per_row_block_dot() -> None:
    vec = [
        make_block(1.0, [i % 16 - 8 for i in range(32)]),
        make_block(0.5, [7 - (i % 16) for i in range(32)]),
    ]

    row0 = [
        make_block(1.0, [3 - (i % 7) for i in range(32)]),
        make_block(-0.5, [(-1 if i % 2 else 1) * (i % 8) for i in range(32)]),
    ]
    row1 = [
        make_block(-1.0, [i % 9 - 4 for i in range(32)]),
        make_block(0.25, [5 - (i % 11) for i in range(32)]),
    ]

    matrix = row0 + row1

    err, rows = q4_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=len(matrix),
        row_count=2,
        row_stride_blocks=2,
        vec_blocks=vec,
        vec_block_capacity=len(vec),
        vec_block_count=2,
    )
    assert err == Q4_0_AVX2_OK

    out0 = {"value": 0}
    err0 = dot_product_blocks_q32_avx2_checked(row0, 2, vec, 2, 2, out0)
    assert err0 == Q4_0_AVX2_OK
    out1 = {"value": 0}
    err1 = dot_product_blocks_q32_avx2_checked(row1, 2, vec, 2, 2, out1)
    assert err1 == Q4_0_AVX2_OK

    assert rows == [out0["value"], out1["value"]]


def test_row_stride_with_padding_blocks() -> None:
    vec = [
        make_block(1.0, [1] * 32),
        make_block(1.0, [2] * 32),
    ]

    row0_live = [
        make_block(1.0, [3] * 32),
        make_block(1.0, [4] * 32),
    ]
    row1_live = [
        make_block(1.0, [5] * 32),
        make_block(1.0, [6] * 32),
    ]

    pad = make_block(1.0, [7] * 32)
    matrix = row0_live + [pad] + row1_live + [pad]

    err, rows = q4_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=len(matrix),
        row_count=2,
        row_stride_blocks=3,
        vec_blocks=vec,
        vec_block_capacity=len(vec),
        vec_block_count=2,
    )
    assert err == Q4_0_AVX2_OK

    exp0 = {"value": 0}
    exp1 = {"value": 0}
    assert dot_product_blocks_q32_avx2_checked(row0_live, 2, vec, 2, 2, exp0) == Q4_0_AVX2_OK
    assert dot_product_blocks_q32_avx2_checked(row1_live, 2, vec, 2, 2, exp1) == Q4_0_AVX2_OK
    assert rows == [exp0["value"], exp1["value"]]


def test_bad_len_guards() -> None:
    vec = [make_block(1.0, [1] * 32)]
    matrix = [make_block(1.0, [1] * 32)]

    err, _ = q4_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=1,
        row_count=1,
        row_stride_blocks=0,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=1,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN

    err, _ = q4_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=1,
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_capacity=0,
        vec_block_count=1,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN

    err, _ = q4_0_dot_rows_avx2_q32_checked(
        matrix_blocks=matrix,
        matrix_block_capacity=0,
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=1,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN


def test_capacity_multiply_overflow_is_rejected() -> None:
    err, _ = q4_0_dot_rows_avx2_q32_checked(
        matrix_blocks=[],
        matrix_block_capacity=I64_MAX,
        row_count=I64_MAX,
        row_stride_blocks=2,
        vec_blocks=[],
        vec_block_capacity=0,
        vec_block_count=0,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW


def test_null_ptr_rejected() -> None:
    err, _ = q4_0_dot_rows_avx2_q32_checked(
        matrix_blocks=None,
        matrix_block_capacity=0,
        row_count=0,
        row_stride_blocks=0,
        vec_blocks=[],
        vec_block_capacity=0,
        vec_block_count=0,
    )
    assert err == Q4_0_AVX2_ERR_NULL_PTR


if __name__ == "__main__":
    test_known_rows_match_per_row_block_dot()
    test_row_stride_with_padding_blocks()
    test_bad_len_guards()
    test_capacity_multiply_overflow_is_rejected()
    test_null_ptr_rejected()
    print("q4_0_avx2_dot_rows_q32_checked_reference_checks=ok")
