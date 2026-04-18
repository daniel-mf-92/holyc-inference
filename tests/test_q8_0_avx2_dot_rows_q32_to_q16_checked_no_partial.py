#!/usr/bin/env python3
"""Parity checks for Q8_0DotRowsAVX2Q32ToQ16CheckedNoPartial semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_avx2_blocks_q32 import (
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_ERR_NULL_PTR,
    Q8_0_AVX2_ERR_OVERFLOW,
    Q8_0_AVX2_OK,
    make_block,
)
from test_q8_0_avx2_rows_q32_to_q16 import q8_0_dot_rows_avx2_q32_to_q16_checked


def q8_0_dot_rows_avx2_q32_to_q16_checked_ptr(
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
        return Q8_0_AVX2_ERR_NULL_PTR

    err, rows_q16 = q8_0_dot_rows_avx2_q32_to_q16_checked(
        matrix_blocks=matrix_blocks,
        matrix_block_capacity=matrix_block_capacity,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_capacity=vec_block_capacity,
        vec_block_count=vec_block_count,
    )
    if err != Q8_0_AVX2_OK:
        return err

    out_holder["rows"] = rows_q16
    return Q8_0_AVX2_OK


def q8_0_dot_rows_avx2_q32_to_q16_checked_no_partial_ptr(
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
        return Q8_0_AVX2_ERR_NULL_PTR

    err, rows_q16 = q8_0_dot_rows_avx2_q32_to_q16_checked(
        matrix_blocks=matrix_blocks,
        matrix_block_capacity=matrix_block_capacity,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_capacity=vec_block_capacity,
        vec_block_count=vec_block_count,
    )
    if err != Q8_0_AVX2_OK:
        return err

    out_holder["rows"] = rows_q16
    return Q8_0_AVX2_OK


def test_no_partial_wrapper_matches_checked_core_success_and_errors() -> None:
    rng = random.Random(2026041810)

    for _ in range(260):
        row_count = rng.randint(0, 6)
        vec_block_count = rng.randint(0, 5)
        row_stride_blocks = vec_block_count + rng.randint(0, 4)
        matrix_capacity = row_count * row_stride_blocks

        matrix = []
        for _ in range(matrix_capacity):
            matrix.append(
                make_block(
                    rng.choice([0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4000, 0xBC00]),
                    [rng.randint(-128, 127) for _ in range(32)],
                )
            )

        vec = [
            make_block(
                rng.choice([0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4000, 0xBC00]),
                [rng.randint(-128, 127) for _ in range(32)],
            )
            for _ in range(vec_block_count)
        ]

        out_no_partial = {"rows": [111, 222, 333]}
        out_core = {"rows": [444, 555, 666]}

        err_no_partial = q8_0_dot_rows_avx2_q32_to_q16_checked_no_partial_ptr(
            matrix_blocks=matrix,
            matrix_block_capacity=matrix_capacity,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_capacity=vec_block_count,
            vec_block_count=vec_block_count,
            out_holder=out_no_partial,
        )
        err_core = q8_0_dot_rows_avx2_q32_to_q16_checked_ptr(
            matrix_blocks=matrix,
            matrix_block_capacity=matrix_capacity,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_capacity=vec_block_count,
            vec_block_count=vec_block_count,
            out_holder=out_core,
        )

        assert err_no_partial == err_core
        if err_no_partial == Q8_0_AVX2_OK:
            assert out_no_partial["rows"] == out_core["rows"]
        else:
            assert out_no_partial["rows"] == [111, 222, 333]


def test_no_partial_wrapper_rejects_null_and_bad_len_without_output_commit() -> None:
    block = make_block(0x3C00, [0] * 32)

    out_holder = {"rows": [9, 9]}
    err = q8_0_dot_rows_avx2_q32_to_q16_checked_no_partial_ptr(
        matrix_blocks=[block],
        matrix_block_capacity=1,
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[block],
        vec_block_capacity=1,
        vec_block_count=1,
        out_holder=out_holder,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out_holder["rows"] == [9, 9]

    err = q8_0_dot_rows_avx2_q32_to_q16_checked_no_partial_ptr(
        matrix_blocks=[block],
        matrix_block_capacity=1,
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=None,
        vec_block_capacity=1,
        vec_block_count=1,
        out_holder=out_holder,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out_holder["rows"] == [9, 9]


def test_no_partial_wrapper_preserves_output_when_core_would_fail_late() -> None:
    vec = [make_block(0x3C00, [127] * 32)]

    # Row 0 is benign, row 1 uses fp16 Inf scale (0x7C00) which drives
    # Q32 accumulation overflow in reference arithmetic.
    row0 = [make_block(0x3C00, [1] + [0] * 31)]
    row1 = [make_block(0x7C00, [127] * 32)]
    matrix = row0 + row1

    out_no_partial = {"rows": [7001, 7002]}
    out_core = {"rows": [8001, 8002]}

    err_no_partial = q8_0_dot_rows_avx2_q32_to_q16_checked_no_partial_ptr(
        matrix_blocks=matrix,
        matrix_block_capacity=len(matrix),
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=1,
        out_holder=out_no_partial,
    )
    err_core = q8_0_dot_rows_avx2_q32_to_q16_checked_ptr(
        matrix_blocks=matrix,
        matrix_block_capacity=len(matrix),
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_capacity=1,
        vec_block_count=1,
        out_holder=out_core,
    )

    assert err_no_partial == err_core == Q8_0_AVX2_ERR_OVERFLOW
    assert out_no_partial["rows"] == [7001, 7002]
    assert out_core["rows"] == [8001, 8002]


def test_no_partial_wrapper_zero_rows_success_no_writes() -> None:
    out_holder = {"rows": [1111, 2222, 3333]}

    err = q8_0_dot_rows_avx2_q32_to_q16_checked_no_partial_ptr(
        matrix_blocks=[],
        matrix_block_capacity=0,
        row_count=0,
        row_stride_blocks=0,
        vec_blocks=[],
        vec_block_capacity=0,
        vec_block_count=0,
        out_holder=out_holder,
    )
    assert err == Q8_0_AVX2_OK
    assert out_holder["rows"] == []


def run() -> None:
    test_no_partial_wrapper_matches_checked_core_success_and_errors()
    test_no_partial_wrapper_rejects_null_and_bad_len_without_output_commit()
    test_no_partial_wrapper_preserves_output_when_core_would_fail_late()
    test_no_partial_wrapper_zero_rows_success_no_writes()
    print("q8_0_avx2_dot_rows_q32_to_q16_checked_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
