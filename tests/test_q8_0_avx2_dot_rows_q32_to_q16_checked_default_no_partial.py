#!/usr/bin/env python3
"""Parity checks for Q8_0DotRowsAVX2Q32ToQ16CheckedDefaultNoPartial."""

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
from test_q8_0_avx2_dot_rows_q32_to_q16_checked_default import (
    q8_0_dot_rows_avx2_q32_to_q16_checked_default_ptr,
)


def q8_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_holder,
):
    if out_holder is None:
        return Q8_0_AVX2_ERR_NULL_PTR
    if row_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    if row_count == 0:
        return q8_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
            matrix_blocks=matrix_blocks,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec_blocks,
            vec_block_count=vec_block_count,
            out_holder=out_holder,
        )

    stage_rows = [0] * row_count
    stage_holder = {"rows": stage_rows}

    err = q8_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
        matrix_blocks=matrix_blocks,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_count=vec_block_count,
        out_holder=stage_holder,
    )
    if err != Q8_0_AVX2_OK:
        return err

    out_holder["rows"] = stage_holder["rows"]
    return Q8_0_AVX2_OK


def test_source_contains_default_no_partial_wrapper_shape() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "Q8_0DotRowsAVX2Q32ToQ16CheckedDefaultNoPartial" in source
    assert "status = Q8_0DotRowsAVX2Q32ToQ16CheckedDefault(matrix_blocks," in source
    assert "staged_rows_q16 = MAlloc(stage_bytes);" in source


def test_null_bad_len_and_zero_row_surfaces() -> None:
    block = make_block(0x3C00, [0] * 32)

    out = {"rows": [7, 8]}
    err = q8_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
        matrix_blocks=[block],
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[block],
        vec_block_count=1,
        out_holder=out,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out["rows"] == [7, 8]

    err = q8_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
        matrix_blocks=[block],
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=[block],
        vec_block_count=1,
        out_holder=None,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    out = {"rows": [11, 22, 33]}
    err = q8_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
        matrix_blocks=[block],
        row_count=0,
        row_stride_blocks=0,
        vec_blocks=[block],
        vec_block_count=0,
        out_holder=out,
    )
    assert err == Q8_0_AVX2_OK
    assert out["rows"] == [11, 22, 33]


def test_overflow_and_bad_layout_preserve_output() -> None:
    out = {"rows": [101, 202]}
    err = q8_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
        matrix_blocks=[],
        row_count=I64_MAX,
        row_stride_blocks=2,
        vec_blocks=[],
        vec_block_count=0,
        out_holder=out,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert out["rows"] == [101, 202]

    block = make_block(0x3C00, [1] * 32)
    out = {"rows": [301, 302]}
    err = q8_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
        matrix_blocks=[block],
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=[block],
        vec_block_count=1,
        out_holder=out,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out["rows"] == [301, 302]


def test_randomized_parity_vs_default_wrapper() -> None:
    rng = random.Random(20260419_501)

    for _ in range(300):
        row_count = rng.randint(0, 8)
        vec_block_count = rng.randint(0, 6)
        row_stride_blocks = vec_block_count + rng.randint(0, 4)

        matrix = []
        for _ in range(row_count * row_stride_blocks):
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

        out_nop = {"rows": [999, 777]}
        out_def = {"rows": [555, 333]}

        err_nop = q8_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_nop,
        )
        err_def = q8_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_def,
        )

        assert err_nop == err_def
        if err_def == Q8_0_AVX2_OK:
            assert out_nop["rows"] == out_def["rows"]
        else:
            assert out_nop["rows"] == [999, 777]


def run() -> None:
    test_source_contains_default_no_partial_wrapper_shape()
    test_null_bad_len_and_zero_row_surfaces()
    test_overflow_and_bad_layout_preserve_output()
    test_randomized_parity_vs_default_wrapper()
    print("q8_0_avx2_dot_rows_q32_to_q16_checked_default_no_partial=ok")


if __name__ == "__main__":
    run()
