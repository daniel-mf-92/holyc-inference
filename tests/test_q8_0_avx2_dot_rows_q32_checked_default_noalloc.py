#!/usr/bin/env python3
"""Parity harness for Q8_0DotRowsAVX2Q32CheckedDefaultNoAlloc."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_avx2_blocks_q32 import (
    I64_MAX,
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_ERR_OVERFLOW,
    Q8_0_AVX2_OK,
)
from test_q8_0_avx2_dot_rows_q32_checked_default import (
    q8_0_dot_rows_avx2_q32_checked_default,
)
from test_q8_0_avx2_dot_rows_q32_checked_default_preflight_noalloc import (
    q8_0_dot_rows_avx2_q32_checked_default_preflight_noalloc,
)
from test_q8_0_avx2_rows_q32 import make_block

I64_MIN = -(1 << 63)


def _try_mul_i64(lhs: int, rhs: int):
    if lhs == 0 or rhs == 0:
        return True, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return False, 0
    return True, product


def q8_0_dot_rows_avx2_q32_checked_default_noalloc(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_rows_q32,
):
    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    stage = {"value": 123456}
    err = q8_0_dot_rows_avx2_q32_checked_default_preflight_noalloc(
        matrix_blocks=matrix_blocks,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_count=vec_block_count,
        out_rows_q32=out_rows_q32,
        out_stage_bytes_holder=stage,
    )
    if err != Q8_0_AVX2_OK:
        return err

    ok, _matrix_block_capacity = _try_mul_i64(row_count, row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    err, rows = q8_0_dot_rows_avx2_q32_checked_default(
        matrix_blocks=matrix_blocks,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_count=vec_block_count,
    )
    if err != Q8_0_AVX2_OK:
        return err

    for i in range(row_count):
        out_rows_q32[i] = rows[i]

    return Q8_0_AVX2_OK


def explicit_composition_reference(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_rows_q32,
):
    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    stage = {"value": 123456}
    err = q8_0_dot_rows_avx2_q32_checked_default_preflight_noalloc(
        matrix_blocks=matrix_blocks,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_count=vec_block_count,
        out_rows_q32=out_rows_q32,
        out_stage_bytes_holder=stage,
    )
    if err != Q8_0_AVX2_OK:
        return err

    ok, _matrix_block_capacity = _try_mul_i64(row_count, row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    err, rows = q8_0_dot_rows_avx2_q32_checked_default(
        matrix_blocks=matrix_blocks,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_count=vec_block_count,
    )
    if err != Q8_0_AVX2_OK:
        return err

    for i in range(row_count):
        out_rows_q32[i] = rows[i]

    return Q8_0_AVX2_OK


def test_source_contains_default_noalloc_wrapper_shape() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0DotRowsAVX2Q32CheckedDefaultNoAlloc(" in source

    body = source.split(
        "I32 Q8_0DotRowsAVX2Q32CheckedDefaultNoAlloc(",
        1,
    )[1].split("I32 Q8_0DotRowsAVX2Q32CheckedDefaultPreflightNoAlloc(", 1)[0]

    assert "Q8_0DotRowsAVX2Q32CheckedDefaultPreflightNoAllocWithMatrixCapacity(" in body
    assert "&matrix_block_capacity," in body
    assert "return Q8_0DotRowsAVX2Q32Checked(matrix_blocks," in body


def test_overflow_and_bad_len_surface_matches_explicit_composition() -> None:
    out_a = [101, 102, 103, 104]
    out_b = [201, 202, 203, 204]

    err_a = q8_0_dot_rows_avx2_q32_checked_default_noalloc(
        matrix_blocks=[],
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=0,
        out_rows_q32=out_a,
    )
    err_b = explicit_composition_reference(
        matrix_blocks=[],
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=0,
        out_rows_q32=out_b,
    )
    assert err_a == err_b == Q8_0_AVX2_ERR_BAD_LEN

    err_a = q8_0_dot_rows_avx2_q32_checked_default_noalloc(
        matrix_blocks=[],
        row_count=(I64_MAX // 8) + 1,
        row_stride_blocks=8,
        vec_blocks=[],
        vec_block_count=0,
        out_rows_q32=out_a,
    )
    err_b = explicit_composition_reference(
        matrix_blocks=[],
        row_count=(I64_MAX // 8) + 1,
        row_stride_blocks=8,
        vec_blocks=[],
        vec_block_count=0,
        out_rows_q32=out_b,
    )
    assert err_a == err_b == Q8_0_AVX2_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260419_540)

    for _ in range(1500):
        mode = rng.randint(0, 9)
        if mode == 0:
            row_count = -rng.randint(1, 8)
            row_stride_blocks = rng.randint(0, 8)
            vec_block_count = rng.randint(0, 8)
            matrix = []
            vec = []
        elif mode == 1:
            row_count = (I64_MAX // 8) + rng.randint(1, 64)
            row_stride_blocks = 8
            vec_block_count = 0
            matrix = []
            vec = []
        else:
            row_count = rng.randint(0, 6)
            vec_block_count = rng.randint(0, 6)
            row_stride_blocks = vec_block_count + rng.randint(0, 4)
            matrix = [
                make_block(
                    rng.choice([0x0000, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4400, 0x7C00, 0xBC00]),
                    [rng.randint(-128, 127) for _ in range(32)],
                )
                for _ in range(max(0, row_count * row_stride_blocks))
            ]
            vec = [
                make_block(
                    rng.choice([0x0000, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4400, 0x7C00, 0xBC00]),
                    [rng.randint(-128, 127) for _ in range(32)],
                )
                for _ in range(max(0, vec_block_count))
            ]

        if row_count < 0 or row_count > 1024:
            out_len = 8
        else:
            out_len = max(4, row_count + 2)
        out_a = [77771] * out_len
        out_b = [88882] * out_len

        err_a = q8_0_dot_rows_avx2_q32_checked_default_noalloc(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_rows_q32=out_a,
        )
        err_b = explicit_composition_reference(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_rows_q32=out_b,
        )

        assert err_a == err_b
        if err_a == Q8_0_AVX2_OK and row_count > 0:
            assert out_a[:row_count] == out_b[:row_count]


def run() -> None:
    test_source_contains_default_noalloc_wrapper_shape()
    test_overflow_and_bad_len_surface_matches_explicit_composition()
    test_randomized_parity_vs_explicit_composition()
    print("q8_0_avx2_dot_rows_q32_checked_default_noalloc=ok")


if __name__ == "__main__":
    run()
