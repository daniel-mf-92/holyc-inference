#!/usr/bin/env python3
"""Parity harness for Q8_0DotRowsAVX2Q32CheckedDefaultNoPartialNoAlloc."""

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
from test_q8_0_avx2_dot_rows_q32_checked_default import (
    q8_0_dot_rows_avx2_q32_checked_default,
)
from test_q8_0_avx2_dot_rows_q32_checked_default_no_partial_preflight_noalloc import (
    q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc,
)

I64_MIN = -(1 << 63)


def _try_mul_i64(lhs: int, rhs: int):
    if lhs == 0 or rhs == 0:
        return True, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return False, 0
    return True, product


def q8_0_dot_rows_avx2_q32_checked_default_no_partial_noalloc(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_rows_q32,
    staged_rows_q32,
    staged_row_capacity: int,
):
    stage_bytes = {"value": 0}
    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc(
        matrix_blocks,
        row_count,
        row_stride_blocks,
        vec_blocks,
        vec_block_count,
        out_rows_q32,
        stage_bytes,
    )
    if err != Q8_0_AVX2_OK:
        return err

    if row_count == 0:
        err, rows = q8_0_dot_rows_avx2_q32_checked_default(
            matrix_blocks,
            row_count,
            row_stride_blocks,
            vec_blocks,
            vec_block_count,
        )
        if err != Q8_0_AVX2_OK:
            return err
        out_rows_q32[:] = rows
        return Q8_0_AVX2_OK

    if matrix_blocks is None or vec_blocks is None or staged_rows_q32 is None:
        return Q8_0_AVX2_ERR_NULL_PTR
    if staged_row_capacity < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if staged_row_capacity < row_count:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, provided_bytes = _try_mul_i64(staged_row_capacity, 8)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW
    if provided_bytes < stage_bytes["value"]:
        return Q8_0_AVX2_ERR_BAD_LEN

    if staged_rows_q32 is out_rows_q32:
        return Q8_0_AVX2_ERR_BAD_LEN

    err, staged = q8_0_dot_rows_avx2_q32_checked_default(
        matrix_blocks,
        row_count,
        row_stride_blocks,
        vec_blocks,
        vec_block_count,
    )
    if err != Q8_0_AVX2_OK:
        return err

    for i in range(row_count):
        staged_rows_q32[i] = staged[i]
        out_rows_q32[i] = staged_rows_q32[i]

    return Q8_0_AVX2_OK


def explicit_staged_composition_reference(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_rows_q32,
    staged_rows_q32,
    staged_row_capacity: int,
):
    if out_rows_q32 is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if row_count > 0 and row_stride_blocks < vec_block_count:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, _ = _try_mul_i64(row_count, row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    if row_count == 0:
        err, rows = q8_0_dot_rows_avx2_q32_checked_default(
            matrix_blocks,
            row_count,
            row_stride_blocks,
            vec_blocks,
            vec_block_count,
        )
        if err != Q8_0_AVX2_OK:
            return err
        out_rows_q32[:] = rows
        return Q8_0_AVX2_OK

    if matrix_blocks is None or vec_blocks is None or staged_rows_q32 is None:
        return Q8_0_AVX2_ERR_NULL_PTR
    if staged_row_capacity < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if staged_row_capacity < row_count:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, required_stage_bytes = _try_mul_i64(row_count, 8)
    if not ok or required_stage_bytes <= 0:
        return Q8_0_AVX2_ERR_OVERFLOW

    ok, provided_stage_bytes = _try_mul_i64(staged_row_capacity, 8)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW
    if provided_stage_bytes < required_stage_bytes:
        return Q8_0_AVX2_ERR_BAD_LEN

    if staged_rows_q32 is out_rows_q32:
        return Q8_0_AVX2_ERR_BAD_LEN

    err, rows = q8_0_dot_rows_avx2_q32_checked_default(
        matrix_blocks,
        row_count,
        row_stride_blocks,
        vec_blocks,
        vec_block_count,
    )
    if err != Q8_0_AVX2_OK:
        return err

    for i in range(row_count):
        staged_rows_q32[i] = rows[i]
        out_rows_q32[i] = staged_rows_q32[i]

    return Q8_0_AVX2_OK


def test_source_contains_noalloc_no_partial_wrapper_shape() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0DotRowsAVX2Q32CheckedDefaultNoPartialNoAlloc(" in source

    body = source.split(
        "I32 Q8_0DotRowsAVX2Q32CheckedDefaultNoPartialNoAlloc(",
        1,
    )[1].split("I32 Q8_0MatMulTiledAVX2Q32Checked(", 1)[0]

    assert "Q8_0DotRowsAVX2Q32CheckedDefaultNoPartialPreflightNoAlloc(" in body
    assert "if (!matrix_blocks || !vec_blocks || !staged_rows_q32)" in body
    assert "if (staged_row_capacity < 0)" in body
    assert "if (staged_row_capacity < required_stage_capacity)" in body
    assert "if (!Q8_0AVX2TryMulI64(staged_row_capacity, sizeof(I64), &provided_stage_bytes))" in body
    assert "if (provided_stage_bytes < required_stage_bytes)" in body
    assert "if (staged_rows_q32 == out_rows_q32)" in body


def test_error_surface_and_no_partial_preservation() -> None:
    rng = random.Random(20260419_545_1)
    matrix = [
        make_block(0x3C00, [rng.randint(-128, 127) for _ in range(32)]),
        make_block(0x3C00, [rng.randint(-128, 127) for _ in range(32)]),
    ]
    vec = [make_block(0x3C00, [rng.randint(-128, 127) for _ in range(32)])]

    out = [901, 902, 903]
    staged = [801, 802, 803]

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_noalloc(
        matrix,
        2,
        1,
        vec,
        1,
        None,
        staged,
        3,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    baseline = list(out)
    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_noalloc(
        matrix,
        2,
        1,
        vec,
        1,
        out,
        None,
        3,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out == baseline

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_noalloc(
        matrix,
        2,
        1,
        vec,
        1,
        out,
        staged,
        -1,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_noalloc(
        matrix,
        2,
        1,
        vec,
        1,
        out,
        staged,
        1,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_noalloc(
        matrix,
        2,
        1,
        vec,
        1,
        out,
        staged,
        (I64_MAX // 8) + 1,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW

    alias = [1, 2, 3]
    alias_baseline = list(alias)
    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_noalloc(
        matrix,
        2,
        1,
        vec,
        1,
        alias,
        alias,
        3,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert alias == alias_baseline


def test_randomized_parity_vs_explicit_staged_composition() -> None:
    rng = random.Random(20260419_545_2)

    for _ in range(700):
        row_count = rng.randint(0, 8)
        vec_block_count = rng.randint(0, 6)

        if row_count == 0:
            row_stride_blocks = rng.randint(0, 4)
        else:
            row_stride_blocks = max(0, vec_block_count + rng.randint(-3, 4))

        matrix_capacity = row_count * row_stride_blocks
        matrix = [
            make_block(
                rng.choice([0x0000, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4400, 0x7C00, 0xBC00]),
                [rng.randint(-128, 127) for _ in range(32)],
            )
            for _ in range(matrix_capacity)
        ]
        vec = [
            make_block(
                rng.choice([0x0000, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4400, 0x7C00, 0xBC00]),
                [rng.randint(-128, 127) for _ in range(32)],
            )
            for _ in range(vec_block_count)
        ]

        if rng.randint(0, 9) == 0:
            staged_capacity = -rng.randint(1, 8)
        elif rng.randint(0, 9) == 1:
            staged_capacity = (I64_MAX // 8) + rng.randint(1, 64)
        else:
            staged_capacity = rng.randint(0, 20)

        out_len = max(4, row_count + 2)
        out_a = [11111] * out_len
        out_b = [55555] * out_len

        staged_is_null = rng.randint(0, 24) == 0
        if staged_is_null:
            staged_a = None
            staged_b = None
        else:
            staged_len = max(12, row_count + 4)
            staged_a = [99991] * staged_len
            staged_b = [12345] * staged_len

        if (not staged_is_null) and rng.randint(0, 30) == 0:
            staged_a = out_a
            staged_b = out_b

        err_a = q8_0_dot_rows_avx2_q32_checked_default_no_partial_noalloc(
            matrix,
            row_count,
            row_stride_blocks,
            vec,
            vec_block_count,
            out_a,
            staged_a,
            staged_capacity,
        )
        err_b = explicit_staged_composition_reference(
            matrix,
            row_count,
            row_stride_blocks,
            vec,
            vec_block_count,
            out_b,
            staged_b,
            staged_capacity,
        )

        assert err_a == err_b
        if err_a == Q8_0_AVX2_OK:
            assert out_a[:row_count] == out_b[:row_count]
        else:
            assert out_a == [11111] * out_len
            assert out_b == [55555] * out_len


def run() -> None:
    test_source_contains_noalloc_no_partial_wrapper_shape()
    test_error_surface_and_no_partial_preservation()
    test_randomized_parity_vs_explicit_staged_composition()
    print("q8_0_avx2_dot_rows_q32_checked_default_no_partial_noalloc=ok")


if __name__ == "__main__":
    run()
