#!/usr/bin/env python3
"""Parity checks for Q8_0DotRowsAVX2Q32CheckedDefaultNoPartialPreflight."""

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
)

I64_MIN = -(1 << 63)


def _try_mul_i64(lhs: int, rhs: int):
    if lhs == 0 or rhs == 0:
        return True, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return False, 0
    return True, product


def q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_rows_q32,
    out_stage_bytes_holder,
):
    if (
        matrix_blocks is None
        or vec_blocks is None
        or out_rows_q32 is None
        or out_stage_bytes_holder is None
    ):
        return Q8_0_AVX2_ERR_NULL_PTR

    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    if row_count > 0 and row_stride_blocks < vec_block_count:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, _ = _try_mul_i64(row_count, row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    if row_count == 0:
        out_stage_bytes_holder["value"] = 0
        return Q8_0_AVX2_OK

    ok, stage_bytes = _try_mul_i64(row_count, 8)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW
    if stage_bytes <= 0:
        return Q8_0_AVX2_ERR_OVERFLOW

    out_stage_bytes_holder["value"] = stage_bytes
    return Q8_0_AVX2_OK


def inline_preflight_reference(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_rows_q32,
    out_stage_bytes_holder,
):
    if (
        matrix_blocks is None
        or vec_blocks is None
        or out_rows_q32 is None
        or out_stage_bytes_holder is None
    ):
        return Q8_0_AVX2_ERR_NULL_PTR

    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    if row_count > 0 and row_stride_blocks < vec_block_count:
        return Q8_0_AVX2_ERR_BAD_LEN

    required_matrix_blocks = row_count * row_stride_blocks
    if required_matrix_blocks < I64_MIN or required_matrix_blocks > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW

    if row_count == 0:
        out_stage_bytes_holder["value"] = 0
        return Q8_0_AVX2_OK

    stage_bytes = row_count * 8
    if stage_bytes < I64_MIN or stage_bytes > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW
    if stage_bytes <= 0:
        return Q8_0_AVX2_ERR_OVERFLOW

    out_stage_bytes_holder["value"] = stage_bytes
    return Q8_0_AVX2_OK


def test_source_contains_preflight_shape() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0DotRowsAVX2Q32CheckedDefaultNoPartialPreflight(" in source
    assert "if (!matrix_blocks || !vec_blocks || !out_rows_q32 || !out_stage_bytes)" in source
    assert "if (row_count > 0 && row_stride_blocks < vec_block_count)" in source
    assert "if (!Q8_0AVX2TryMulI64(row_count, row_stride_blocks, &required_matrix_blocks))" in source
    assert "if (!Q8_0AVX2TryMulI64(row_count, sizeof(I64), &stage_bytes))" in source


def test_preflight_null_and_bad_len_contract() -> None:
    out_rows = [1, 2, 3]
    out_stage = {"value": 999}

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight(
        matrix_blocks=None,
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=1,
        out_rows_q32=out_rows,
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out_stage["value"] == 999

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight(
        matrix_blocks=[],
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=1,
        out_rows_q32=None,
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight(
        matrix_blocks=[],
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=1,
        out_rows_q32=out_rows,
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out_stage["value"] == 999

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight(
        matrix_blocks=[],
        row_count=2,
        row_stride_blocks=-1,
        vec_blocks=[],
        vec_block_count=1,
        out_rows_q32=out_rows,
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN



def test_preflight_rejects_stride_shorter_than_vector_blocks() -> None:
    out_stage = {"value": 555}
    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight(
        matrix_blocks=[0],
        row_count=3,
        row_stride_blocks=1,
        vec_blocks=[0, 0],
        vec_block_count=2,
        out_rows_q32=[0, 0, 0],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out_stage["value"] == 555


def test_preflight_zero_rows_sets_zero_stage_bytes() -> None:
    out_stage = {"value": 777}
    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight(
        matrix_blocks=[],
        row_count=0,
        row_stride_blocks=123,
        vec_blocks=[],
        vec_block_count=456,
        out_rows_q32=[],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_OK
    assert out_stage["value"] == 0


def test_preflight_overflow_surface() -> None:
    out_stage = {"value": 31337}

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight(
        matrix_blocks=[0],
        row_count=(I64_MAX // 8) + 1,
        row_stride_blocks=8,
        vec_blocks=[],
        vec_block_count=0,
        out_rows_q32=[0],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert out_stage["value"] == 31337

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight(
        matrix_blocks=[0],
        row_count=I64_MAX,
        row_stride_blocks=2,
        vec_blocks=[0],
        vec_block_count=1,
        out_rows_q32=[0],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert out_stage["value"] == 31337


def test_randomized_parity_vs_inline_preflight() -> None:
    rng = random.Random(20260419_513)

    for _ in range(900):
        chooser = rng.randint(0, 10)
        if chooser == 0:
            row_count = -rng.randint(1, 128)
        elif chooser == 1:
            row_count = (I64_MAX // 8) + rng.randint(1, 4096)
        else:
            row_count = rng.randint(0, 1_500_000)

        if rng.randint(0, 9) == 0:
            row_stride_blocks = -rng.randint(1, 256)
        else:
            row_stride_blocks = rng.randint(0, 1_500_000)

        if rng.randint(0, 11) == 0:
            vec_block_count = -rng.randint(1, 64)
        else:
            vec_block_count = rng.randint(0, 1_500_000)

        out_a = {"value": 111111}
        out_b = {"value": 222222}

        err_a = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight(
            matrix_blocks=[],
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=[],
            vec_block_count=vec_block_count,
            out_rows_q32=[],
            out_stage_bytes_holder=out_a,
        )
        err_b = inline_preflight_reference(
            matrix_blocks=[],
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=[],
            vec_block_count=vec_block_count,
            out_rows_q32=[],
            out_stage_bytes_holder=out_b,
        )

        assert err_a == err_b
        if err_a == Q8_0_AVX2_OK:
            assert out_a["value"] == out_b["value"]
        else:
            assert out_a["value"] == 111111
            assert out_b["value"] == 222222


def run() -> None:
    test_source_contains_preflight_shape()
    test_preflight_null_and_bad_len_contract()
    test_preflight_rejects_stride_shorter_than_vector_blocks()
    test_preflight_zero_rows_sets_zero_stage_bytes()
    test_preflight_overflow_surface()
    test_randomized_parity_vs_inline_preflight()
    print("q8_0_avx2_dot_rows_q32_checked_default_no_partial_preflight=ok")


if __name__ == "__main__":
    run()
