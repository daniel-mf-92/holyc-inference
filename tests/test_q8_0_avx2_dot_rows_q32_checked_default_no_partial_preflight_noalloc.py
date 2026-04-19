#!/usr/bin/env python3
"""Parity checks for Q8_0DotRowsAVX2Q32CheckedDefaultNoPartialPreflightNoAlloc."""

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


def inline_noalloc_no_partial_preflight_reference(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_rows_q32,
    out_stage_bytes_holder,
):
    # IQ-529 contract: payload pointers are intentionally optional for preflight.
    if out_rows_q32 is None or out_stage_bytes_holder is None:
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


def q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_rows_q32,
    out_stage_bytes_holder,
):
    return inline_noalloc_no_partial_preflight_reference(
        matrix_blocks,
        row_count,
        row_stride_blocks,
        vec_blocks,
        vec_block_count,
        out_rows_q32,
        out_stage_bytes_holder,
    )


def test_source_contains_noalloc_no_partial_preflight_shape() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0DotRowsAVX2Q32CheckedDefaultNoPartialPreflightNoAlloc(" in source

    body = source.split(
        "I32 Q8_0DotRowsAVX2Q32CheckedDefaultNoPartialPreflightNoAlloc(",
        1,
    )[1].split("I32 Q8_0DotRowsAVX2Q32CheckedDefaultNoPartialPreflight(", 1)[0]

    assert "if (!out_rows_q32 || !out_stage_bytes)" in body
    assert "if (row_count < 0 || row_stride_blocks < 0 || vec_block_count < 0)" in body
    assert "if (row_count > 0 && row_stride_blocks < vec_block_count)" in body
    assert "if (!Q8_0AVX2TryMulI64(row_count, row_stride_blocks, &required_matrix_blocks))" in body
    assert "if (!Q8_0AVX2TryMulI64(row_count, sizeof(I64), &stage_bytes))" in body

    # Payload pointers must remain optional in this no-allocation metadata preflight.
    assert "if (!matrix_blocks || !vec_blocks || !out_rows_q32 || !out_stage_bytes)" not in body


def test_preflight_allows_null_payload_pointers() -> None:
    out_stage = {"value": 31337}

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc(
        matrix_blocks=None,
        row_count=3,
        row_stride_blocks=5,
        vec_blocks=None,
        vec_block_count=4,
        out_rows_q32=[0, 0, 0],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_OK
    assert out_stage["value"] == 24


def test_preflight_rejects_null_output_metadata_sinks() -> None:
    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc(
        matrix_blocks=[],
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=1,
        out_rows_q32=None,
        out_stage_bytes_holder={"value": 1},
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc(
        matrix_blocks=[],
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=1,
        out_rows_q32=[0],
        out_stage_bytes_holder=None,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR


def test_error_surface_bad_len_and_overflow() -> None:
    out_stage = {"value": 123456}

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc(
        matrix_blocks=[],
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=1,
        out_rows_q32=[0],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out_stage["value"] == 123456

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc(
        matrix_blocks=[],
        row_count=3,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=2,
        out_rows_q32=[0, 0, 0],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out_stage["value"] == 123456

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc(
        matrix_blocks=[],
        row_count=(I64_MAX // 8) + 1,
        row_stride_blocks=8,
        vec_blocks=[],
        vec_block_count=0,
        out_rows_q32=[0],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert out_stage["value"] == 123456


def test_zero_rows_sets_stage_bytes_zero() -> None:
    out_stage = {"value": 777}
    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc(
        matrix_blocks=None,
        row_count=0,
        row_stride_blocks=123,
        vec_blocks=None,
        vec_block_count=456,
        out_rows_q32=[],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_OK
    assert out_stage["value"] == 0


def test_randomized_parity_vs_inline_reference() -> None:
    rng = random.Random(20260419_529)

    for _ in range(1600):
        chooser = rng.randint(0, 10)
        if chooser == 0:
            row_count = -rng.randint(1, 128)
        elif chooser == 1:
            row_count = (I64_MAX // 8) + rng.randint(1, 4096)
        else:
            row_count = rng.randint(0, 2_000_000)

        if rng.randint(0, 9) == 0:
            row_stride_blocks = -rng.randint(1, 256)
        else:
            row_stride_blocks = rng.randint(0, 2_000_000)

        if rng.randint(0, 11) == 0:
            vec_block_count = -rng.randint(1, 64)
        else:
            vec_block_count = rng.randint(0, 2_000_000)

        matrix_blocks = None if rng.randint(0, 1) == 0 else []
        vec_blocks = None if rng.randint(0, 1) == 0 else []

        out_impl = {"value": 111111}
        out_ref = {"value": 222222}

        err_impl = q8_0_dot_rows_avx2_q32_checked_default_no_partial_preflight_noalloc(
            matrix_blocks=matrix_blocks,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec_blocks,
            vec_block_count=vec_block_count,
            out_rows_q32=[],
            out_stage_bytes_holder=out_impl,
        )
        err_ref = inline_noalloc_no_partial_preflight_reference(
            matrix_blocks=matrix_blocks,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec_blocks,
            vec_block_count=vec_block_count,
            out_rows_q32=[],
            out_stage_bytes_holder=out_ref,
        )

        assert err_impl == err_ref
        if err_impl == Q8_0_AVX2_OK:
            assert out_impl["value"] == out_ref["value"]
        else:
            assert out_impl["value"] == 111111
            assert out_ref["value"] == 222222


def run() -> None:
    test_source_contains_noalloc_no_partial_preflight_shape()
    test_preflight_allows_null_payload_pointers()
    test_preflight_rejects_null_output_metadata_sinks()
    test_error_surface_bad_len_and_overflow()
    test_zero_rows_sets_stage_bytes_zero()
    test_randomized_parity_vs_inline_reference()
    print("q8_0_avx2_dot_rows_q32_checked_default_no_partial_preflight_noalloc=ok")


if __name__ == "__main__":
    run()
