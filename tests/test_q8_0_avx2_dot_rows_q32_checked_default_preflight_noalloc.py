#!/usr/bin/env python3
"""Parity checks for Q8_0DotRowsAVX2Q32CheckedDefaultPreflightNoAlloc."""

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
from test_q8_0_avx2_dot_rows_q32_checked_default_no_partial_preflight import (
    inline_preflight_reference,
)

I64_MIN = -(1 << 63)


def _try_mul_i64(lhs: int, rhs: int):
    if lhs == 0 or rhs == 0:
        return True, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return False, 0
    return True, product


def q8_0_dot_rows_avx2_q32_checked_default_preflight_noalloc(
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


def q8_0_dot_rows_avx2_q32_checked_default(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_rows_q32,
):
    stage = {"value": -1}
    err = q8_0_dot_rows_avx2_q32_checked_default_preflight_noalloc(
        matrix_blocks,
        row_count,
        row_stride_blocks,
        vec_blocks,
        vec_block_count,
        out_rows_q32,
        stage,
    )
    if err != Q8_0_AVX2_OK:
        return err

    for i in range(row_count):
        out_rows_q32[i] = (i + 1) * 17
    return Q8_0_AVX2_OK


def q8_0_dot_rows_avx2_q32_checked_default_inline_reference(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_rows_q32,
):
    stage = {"value": -1}
    err = inline_preflight_reference(
        matrix_blocks,
        row_count,
        row_stride_blocks,
        vec_blocks,
        vec_block_count,
        out_rows_q32,
        stage,
    )
    if err != Q8_0_AVX2_OK:
        return err

    for i in range(row_count):
        out_rows_q32[i] = (i + 1) * 17
    return Q8_0_AVX2_OK


def test_source_contains_new_preflight_and_default_uses_it() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")

    assert "I32 Q8_0DotRowsAVX2Q32CheckedDefaultPreflightNoAlloc(" in source
    body = source.split(
        "I32 Q8_0DotRowsAVX2Q32CheckedDefaultPreflightNoAlloc(",
        1,
    )[1].split("I32 Q8_0DotRowsAVX2Q32CheckedDefaultNoPartial(", 1)[0]
    assert "if (row_count > 0 && row_stride_blocks < vec_block_count)" in body
    assert "if (!Q8_0AVX2TryMulI64(row_count, row_stride_blocks, &required_matrix_blocks))" in body
    assert "if (!Q8_0AVX2TryMulI64(row_count, sizeof(I64), &stage_bytes))" in body

    default_body = source.split(
        "I32 Q8_0DotRowsAVX2Q32CheckedDefault(",
        1,
    )[1].split("I32 Q8_0DotRowsAVX2Q32CheckedDefaultPreflightNoAlloc(", 1)[0]
    assert "Q8_0DotRowsAVX2Q32CheckedDefaultPreflightNoAlloc(" in default_body


def test_error_surface_matches_inline_reference() -> None:
    out_stage = {"value": 123456}

    err = q8_0_dot_rows_avx2_q32_checked_default_preflight_noalloc(
        matrix_blocks=None,
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=1,
        out_rows_q32=[0],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out_stage["value"] == 123456

    err = q8_0_dot_rows_avx2_q32_checked_default_preflight_noalloc(
        matrix_blocks=[],
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[],
        vec_block_count=1,
        out_rows_q32=[0],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err = q8_0_dot_rows_avx2_q32_checked_default_preflight_noalloc(
        matrix_blocks=[],
        row_count=(I64_MAX // 8) + 1,
        row_stride_blocks=8,
        vec_blocks=[],
        vec_block_count=0,
        out_rows_q32=[0],
        out_stage_bytes_holder=out_stage,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW


def test_randomized_preflight_parity_vs_inline_reference() -> None:
    rng = random.Random(20260419_525)

    for _ in range(3000):
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

        out_new = {"value": 10101}
        out_ref = {"value": 20202}

        err_new = q8_0_dot_rows_avx2_q32_checked_default_preflight_noalloc(
            matrix_blocks=[],
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=[],
            vec_block_count=vec_block_count,
            out_rows_q32=[],
            out_stage_bytes_holder=out_new,
        )
        err_ref = inline_preflight_reference(
            matrix_blocks=[],
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=[],
            vec_block_count=vec_block_count,
            out_rows_q32=[],
            out_stage_bytes_holder=out_ref,
        )

        assert err_new == err_ref
        if err_new == Q8_0_AVX2_OK:
            assert out_new["value"] == out_ref["value"]
        else:
            assert out_new["value"] == 10101
            assert out_ref["value"] == 20202


def test_randomized_default_wrapper_parity_vs_inline_reference() -> None:
    rng = random.Random(20260419_5259)

    for _ in range(2500):
        row_count = rng.randint(0, 256)
        vec_block_count = rng.randint(0, 256)
        row_stride_blocks = rng.randint(0, 300)

        out_new = [0x1111] * max(row_count, 1)
        out_ref = [0x2222] * max(row_count, 1)

        err_new = q8_0_dot_rows_avx2_q32_checked_default(
            matrix_blocks=[0],
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=[0],
            vec_block_count=vec_block_count,
            out_rows_q32=out_new,
        )
        err_ref = q8_0_dot_rows_avx2_q32_checked_default_inline_reference(
            matrix_blocks=[0],
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=[0],
            vec_block_count=vec_block_count,
            out_rows_q32=out_ref,
        )

        assert err_new == err_ref
        if err_new == Q8_0_AVX2_OK:
            assert out_new[:row_count] == out_ref[:row_count]
        else:
            assert all(v == 0x1111 for v in out_new)
            assert all(v == 0x2222 for v in out_ref)


def run() -> None:
    test_source_contains_new_preflight_and_default_uses_it()
    test_error_surface_matches_inline_reference()
    test_randomized_preflight_parity_vs_inline_reference()
    test_randomized_default_wrapper_parity_vs_inline_reference()
    print("q8_0_avx2_dot_rows_q32_checked_default_preflight_noalloc=ok")


if __name__ == "__main__":
    run()
