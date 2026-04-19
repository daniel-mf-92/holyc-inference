#!/usr/bin/env python3
"""Parity harness for ...DefaultNoPartialPreflightNoAlloc."""

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
)
from test_q8_0_matmul_tiled_avx2_q32 import Q8_0_AVX2_OK

I64_MIN = -(1 << 63)


def try_mul_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    value = lhs * rhs
    if value < I64_MIN or value > I64_MAX:
        return False, 0
    return True, value


def q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight_noalloc(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_mat_q32,
    out_row_stride_cols: int,
    out_staged_out_capacity: list[int] | None,
    out_staged_out_bytes: list[int] | None,
) -> int:
    if out_mat_q32 is None or out_staged_out_capacity is None or out_staged_out_bytes is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, lhs_block_capacity = try_mul_i64(lhs_rows, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    ok, rhs_block_capacity = try_mul_i64(rhs_cols, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    ok, staged_out_capacity = try_mul_i64(lhs_rows, out_row_stride_cols)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    _ = lhs_matrix_blocks
    _ = rhs_col_blocks
    _ = lhs_block_capacity
    _ = rhs_block_capacity

    if lhs_rows == 0 or rhs_cols == 0:
        out_staged_out_capacity[0] = staged_out_capacity
        out_staged_out_bytes[0] = 0
        return Q8_0_AVX2_OK

    if staged_out_capacity <= 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, staged_out_bytes = try_mul_i64(staged_out_capacity, 8)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW
    if staged_out_bytes <= 0:
        return Q8_0_AVX2_ERR_OVERFLOW

    out_staged_out_capacity[0] = staged_out_capacity
    out_staged_out_bytes[0] = staged_out_bytes
    return Q8_0_AVX2_OK


def inline_noalloc_preflight_reference(
    lhs_matrix_blocks,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_mat_q32,
    out_row_stride_cols: int,
    out_staged_out_capacity: list[int] | None,
    out_staged_out_bytes: list[int] | None,
) -> int:
    if out_mat_q32 is None or out_staged_out_capacity is None or out_staged_out_bytes is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    lhs_block_capacity = lhs_rows * lhs_row_stride_blocks
    if lhs_block_capacity < I64_MIN or lhs_block_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW

    rhs_block_capacity = rhs_cols * rhs_col_stride_blocks
    if rhs_block_capacity < I64_MIN or rhs_block_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW

    staged_out_capacity = lhs_rows * out_row_stride_cols
    if staged_out_capacity < I64_MIN or staged_out_capacity > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW

    _ = lhs_matrix_blocks
    _ = rhs_col_blocks
    _ = lhs_block_capacity
    _ = rhs_block_capacity

    if lhs_rows == 0 or rhs_cols == 0:
        out_staged_out_capacity[0] = staged_out_capacity
        out_staged_out_bytes[0] = 0
        return Q8_0_AVX2_OK

    if staged_out_capacity <= 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    staged_out_bytes = staged_out_capacity * 8
    if staged_out_bytes < I64_MIN or staged_out_bytes > I64_MAX:
        return Q8_0_AVX2_ERR_OVERFLOW
    if staged_out_bytes <= 0:
        return Q8_0_AVX2_ERR_OVERFLOW

    out_staged_out_capacity[0] = staged_out_capacity
    out_staged_out_bytes[0] = staged_out_bytes
    return Q8_0_AVX2_OK


def test_source_contains_noalloc_no_partial_preflight_shape() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialPreflightNoAlloc(" in source

    body = source.split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialPreflightNoAlloc(",
        1,
    )[1].split("I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialNoAllocDefaultStage(", 1)[0]

    assert "if (!out_mat_q32 || !out_staged_out_capacity || !out_staged_out_bytes)" in body
    assert "if (lhs_rows < 0 || lhs_row_stride_blocks < 0)" in body
    assert "if (rhs_cols < 0 || rhs_col_stride_blocks < 0)" in body
    assert "if (k_block_count < 0 || out_row_stride_cols < 0)" in body
    assert "if (!Q8_0AVX2TryMulI64(lhs_rows, lhs_row_stride_blocks, &lhs_block_capacity))" in body
    assert "if (!Q8_0AVX2TryMulI64(rhs_cols, rhs_col_stride_blocks, &rhs_block_capacity))" in body
    assert "if (!Q8_0AVX2TryMulI64(lhs_rows, out_row_stride_cols, &staged_out_capacity))" in body
    assert "if (!lhs_rows || !rhs_cols)" in body
    assert "if (!Q8_0AVX2TryMulI64(staged_out_capacity, sizeof(I64), &staged_out_bytes))" in body


def test_preflight_allows_null_payload_matrix_pointers() -> None:
    cap = [31337]
    bts = [31338]

    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight_noalloc(
        lhs_matrix_blocks=None,
        lhs_rows=3,
        lhs_row_stride_blocks=4,
        rhs_col_blocks=None,
        rhs_cols=2,
        rhs_col_stride_blocks=5,
        k_block_count=3,
        out_mat_q32=[0] * 12,
        out_row_stride_cols=6,
        out_staged_out_capacity=cap,
        out_staged_out_bytes=bts,
    )
    assert err == Q8_0_AVX2_OK
    assert cap[0] == 18
    assert bts[0] == 144


def test_preflight_rejects_missing_output_sinks() -> None:
    cap = [1]
    bts = [2]

    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight_noalloc(
            lhs_matrix_blocks=[],
            lhs_rows=1,
            lhs_row_stride_blocks=1,
            rhs_col_blocks=[],
            rhs_cols=1,
            rhs_col_stride_blocks=1,
            k_block_count=1,
            out_mat_q32=None,
            out_row_stride_cols=1,
            out_staged_out_capacity=cap,
            out_staged_out_bytes=bts,
        )
        == Q8_0_AVX2_ERR_NULL_PTR
    )
    assert cap[0] == 1
    assert bts[0] == 2


def test_preflight_bad_len_overflow_and_zero_span_surfaces() -> None:
    cap = [777]
    bts = [888]

    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight_noalloc(
        lhs_matrix_blocks=[],
        lhs_rows=-1,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=[],
        rhs_cols=1,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        out_mat_q32=[0],
        out_row_stride_cols=1,
        out_staged_out_capacity=cap,
        out_staged_out_bytes=bts,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert cap[0] == 777
    assert bts[0] == 888

    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight_noalloc(
        lhs_matrix_blocks=[],
        lhs_rows=(I64_MAX // 8) + 1,
        lhs_row_stride_blocks=8,
        rhs_col_blocks=[],
        rhs_cols=1,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        out_mat_q32=[0],
        out_row_stride_cols=1,
        out_staged_out_capacity=cap,
        out_staged_out_bytes=bts,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert cap[0] == 777
    assert bts[0] == 888

    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight_noalloc(
        lhs_matrix_blocks=[],
        lhs_rows=0,
        lhs_row_stride_blocks=123,
        rhs_col_blocks=[],
        rhs_cols=7,
        rhs_col_stride_blocks=9,
        k_block_count=4,
        out_mat_q32=[],
        out_row_stride_cols=456,
        out_staged_out_capacity=cap,
        out_staged_out_bytes=bts,
    )
    assert err == Q8_0_AVX2_OK
    assert cap[0] == 0
    assert bts[0] == 0


def test_randomized_parity_vs_inline_reference() -> None:
    rng = random.Random(20260419_534)

    for _ in range(2200):
        lhs_rows = rng.randint(-2, 100_000)
        lhs_stride = rng.randint(-2, 100_000)
        rhs_cols = rng.randint(-2, 100_000)
        rhs_stride = rng.randint(-2, 100_000)
        k_blocks = rng.randint(-2, 100_000)
        out_stride = rng.randint(-2, 100_000)

        out = None if rng.randint(0, 19) == 0 else [0] * 8

        cap_impl = [111]
        bts_impl = [222]
        cap_ref = [333]
        bts_ref = [444]

        got = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight_noalloc(
            lhs_matrix_blocks=None if rng.randint(0, 1) == 0 else [],
            lhs_rows=lhs_rows,
            lhs_row_stride_blocks=lhs_stride,
            rhs_col_blocks=None if rng.randint(0, 1) == 0 else [],
            rhs_cols=rhs_cols,
            rhs_col_stride_blocks=rhs_stride,
            k_block_count=k_blocks,
            out_mat_q32=out,
            out_row_stride_cols=out_stride,
            out_staged_out_capacity=cap_impl,
            out_staged_out_bytes=bts_impl,
        )
        want = inline_noalloc_preflight_reference(
            lhs_matrix_blocks=None,
            lhs_rows=lhs_rows,
            lhs_row_stride_blocks=lhs_stride,
            rhs_col_blocks=None,
            rhs_cols=rhs_cols,
            rhs_col_stride_blocks=rhs_stride,
            k_block_count=k_blocks,
            out_mat_q32=out,
            out_row_stride_cols=out_stride,
            out_staged_out_capacity=cap_ref,
            out_staged_out_bytes=bts_ref,
        )

        assert got == want
        if got == Q8_0_AVX2_OK:
            assert cap_impl[0] == cap_ref[0]
            assert bts_impl[0] == bts_ref[0]
        else:
            assert cap_impl[0] == 111
            assert bts_impl[0] == 222
            assert cap_ref[0] == 333
            assert bts_ref[0] == 444


def run() -> None:
    test_source_contains_noalloc_no_partial_preflight_shape()
    test_preflight_allows_null_payload_matrix_pointers()
    test_preflight_rejects_missing_output_sinks()
    test_preflight_bad_len_overflow_and_zero_span_surfaces()
    test_randomized_parity_vs_inline_reference()
    print("q8_0_avx2_matmul_tiled_q32_checked_default_no_partial_preflight_noalloc=ok")


if __name__ == "__main__":
    run()
