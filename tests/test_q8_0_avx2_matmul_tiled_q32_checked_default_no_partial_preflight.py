#!/usr/bin/env python3
"""Parity harness for Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialPreflight."""

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
from test_q8_0_avx2_matmul_tiled_q32_checked_default_preflight import (
    q8_0_matmul_tiled_avx2_q32_checked_default_preflight,
)
from test_q8_0_matmul_tiled_avx2_q32 import Q8_0_AVX2_OK, build_matrix_cols_as_blocks, build_matrix_rows_as_blocks


I64_MIN = -(1 << 63)


def try_mul_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    value = lhs * rhs
    if value < I64_MIN or value > I64_MAX:
        return False, 0
    return True, value


def q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight(
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
    if (
        lhs_matrix_blocks is None
        or rhs_col_blocks is None
        or out_mat_q32 is None
        or out_staged_out_capacity is None
        or out_staged_out_bytes is None
    ):
        return Q8_0_AVX2_ERR_NULL_PTR

    lhs_capacity = [0]
    rhs_capacity = [0]
    staged_capacity = [0]
    err = q8_0_matmul_tiled_avx2_q32_checked_default_preflight(
        lhs_matrix_blocks,
        lhs_rows,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_cols,
        rhs_col_stride_blocks,
        k_block_count,
        out_mat_q32,
        out_row_stride_cols,
        lhs_capacity,
        rhs_capacity,
        staged_capacity,
    )
    if err != Q8_0_AVX2_OK:
        return err

    if lhs_rows == 0 or rhs_cols == 0:
        out_staged_out_capacity[0] = staged_capacity[0]
        out_staged_out_bytes[0] = 0
        return Q8_0_AVX2_OK

    if staged_capacity[0] <= 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, staged_bytes = try_mul_i64(staged_capacity[0], 8)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW
    if staged_bytes <= 0:
        return Q8_0_AVX2_ERR_OVERFLOW

    out_staged_out_capacity[0] = staged_capacity[0]
    out_staged_out_bytes[0] = staged_bytes
    return Q8_0_AVX2_OK


def explicit_inline_preflight_reference(
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
    if (
        lhs_matrix_blocks is None
        or rhs_col_blocks is None
        or out_mat_q32 is None
        or out_staged_out_capacity is None
        or out_staged_out_bytes is None
    ):
        return Q8_0_AVX2_ERR_NULL_PTR

    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, _ = try_mul_i64(lhs_rows, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW
    ok, _ = try_mul_i64(rhs_cols, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW
    ok, staged_capacity = try_mul_i64(lhs_rows, out_row_stride_cols)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW

    if lhs_rows == 0 or rhs_cols == 0:
        out_staged_out_capacity[0] = staged_capacity
        out_staged_out_bytes[0] = 0
        return Q8_0_AVX2_OK

    if staged_capacity <= 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    ok, staged_bytes = try_mul_i64(staged_capacity, 8)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW
    if staged_bytes <= 0:
        return Q8_0_AVX2_ERR_OVERFLOW

    out_staged_out_capacity[0] = staged_capacity
    out_staged_out_bytes[0] = staged_bytes
    return Q8_0_AVX2_OK


def test_source_contains_no_partial_preflight_shape() -> None:
    source = pathlib.Path("src/quant/q8_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialPreflight(" in source

    body = source.split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialPreflight(",
        1,
    )[1].split(
        "I32 Q8_0DotBlocksAVX2Q32ToQ16Checked(",
        1,
    )[0]
    assert "!out_staged_out_capacity || !out_staged_out_bytes" in body
    assert "status = Q8_0MatMulTiledAVX2Q32CheckedDefaultPreflight(" in body
    assert "if (!lhs_rows || !rhs_cols)" in body
    assert "if (staged_out_capacity <= 0)" in body
    assert "if (!Q8_0AVX2TryMulI64(staged_out_capacity, sizeof(I64), &staged_out_bytes))" in body

    no_partial_body = source.split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartial(",
        1,
    )[1].split(
        "I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialPreflight(",
        1,
    )[0]
    assert "status = Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartialPreflight(" in no_partial_body


def test_preflight_null_and_bad_len_surfaces() -> None:
    rng = random.Random(20260419_522_1)
    lhs = build_matrix_rows_as_blocks(2, 2, 2, rng)
    rhs = build_matrix_cols_as_blocks(2, 2, 2, rng)
    out = [0] * 8

    cap = [111]
    bts = [222]
    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight(
            None,
            2,
            2,
            rhs,
            2,
            2,
            2,
            out,
            4,
            cap,
            bts,
        )
        == Q8_0_AVX2_ERR_NULL_PTR
    )
    assert cap[0] == 111 and bts[0] == 222

    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight(
            lhs,
            -1,
            2,
            rhs,
            2,
            2,
            2,
            out,
            4,
            cap,
            bts,
        )
        == Q8_0_AVX2_ERR_BAD_LEN
    )

    assert (
        q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight(
            lhs,
            2,
            2,
            rhs,
            2,
            2,
            2,
            out,
            -1,
            cap,
            bts,
        )
        == Q8_0_AVX2_ERR_BAD_LEN
    )


def test_preflight_zero_span_and_overflow_fixtures() -> None:
    rng = random.Random(20260419_522_2)
    lhs = build_matrix_rows_as_blocks(1, 1, 1, rng)
    rhs = build_matrix_cols_as_blocks(1, 1, 1, rng)

    cap = [999]
    bts = [999]
    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight(
        lhs,
        0,
        7,
        rhs,
        3,
        5,
        1,
        [0],
        9,
        cap,
        bts,
    )
    assert err == Q8_0_AVX2_OK
    assert cap[0] == 0
    assert bts[0] == 0

    cap = [123]
    bts = [456]
    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight(
        lhs,
        I64_MAX,
        2,
        rhs,
        1,
        1,
        1,
        [0],
        1,
        cap,
        bts,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert cap[0] == 123
    assert bts[0] == 456

    cap = [123]
    bts = [456]
    err = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight(
        lhs,
        (I64_MAX // 8) + 1,
        1,
        rhs,
        1,
        1,
        1,
        [0],
        1,
        cap,
        bts,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert cap[0] == 123
    assert bts[0] == 456


def test_randomized_parity_vs_inline_reference() -> None:
    rng = random.Random(20260419_522_3)

    for _ in range(2500):
        chooser = rng.randint(0, 12)
        if chooser == 0:
            lhs_rows = -rng.randint(1, 128)
        else:
            lhs_rows = rng.randint(0, 6000)

        chooser = rng.randint(0, 12)
        if chooser == 1:
            lhs_row_stride_blocks = -rng.randint(1, 128)
        else:
            lhs_row_stride_blocks = rng.randint(0, 128)

        chooser = rng.randint(0, 12)
        if chooser == 2:
            rhs_cols = -rng.randint(1, 128)
        else:
            rhs_cols = rng.randint(0, 6000)

        chooser = rng.randint(0, 12)
        if chooser == 3:
            rhs_col_stride_blocks = -rng.randint(1, 128)
        else:
            rhs_col_stride_blocks = rng.randint(0, 128)

        chooser = rng.randint(0, 12)
        if chooser == 4:
            k_block_count = -rng.randint(1, 64)
        else:
            k_block_count = rng.randint(0, 128)

        chooser = rng.randint(0, 12)
        if chooser == 5:
            out_row_stride_cols = -rng.randint(1, 128)
        else:
            out_row_stride_cols = rng.randint(0, 128)

        null_mode = rng.randint(0, 24)
        lhs = None if null_mode == 0 else [0]
        rhs = None if null_mode == 1 else [0]
        out = None if null_mode == 2 else [0, 0, 0, 0]
        out_cap = None if null_mode == 3 else [rng.randint(0, 1000)]
        out_bytes = None if null_mode == 4 else [rng.randint(0, 1000)]

        got_cap = None if out_cap is None else list(out_cap)
        got_bytes = None if out_bytes is None else list(out_bytes)
        want_cap = None if out_cap is None else list(out_cap)
        want_bytes = None if out_bytes is None else list(out_bytes)

        got = q8_0_matmul_tiled_avx2_q32_checked_default_no_partial_preflight(
            lhs,
            lhs_rows,
            lhs_row_stride_blocks,
            rhs,
            rhs_cols,
            rhs_col_stride_blocks,
            k_block_count,
            out,
            out_row_stride_cols,
            got_cap,
            got_bytes,
        )
        want = explicit_inline_preflight_reference(
            lhs,
            lhs_rows,
            lhs_row_stride_blocks,
            rhs,
            rhs_cols,
            rhs_col_stride_blocks,
            k_block_count,
            out,
            out_row_stride_cols,
            want_cap,
            want_bytes,
        )

        assert got == want
        assert got_cap == want_cap
        assert got_bytes == want_bytes


def run() -> None:
    test_source_contains_no_partial_preflight_shape()
    test_preflight_null_and_bad_len_surfaces()
    test_preflight_zero_span_and_overflow_fixtures()
    test_randomized_parity_vs_inline_reference()
    print("q8_0_avx2_matmul_tiled_q32_checked_default_no_partial_preflight=ok")


if __name__ == "__main__":
    run()
