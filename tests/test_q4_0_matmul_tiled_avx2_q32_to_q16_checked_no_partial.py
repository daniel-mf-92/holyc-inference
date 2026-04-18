#!/usr/bin/env python3
"""Parity checks for Q4_0MatMulTiledAVX2Q32ToQ16CheckedNoPartial semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_avx2_dot_q32_checked import (
    Q4_0_AVX2_ERR_BAD_LEN,
    Q4_0_AVX2_ERR_NULL_PTR,
    Q4_0_AVX2_OK,
    half_bits,
    pack_q4_signed,
)
from test_q4_0_avx2_dot_q32_to_q16_checked_default import (
    q4_0_dot_blocks_avx2_q32_to_q16_checked_ptr,
)
from test_q4_0_avx2_dot_rows_q32_checked import (
    I64_MAX,
    Q4_0_AVX2_ERR_OVERFLOW,
)
from test_q4_0_matmul_tiled_avx2_q32_checked import (
    build_matrix_cols_as_blocks,
    build_matrix_rows_as_blocks,
)
from test_q4_0_matmul_tiled_avx2_q32_to_q16_checked_default_tiles import (
    q4_0_try_add_i64_nonneg,
    q4_0_try_mul_i64_nonneg,
)


def q4_0_matmul_tiled_avx2_q32_to_q16_checked_ptr(
    lhs_matrix_blocks,
    lhs_block_capacity: int,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_block_capacity: int,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    tile_rows: int,
    tile_cols: int,
    out_capacity: int,
    out_row_stride_cols: int,
    out_holder,
):
    if lhs_matrix_blocks is None or rhs_col_blocks is None or out_holder is None:
        return Q4_0_AVX2_ERR_NULL_PTR

    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_capacity < 0:
        return Q4_0_AVX2_ERR_BAD_LEN
    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q4_0_AVX2_ERR_BAD_LEN
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q4_0_AVX2_ERR_BAD_LEN
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q4_0_AVX2_ERR_BAD_LEN
    if tile_rows <= 0 or tile_cols <= 0:
        return Q4_0_AVX2_ERR_BAD_LEN

    if lhs_rows > 0 and lhs_row_stride_blocks < k_block_count:
        return Q4_0_AVX2_ERR_BAD_LEN
    if rhs_cols > 0 and rhs_col_stride_blocks < k_block_count:
        return Q4_0_AVX2_ERR_BAD_LEN
    if lhs_rows > 0 and out_row_stride_cols < rhs_cols:
        return Q4_0_AVX2_ERR_BAD_LEN

    ok, required_lhs_blocks = q4_0_try_mul_i64_nonneg(lhs_rows, lhs_row_stride_blocks)
    if not ok:
        return Q4_0_AVX2_ERR_OVERFLOW
    if required_lhs_blocks > lhs_block_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN

    ok, required_rhs_blocks = q4_0_try_mul_i64_nonneg(rhs_cols, rhs_col_stride_blocks)
    if not ok:
        return Q4_0_AVX2_ERR_OVERFLOW
    if required_rhs_blocks > rhs_block_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN

    ok, required_out_cells = q4_0_try_mul_i64_nonneg(lhs_rows, out_row_stride_cols)
    if not ok:
        return Q4_0_AVX2_ERR_OVERFLOW
    if required_out_cells > out_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN

    if len(out_holder["mat"]) < out_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN

    row_tile_start = 0
    while row_tile_start < lhs_rows:
        ok, row_tile_end = q4_0_try_add_i64_nonneg(row_tile_start, tile_rows)
        if not ok:
            return Q4_0_AVX2_ERR_OVERFLOW
        row_tile_end = min(row_tile_end, lhs_rows)

        col_tile_start = 0
        while col_tile_start < rhs_cols:
            ok, col_tile_end = q4_0_try_add_i64_nonneg(col_tile_start, tile_cols)
            if not ok:
                return Q4_0_AVX2_ERR_OVERFLOW
            col_tile_end = min(col_tile_end, rhs_cols)

            for row_index in range(row_tile_start, row_tile_end):
                ok, lhs_row_base = q4_0_try_mul_i64_nonneg(row_index, lhs_row_stride_blocks)
                if not ok:
                    return Q4_0_AVX2_ERR_OVERFLOW
                ok, out_row_base = q4_0_try_mul_i64_nonneg(row_index, out_row_stride_cols)
                if not ok:
                    return Q4_0_AVX2_ERR_OVERFLOW

                for col_index in range(col_tile_start, col_tile_end):
                    ok, rhs_col_base = q4_0_try_mul_i64_nonneg(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q4_0_AVX2_ERR_OVERFLOW
                    ok, out_index = q4_0_try_add_i64_nonneg(out_row_base, col_index)
                    if not ok:
                        return Q4_0_AVX2_ERR_OVERFLOW

                    dot_holder = {"value": 0}
                    status = q4_0_dot_blocks_avx2_q32_to_q16_checked_ptr(
                        lhs_blocks=lhs_matrix_blocks[lhs_row_base : lhs_row_base + k_block_count],
                        rhs_blocks=rhs_col_blocks[rhs_col_base : rhs_col_base + k_block_count],
                        block_count=k_block_count,
                        out_holder=dot_holder,
                    )
                    if status != Q4_0_AVX2_OK:
                        return status

                    out_holder["mat"][out_index] = dot_holder["value"]

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    return Q4_0_AVX2_OK


def q4_0_matmul_tiled_avx2_q32_to_q16_checked_no_partial_ptr(
    lhs_matrix_blocks,
    lhs_block_capacity: int,
    lhs_rows: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_block_capacity: int,
    rhs_cols: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    tile_rows: int,
    tile_cols: int,
    out_capacity: int,
    out_row_stride_cols: int,
    out_holder,
):
    if lhs_matrix_blocks is None or rhs_col_blocks is None or out_holder is None:
        return Q4_0_AVX2_ERR_NULL_PTR
    if lhs_rows < 0 or out_row_stride_cols < 0:
        return Q4_0_AVX2_ERR_BAD_LEN

    if lhs_rows == 0:
        return q4_0_matmul_tiled_avx2_q32_to_q16_checked_ptr(
            lhs_matrix_blocks=lhs_matrix_blocks,
            lhs_block_capacity=lhs_block_capacity,
            lhs_rows=lhs_rows,
            lhs_row_stride_blocks=lhs_row_stride_blocks,
            rhs_col_blocks=rhs_col_blocks,
            rhs_block_capacity=rhs_block_capacity,
            rhs_cols=rhs_cols,
            rhs_col_stride_blocks=rhs_col_stride_blocks,
            k_block_count=k_block_count,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            out_capacity=out_capacity,
            out_row_stride_cols=out_row_stride_cols,
            out_holder=out_holder,
        )

    ok, required_out_cells = q4_0_try_mul_i64_nonneg(lhs_rows, out_row_stride_cols)
    if not ok:
        return Q4_0_AVX2_ERR_OVERFLOW
    if required_out_cells > out_capacity:
        return Q4_0_AVX2_ERR_BAD_LEN

    if required_out_cells == 0:
        return q4_0_matmul_tiled_avx2_q32_to_q16_checked_ptr(
            lhs_matrix_blocks=lhs_matrix_blocks,
            lhs_block_capacity=lhs_block_capacity,
            lhs_rows=lhs_rows,
            lhs_row_stride_blocks=lhs_row_stride_blocks,
            rhs_col_blocks=rhs_col_blocks,
            rhs_block_capacity=rhs_block_capacity,
            rhs_cols=rhs_cols,
            rhs_col_stride_blocks=rhs_col_stride_blocks,
            k_block_count=k_block_count,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            out_capacity=out_capacity,
            out_row_stride_cols=out_row_stride_cols,
            out_holder=out_holder,
        )

    staged = {"mat": [0] * required_out_cells}
    for out_index in range(required_out_cells):
        staged["mat"][out_index] = out_holder["mat"][out_index]

    status = q4_0_matmul_tiled_avx2_q32_to_q16_checked_ptr(
        lhs_matrix_blocks=lhs_matrix_blocks,
        lhs_block_capacity=lhs_block_capacity,
        lhs_rows=lhs_rows,
        lhs_row_stride_blocks=lhs_row_stride_blocks,
        rhs_col_blocks=rhs_col_blocks,
        rhs_block_capacity=rhs_block_capacity,
        rhs_cols=rhs_cols,
        rhs_col_stride_blocks=rhs_col_stride_blocks,
        k_block_count=k_block_count,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        out_capacity=required_out_cells,
        out_row_stride_cols=out_row_stride_cols,
        out_holder=staged,
    )
    if status != Q4_0_AVX2_OK:
        return status

    if len(out_holder["mat"]) < required_out_cells:
        return Q4_0_AVX2_ERR_BAD_LEN

    for out_index in range(required_out_cells):
        out_holder["mat"][out_index] = staged["mat"][out_index]

    return Q4_0_AVX2_OK


def test_no_partial_wrapper_matches_checked_core_randomized() -> None:
    rng = random.Random(2026041818)

    for _ in range(220):
        lhs_rows = rng.randint(0, 6)
        rhs_cols = rng.randint(0, 6)
        k_block_count = rng.randint(0, 5)
        lhs_row_stride_blocks = k_block_count + rng.randint(0, 3)
        rhs_col_stride_blocks = k_block_count + rng.randint(0, 3)
        out_row_stride_cols = rhs_cols + rng.randint(0, 3)
        tile_rows = rng.randint(1, 3)
        tile_cols = rng.randint(1, 3)

        lhs = build_matrix_rows_as_blocks(lhs_rows, lhs_row_stride_blocks, k_block_count, rng)
        rhs = build_matrix_cols_as_blocks(rhs_cols, rhs_col_stride_blocks, k_block_count, rng)

        out_capacity = lhs_rows * out_row_stride_cols
        out_no_partial = {"mat": [111] * (out_capacity + 3)}
        out_core = {"mat": [222] * (out_capacity + 3)}

        err_no_partial = q4_0_matmul_tiled_avx2_q32_to_q16_checked_no_partial_ptr(
            lhs_matrix_blocks=lhs,
            lhs_block_capacity=len(lhs),
            lhs_rows=lhs_rows,
            lhs_row_stride_blocks=lhs_row_stride_blocks,
            rhs_col_blocks=rhs,
            rhs_block_capacity=len(rhs),
            rhs_cols=rhs_cols,
            rhs_col_stride_blocks=rhs_col_stride_blocks,
            k_block_count=k_block_count,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            out_capacity=out_capacity,
            out_row_stride_cols=out_row_stride_cols,
            out_holder=out_no_partial,
        )
        err_core = q4_0_matmul_tiled_avx2_q32_to_q16_checked_ptr(
            lhs_matrix_blocks=lhs,
            lhs_block_capacity=len(lhs),
            lhs_rows=lhs_rows,
            lhs_row_stride_blocks=lhs_row_stride_blocks,
            rhs_col_blocks=rhs,
            rhs_block_capacity=len(rhs),
            rhs_cols=rhs_cols,
            rhs_col_stride_blocks=rhs_col_stride_blocks,
            k_block_count=k_block_count,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            out_capacity=out_capacity,
            out_row_stride_cols=out_row_stride_cols,
            out_holder=out_core,
        )

        assert err_no_partial == err_core
        if err_no_partial == Q4_0_AVX2_OK:
            for row_index in range(lhs_rows):
                row_base = row_index * out_row_stride_cols
                assert (
                    out_no_partial["mat"][row_base : row_base + rhs_cols]
                    == out_core["mat"][row_base : row_base + rhs_cols]
                )

                assert (
                    out_no_partial["mat"][row_base + rhs_cols : row_base + out_row_stride_cols]
                    == [111] * (out_row_stride_cols - rhs_cols)
                )
                assert (
                    out_core["mat"][row_base + rhs_cols : row_base + out_row_stride_cols]
                    == [222] * (out_row_stride_cols - rhs_cols)
                )

            assert out_no_partial["mat"][out_capacity:] == [111] * 3
            assert out_core["mat"][out_capacity:] == [222] * 3
        else:
            assert out_no_partial["mat"] == [111] * (out_capacity + 3)


def test_no_partial_wrapper_preserves_output_on_late_truncation_failure() -> None:
    lhs = [(half_bits(1.0), pack_q4_signed([1] * 32))]

    rhs_good = (half_bits(1.0), pack_q4_signed([1] * 32))
    rhs_bad = (half_bits(1.0), bytes([0x88] * 15))
    rhs = [rhs_good, rhs_bad]

    out_no_partial = {"mat": [7001, 7002]}
    out_core_inplace = {"mat": [8001, 8002]}

    err_no_partial = q4_0_matmul_tiled_avx2_q32_to_q16_checked_no_partial_ptr(
        lhs_matrix_blocks=lhs,
        lhs_block_capacity=1,
        lhs_rows=1,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=rhs,
        rhs_block_capacity=2,
        rhs_cols=2,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        tile_rows=1,
        tile_cols=1,
        out_capacity=2,
        out_row_stride_cols=2,
        out_holder=out_no_partial,
    )
    err_core = q4_0_matmul_tiled_avx2_q32_to_q16_checked_ptr(
        lhs_matrix_blocks=lhs,
        lhs_block_capacity=1,
        lhs_rows=1,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=rhs,
        rhs_block_capacity=2,
        rhs_cols=2,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        tile_rows=1,
        tile_cols=1,
        out_capacity=2,
        out_row_stride_cols=2,
        out_holder=out_core_inplace,
    )

    assert err_no_partial == err_core == Q4_0_AVX2_ERR_BAD_LEN
    assert out_no_partial["mat"] == [7001, 7002]
    assert out_core_inplace["mat"][0] != 8001
    assert out_core_inplace["mat"][1] == 8002


def test_no_partial_wrapper_null_bad_len_overflow_paths() -> None:
    out_holder = {"mat": [9, 9]}

    err = q4_0_matmul_tiled_avx2_q32_to_q16_checked_no_partial_ptr(
        lhs_matrix_blocks=None,
        lhs_block_capacity=0,
        lhs_rows=0,
        lhs_row_stride_blocks=0,
        rhs_col_blocks=[],
        rhs_block_capacity=0,
        rhs_cols=0,
        rhs_col_stride_blocks=0,
        k_block_count=0,
        tile_rows=1,
        tile_cols=1,
        out_capacity=0,
        out_row_stride_cols=0,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_NULL_PTR
    assert out_holder["mat"] == [9, 9]

    err = q4_0_matmul_tiled_avx2_q32_to_q16_checked_no_partial_ptr(
        lhs_matrix_blocks=[],
        lhs_block_capacity=0,
        lhs_rows=-1,
        lhs_row_stride_blocks=0,
        rhs_col_blocks=[],
        rhs_block_capacity=0,
        rhs_cols=0,
        rhs_col_stride_blocks=0,
        k_block_count=0,
        tile_rows=1,
        tile_cols=1,
        out_capacity=0,
        out_row_stride_cols=0,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN
    assert out_holder["mat"] == [9, 9]

    err = q4_0_matmul_tiled_avx2_q32_to_q16_checked_no_partial_ptr(
        lhs_matrix_blocks=[],
        lhs_block_capacity=0,
        lhs_rows=I64_MAX,
        lhs_row_stride_blocks=0,
        rhs_col_blocks=[],
        rhs_block_capacity=0,
        rhs_cols=0,
        rhs_col_stride_blocks=0,
        k_block_count=0,
        tile_rows=1,
        tile_cols=1,
        out_capacity=0,
        out_row_stride_cols=2,
        out_holder=out_holder,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert out_holder["mat"] == [9, 9]


def run() -> None:
    test_no_partial_wrapper_matches_checked_core_randomized()
    test_no_partial_wrapper_preserves_output_on_late_truncation_failure()
    test_no_partial_wrapper_null_bad_len_overflow_paths()
    print("q4_0_matmul_tiled_avx2_q32_to_q16_checked_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
