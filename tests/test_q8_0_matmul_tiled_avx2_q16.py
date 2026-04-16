#!/usr/bin/env python3
"""Reference checks for Q8_0MatMulQ16TiledAVX2Checked semantics."""

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
from test_q8_0_avx2_blocks_q32_to_q16 import q8_0_dot_blocks_avx2_q32_to_q16_checked
from test_q8_0_matmul_tiled_avx2_q32 import q8_0_matmul_tiled_avx2_q32_checked


def try_mul_i64_nonneg(lhs: int, rhs: int):
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def try_add_i64_nonneg(lhs: int, rhs: int):
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def compute_tile_end_checked(tile_start: int, tile_span: int, axis_len: int):
    if tile_start < 0 or tile_span < 0 or axis_len < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0
    if tile_start > axis_len:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    ok, tile_end = try_add_i64_nonneg(tile_start, tile_span)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, 0

    return Q8_0_AVX2_OK, min(tile_end, axis_len)


def compute_out_index_checked(out_row_base: int, col_index: int):
    if out_row_base < 0 or col_index < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    ok, out_index = try_add_i64_nonneg(out_row_base, col_index)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, 0

    return Q8_0_AVX2_OK, out_index


def round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return (value + (1 << (shift - 1))) >> shift
    return -(((-value) + (1 << (shift - 1))) >> shift)


def q8_0_matmul_tiled_avx2_q16_checked(
    lhs_blocks,
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    tile_rows: int,
    tile_cols: int,
    out_cell_capacity: int,
    out_row_stride_cells: int,
):
    if lhs_blocks is None or rhs_col_blocks is None:
        return Q8_0_AVX2_ERR_NULL_PTR, []

    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_cell_capacity < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if tile_rows <= 0 or tile_cols <= 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if row_count > 0 and out_row_stride_cells < col_count:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    ok, lhs_required = try_mul_i64_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, []
    ok, rhs_required = try_mul_i64_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, []
    ok, out_required = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, []

    if lhs_required > lhs_block_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if rhs_required > rhs_block_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if out_required > out_cell_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    out_cells_q16 = [0] * out_required

    row_tile_start = 0
    while row_tile_start < row_count:
        err, row_tile_end = compute_tile_end_checked(row_tile_start, tile_rows, row_count)
        if err != Q8_0_AVX2_OK:
            return err, []

        col_tile_start = 0
        while col_tile_start < col_count:
            err, col_tile_end = compute_tile_end_checked(col_tile_start, tile_cols, col_count)
            if err != Q8_0_AVX2_OK:
                return err, []

            for row_index in range(row_tile_start, row_tile_end):
                ok, lhs_row_base = try_mul_i64_nonneg(row_index, lhs_row_stride_blocks)
                if not ok:
                    return Q8_0_AVX2_ERR_OVERFLOW, []
                ok, out_row_base = try_mul_i64_nonneg(row_index, out_row_stride_cells)
                if not ok:
                    return Q8_0_AVX2_ERR_OVERFLOW, []

                lhs_row_slice = lhs_blocks[lhs_row_base : lhs_row_base + k_block_count]
                if len(lhs_row_slice) != k_block_count:
                    return Q8_0_AVX2_ERR_BAD_LEN, []

                for col_index in range(col_tile_start, col_tile_end):
                    ok, rhs_col_base = try_mul_i64_nonneg(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q8_0_AVX2_ERR_OVERFLOW, []
                    err, out_index = compute_out_index_checked(out_row_base, col_index)
                    if err != Q8_0_AVX2_OK:
                        return err, []

                    rhs_col_slice = rhs_col_blocks[rhs_col_base : rhs_col_base + k_block_count]
                    if len(rhs_col_slice) != k_block_count:
                        return Q8_0_AVX2_ERR_BAD_LEN, []

                    err, cell_dot_q16 = q8_0_dot_blocks_avx2_q32_to_q16_checked(
                        lhs_row_slice,
                        rhs_col_slice,
                        k_block_count,
                    )
                    if err != Q8_0_AVX2_OK:
                        return err, []

                    out_cells_q16[out_index] = cell_dot_q16

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    return Q8_0_AVX2_OK, out_cells_q16


def q8_0_matmul_q16_reference_untiled(
    lhs_blocks,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_row_stride_cells: int,
):
    out = [0] * (row_count * out_row_stride_cells)
    for row_index in range(row_count):
        lhs_base = row_index * lhs_row_stride_blocks
        lhs_row = lhs_blocks[lhs_base : lhs_base + k_block_count]
        if len(lhs_row) != k_block_count:
            return Q8_0_AVX2_ERR_BAD_LEN, []

        for col_index in range(col_count):
            rhs_base = col_index * rhs_col_stride_blocks
            rhs_col = rhs_col_blocks[rhs_base : rhs_base + k_block_count]
            if len(rhs_col) != k_block_count:
                return Q8_0_AVX2_ERR_BAD_LEN, []

            err, dot_q16 = q8_0_dot_blocks_avx2_q32_to_q16_checked(lhs_row, rhs_col, k_block_count)
            if err != Q8_0_AVX2_OK:
                return err, []

            out[row_index * out_row_stride_cells + col_index] = dot_q16

    return Q8_0_AVX2_OK, out


def test_known_matrix_matches_q32_then_single_round() -> None:
    lhs = [
        make_block(0x3C00, [i - 16 for i in range(32)]),
        make_block(0x3800, [16 - i for i in range(32)]),
        make_block(0x3555, [(-1) ** i * (i % 9) for i in range(32)]),
        make_block(0x3A00, [7 - (i % 13) for i in range(32)]),
    ]
    rhs = [
        make_block(0x3C00, [2 * (i - 16) for i in range(32)]),
        make_block(0x3400, [i % 7 - 3 for i in range(32)]),
        make_block(0x3000, [(-1) ** (i + 1) * (i % 11) for i in range(32)]),
        make_block(0x4000, [5 - (i % 6) for i in range(32)]),
    ]

    err, got = q8_0_matmul_tiled_avx2_q16_checked(
        lhs_blocks=lhs,
        lhs_block_capacity=len(lhs),
        row_count=2,
        lhs_row_stride_blocks=2,
        rhs_col_blocks=rhs,
        rhs_block_capacity=len(rhs),
        col_count=2,
        rhs_col_stride_blocks=2,
        k_block_count=2,
        tile_rows=1,
        tile_cols=2,
        out_cell_capacity=4,
        out_row_stride_cells=2,
    )
    assert err == Q8_0_AVX2_OK

    err, ref = q8_0_matmul_q16_reference_untiled(
        lhs_blocks=lhs,
        row_count=2,
        lhs_row_stride_blocks=2,
        rhs_col_blocks=rhs,
        col_count=2,
        rhs_col_stride_blocks=2,
        k_block_count=2,
        out_row_stride_cells=2,
    )
    assert err == Q8_0_AVX2_OK
    assert got == ref


def test_q16_output_matches_q32_matmul_single_rounding_per_cell() -> None:
    rng = random.Random(2026041604)
    fp16_scales = [0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00, 0x4000, 0x4400, 0xB800, 0xBC00]

    for _ in range(180):
        row_count = rng.randint(1, 7)
        col_count = rng.randint(1, 7)
        k_block_count = rng.randint(1, 6)

        lhs_stride = k_block_count + rng.randint(0, 2)
        rhs_stride = k_block_count + rng.randint(0, 2)
        out_stride = col_count + rng.randint(0, 2)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        lhs = [make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]) for _ in range(lhs_capacity)]
        rhs = [make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]) for _ in range(rhs_capacity)]

        tile_rows = rng.randint(1, 4)
        tile_cols = rng.randint(1, 4)

        err, got_q16 = q8_0_matmul_tiled_avx2_q16_checked(
            lhs_blocks=lhs,
            lhs_block_capacity=lhs_capacity,
            row_count=row_count,
            lhs_row_stride_blocks=lhs_stride,
            rhs_col_blocks=rhs,
            rhs_block_capacity=rhs_capacity,
            col_count=col_count,
            rhs_col_stride_blocks=rhs_stride,
            k_block_count=k_block_count,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            out_cell_capacity=out_capacity,
            out_row_stride_cells=out_stride,
        )
        assert err == Q8_0_AVX2_OK

        err, ref_q32 = q8_0_matmul_tiled_avx2_q32_checked(
            lhs_matrix_blocks=lhs,
            lhs_block_capacity=lhs_capacity,
            lhs_rows=row_count,
            lhs_row_stride_blocks=lhs_stride,
            rhs_col_blocks=rhs,
            rhs_block_capacity=rhs_capacity,
            rhs_cols=col_count,
            rhs_col_stride_blocks=rhs_stride,
            k_block_count=k_block_count,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            out_capacity=out_capacity,
            out_row_stride_cols=out_stride,
        )
        assert err == Q8_0_AVX2_OK

        # Q16 AVX2 matmul contract: each cell rounds once from full-cell Q32.
        expected_q16 = [0] * len(ref_q32)
        for r in range(row_count):
            for c in range(col_count):
                idx = r * out_stride + c
                expected_q16[idx] = round_shift_right_signed(ref_q32[idx], 16)

        assert got_q16 == expected_q16


def test_tile_and_stride_invariants() -> None:
    lhs = [make_block(0x3C00, [1] * 32)] * 4
    rhs = [make_block(0x3C00, [1] * 32)] * 4

    err, _ = q8_0_matmul_tiled_avx2_q16_checked(
        lhs_blocks=lhs,
        lhs_block_capacity=4,
        row_count=2,
        lhs_row_stride_blocks=1,
        rhs_col_blocks=rhs,
        rhs_block_capacity=4,
        col_count=2,
        rhs_col_stride_blocks=2,
        k_block_count=2,
        tile_rows=1,
        tile_cols=1,
        out_cell_capacity=4,
        out_row_stride_cells=2,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_matmul_tiled_avx2_q16_checked(
        lhs_blocks=lhs,
        lhs_block_capacity=4,
        row_count=2,
        lhs_row_stride_blocks=2,
        rhs_col_blocks=rhs,
        rhs_block_capacity=4,
        col_count=2,
        rhs_col_stride_blocks=2,
        k_block_count=2,
        tile_rows=0,
        tile_cols=1,
        out_cell_capacity=4,
        out_row_stride_cells=2,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_matmul_tiled_avx2_q16_checked(
        lhs_blocks=lhs,
        lhs_block_capacity=4,
        row_count=2,
        lhs_row_stride_blocks=2,
        rhs_col_blocks=rhs,
        rhs_block_capacity=4,
        col_count=2,
        rhs_col_stride_blocks=2,
        k_block_count=2,
        tile_rows=1,
        tile_cols=1,
        out_cell_capacity=4,
        out_row_stride_cells=1,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN


def test_error_paths() -> None:
    err, _ = q8_0_matmul_tiled_avx2_q16_checked(None, 0, 0, 0, [], 0, 0, 0, 0, 1, 1, 0, 0)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_matmul_tiled_avx2_q16_checked([], 0, 0, 0, None, 0, 0, 0, 0, 1, 1, 0, 0)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_matmul_tiled_avx2_q16_checked([], -1, 0, 0, [], 0, 0, 0, 0, 1, 1, 0, 0)
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_matmul_tiled_avx2_q16_checked([], I64_MAX, I64_MAX, 2, [], 0, 0, 0, 0, 1, 1, 0, 0)
    assert err == Q8_0_AVX2_ERR_OVERFLOW


def test_q16_q32_shared_preflight_error_surface() -> None:
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(9)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(9)]

    # Preflight-equivalent invalid configurations should fail identically in
    # Q32 and Q16 tiled AVX2 matmul entry points.
    scenarios = [
        dict(row_count=-1, col_count=2, k_block_count=1, lhs_stride=1, rhs_stride=1, out_stride=2, tile_rows=1, tile_cols=1, lhs_cap=9, rhs_cap=9, out_cap=9),
        dict(row_count=2, col_count=2, k_block_count=3, lhs_stride=2, rhs_stride=3, out_stride=2, tile_rows=1, tile_cols=1, lhs_cap=9, rhs_cap=9, out_cap=9),
        dict(row_count=2, col_count=2, k_block_count=2, lhs_stride=2, rhs_stride=2, out_stride=1, tile_rows=1, tile_cols=1, lhs_cap=9, rhs_cap=9, out_cap=9),
        dict(row_count=2, col_count=2, k_block_count=2, lhs_stride=2, rhs_stride=2, out_stride=2, tile_rows=0, tile_cols=1, lhs_cap=9, rhs_cap=9, out_cap=9),
        dict(row_count=I64_MAX, col_count=2, k_block_count=1, lhs_stride=2, rhs_stride=1, out_stride=2, tile_rows=1, tile_cols=1, lhs_cap=9, rhs_cap=9, out_cap=9),
        dict(row_count=3, col_count=3, k_block_count=2, lhs_stride=3, rhs_stride=3, out_stride=3, tile_rows=2, tile_cols=2, lhs_cap=8, rhs_cap=9, out_cap=9),
        dict(row_count=3, col_count=3, k_block_count=2, lhs_stride=3, rhs_stride=3, out_stride=3, tile_rows=2, tile_cols=2, lhs_cap=9, rhs_cap=8, out_cap=9),
        dict(row_count=3, col_count=3, k_block_count=2, lhs_stride=3, rhs_stride=3, out_stride=3, tile_rows=2, tile_cols=2, lhs_cap=9, rhs_cap=9, out_cap=8),
    ]

    for scenario in scenarios:
        err_q16, _ = q8_0_matmul_tiled_avx2_q16_checked(
            lhs_blocks=lhs,
            lhs_block_capacity=scenario["lhs_cap"],
            row_count=scenario["row_count"],
            lhs_row_stride_blocks=scenario["lhs_stride"],
            rhs_col_blocks=rhs,
            rhs_block_capacity=scenario["rhs_cap"],
            col_count=scenario["col_count"],
            rhs_col_stride_blocks=scenario["rhs_stride"],
            k_block_count=scenario["k_block_count"],
            tile_rows=scenario["tile_rows"],
            tile_cols=scenario["tile_cols"],
            out_cell_capacity=scenario["out_cap"],
            out_row_stride_cells=scenario["out_stride"],
        )

        err_q32, _ = q8_0_matmul_tiled_avx2_q32_checked(
            lhs_matrix_blocks=lhs,
            lhs_block_capacity=scenario["lhs_cap"],
            lhs_rows=scenario["row_count"],
            lhs_row_stride_blocks=scenario["lhs_stride"],
            rhs_col_blocks=rhs,
            rhs_block_capacity=scenario["rhs_cap"],
            rhs_cols=scenario["col_count"],
            rhs_col_stride_blocks=scenario["rhs_stride"],
            k_block_count=scenario["k_block_count"],
            tile_rows=scenario["tile_rows"],
            tile_cols=scenario["tile_cols"],
            out_capacity=scenario["out_cap"],
            out_row_stride_cols=scenario["out_stride"],
        )

        assert err_q16 == err_q32


def test_compute_tile_end_checked_contract() -> None:
    err, tile_end = compute_tile_end_checked(2, 3, 9)
    assert err == Q8_0_AVX2_OK
    assert tile_end == 5

    err, tile_end = compute_tile_end_checked(7, 5, 9)
    assert err == Q8_0_AVX2_OK
    assert tile_end == 9

    err, _ = compute_tile_end_checked(-1, 3, 9)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    err, _ = compute_tile_end_checked(1, -3, 9)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    err, _ = compute_tile_end_checked(1, 3, -9)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    err, _ = compute_tile_end_checked(10, 1, 9)
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = compute_tile_end_checked(I64_MAX - 1, 5, I64_MAX)
    assert err == Q8_0_AVX2_ERR_OVERFLOW


def run() -> None:
    test_known_matrix_matches_q32_then_single_round()
    test_q16_output_matches_q32_matmul_single_rounding_per_cell()
    test_tile_and_stride_invariants()
    test_error_paths()
    test_q16_q32_shared_preflight_error_surface()
    test_compute_tile_end_checked_contract()
    print("q8_0_matmul_tiled_avx2_q16_reference_checks=ok")


if __name__ == "__main__":
    run()
