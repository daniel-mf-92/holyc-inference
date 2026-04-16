#!/usr/bin/env python3
"""Parity harness for checked tiled Q8_0 matmul semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_q8_0_dot import (
    Q8_0_ERR_BAD_DST_LEN,
    Q8_0_ERR_OVERFLOW,
    Q8_0_I64_MAX,
    Q8_0_OK,
    Q8_0_VALUES_PER_BLOCK,
    dot_product_blocks_q16_accumulate_checked,
    half_bits,
    pack_signed,
)
from test_q8_0_avx2_blocks_q32 import (
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_OK,
    q8_0_dot_blocks_avx2_q32_checked,
)


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > Q8_0_I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def try_add_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > Q8_0_I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def q8_0_matmul_q16_tiled_checked(
    lhs_blocks: list[tuple[int, bytes]],
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks: list[tuple[int, bytes]],
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    tile_rows: int,
    tile_cols: int,
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> tuple[int, list[int]]:
    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_cell_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN, []
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return Q8_0_ERR_BAD_DST_LEN, []
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q8_0_ERR_BAD_DST_LEN, []
    if tile_rows <= 0 or tile_cols <= 0:
        return Q8_0_ERR_BAD_DST_LEN, []

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return Q8_0_ERR_BAD_DST_LEN, []
    if row_count > 0 and out_row_stride_cells < col_count:
        return Q8_0_ERR_BAD_DST_LEN, []

    ok, lhs_required = try_mul_i64_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_ERR_OVERFLOW, []
    ok, rhs_required = try_mul_i64_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_ERR_OVERFLOW, []
    ok, out_required = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q8_0_ERR_OVERFLOW, []

    if lhs_required > lhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN, []
    if rhs_required > rhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN, []
    if out_required > out_cell_capacity:
        return Q8_0_ERR_BAD_DST_LEN, []

    out_cells_q16 = [0] * out_required

    row_tile_start = 0
    while row_tile_start < row_count:
        ok, row_tile_end = try_add_i64_nonneg(row_tile_start, tile_rows)
        if not ok:
            return Q8_0_ERR_OVERFLOW, []
        row_tile_end = min(row_tile_end, row_count)

        col_tile_start = 0
        while col_tile_start < col_count:
            ok, col_tile_end = try_add_i64_nonneg(col_tile_start, tile_cols)
            if not ok:
                return Q8_0_ERR_OVERFLOW, []
            col_tile_end = min(col_tile_end, col_count)

            for row_index in range(row_tile_start, row_tile_end):
                ok, lhs_row_base = try_mul_i64_nonneg(row_index, lhs_row_stride_blocks)
                if not ok:
                    return Q8_0_ERR_OVERFLOW, []
                ok, out_row_base = try_mul_i64_nonneg(row_index, out_row_stride_cells)
                if not ok:
                    return Q8_0_ERR_OVERFLOW, []

                lhs_row_slice = lhs_blocks[lhs_row_base : lhs_row_base + k_block_count]
                if len(lhs_row_slice) != k_block_count:
                    return Q8_0_ERR_BAD_DST_LEN, []

                for col_index in range(col_tile_start, col_tile_end):
                    ok, rhs_col_base = try_mul_i64_nonneg(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q8_0_ERR_OVERFLOW, []
                    ok, out_index = try_add_i64_nonneg(out_row_base, col_index)
                    if not ok:
                        return Q8_0_ERR_OVERFLOW, []

                    rhs_col_slice = rhs_col_blocks[rhs_col_base : rhs_col_base + k_block_count]
                    if len(rhs_col_slice) != k_block_count:
                        return Q8_0_ERR_BAD_DST_LEN, []

                    err, cell_dot_q16 = dot_product_blocks_q16_accumulate_checked(
                        lhs_row_slice,
                        rhs_col_slice,
                        0,
                    )
                    if err != Q8_0_OK:
                        return err, []

                    out_cells_q16[out_index] = cell_dot_q16

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    return Q8_0_OK, out_cells_q16


def q8_0_matmul_q16_reference_untiled(
    lhs_blocks: list[tuple[int, bytes]],
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks: list[tuple[int, bytes]],
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_row_stride_cells: int,
) -> tuple[int, list[int]]:
    out_required = row_count * out_row_stride_cells
    out = [0] * out_required

    for row_index in range(row_count):
        lhs_row_base = row_index * lhs_row_stride_blocks
        lhs_row_slice = lhs_blocks[lhs_row_base : lhs_row_base + k_block_count]
        if len(lhs_row_slice) != k_block_count:
            return Q8_0_ERR_BAD_DST_LEN, []

        for col_index in range(col_count):
            rhs_col_base = col_index * rhs_col_stride_blocks
            rhs_col_slice = rhs_col_blocks[rhs_col_base : rhs_col_base + k_block_count]
            if len(rhs_col_slice) != k_block_count:
                return Q8_0_ERR_BAD_DST_LEN, []

            err, dot_q16 = dot_product_blocks_q16_accumulate_checked(lhs_row_slice, rhs_col_slice, 0)
            if err != Q8_0_OK:
                return err, []

            out[row_index * out_row_stride_cells + col_index] = dot_q16

    return Q8_0_OK, out


def q8_0_matmul_q32_tiled_avx2_checked(
    lhs_blocks: list[tuple[int, bytes]],
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks: list[tuple[int, bytes]],
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    tile_rows: int,
    tile_cols: int,
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> tuple[int, list[int]]:
    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_cell_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN, []
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return Q8_0_ERR_BAD_DST_LEN, []
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q8_0_ERR_BAD_DST_LEN, []
    if tile_rows <= 0 or tile_cols <= 0:
        return Q8_0_ERR_BAD_DST_LEN, []

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return Q8_0_ERR_BAD_DST_LEN, []
    if row_count > 0 and out_row_stride_cells < col_count:
        return Q8_0_ERR_BAD_DST_LEN, []

    ok, lhs_required = try_mul_i64_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_ERR_OVERFLOW, []
    ok, rhs_required = try_mul_i64_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_ERR_OVERFLOW, []
    ok, out_required = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q8_0_ERR_OVERFLOW, []

    if lhs_required > lhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN, []
    if rhs_required > rhs_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN, []
    if out_required > out_cell_capacity:
        return Q8_0_ERR_BAD_DST_LEN, []

    out_cells_q32 = [0] * out_required

    row_tile_start = 0
    while row_tile_start < row_count:
        ok, row_tile_end = try_add_i64_nonneg(row_tile_start, tile_rows)
        if not ok:
            return Q8_0_ERR_OVERFLOW, []
        row_tile_end = min(row_tile_end, row_count)

        col_tile_start = 0
        while col_tile_start < col_count:
            ok, col_tile_end = try_add_i64_nonneg(col_tile_start, tile_cols)
            if not ok:
                return Q8_0_ERR_OVERFLOW, []
            col_tile_end = min(col_tile_end, col_count)

            for row_index in range(row_tile_start, row_tile_end):
                ok, lhs_row_base = try_mul_i64_nonneg(row_index, lhs_row_stride_blocks)
                if not ok:
                    return Q8_0_ERR_OVERFLOW, []
                ok, out_row_base = try_mul_i64_nonneg(row_index, out_row_stride_cells)
                if not ok:
                    return Q8_0_ERR_OVERFLOW, []

                lhs_row_slice = lhs_blocks[lhs_row_base : lhs_row_base + k_block_count]
                if len(lhs_row_slice) != k_block_count:
                    return Q8_0_ERR_BAD_DST_LEN, []

                for col_index in range(col_tile_start, col_tile_end):
                    ok, rhs_col_base = try_mul_i64_nonneg(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q8_0_ERR_OVERFLOW, []
                    ok, out_index = try_add_i64_nonneg(out_row_base, col_index)
                    if not ok:
                        return Q8_0_ERR_OVERFLOW, []

                    rhs_col_slice = rhs_col_blocks[rhs_col_base : rhs_col_base + k_block_count]
                    if len(rhs_col_slice) != k_block_count:
                        return Q8_0_ERR_BAD_DST_LEN, []

                    lhs_blocks_avx2 = [{"d_fp16": scale, "qs": list(qs)} for scale, qs in lhs_row_slice]
                    rhs_blocks_avx2 = [{"d_fp16": scale, "qs": list(qs)} for scale, qs in rhs_col_slice]

                    err, cell_dot_q32 = q8_0_dot_blocks_avx2_q32_checked(
                        lhs_blocks_avx2,
                        rhs_blocks_avx2,
                        k_block_count,
                    )
                    if err != Q8_0_AVX2_OK:
                        if err == Q8_0_AVX2_ERR_BAD_LEN:
                            return Q8_0_ERR_BAD_DST_LEN, []
                        return Q8_0_ERR_OVERFLOW, []

                    out_cells_q32[out_index] = cell_dot_q32

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    return Q8_0_OK, out_cells_q32


def q8_0_matmul_q32_reference_untiled(
    lhs_blocks: list[tuple[int, bytes]],
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks: list[tuple[int, bytes]],
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_row_stride_cells: int,
) -> tuple[int, list[int]]:
    out_required = row_count * out_row_stride_cells
    out = [0] * out_required

    for row_index in range(row_count):
        lhs_row_base = row_index * lhs_row_stride_blocks
        lhs_row_slice = lhs_blocks[lhs_row_base : lhs_row_base + k_block_count]
        if len(lhs_row_slice) != k_block_count:
            return Q8_0_ERR_BAD_DST_LEN, []

        for col_index in range(col_count):
            rhs_col_base = col_index * rhs_col_stride_blocks
            rhs_col_slice = rhs_col_blocks[rhs_col_base : rhs_col_base + k_block_count]
            if len(rhs_col_slice) != k_block_count:
                return Q8_0_ERR_BAD_DST_LEN, []

            lhs_blocks_avx2 = [{"d_fp16": scale, "qs": list(qs)} for scale, qs in lhs_row_slice]
            rhs_blocks_avx2 = [{"d_fp16": scale, "qs": list(qs)} for scale, qs in rhs_col_slice]

            err, dot_q32 = q8_0_dot_blocks_avx2_q32_checked(lhs_blocks_avx2, rhs_blocks_avx2, k_block_count)
            if err != Q8_0_AVX2_OK:
                if err == Q8_0_AVX2_ERR_BAD_LEN:
                    return Q8_0_ERR_BAD_DST_LEN, []
                return Q8_0_ERR_OVERFLOW, []

            out[row_index * out_row_stride_cells + col_index] = dot_q32

    return Q8_0_OK, out


def make_block(rng: random.Random, *, scale: float | None = None) -> tuple[int, bytes]:
    if scale is None:
        scale = rng.uniform(-2.0, 2.0)
    qs = [rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)]
    return half_bits(scale), pack_signed(qs)


def test_tiled_matches_untiled_randomized() -> None:
    rng = random.Random(2026041608)

    for _ in range(220):
        row_count = rng.randint(1, 7)
        col_count = rng.randint(1, 7)
        k_block_count = rng.randint(1, 6)

        lhs_row_stride_blocks = k_block_count + rng.randint(0, 3)
        rhs_col_stride_blocks = k_block_count + rng.randint(0, 3)
        out_row_stride_cells = col_count + rng.randint(0, 3)

        lhs_capacity = row_count * lhs_row_stride_blocks
        rhs_capacity = col_count * rhs_col_stride_blocks
        out_capacity = row_count * out_row_stride_cells

        lhs_blocks = [make_block(rng) for _ in range(lhs_capacity)]
        rhs_col_blocks = [make_block(rng) for _ in range(rhs_capacity)]

        tile_rows = rng.randint(1, max(1, row_count + 2))
        tile_cols = rng.randint(1, max(1, col_count + 2))

        err_tiled, out_tiled = q8_0_matmul_q16_tiled_checked(
            lhs_blocks,
            lhs_capacity,
            row_count,
            lhs_row_stride_blocks,
            rhs_col_blocks,
            rhs_capacity,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            tile_rows,
            tile_cols,
            out_capacity,
            out_row_stride_cells,
        )
        assert err_tiled == Q8_0_OK

        err_ref, out_ref = q8_0_matmul_q16_reference_untiled(
            lhs_blocks,
            row_count,
            lhs_row_stride_blocks,
            rhs_col_blocks,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            out_row_stride_cells,
        )
        assert err_ref == Q8_0_OK
        assert out_tiled == out_ref


def test_rejects_bad_dimensions_and_capacities() -> None:
    rng = random.Random(2026041609)
    lhs_blocks = [make_block(rng) for _ in range(6)]
    rhs_blocks = [make_block(rng) for _ in range(6)]

    err, _ = q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        -1,
        2,
        3,
        rhs_blocks,
        6,
        2,
        3,
        2,
        1,
        1,
        6,
        2,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        6,
        2,
        1,
        rhs_blocks,
        6,
        2,
        3,
        2,
        1,
        1,
        6,
        2,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        6,
        2,
        3,
        rhs_blocks,
        6,
        2,
        3,
        2,
        0,
        1,
        6,
        2,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        5,
        2,
        3,
        rhs_blocks,
        6,
        2,
        3,
        2,
        1,
        1,
        6,
        2,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN


def test_reports_extent_multiply_overflow() -> None:
    rng = random.Random(2026041610)
    lhs_blocks = [make_block(rng)]
    rhs_blocks = [make_block(rng)]

    err, _ = q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        Q8_0_I64_MAX,
        Q8_0_I64_MAX,
        2,
        rhs_blocks,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert err == Q8_0_ERR_OVERFLOW


def test_reports_cell_accumulator_overflow() -> None:
    max_pos = pack_signed([127] * Q8_0_VALUES_PER_BLOCK)
    vec_col_blocks = [
        (half_bits(65504.0), max_pos),
        (half_bits(65504.0), max_pos),
    ]
    lhs_blocks = [
        (half_bits(65504.0), max_pos),
        (half_bits(65504.0), max_pos),
    ]

    err, _ = q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        2,
        1,
        2,
        vec_col_blocks,
        2,
        1,
        2,
        2,
        1,
        1,
        1,
        1,
    )
    assert err == Q8_0_ERR_OVERFLOW


def test_q32_avx2_tiled_matches_untiled_randomized() -> None:
    rng = random.Random(2026041611)

    for _ in range(180):
        row_count = rng.randint(1, 6)
        col_count = rng.randint(1, 6)
        k_block_count = rng.randint(1, 5)

        lhs_row_stride_blocks = k_block_count + rng.randint(0, 2)
        rhs_col_stride_blocks = k_block_count + rng.randint(0, 2)
        out_row_stride_cells = col_count + rng.randint(0, 2)

        lhs_capacity = row_count * lhs_row_stride_blocks
        rhs_capacity = col_count * rhs_col_stride_blocks
        out_capacity = row_count * out_row_stride_cells

        lhs_blocks = [make_block(rng) for _ in range(lhs_capacity)]
        rhs_col_blocks = [make_block(rng) for _ in range(rhs_capacity)]

        tile_rows = rng.randint(1, max(1, row_count + 2))
        tile_cols = rng.randint(1, max(1, col_count + 2))

        err_tiled, out_tiled = q8_0_matmul_q32_tiled_avx2_checked(
            lhs_blocks,
            lhs_capacity,
            row_count,
            lhs_row_stride_blocks,
            rhs_col_blocks,
            rhs_capacity,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            tile_rows,
            tile_cols,
            out_capacity,
            out_row_stride_cells,
        )
        assert err_tiled == Q8_0_OK

        err_ref, out_ref = q8_0_matmul_q32_reference_untiled(
            lhs_blocks,
            row_count,
            lhs_row_stride_blocks,
            rhs_col_blocks,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            out_row_stride_cells,
        )
        assert err_ref == Q8_0_OK
        assert out_tiled == out_ref


def test_q32_avx2_reports_extent_multiply_overflow() -> None:
    rng = random.Random(2026041612)
    lhs_blocks = [make_block(rng)]
    rhs_blocks = [make_block(rng)]

    err, _ = q8_0_matmul_q32_tiled_avx2_checked(
        lhs_blocks,
        Q8_0_I64_MAX,
        Q8_0_I64_MAX,
        2,
        rhs_blocks,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert err == Q8_0_ERR_OVERFLOW


def run() -> None:
    test_tiled_matches_untiled_randomized()
    test_rejects_bad_dimensions_and_capacities()
    test_reports_extent_multiply_overflow()
    test_reports_cell_accumulator_overflow()
    test_q32_avx2_tiled_matches_untiled_randomized()
    test_q32_avx2_reports_extent_multiply_overflow()
    print("q8_0_matmul_tiled_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
