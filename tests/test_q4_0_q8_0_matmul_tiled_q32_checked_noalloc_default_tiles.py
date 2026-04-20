#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTiles."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    I64_MAX,
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    dot_blocks_avx2_q32_checked,
    make_q4_block,
    make_q8_block,
    q4_0_q8_0_matmul_tiled_avx2_q32_checked,
)


def try_add_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles(
    lhs_q4_blocks,
    lhs_q4_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_q8_col_blocks,
    rhs_q8_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells_q32: list[int] | None,
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> int:
    if lhs_q4_blocks is None or rhs_q8_col_blocks is None or out_cells_q32 is None:
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    if lhs_q4_block_capacity < 0 or rhs_q8_block_capacity < 0 or out_cell_capacity < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if row_count > 0 and out_row_stride_cells < col_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, lhs_required_blocks = try_mul_i64_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, rhs_required_blocks = try_mul_i64_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, out_required_cells = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    if lhs_required_blocks > lhs_q4_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if rhs_required_blocks > rhs_q8_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if out_required_cells > out_cell_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    # Strict no-partial contract: validate every dot path before first write.
    row_tile_start = 0
    while row_tile_start < row_count:
        ok, row_tile_end = try_add_i64_nonneg(row_tile_start, 4)
        if not ok:
            return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
        row_tile_end = min(row_tile_end, row_count)

        col_tile_start = 0
        while col_tile_start < col_count:
            ok, col_tile_end = try_add_i64_nonneg(col_tile_start, 4)
            if not ok:
                return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
            col_tile_end = min(col_tile_end, col_count)

            for row_index in range(row_tile_start, row_tile_end):
                ok, lhs_row_base = try_mul_i64_nonneg(row_index, lhs_row_stride_blocks)
                if not ok:
                    return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

                for col_index in range(col_tile_start, col_tile_end):
                    ok, rhs_col_base = try_mul_i64_nonneg(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

                    err, _ = dot_blocks_avx2_q32_checked(
                        lhs_blocks=lhs_q4_blocks[lhs_row_base : lhs_row_base + k_block_count],
                        rhs_blocks=rhs_q8_col_blocks[rhs_col_base : rhs_col_base + k_block_count],
                        block_count=k_block_count,
                    )
                    if err != Q4_0_Q8_0_AVX2_OK:
                        return err

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    err, staged = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
        lhs_q4_blocks=lhs_q4_blocks,
        lhs_q4_block_capacity=lhs_q4_block_capacity,
        row_count=row_count,
        lhs_row_stride_blocks=lhs_row_stride_blocks,
        rhs_q8_col_blocks=rhs_q8_col_blocks,
        rhs_q8_block_capacity=rhs_q8_block_capacity,
        col_count=col_count,
        rhs_col_stride_blocks=rhs_col_stride_blocks,
        k_block_count=k_block_count,
        tile_rows=4,
        tile_cols=4,
        out_cell_capacity=out_cell_capacity,
        out_row_stride_cells=out_row_stride_cells,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    for index in range(out_required_cells):
        out_cells_q32[index] = staged[index]

    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(
    lhs_q4_blocks,
    lhs_q4_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_q8_col_blocks,
    rhs_q8_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells_q32: list[int] | None,
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> int:
    if out_cells_q32 is None:
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    sentinel = list(out_cells_q32)
    err = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        out_cells_q32[:] = sentinel
    return err


def test_source_contains_default_tiles_wrapper() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    signature = "I32 Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTiles("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M" in source
    assert "Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N" in source
    assert "Q4_0_Q8_0_MATMUL_DEFAULT_TILE_K" in source
    assert "Q4_0_Q8_0DotBlocksAVX2Q32Checked(" in body
    assert "return Q4_0Q8_0MatMulQ32TiledAVX2Checked(" in body


def test_known_vector_and_no_partial_on_error() -> None:
    rng = random.Random(2026042001)

    row_count = 3
    col_count = 4
    k_block_count = 2
    lhs_stride = 3
    rhs_stride = 3
    out_stride = 5

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]

    out_capacity = row_count * out_stride
    out = [123456] * out_capacity

    err = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        out,
        out_capacity,
        out_stride,
    )
    assert err == Q4_0_Q8_0_AVX2_OK

    err_ref, ref = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
        lhs_q4_blocks=lhs,
        lhs_q4_block_capacity=len(lhs),
        row_count=row_count,
        lhs_row_stride_blocks=lhs_stride,
        rhs_q8_col_blocks=rhs,
        rhs_q8_block_capacity=len(rhs),
        col_count=col_count,
        rhs_col_stride_blocks=rhs_stride,
        k_block_count=k_block_count,
        tile_rows=4,
        tile_cols=4,
        out_cell_capacity=out_capacity,
        out_row_stride_cells=out_stride,
    )
    assert err_ref == Q4_0_Q8_0_AVX2_OK
    assert out == ref

    sentinel = [99] * out_capacity
    out2 = sentinel.copy()
    err = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        out2,
        out_capacity - 1,
        out_stride,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert out2 == sentinel


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420684)

    for _ in range(1200):
        row_count = rng.randint(0, 8)
        col_count = rng.randint(0, 8)
        k_block_count = rng.randint(0, 6)

        lhs_stride = rng.randint(0, 9)
        rhs_stride = rng.randint(0, 9)
        out_stride = rng.randint(0, 10)

        if row_count > 0 and k_block_count > 0 and lhs_stride < k_block_count:
            lhs_stride = k_block_count + rng.randint(0, 2)
        if col_count > 0 and k_block_count > 0 and rhs_stride < k_block_count:
            rhs_stride = k_block_count + rng.randint(0, 2)
        if row_count > 0 and col_count > 0 and out_stride < col_count:
            out_stride = col_count + rng.randint(0, 3)

        lhs_need = row_count * lhs_stride
        rhs_need = col_count * rhs_stride
        out_need = row_count * out_stride

        lhs_cap = max(0, lhs_need + rng.randint(-2, 3))
        rhs_cap = max(0, rhs_need + rng.randint(-2, 3))
        out_cap = max(0, out_need + rng.randint(-2, 3))

        lhs = [make_q4_block(rng) for _ in range(lhs_cap)]
        rhs = [make_q8_block(rng) for _ in range(rhs_cap)]

        out_a = [-(10**9)] * out_cap
        out_b = out_a.copy()

        err_a = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles(
            lhs,
            lhs_cap,
            row_count,
            lhs_stride,
            rhs,
            rhs_cap,
            col_count,
            rhs_stride,
            k_block_count,
            out_a,
            out_cap,
            out_stride,
        )
        err_b = explicit_checked_composition(
            lhs,
            lhs_cap,
            row_count,
            lhs_stride,
            rhs,
            rhs_cap,
            col_count,
            rhs_stride,
            k_block_count,
            out_b,
            out_cap,
            out_stride,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_default_tiles_wrapper()
    test_known_vector_and_no_partial_on_error()
    test_randomized_parity_against_explicit_composition()
    print("ok")
