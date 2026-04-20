#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesCommitOnly."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    make_q4_block,
    make_q8_block,
    q4_0_q8_0_matmul_tiled_avx2_q32_checked,
)
from test_q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only import (
    q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only,
)


def q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_commit_only(
    lhs_q4_blocks,
    lhs_q4_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_q8_col_blocks,
    rhs_q8_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells_q32,
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> int:
    if lhs_q4_blocks is None or rhs_q8_col_blocks is None or out_cells_q32 is None:
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    required_lhs = [0]
    required_rhs = [0]
    required_out = [0]
    tile_m = [0]
    tile_n = [0]
    tile_k = [0]

    status = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only(
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
        required_lhs,
        required_rhs,
        required_out,
        tile_m,
        tile_n,
        tile_k,
    )
    if status != Q4_0_Q8_0_AVX2_OK:
        return status

    if tile_m[0] != 4 or tile_n[0] != 4 or tile_k[0] != 8:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    status, staged = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
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
    if status != Q4_0_Q8_0_AVX2_OK:
        return status

    for index in range(len(staged)):
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
    out_cells_q32,
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> int:
    if out_cells_q32 is None:
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    sentinel = list(out_cells_q32)
    required_lhs = [0]
    required_rhs = [0]
    required_out = [0]
    tile_m = [0]
    tile_n = [0]
    tile_k = [0]

    status = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only(
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
        required_lhs,
        required_rhs,
        required_out,
        tile_m,
        tile_n,
        tile_k,
    )
    if status != Q4_0_Q8_0_AVX2_OK:
        out_cells_q32[:] = sentinel
        return status

    if tile_m[0] != 4 or tile_n[0] != 4 or tile_k[0] != 8:
        out_cells_q32[:] = sentinel
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    status, staged = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
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
    if status != Q4_0_Q8_0_AVX2_OK:
        out_cells_q32[:] = sentinel
        return status

    for index in range(len(staged)):
        out_cells_q32[index] = staged[index]

    return Q4_0_Q8_0_AVX2_OK


def test_source_contains_commit_only_signature() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    signature = "I32 Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesCommitOnly("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesPreflightOnly(" in body
    assert "if (status != Q4_0_Q8_0_MATMUL_OK)" in body
    assert "if (tile_m != Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M ||" in body
    assert "return Q4_0Q8_0MatMulQ32TiledAVX2Checked(" in body


def test_known_vector_and_no_partial_error_behavior() -> None:
    rng = random.Random(2026042011)

    row_count = 4
    col_count = 3
    k_block_count = 2
    lhs_stride = 3
    rhs_stride = 3
    out_stride = 6

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]

    out_capacity = row_count * out_stride
    out = [654321] * out_capacity

    err = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_commit_only(
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

    sentinel = [77] * out_capacity
    out_bad = sentinel.copy()
    err = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_commit_only(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        out_bad,
        out_capacity - 1,
        out_stride,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert out_bad == sentinel


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420888)

    for _ in range(1400):
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

        err_a = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_commit_only(
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


def test_overflow_passthrough_keeps_output_unchanged() -> None:
    lhs = []
    rhs = []
    sentinel = [101, 202, 303]
    out = sentinel.copy()

    err = q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_commit_only(
        lhs,
        0,
        0x7FFFFFFFFFFFFFFF,
        2,
        rhs,
        0,
        1,
        1,
        0,
        out,
        len(out),
        1,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    assert out == sentinel


if __name__ == "__main__":
    test_source_contains_commit_only_signature()
    test_known_vector_and_no_partial_error_behavior()
    test_randomized_parity_against_explicit_composition()
    test_overflow_passthrough_keeps_output_unchanged()
    print("ok")
