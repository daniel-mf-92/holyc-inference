#!/usr/bin/env python3
"""Parity harness for MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTilesPreflightOnly (IQ-1012)."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only import (
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only,
)
from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    I64_MAX,
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    make_q4_block,
    make_q8_block,
)

DEFAULT_TILE_M = 4
DEFAULT_TILE_N = 4


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only(
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
    out_required_out_cells,
    out_required_out_bytes,
    out_tile_rows,
    out_tile_cols,
) -> int:
    if (
        out_required_out_cells is None
        or out_required_out_bytes is None
        or out_tile_rows is None
        or out_tile_cols is None
    ):
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    if (
        out_required_out_cells is out_required_out_bytes
        or out_required_out_cells is out_tile_rows
        or out_required_out_cells is out_tile_cols
        or out_required_out_bytes is out_tile_rows
        or out_required_out_bytes is out_tile_cols
        or out_tile_rows is out_tile_cols
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    if (
        out_required_out_cells is lhs_q4_blocks
        or out_required_out_cells is rhs_q8_col_blocks
        or out_required_out_cells is out_cells_q32
        or out_required_out_bytes is lhs_q4_blocks
        or out_required_out_bytes is rhs_q8_col_blocks
        or out_required_out_bytes is out_cells_q32
        or out_tile_rows is lhs_q4_blocks
        or out_tile_rows is rhs_q8_col_blocks
        or out_tile_rows is out_cells_q32
        or out_tile_cols is lhs_q4_blocks
        or out_tile_cols is rhs_q8_col_blocks
        or out_tile_cols is out_cells_q32
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    snapshot_row_count = row_count
    snapshot_col_count = col_count
    snapshot_k_block_count = k_block_count
    snapshot_out_capacity = out_cell_capacity

    staged_required_out_cells = [0]
    staged_required_out_bytes = [0]
    staged_tile_rows = [0]
    staged_tile_cols = [0]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        DEFAULT_TILE_M,
        DEFAULT_TILE_N,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        staged_required_out_cells,
        staged_required_out_bytes,
        staged_tile_rows,
        staged_tile_cols,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    if (
        snapshot_row_count != row_count
        or snapshot_col_count != col_count
        or snapshot_k_block_count != k_block_count
        or snapshot_out_capacity != out_cell_capacity
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, canonical_required_out_cells = try_mul_i64_nonneg(snapshot_row_count, out_row_stride_cells)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    if canonical_required_out_cells > snapshot_out_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, canonical_required_out_bytes = try_mul_i64_nonneg(canonical_required_out_cells, 8)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    if (
        staged_required_out_cells[0] != canonical_required_out_cells
        or staged_required_out_bytes[0] != canonical_required_out_bytes
        or staged_tile_rows[0] != DEFAULT_TILE_M
        or staged_tile_cols[0] != DEFAULT_TILE_N
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_required_out_cells[0] = staged_required_out_cells[0]
    out_required_out_bytes[0] = staged_required_out_bytes[0]
    out_tile_rows[0] = staged_tile_rows[0]
    out_tile_cols[0] = staged_tile_cols[0]
    return Q4_0_Q8_0_AVX2_OK


def test_source_contains_iq1012_default_preflight_wrapper() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTilesPreflightOnly("
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI32 ", 1)[0]

    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly(" in body
    assert "Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M" in body
    assert "Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N" in body
    assert "out_required_out_cells == (I64 *)lhs_q4_blocks" in body
    assert "staged_required_out_cells != canonical_required_out_cells" in body
    assert "*out_required_out_cells = staged_required_out_cells;" in body
    assert "*out_required_out_bytes = staged_required_out_bytes;" in body
    assert "*out_tile_rows = staged_tile_rows;" in body
    assert "*out_tile_cols = staged_tile_cols;" in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnly(" not in body


def test_known_vector_and_zero_write_contract() -> None:
    rng = random.Random(202604221012)
    row_count = 5
    col_count = 6
    k_block_count = 3
    lhs_stride = 4
    rhs_stride = 5
    out_stride = 7

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out_capacity = row_count * out_stride
    out = [0x6A6A] * out_capacity
    out_before = list(out)

    req_cells = [0x1111]
    req_bytes = [0x2222]
    tile_rows = [0x3333]
    tile_cols = [0x4444]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only(
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
        req_cells,
        req_bytes,
        tile_rows,
        tile_cols,
    )
    assert err == Q4_0_Q8_0_AVX2_OK
    assert out == out_before
    assert req_cells == [row_count * out_stride]
    assert req_bytes == [row_count * out_stride * 8]
    assert tile_rows == [DEFAULT_TILE_M]
    assert tile_cols == [DEFAULT_TILE_N]


def test_randomized_parity_vs_explicit_default_composition() -> None:
    rng = random.Random(202604221013)

    for i in range(500):
        row_count = rng.randint(0, 8)
        col_count = rng.randint(0, 8)
        k_block_count = rng.randint(0, 5)

        lhs_stride = k_block_count + rng.randint(0, 3)
        rhs_stride = k_block_count + rng.randint(0, 3)
        out_stride = col_count + rng.randint(0, 3)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        local_rng = random.Random(20260422101300 + i)
        lhs = [make_q4_block(local_rng) for _ in range(lhs_capacity)]
        rhs = [make_q8_block(local_rng) for _ in range(rhs_capacity)]

        out_a = [0xAAAA] * out_capacity
        out_b = [0xAAAA] * out_capacity

        req_cells_a = [0x10]
        req_cells_b = [0x10]
        req_bytes_a = [0x20]
        req_bytes_b = [0x20]
        tile_rows_a = [0x30]
        tile_rows_b = [0x30]
        tile_cols_a = [0x40]
        tile_cols_b = [0x40]

        err_a = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only(
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            out_a,
            out_capacity,
            out_stride,
            req_cells_a,
            req_bytes_a,
            tile_rows_a,
            tile_cols_a,
        )

        err_b = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            DEFAULT_TILE_M,
            DEFAULT_TILE_N,
            out_b,
            out_capacity,
            out_stride,
            req_cells_b,
            req_bytes_b,
            tile_rows_b,
            tile_cols_b,
        )

        assert err_a == err_b
        assert out_a == out_b
        assert req_cells_a == req_cells_b
        assert req_bytes_a == req_bytes_b
        assert tile_rows_a == tile_rows_b
        assert tile_cols_a == tile_cols_b
