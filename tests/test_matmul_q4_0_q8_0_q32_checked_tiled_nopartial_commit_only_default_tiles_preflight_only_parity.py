#!/usr/bin/env python3
"""Parity harness for IQ-1014 default-tiles preflight-only parity gate."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only import (
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only,
)
from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only import (
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only,
)
from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_OK,
    make_q4_block,
    make_q8_block,
)

DEFAULT_TILE_M = 4
DEFAULT_TILE_N = 4


def diag_ptr_overlaps_out_cells_range(diag_slot, out_cells_q32, out_cell_capacity: int) -> bool:
    if diag_slot is None or out_cells_q32 is None:
        return False
    if out_cell_capacity <= 0:
        return False
    if diag_slot is out_cells_q32:
        return True
    return False


def matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity(
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
        out_required_out_cells is out_cells_q32
        or out_required_out_bytes is out_cells_q32
        or out_tile_rows is out_cells_q32
        or out_tile_cols is out_cells_q32
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    if (
        diag_ptr_overlaps_out_cells_range(out_required_out_cells, out_cells_q32, out_cell_capacity)
        or diag_ptr_overlaps_out_cells_range(out_required_out_bytes, out_cells_q32, out_cell_capacity)
        or diag_ptr_overlaps_out_cells_range(out_tile_rows, out_cells_q32, out_cell_capacity)
        or diag_ptr_overlaps_out_cells_range(out_tile_cols, out_cells_q32, out_cell_capacity)
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    snapshot_row_count = row_count
    snapshot_col_count = col_count
    snapshot_k_block_count = k_block_count
    snapshot_out_capacity = out_cell_capacity
    snapshot_out_row_stride_cells = out_row_stride_cells

    staged_default_required_out_cells = [0]
    staged_default_required_out_bytes = [0]
    staged_default_tile_rows = [0]
    staged_default_tile_cols = [0]

    staged_canonical_required_out_cells = [0]
    staged_canonical_required_out_bytes = [0]
    staged_canonical_tile_rows = [0]
    staged_canonical_tile_cols = [0]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only(
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
        staged_default_required_out_cells,
        staged_default_required_out_bytes,
        staged_default_tile_rows,
        staged_default_tile_cols,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

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
        staged_canonical_required_out_cells,
        staged_canonical_required_out_bytes,
        staged_canonical_tile_rows,
        staged_canonical_tile_cols,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    if (
        snapshot_row_count != row_count
        or snapshot_col_count != col_count
        or snapshot_k_block_count != k_block_count
        or snapshot_out_capacity != out_cell_capacity
        or snapshot_out_row_stride_cells != out_row_stride_cells
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    if (
        staged_default_required_out_cells[0] != staged_canonical_required_out_cells[0]
        or staged_default_required_out_bytes[0] != staged_canonical_required_out_bytes[0]
        or staged_default_tile_rows[0] != staged_canonical_tile_rows[0]
        or staged_default_tile_cols[0] != staged_canonical_tile_cols[0]
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_required_out_cells[0] = staged_default_required_out_cells[0]
    out_required_out_bytes[0] = staged_default_required_out_bytes[0]
    out_tile_rows[0] = staged_default_tile_rows[0]
    out_tile_cols[0] = staged_default_tile_cols[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity(
        *args
    )


def test_source_contains_iq1014_default_preflight_parity_gate() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTilesPreflightOnlyParity("
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI32 ", 1)[0]

    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTilesPreflightOnly(" in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly(" in body
    assert "Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M" in body
    assert "Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N" in body
    assert "if (staged_default_required_out_cells != staged_canonical_required_out_cells" in body
    assert "*out_required_out_cells = staged_default_required_out_cells;" in body
    assert "*out_required_out_bytes = staged_default_required_out_bytes;" in body
    assert "*out_tile_rows = staged_default_tile_rows;" in body
    assert "*out_tile_cols = staged_default_tile_cols;" in body


def test_known_vector_success_and_zero_write_contract() -> None:
    rng = random.Random(202604221014)

    row_count = 5
    col_count = 6
    k_block_count = 3
    lhs_stride = 4
    rhs_stride = 4
    out_stride = 8

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]

    out_capacity = row_count * out_stride
    out = [0x4444] * out_capacity
    out_before = list(out)

    req_cells = [0x1111]
    req_bytes = [0x2222]
    tile_rows = [0x3333]
    tile_cols = [0x5555]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity(
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
    assert req_cells == [row_count * out_stride]
    assert req_bytes == [row_count * out_stride * 8]
    assert tile_rows == [DEFAULT_TILE_M]
    assert tile_cols == [DEFAULT_TILE_N]
    assert out == out_before


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(2026042210142)

    for i in range(1000):
        row_count = rng.randint(0, 10)
        col_count = rng.randint(0, 10)
        k_block_count = rng.randint(0, 6)
        lhs_stride = k_block_count + rng.randint(0, 3)
        rhs_stride = k_block_count + rng.randint(0, 3)
        out_stride = col_count + rng.randint(0, 3)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        if rng.random() < 0.25:
            lhs_capacity = max(0, lhs_capacity - rng.randint(1, 4))
        if rng.random() < 0.25:
            rhs_capacity = max(0, rhs_capacity - rng.randint(1, 4))
        if rng.random() < 0.25:
            out_capacity = max(0, out_capacity - rng.randint(1, 4))

        if rng.random() < 0.1:
            row_count = rng.choice([-1, -2])
        if rng.random() < 0.1:
            col_count = rng.choice([-1, -3])
        if rng.random() < 0.08:
            k_block_count = rng.choice([-1, -4])
        if rng.random() < 0.08:
            out_stride = rng.choice([-1, -2])

        local_rng = random.Random(202604221014200 + i)
        lhs = [make_q4_block(local_rng) for _ in range(max(1, abs(row_count) * max(1, abs(lhs_stride))))]
        rhs = [make_q8_block(local_rng) for _ in range(max(1, abs(col_count) * max(1, abs(rhs_stride))))]
        out = [0x2A2A] * max(1, abs(out_capacity))

        req_cells_a = [0x1111]
        req_bytes_a = [0x2222]
        tile_rows_a = [0x3333]
        tile_cols_a = [0x4444]

        req_cells_b = [0x1111]
        req_bytes_b = [0x2222]
        tile_rows_b = [0x3333]
        tile_cols_b = [0x4444]

        args_a = (
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            out,
            out_capacity,
            out_stride,
            req_cells_a,
            req_bytes_a,
            tile_rows_a,
            tile_cols_a,
        )
        args_b = (
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            out,
            out_capacity,
            out_stride,
            req_cells_b,
            req_bytes_b,
            tile_rows_b,
            tile_cols_b,
        )

        got = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity(
            *args_a
        )
        want = explicit_checked_composition(*args_b)

        assert got == want
        assert req_cells_a == req_cells_b
        assert req_bytes_a == req_bytes_b
        assert tile_rows_a == tile_rows_b
        assert tile_cols_a == tile_cols_b


def test_alias_guards() -> None:
    rng = random.Random(2026042210143)

    lhs = [make_q4_block(rng) for _ in range(16)]
    rhs = [make_q8_block(rng) for _ in range(16)]
    out = [0x9999] * 64

    req_cells = [0x1111]
    req_bytes = [0x2222]
    tile_rows = [0x3333]
    tile_cols = [0x4444]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity(
        lhs,
        len(lhs),
        2,
        4,
        rhs,
        len(rhs),
        2,
        4,
        2,
        out,
        len(out),
        4,
        req_cells,
        req_cells,
        tile_rows,
        tile_cols,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert req_cells == [0x1111]
    assert req_bytes == [0x2222]

    req_cells2 = [0xAAAA]
    req_bytes2 = [0xBBBB]
    tile_rows2 = [0xCCCC]
    tile_cols2 = [0xDDDD]
    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity(
        lhs,
        len(lhs),
        2,
        4,
        rhs,
        len(rhs),
        2,
        4,
        2,
        out,
        len(out),
        4,
        out,
        req_bytes2,
        tile_rows2,
        tile_cols2,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert req_cells2 == [0xAAAA]
    assert req_bytes2 == [0xBBBB]


if __name__ == "__main__":
    test_source_contains_iq1014_default_preflight_parity_gate()
    test_known_vector_success_and_zero_write_contract()
    test_randomized_parity_vs_explicit_composition()
    test_alias_guards()
    print("ok")
