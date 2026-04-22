#!/usr/bin/env python3
"""Parity harness for ...PreflightOnlyParityCommitOnlyPreflightOnlyParity (IQ-1006)."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only import (  # noqa: E402
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only,
)
from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only import (  # noqa: E402
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only,
)
from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (  # noqa: E402
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    make_q4_block,
    make_q8_block,
)


def matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
    lhs_q4_blocks,
    lhs_q4_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_q8_col_blocks,
    rhs_q8_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    tile_row_span: int,
    tile_col_span: int,
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

    snapshot_lhs_q4_block_capacity = lhs_q4_block_capacity
    snapshot_row_count = row_count
    snapshot_lhs_row_stride_blocks = lhs_row_stride_blocks
    snapshot_rhs_q8_block_capacity = rhs_q8_block_capacity
    snapshot_col_count = col_count
    snapshot_rhs_col_stride_blocks = rhs_col_stride_blocks
    snapshot_k_block_count = k_block_count
    snapshot_tile_row_span = tile_row_span
    snapshot_tile_col_span = tile_col_span
    snapshot_out_capacity = out_cell_capacity
    snapshot_out_row_stride_cells = out_row_stride_cells

    preflight_required_out_cells = [0]
    preflight_required_out_bytes = [0]
    preflight_tile_rows = [0]
    preflight_tile_cols = [0]

    canonical_required_out_cells = [0]
    canonical_required_out_bytes = [0]
    canonical_tile_rows = [0]
    canonical_tile_cols = [0]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        tile_row_span,
        tile_col_span,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        preflight_required_out_cells,
        preflight_required_out_bytes,
        preflight_tile_rows,
        preflight_tile_cols,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        tile_row_span,
        tile_col_span,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        canonical_required_out_cells,
        canonical_required_out_bytes,
        canonical_tile_rows,
        canonical_tile_cols,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    if (
        snapshot_lhs_q4_block_capacity != lhs_q4_block_capacity
        or snapshot_row_count != row_count
        or snapshot_lhs_row_stride_blocks != lhs_row_stride_blocks
        or snapshot_rhs_q8_block_capacity != rhs_q8_block_capacity
        or snapshot_col_count != col_count
        or snapshot_rhs_col_stride_blocks != rhs_col_stride_blocks
        or snapshot_k_block_count != k_block_count
        or snapshot_tile_row_span != tile_row_span
        or snapshot_tile_col_span != tile_col_span
        or snapshot_out_capacity != out_cell_capacity
        or snapshot_out_row_stride_cells != out_row_stride_cells
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    if (
        preflight_required_out_cells[0] != canonical_required_out_cells[0]
        or preflight_required_out_bytes[0] != canonical_required_out_bytes[0]
        or preflight_tile_rows[0] != canonical_tile_rows[0]
        or preflight_tile_cols[0] != canonical_tile_cols[0]
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_required_out_cells[0] = canonical_required_out_cells[0]
    out_required_out_bytes[0] = canonical_required_out_bytes[0]
    out_tile_rows[0] = canonical_tile_rows[0]
    out_tile_cols[0] = canonical_tile_cols[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    preflight_required_out_cells = [0x11]
    preflight_required_out_bytes = [0x12]
    preflight_tile_rows = [0x13]
    preflight_tile_cols = [0x14]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        *args[:-4],
        preflight_required_out_cells,
        preflight_required_out_bytes,
        preflight_tile_rows,
        preflight_tile_cols,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    canonical_required_out_cells = [0x21]
    canonical_required_out_bytes = [0x22]
    canonical_tile_rows = [0x23]
    canonical_tile_cols = [0x24]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only(
        *args[:-4],
        canonical_required_out_cells,
        canonical_required_out_bytes,
        canonical_tile_rows,
        canonical_tile_cols,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    if (
        preflight_required_out_cells[0] != canonical_required_out_cells[0]
        or preflight_required_out_bytes[0] != canonical_required_out_bytes[0]
        or preflight_tile_rows[0] != canonical_tile_rows[0]
        or preflight_tile_cols[0] != canonical_tile_cols[0]
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_required_out_cells = args[-4]
    out_required_out_bytes = args[-3]
    out_tile_rows = args[-2]
    out_tile_cols = args[-1]

    out_required_out_cells[0] = canonical_required_out_cells[0]
    out_required_out_bytes[0] = canonical_required_out_bytes[0]
    out_tile_rows[0] = canonical_tile_rows[0]
    out_tile_cols[0] = canonical_tile_cols[0]
    return Q4_0_Q8_0_AVX2_OK


def test_source_contains_iq1006_signature_and_parity_gate_contract() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = (
        "I32 "
        "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    )
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI32 ", 1)[0]

    assert "// IQ-1006 strict parity gate:" in body
    assert (
        "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    )
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (preflight_required_out_cells != canonical_required_out_cells ||" in body
    assert "if (snapshot_lhs_q4_block_capacity != lhs_q4_block_capacity ||" in body
    assert "snapshot_rhs_q8_block_capacity != rhs_q8_block_capacity ||" in body
    assert "snapshot_out_row_stride_cells != out_row_stride_cells)" in body
    assert "*out_required_out_cells = canonical_required_out_cells;" in body
    assert "*out_required_out_bytes = canonical_required_out_bytes;" in body
    assert "*out_tile_rows = canonical_tile_rows;" in body
    assert "*out_tile_cols = canonical_tile_cols;" in body


def test_known_vector_success_and_null_rejection() -> None:
    rng = random.Random(2026042210061)

    row_count = 5
    col_count = 4
    k_block_count = 3
    lhs_stride = 4
    rhs_stride = 5
    out_stride = 8
    tile_rows = 2
    tile_cols = 3

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out_capacity = row_count * out_stride

    out = [0x6161] * out_capacity
    req_cells = [0x1111]
    req_bytes = [0x2222]
    got_tile_rows = [0x3333]
    got_tile_cols = [0x4444]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        tile_rows,
        tile_cols,
        out,
        out_capacity,
        out_stride,
        req_cells,
        req_bytes,
        got_tile_rows,
        got_tile_cols,
    )
    assert err == Q4_0_Q8_0_AVX2_OK
    assert req_cells == [row_count * out_stride]
    assert req_bytes == [row_count * out_stride * 8]
    assert got_tile_rows == [tile_rows]
    assert got_tile_cols == [tile_cols]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        tile_rows,
        tile_cols,
        out,
        out_capacity,
        out_stride,
        None,
        req_bytes,
        got_tile_rows,
        got_tile_cols,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_NULL_PTR


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(2026042210062)

    for i in range(700):
        row_count = rng.randint(0, 8)
        col_count = rng.randint(0, 8)
        k_block_count = rng.randint(0, 5)
        lhs_stride = k_block_count + rng.randint(0, 3)
        rhs_stride = k_block_count + rng.randint(0, 3)
        out_stride = col_count + rng.randint(0, 3)
        tile_rows = rng.randint(1, 5)
        tile_cols = rng.randint(1, 5)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        if rng.random() < 0.2:
            lhs_capacity = max(0, lhs_capacity - rng.randint(1, 2))
        if rng.random() < 0.2:
            rhs_capacity = max(0, rhs_capacity - rng.randint(1, 2))
        if rng.random() < 0.2:
            out_capacity = max(0, out_capacity - rng.randint(1, 2))

        local_rng = random.Random(202604221006200 + i)
        lhs = [make_q4_block(local_rng) for _ in range(max(1, row_count * max(1, lhs_stride)))]
        rhs = [make_q8_block(local_rng) for _ in range(max(1, col_count * max(1, rhs_stride)))]

        out_a = [0xAAAA] * max(1, row_count * max(1, out_stride))
        out_b = [0xAAAA] * len(out_a)

        req_cells_a = [0x10]
        req_cells_b = [0x10]
        req_bytes_a = [0x20]
        req_bytes_b = [0x20]
        tile_rows_a = [0x30]
        tile_rows_b = [0x30]
        tile_cols_a = [0x40]
        tile_cols_b = [0x40]

        err_a = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            tile_rows,
            tile_cols,
            out_a,
            out_capacity,
            out_stride,
            req_cells_a,
            req_bytes_a,
            tile_rows_a,
            tile_cols_a,
        )

        err_b = explicit_checked_composition(
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            tile_rows,
            tile_cols,
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


def test_overflow_vector_keeps_diagnostics_unpublished() -> None:
    req_cells = [801]
    req_bytes = [802]
    tile_rows = [803]
    tile_cols = [804]
    out = [0x9999] * 8

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        [],
        1 << 62,
        1 << 62,
        3,
        [],
        1 << 62,
        2,
        3,
        1,
        2,
        2,
        out,
        1 << 62,
        1 << 62,
        req_cells,
        req_bytes,
        tile_rows,
        tile_cols,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    assert req_cells == [801]
    assert req_bytes == [802]
    assert tile_rows == [803]
    assert tile_cols == [804]
