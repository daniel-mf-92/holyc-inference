#!/usr/bin/env python3
"""Commit-only parity harness for IQ-1039 default-tiles diagnostics wrapper."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only import (  # noqa: E402
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only,
)
from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only_parity import (  # noqa: E402
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only_parity,
)
from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (  # noqa: E402
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    make_q4_block,
    make_q8_block,
)


U64_MAX = 0xFFFFFFFFFFFFFFFF


def try_mul_i64_nonneg(a: int, b: int) -> tuple[bool, int]:
    if a < 0 or b < 0:
        return False, 0
    if a > 0 and b > ((1 << 63) - 1) // a:
        return False, 0
    return True, a * b


def diag_ptr_overlaps_out_cells_range(diag_slot, out_cells_q32, out_cell_capacity: int) -> bool:
    if diag_slot is None or out_cells_q32 is None:
        return False
    if out_cell_capacity <= 0:
        return False
    return diag_slot is out_cells_q32


def matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
        out_required_out_cells is lhs_q4_blocks
        or out_required_out_cells is rhs_q8_col_blocks
        or out_required_out_bytes is lhs_q4_blocks
        or out_required_out_bytes is rhs_q8_col_blocks
        or out_tile_rows is lhs_q4_blocks
        or out_tile_rows is rhs_q8_col_blocks
        or out_tile_cols is lhs_q4_blocks
        or out_tile_cols is rhs_q8_col_blocks
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

    staged_parity_required_out_cells = [0]
    staged_parity_required_out_bytes = [0]
    staged_parity_tile_rows = [0]
    staged_parity_tile_cols = [0]

    canonical_required_out_cells = [0]
    canonical_required_out_bytes = [0]
    canonical_tile_rows = [0]
    canonical_tile_cols = [0]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only_parity(
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
        staged_parity_required_out_cells,
        staged_parity_required_out_bytes,
        staged_parity_tile_rows,
        staged_parity_tile_cols,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only(
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
        canonical_required_out_cells,
        canonical_required_out_bytes,
        canonical_tile_rows,
        canonical_tile_cols,
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

    ok, staged_expected_out_bytes = try_mul_i64_nonneg(staged_parity_required_out_cells[0], 8)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    if staged_parity_required_out_bytes[0] != staged_expected_out_bytes:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, canonical_expected_out_bytes = try_mul_i64_nonneg(canonical_required_out_cells[0], 8)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    if canonical_required_out_bytes[0] != canonical_expected_out_bytes:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    if (
        staged_parity_required_out_cells[0] != canonical_required_out_cells[0]
        or staged_parity_required_out_bytes[0] != canonical_required_out_bytes[0]
        or staged_parity_tile_rows[0] != canonical_tile_rows[0]
        or staged_parity_tile_cols[0] != canonical_tile_cols[0]
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_required_out_cells[0] = staged_parity_required_out_cells[0]
    out_required_out_bytes[0] = staged_parity_required_out_bytes[0]
    out_tile_rows[0] = staged_parity_tile_rows[0]
    out_tile_cols[0] = staged_parity_tile_cols[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        *args
    )


def test_source_contains_iq1039_signature_and_contract() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = (
        "I32 "
        "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTilesPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
    )
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI32 ", 1)[0]

    assert "// IQ-1039 commit-only parity wrapper:" in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTilesPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTilesPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "if (snapshot_row_count != row_count ||" in body
    assert "if (!Q4_0Q8_0MatMulTryMulI64NonNeg(staged_parity_required_out_cells," in body
    assert "if (!Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_required_out_cells," in body
    assert "if (staged_parity_required_out_cells != canonical_required_out_cells ||" in body
    assert "*out_required_out_cells = staged_parity_required_out_cells;" in body


def test_known_vector_success_and_alias_rejection() -> None:
    rng = random.Random(202604221039)

    row_count = 5
    col_count = 4
    k_block_count = 3
    lhs_stride = 4
    rhs_stride = 4
    out_stride = 6

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]

    out_capacity = row_count * out_stride
    out = [0xC0DE] * out_capacity
    out_before = list(out)

    req_cells = [0x1111]
    req_bytes = [0x2222]
    tile_rows = [0x3333]
    tile_cols = [0x4444]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
    assert tile_rows == [4]
    assert tile_cols == [4]
    assert out == out_before

    bad = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
        out,
        req_bytes,
        tile_rows,
        tile_cols,
    )
    assert bad == Q4_0_Q8_0_AVX2_ERR_BAD_LEN


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(2026042210392)

    for i in range(1000):
        row_count = rng.randint(0, 9)
        col_count = rng.randint(0, 9)
        k_block_count = rng.randint(0, 6)
        lhs_stride = k_block_count + rng.randint(0, 3)
        rhs_stride = k_block_count + rng.randint(0, 3)
        out_stride = col_count + rng.randint(0, 3)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        if rng.random() < 0.22:
            lhs_capacity = max(0, lhs_capacity - rng.randint(1, 4))
        if rng.random() < 0.22:
            rhs_capacity = max(0, rhs_capacity - rng.randint(1, 4))
        if rng.random() < 0.22:
            out_capacity = max(0, out_capacity - rng.randint(1, 4))

        if rng.random() < 0.08:
            row_count = rng.choice([-1, -2])
        if rng.random() < 0.08:
            col_count = rng.choice([-1, -3])
        if rng.random() < 0.08:
            k_block_count = rng.choice([-1, -4])
        if rng.random() < 0.08:
            out_stride = rng.choice([-1, -2])

        local_rng = random.Random(202604221039200 + i)
        lhs = [make_q4_block(local_rng) for _ in range(max(1, abs(row_count) * max(1, abs(lhs_stride))))]
        rhs = [make_q8_block(local_rng) for _ in range(max(1, abs(col_count) * max(1, abs(rhs_stride))))]
        out = [local_rng.randint(-2000, 2000) for _ in range(max(1, abs(out_capacity)))]

        req_cells_a = [0x1111111111111111]
        req_bytes_a = [0x2222222222222222]
        tile_rows_a = [0x3333333333333333]
        tile_cols_a = [0x4444444444444444]

        req_cells_b = list(req_cells_a)
        req_bytes_b = list(req_bytes_a)
        tile_rows_b = list(tile_rows_a)
        tile_cols_b = list(tile_cols_a)

        err_a = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
            out,
            out_capacity,
            out_stride,
            req_cells_b,
            req_bytes_b,
            tile_rows_b,
            tile_cols_b,
        )

        assert err_a == err_b
        assert req_cells_a == req_cells_b
        assert req_bytes_a == req_bytes_b
        assert tile_rows_a == tile_rows_b
        assert tile_cols_a == tile_cols_b


if __name__ == "__main__":
    test_source_contains_iq1039_signature_and_contract()
    test_known_vector_success_and_alias_rejection()
    test_randomized_parity_vs_explicit_composition()
    print(
        "matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_default_tiles_preflight_only_parity_commit_only_preflight_only_parity_commit_only=ok"
    )
