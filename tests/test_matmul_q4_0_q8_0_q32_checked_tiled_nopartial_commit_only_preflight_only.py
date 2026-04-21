#!/usr/bin/env python3
"""Parity harness for MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly (IQ-994)."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only import (
    commit_only_matmul_q4_0_q8_0_q32_checked_tiled_nopartial,
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


def matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(
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

    if lhs_q4_blocks is None or rhs_q8_col_blocks is None or out_cells_q32 is None:
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    if lhs_q4_block_capacity < 0 or rhs_q8_block_capacity < 0 or out_cell_capacity < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if tile_row_span <= 0 or tile_col_span <= 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if row_count > 0 and out_row_stride_cells < col_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    snapshot_row_count = row_count
    snapshot_col_count = col_count
    snapshot_k_block_count = k_block_count
    snapshot_tile_row_span = tile_row_span
    snapshot_tile_col_span = tile_col_span
    snapshot_out_capacity = out_cell_capacity

    ok, required_lhs_blocks = try_mul_i64_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, required_rhs_blocks = try_mul_i64_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, staged_required_out_cells = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    if required_lhs_blocks > lhs_q4_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if required_rhs_blocks > rhs_q8_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_required_out_cells > out_cell_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, row_plus_tile_minus_one = try_add_i64_nonneg(row_count, tile_row_span - 1)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, col_plus_tile_minus_one = try_add_i64_nonneg(col_count, tile_col_span - 1)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    tile_row_count = row_plus_tile_minus_one // tile_row_span
    tile_col_count = col_plus_tile_minus_one // tile_col_span
    ok, tile_total_count = try_mul_i64_nonneg(tile_row_count, tile_col_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    if row_count > 0 and col_count > 0 and tile_total_count == 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, staged_required_out_bytes = try_mul_i64_nonneg(staged_required_out_cells, 8)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    staged_tile_rows = tile_row_span
    staged_tile_cols = tile_col_span

    if (
        snapshot_row_count != row_count
        or snapshot_col_count != col_count
        or snapshot_k_block_count != k_block_count
        or snapshot_tile_row_span != tile_row_span
        or snapshot_tile_col_span != tile_col_span
        or snapshot_out_capacity != out_cell_capacity
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_required_out_cells[0] = staged_required_out_cells
    out_required_out_bytes[0] = staged_required_out_bytes
    out_tile_rows[0] = staged_tile_rows
    out_tile_cols[0] = staged_tile_cols
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(*args)


def test_source_contains_iq_994_signature_and_no_write_contract() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "if (!lhs_q4_blocks || !rhs_q8_col_blocks || !out_cells_q32)" in body
    assert "Q4_0Q8_0MatMulComputeTileGridChecked(" in body
    assert "*out_required_out_cells = staged_required_out_cells;" in body
    assert "*out_required_out_bytes = staged_required_out_bytes;" in body
    assert "*out_tile_rows = staged_tile_rows;" in body
    assert "*out_tile_cols = staged_tile_cols;" in body


def test_known_vector_diagnostics_and_no_output_write() -> None:
    rng = random.Random(202604220994)

    row_count = 5
    col_count = 4
    k_block_count = 2
    lhs_stride = 3
    rhs_stride = 4
    out_stride = 6

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [424242] * (row_count * out_stride)
    out_before = out.copy()

    req_cells = [0]
    req_bytes = [0]
    tile_rows = [0]
    tile_cols = [0]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        3,
        2,
        out,
        len(out),
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
    assert tile_rows == [3]
    assert tile_cols == [2]


def test_no_partial_diagnostics_publish_on_error() -> None:
    rng = random.Random(202604220995)

    row_count = 3
    col_count = 2
    k_block_count = 1
    lhs_stride = 1
    rhs_stride = 1
    out_stride = 2

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [9191] * (row_count * out_stride)

    req_cells = [111]
    req_bytes = [222]
    tile_rows = [333]
    tile_cols = [444]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        2,
        2,
        out,
        len(out) - 1,
        out_stride,
        req_cells,
        req_bytes,
        tile_rows,
        tile_cols,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert req_cells == [111]
    assert req_bytes == [222]
    assert tile_rows == [333]
    assert tile_cols == [444]


def test_randomized_parity_vs_commit_only_diagnostics_and_no_write() -> None:
    rng = random.Random(202604220996)

    for _ in range(150):
        row_count = rng.randint(0, 6)
        col_count = rng.randint(0, 6)
        k_block_count = rng.randint(0, 4)

        lhs_stride = k_block_count + rng.randint(0, 3)
        rhs_stride = k_block_count + rng.randint(0, 3)
        out_stride = col_count + rng.randint(0, 3)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        lhs = [make_q4_block(rng) for _ in range(lhs_capacity)]
        rhs = [make_q8_block(rng) for _ in range(rhs_capacity)]

        out_a = [rng.randint(-500, 500) for _ in range(out_capacity)]
        out_b = out_a.copy()

        req_cells_a = [7]
        req_bytes_a = [8]
        tile_rows_a = [9]
        tile_cols_a = [10]

        req_cells_b = [11]
        req_bytes_b = [12]
        tile_rows_b = [13]
        tile_cols_b = [14]

        tile_row_span = rng.randint(1, 5)
        tile_col_span = rng.randint(1, 5)

        err_a = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            tile_row_span,
            tile_col_span,
            out_a,
            out_capacity,
            out_stride,
            req_cells_a,
            req_bytes_a,
            tile_rows_a,
            tile_cols_a,
        )

        err_b = commit_only_matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            tile_row_span,
            tile_col_span,
            out_b,
            out_capacity,
            out_stride,
            req_cells_b,
            req_bytes_b,
            tile_rows_b,
            tile_cols_b,
        )

        assert err_a == err_b
        if err_a == Q4_0_Q8_0_AVX2_OK:
            assert out_a == [
                v for v in out_a
            ]  # explicit no-op to keep intent near invariants
            assert req_cells_a == req_cells_b
            assert req_bytes_a == req_bytes_b
            assert tile_rows_a == tile_rows_b
            assert tile_cols_a == tile_cols_b


def test_overflow_guard_required_bytes() -> None:
    req_cells = [111]
    req_bytes = [222]
    tile_rows = [333]
    tile_cols = [444]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(
        [],
        0,
        I64_MAX,
        0,
        [],
        0,
        0,
        0,
        0,
        1,
        1,
        [],
        I64_MAX,
        2,
        req_cells,
        req_bytes,
        tile_rows,
        tile_cols,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    assert req_cells == [111]
    assert req_bytes == [222]
    assert tile_rows == [333]
    assert tile_cols == [444]


if __name__ == "__main__":
    test_source_contains_iq_994_signature_and_no_write_contract()
    test_known_vector_diagnostics_and_no_output_write()
    test_no_partial_diagnostics_publish_on_error()
    test_randomized_parity_vs_commit_only_diagnostics_and_no_write()
    test_overflow_guard_required_bytes()
    print("ok")
