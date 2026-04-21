#!/usr/bin/env python3
"""Parity harness for MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnly (IQ-993)."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial import (
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial,
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


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def commit_only_matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
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

    snapshot_row_count = row_count
    snapshot_col_count = col_count
    snapshot_k_block_count = k_block_count
    snapshot_tile_row_span = tile_row_span
    snapshot_tile_col_span = tile_col_span
    snapshot_out_capacity = out_cell_capacity

    staged_required_out_cells = [out_required_out_cells[0]]
    staged_required_out_bytes = [out_required_out_bytes[0]]
    staged_tile_rows = [out_tile_rows[0]]
    staged_tile_cols = [out_tile_cols[0]]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
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
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    ok, required_out_cells = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    if required_out_cells > out_cell_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, required_out_bytes = try_mul_i64_nonneg(required_out_cells, 8)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    staged_required_out_cells[0] = required_out_cells
    staged_required_out_bytes[0] = required_out_bytes
    staged_tile_rows[0] = tile_row_span
    staged_tile_cols[0] = tile_col_span

    if snapshot_row_count != row_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_col_count != col_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_k_block_count != k_block_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_tile_row_span != tile_row_span:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_tile_col_span != tile_col_span:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_out_capacity != out_cell_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_required_out_cells[0] = staged_required_out_cells[0]
    out_required_out_bytes[0] = staged_required_out_bytes[0]
    out_tile_rows[0] = staged_tile_rows[0]
    out_tile_cols[0] = staged_tile_cols[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return commit_only_matmul_q4_0_q8_0_q32_checked_tiled_nopartial(*args)


def test_source_contains_iq_993_signature_and_commit_atomic_publish() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnly("
    assert sig in source
    body = source.rsplit(sig, 1)[1]
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartial(" in body
    assert "snapshot_tile_rows" in body
    assert "snapshot_tile_cols" in body
    assert "*out_required_out_cells = staged_required_out_cells;" in body
    assert "*out_required_out_bytes = staged_required_out_bytes;" in body
    assert "*out_tile_rows = staged_tile_rows;" in body
    assert "*out_tile_cols = staged_tile_cols;" in body


def test_known_vector_and_no_partial_diagnostics_on_error() -> None:
    rng = random.Random(202604220993)

    row_count = 4
    col_count = 3
    k_block_count = 2
    lhs_stride = 3
    rhs_stride = 3
    out_stride = 5
    tile_rows = 3
    tile_cols = 2

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]

    out_capacity = row_count * out_stride
    out = [31337] * out_capacity
    req_cells = [777]
    req_bytes = [888]
    got_tile_rows = [999]
    got_tile_cols = [111]

    err = commit_only_matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
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

    out_fail = [1212] * out_capacity
    req_cells_fail = [501]
    req_bytes_fail = [502]
    got_tile_rows_fail = [503]
    got_tile_cols_fail = [504]
    err = commit_only_matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
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
        out_fail,
        out_capacity - 1,
        out_stride,
        req_cells_fail,
        req_bytes_fail,
        got_tile_rows_fail,
        got_tile_cols_fail,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert out_fail == [1212] * out_capacity
    assert req_cells_fail == [501]
    assert req_bytes_fail == [502]
    assert got_tile_rows_fail == [503]
    assert got_tile_cols_fail == [504]


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(202604220994)

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

        local_rng = random.Random(20260422099400 + i)
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

        err_a = commit_only_matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
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


def test_overflow_guard_for_required_out_bytes() -> None:
    req_cells = [11]
    req_bytes = [22]
    tile_rows = [33]
    tile_cols = [44]

    err = commit_only_matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
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
    assert req_cells == [11]
    assert req_bytes == [22]
    assert tile_rows == [33]
    assert tile_cols == [44]


if __name__ == "__main__":
    test_source_contains_iq_993_signature_and_commit_atomic_publish()
    test_known_vector_and_no_partial_diagnostics_on_error()
    test_randomized_parity_against_explicit_composition()
    test_overflow_guard_for_required_out_bytes()
    print("ok")
