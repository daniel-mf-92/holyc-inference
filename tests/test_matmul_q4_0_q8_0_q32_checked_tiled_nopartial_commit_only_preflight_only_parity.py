#!/usr/bin/env python3
"""Parity harness for MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParity (IQ-995)."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial import (  # noqa: E402
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial,
)
from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only import (  # noqa: E402
    commit_only_matmul_q4_0_q8_0_q32_checked_tiled_nopartial,
)
from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only import (  # noqa: E402
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only,
)
from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (  # noqa: E402
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    make_q4_block,
    make_q8_block,
)


def matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity(
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

    snapshot_row_count = row_count
    snapshot_col_count = col_count
    snapshot_k_block_count = k_block_count
    snapshot_tile_row_span = tile_row_span
    snapshot_tile_col_span = tile_col_span
    snapshot_out_capacity = out_cell_capacity

    staged_pre_required_out_cells = [0]
    staged_pre_required_out_bytes = [0]
    staged_pre_tile_rows = [0]
    staged_pre_tile_cols = [0]

    staged_commit_required_out_cells = [0]
    staged_commit_required_out_bytes = [0]
    staged_commit_tile_rows = [0]
    staged_commit_tile_cols = [0]

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
        tile_row_span,
        tile_col_span,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        staged_pre_required_out_cells,
        staged_pre_required_out_bytes,
        staged_pre_tile_rows,
        staged_pre_tile_cols,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    err = commit_only_matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
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
        staged_commit_required_out_cells,
        staged_commit_required_out_bytes,
        staged_commit_tile_rows,
        staged_commit_tile_cols,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

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

    if staged_pre_required_out_cells[0] != staged_commit_required_out_cells[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_pre_required_out_bytes[0] != staged_commit_required_out_bytes[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_pre_tile_rows[0] != staged_commit_tile_rows[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_pre_tile_cols[0] != staged_commit_tile_cols[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_required_out_cells[0] = staged_pre_required_out_cells[0]
    out_required_out_bytes[0] = staged_pre_required_out_bytes[0]
    out_tile_rows[0] = staged_pre_tile_rows[0]
    out_tile_cols[0] = staged_pre_tile_cols[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity(*args)


def test_source_contains_iq995_signature_and_parity_contract() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.rsplit(sig, 1)[1]

    assert "// IQ-995 diagnostics-only parity gate:" in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly(" in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnly(" in body
    assert "if (staged_pre_required_out_cells != staged_commit_required_out_cells" in body
    assert "staged_pre_required_out_bytes != staged_commit_required_out_bytes" in body
    assert "staged_pre_tile_rows != staged_commit_tile_rows" in body
    assert "staged_pre_tile_cols != staged_commit_tile_cols" in body
    assert "*out_required_out_cells = staged_pre_required_out_cells;" in body
    assert "*out_required_out_bytes = staged_pre_required_out_bytes;" in body
    assert "*out_tile_rows = staged_pre_tile_rows;" in body
    assert "*out_tile_cols = staged_pre_tile_cols;" in body


def test_known_vector_success_and_bad_capacity_no_partial_diagnostics() -> None:
    rng = random.Random(20260422_9951)

    row_count = 4
    col_count = 5
    k_block_count = 3
    lhs_stride = 4
    rhs_stride = 5
    out_stride = 7
    tile_rows = 2
    tile_cols = 3

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out_capacity = row_count * out_stride

    out = [0x5151] * out_capacity
    req_cells = [0x1111]
    req_bytes = [0x2222]
    got_tile_rows = [0x3333]
    got_tile_cols = [0x4444]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity(
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

    out_fail = [0x8181] * out_capacity
    out_fail_before = list(out_fail)
    req_cells_fail = [91]
    req_bytes_fail = [92]
    got_tile_rows_fail = [93]
    got_tile_cols_fail = [94]
    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity(
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
    assert out_fail == out_fail_before
    assert req_cells_fail == [91]
    assert req_bytes_fail == [92]
    assert got_tile_rows_fail == [93]
    assert got_tile_cols_fail == [94]


def test_parity_matches_commit_only_and_base_output() -> None:
    rng = random.Random(20260422_9952)

    for i in range(800):
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

        local_rng = random.Random(20260422_995200 + i)
        lhs = [make_q4_block(local_rng) for _ in range(max(1, row_count * lhs_stride))]
        rhs = [make_q8_block(local_rng) for _ in range(max(1, col_count * rhs_stride))]

        out_a = [0xAAAA] * max(1, row_count * max(1, out_stride))
        out_b = [0xAAAA] * len(out_a)
        out_c = [0xAAAA] * len(out_a)

        req_cells_a = [0x10]
        req_cells_b = [0x10]
        req_bytes_a = [0x20]
        req_bytes_b = [0x20]
        tile_rows_a = [0x30]
        tile_rows_b = [0x30]
        tile_cols_a = [0x40]
        tile_cols_b = [0x40]

        err_a = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity(
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
        err_base = matmul_q4_0_q8_0_q32_checked_tiled_nopartial(
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
            out_c,
            out_capacity,
            out_stride,
        )

        assert err_a == err_b
        assert out_a == out_b
        assert req_cells_a == req_cells_b
        assert req_bytes_a == req_bytes_b
        assert tile_rows_a == tile_rows_b
        assert tile_cols_a == tile_cols_b

        if err_a == Q4_0_Q8_0_AVX2_OK:
            assert err_base == Q4_0_Q8_0_AVX2_OK
            assert out_a == out_c
        elif err_base == Q4_0_Q8_0_AVX2_OK:
            assert err_a != Q4_0_Q8_0_AVX2_OK


def test_overflow_vector_keeps_diagnostics_unpublished() -> None:
    req_cells = [701]
    req_bytes = [702]
    tile_rows = [703]
    tile_cols = [704]
    out = [0x9999] * 8

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity(
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
    assert req_cells == [701]
    assert req_bytes == [702]
    assert tile_rows == [703]
    assert tile_cols == [704]


def test_aliasing_diagnostic_destinations_rejected() -> None:
    rng = random.Random(9954)
    lhs = [make_q4_block(rng)]
    rhs = [make_q8_block(rng)]
    out = [0]

    shared = [0]
    status = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity(
        lhs,
        1,
        1,
        1,
        rhs,
        1,
        1,
        1,
        1,
        1,
        1,
        out,
        1,
        1,
        shared,
        shared,
        [0],
        [0],
    )
    assert status == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
