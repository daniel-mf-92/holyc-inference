#!/usr/bin/env python3
"""Harness for MatMulQ4_0Q8_0Q32CheckedTiledNoPartial...ParityCommitOnlyPreflightOnly (IQ-1013)."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only import (  # noqa: E402
    matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only,
)
from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (  # noqa: E402
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


def try_add_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def compute_tile_grid_checked(
    row_count: int,
    col_count: int,
    tile_rows: int,
    tile_cols: int,
) -> tuple[bool, int, int, int]:
    if tile_rows <= 0 or tile_cols <= 0:
        return False, 0, 0, 0

    ok, row_numer = try_add_i64_nonneg(row_count, tile_rows - 1)
    if not ok:
        return False, 0, 0, 0
    ok, col_numer = try_add_i64_nonneg(col_count, tile_cols - 1)
    if not ok:
        return False, 0, 0, 0

    row_tile_count = row_numer // tile_rows
    col_tile_count = col_numer // tile_cols

    ok, tile_total_count = try_mul_i64_nonneg(row_tile_count, col_tile_count)
    if not ok:
        return False, 0, 0, 0

    return True, row_tile_count, col_tile_count, tile_total_count


def matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    snapshot_out_row_stride_cells = out_row_stride_cells

    ok, required_lhs_blocks = try_mul_i64_nonneg(snapshot_row_count, lhs_row_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, required_rhs_blocks = try_mul_i64_nonneg(snapshot_col_count, rhs_col_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, staged_required_out_cells = try_mul_i64_nonneg(snapshot_row_count, snapshot_out_row_stride_cells)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    if required_lhs_blocks > lhs_q4_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if required_rhs_blocks > rhs_q8_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_required_out_cells > snapshot_out_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, _tile_rows, _tile_cols, tile_total_count = compute_tile_grid_checked(
        snapshot_row_count,
        snapshot_col_count,
        snapshot_tile_row_span,
        snapshot_tile_col_span,
    )
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    if snapshot_row_count and snapshot_col_count and not tile_total_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, staged_required_out_bytes = try_mul_i64_nonneg(staged_required_out_cells, 8)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    staged_tile_rows = snapshot_tile_row_span
    staged_tile_cols = snapshot_tile_col_span

    if (
        snapshot_row_count != row_count
        or snapshot_col_count != col_count
        or snapshot_k_block_count != k_block_count
        or snapshot_tile_row_span != tile_row_span
        or snapshot_tile_col_span != tile_col_span
        or snapshot_out_capacity != out_cell_capacity
        or snapshot_out_row_stride_cells != out_row_stride_cells
    ):
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_required_out_cells[0] = staged_required_out_cells
    out_required_out_bytes[0] = staged_required_out_bytes
    out_tile_rows[0] = staged_tile_rows
    out_tile_cols[0] = staged_tile_cols
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only(
        *args
    )


def test_source_contains_iq1013_signature_and_no_write_recompute_contract() -> None:
    source = pathlib.Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = (
        "I32 "
        "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    )
    assert sig in source
    body = source.rsplit(sig, 1)[1].split("\nI32 ", 1)[0]

    assert "// IQ-1013 diagnostics-only no-write companion:" in body
    assert "if (!Q4_0Q8_0MatMulTryMulI64NonNeg(snapshot_row_count," in body
    assert "snapshot_lhs_q4_block_capacity = lhs_q4_block_capacity;" in body
    assert "snapshot_out_row_stride_cells = out_row_stride_cells;" in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (staged_required_out_cells != canonical_required_out_cells ||" in body
    assert "snapshot_lhs_q4_block_capacity != lhs_q4_block_capacity" in body
    assert "*out_required_out_cells = staged_required_out_cells;" in body
    assert "*out_required_out_bytes = staged_required_out_bytes;" in body
    assert "*out_tile_rows = staged_tile_rows;" in body
    assert "*out_tile_cols = staged_tile_cols;" in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly(" not in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body


def test_known_vector_success_and_null_rejection() -> None:
    rng = random.Random(2026042210051)

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

    out = [0x5151] * out_capacity
    out_before = list(out)
    req_cells = [0x1111]
    req_bytes = [0x2222]
    got_tile_rows = [0x3333]
    got_tile_cols = [0x4444]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    assert out == out_before

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    rng = random.Random(2026042210052)

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

        local_rng = random.Random(202604221005200 + i)
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

        err_a = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    req_cells = [701]
    req_bytes = [702]
    tile_rows = [703]
    tile_cols = [704]
    out = [0x9999] * 8

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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


def test_success_path_never_writes_output_payload() -> None:
    rng = random.Random(2026042210135)

    row_count = 6
    col_count = 5
    k_block_count = 3
    lhs_stride = 4
    rhs_stride = 5
    out_stride = 7

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x6A6A] * (row_count * out_stride)
    out_before = list(out)

    req_cells = [0x55]
    req_bytes = [0x66]
    tile_rows = [0x77]
    tile_cols = [0x88]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
        3,
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


def test_output_alias_rejection() -> None:
    rng = random.Random(2026042210136)

    row_count = 3
    col_count = 2
    k_block_count = 2
    lhs_stride = 2
    rhs_stride = 2
    out_stride = 4

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x5050] * (row_count * out_stride)

    diag = out
    independent = [0xAAAA]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs,
        len(lhs),
        row_count,
        lhs_stride,
        rhs,
        len(rhs),
        col_count,
        rhs_stride,
        k_block_count,
        1,
        1,
        out,
        len(out),
        out_stride,
        diag,
        independent,
        [0xBBBB],
        [0xCCCC],
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN


def test_randomized_no_publish_on_failure_contract() -> None:
    rng = random.Random(2026042210053)

    for i in range(900):
        row_count = rng.randint(0, 12)
        col_count = rng.randint(0, 12)
        k_block_count = rng.randint(0, 8)

        lhs_stride = k_block_count + rng.randint(0, 4)
        rhs_stride = k_block_count + rng.randint(0, 4)
        out_stride = col_count + rng.randint(0, 4)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        if rng.random() < 0.35:
            lhs_capacity = max(0, lhs_capacity - rng.randint(1, 3))
        if rng.random() < 0.35:
            rhs_capacity = max(0, rhs_capacity - rng.randint(1, 3))
        if rng.random() < 0.35:
            out_capacity = max(0, out_capacity - rng.randint(1, 3))

        tile_rows = rng.randint(1, 6)
        tile_cols = rng.randint(1, 6)

        local_rng = random.Random(2026042210053000 + i)
        lhs = [make_q4_block(local_rng) for _ in range(max(1, row_count * max(1, lhs_stride)))]
        rhs = [make_q8_block(local_rng) for _ in range(max(1, col_count * max(1, rhs_stride)))]
        out = [0xEEEE] * max(1, row_count * max(1, out_stride))

        req_cells = [0xA1]
        req_bytes = [0xA2]
        got_tile_rows = [0xA3]
        got_tile_cols = [0xA4]

        err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
            out,
            out_capacity,
            out_stride,
            req_cells,
            req_bytes,
            got_tile_rows,
            got_tile_cols,
        )

        if err != Q4_0_Q8_0_AVX2_OK:
            assert req_cells == [0xA1]
            assert req_bytes == [0xA2]
            assert got_tile_rows == [0xA3]
            assert got_tile_cols == [0xA4]


def test_pointer_alias_rejections_for_diagnostics_outputs() -> None:
    rng = random.Random(2026042210054)

    lhs = [make_q4_block(rng) for _ in range(6)]
    rhs = [make_q8_block(rng) for _ in range(6)]
    out = [0] * 12

    shared = [0x1234]
    independent = [0x5678]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs,
        len(lhs),
        2,
        3,
        rhs,
        len(rhs),
        2,
        3,
        2,
        1,
        1,
        out,
        len(out),
        6,
        shared,
        shared,
        independent,
        [0x9ABC],
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs,
        len(lhs),
        2,
        3,
        rhs,
        len(rhs),
        2,
        3,
        2,
        1,
        1,
        out,
        len(out),
        6,
        shared,
        independent,
        shared,
        [0x9ABC],
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
