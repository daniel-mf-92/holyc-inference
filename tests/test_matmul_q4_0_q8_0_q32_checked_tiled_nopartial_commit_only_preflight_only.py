#!/usr/bin/env python3
"""Parity harness for MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly (IQ-994)."""

from __future__ import annotations

import random
from pathlib import Path

import test_q4_0_q8_0_matmul_tiled_avx2_q32 as ref
import test_q4_0_q8_0_dot_kernel as dot_ref


OK = 0
ERR_NULL_PTR = 1
ERR_BAD_DST_LEN = 2
ERR_OVERFLOW = 3
I64_MAX = (1 << 63) - 1


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

    ok, tile_count = try_mul_i64_nonneg(row_tile_count, col_tile_count)
    if not ok:
        return False, 0, 0, 0

    return True, row_tile_count, col_tile_count, tile_count


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
        return ERR_NULL_PTR

    if lhs_q4_blocks is None or rhs_q8_col_blocks is None or out_cells_q32 is None:
        return ERR_NULL_PTR

    if lhs_q4_block_capacity < 0 or rhs_q8_block_capacity < 0 or out_cell_capacity < 0:
        return ERR_BAD_DST_LEN
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return ERR_BAD_DST_LEN
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return ERR_BAD_DST_LEN
    if tile_row_span <= 0 or tile_col_span <= 0:
        return ERR_BAD_DST_LEN

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return ERR_BAD_DST_LEN
    if row_count > 0 and out_row_stride_cells < col_count:
        return ERR_BAD_DST_LEN

    snapshot_row_count = row_count
    snapshot_col_count = col_count
    snapshot_k_block_count = k_block_count
    snapshot_tile_row_span = tile_row_span
    snapshot_tile_col_span = tile_col_span
    snapshot_out_capacity = out_cell_capacity

    ok, required_lhs_blocks = try_mul_i64_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return ERR_OVERFLOW
    ok, required_rhs_blocks = try_mul_i64_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return ERR_OVERFLOW
    ok, staged_required_out_cells = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return ERR_OVERFLOW

    if required_lhs_blocks > lhs_q4_block_capacity:
        return ERR_BAD_DST_LEN
    if required_rhs_blocks > rhs_q8_block_capacity:
        return ERR_BAD_DST_LEN
    if staged_required_out_cells > out_cell_capacity:
        return ERR_BAD_DST_LEN

    ok, _tile_row_count, _tile_col_count, tile_total_count = compute_tile_grid_checked(
        row_count,
        col_count,
        tile_row_span,
        tile_col_span,
    )
    if not ok:
        return ERR_OVERFLOW
    if row_count and col_count and not tile_total_count:
        return ERR_BAD_DST_LEN

    ok, staged_required_out_bytes = try_mul_i64_nonneg(staged_required_out_cells, 8)
    if not ok:
        return ERR_OVERFLOW

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
        return ERR_BAD_DST_LEN

    out_required_out_cells[0] = staged_required_out_cells
    out_required_out_bytes[0] = staged_required_out_bytes
    out_tile_rows[0] = staged_tile_rows
    out_tile_cols[0] = staged_tile_cols
    return OK


def explicit_expected_tuple(
    row_count: int,
    out_row_stride_cells: int,
    tile_row_span: int,
    tile_col_span: int,
) -> tuple[int, int, int, int]:
    required_out_cells = row_count * out_row_stride_cells
    return required_out_cells, required_out_cells * 8, tile_row_span, tile_col_span


def make_q4_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.0, 2.0)
    vals = [rng.randrange(-8, 8) for _ in range(32)]
    return dot_ref.half_bits(scale), dot_ref.pack_q4_from_signed(vals)


def make_q8_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.0, 2.0)
    vals = [rng.randrange(-128, 128) for _ in range(32)]
    return dot_ref.half_bits(scale), dot_ref.pack_q8_signed(vals)


def test_source_contains_iq994_preflight_contract() -> None:
    source = Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    parts = source.split(sig)
    assert len(parts) >= 3
    body = parts[2].split("\nI32 ", 1)[0]

    assert "// IQ-994 diagnostics-only no-write preflight:" in body
    assert "Q4_0Q8_0MatMulComputeTileGridChecked(" in body
    assert "if (row_count && col_count && !tile_total_count)" in body
    assert "if (snapshot_row_count != row_count" in body
    assert "*out_required_out_cells = staged_required_out_cells;" in body
    assert "*out_required_out_bytes = staged_required_out_bytes;" in body
    assert "*out_tile_rows = staged_tile_rows;" in body
    assert "*out_tile_cols = staged_tile_cols;" in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartial(" not in body
    assert "MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnly(" not in body


def test_known_vector_and_zero_write_buffer() -> None:
    rng = random.Random(20260422_9941)
    row_count = 5
    col_count = 7
    k_block_count = 4
    lhs_stride = 6
    rhs_stride = 4
    out_stride = 9
    tile_rows = 3
    tile_cols = 2

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out_cells = [0x7A7A7A7A] * (row_count * out_stride)
    out_before = list(out_cells)

    out_req_cells = [0x1111]
    out_req_bytes = [0x2222]
    out_tile_rows = [0x3333]
    out_tile_cols = [0x4444]

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
        tile_rows,
        tile_cols,
        out_cells,
        len(out_cells),
        out_stride,
        out_req_cells,
        out_req_bytes,
        out_tile_rows,
        out_tile_cols,
    )
    assert err == OK
    assert out_cells == out_before

    exp_cells, exp_bytes, exp_tile_rows, exp_tile_cols = explicit_expected_tuple(
        row_count,
        out_stride,
        tile_rows,
        tile_cols,
    )
    assert out_req_cells == [exp_cells]
    assert out_req_bytes == [exp_bytes]
    assert out_tile_rows == [exp_tile_rows]
    assert out_tile_cols == [exp_tile_cols]


def test_error_paths_do_not_publish_partial_outputs() -> None:
    rng = random.Random(20260422_9942)
    lhs = [make_q4_block(rng) for _ in range(8)]
    rhs = [make_q8_block(rng) for _ in range(8)]
    out_cells = [0xABCD] * 16

    out_req_cells = [0x1111]
    out_req_bytes = [0x2222]
    out_tile_rows = [0x3333]
    out_tile_cols = [0x4444]

    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(
        lhs,
        len(lhs),
        4,
        2,
        rhs,
        len(rhs),
        4,
        2,
        3,
        2,
        2,
        out_cells,
        len(out_cells),
        4,
        out_req_cells,
        out_req_bytes,
        out_tile_rows,
        out_tile_cols,
    )
    assert err == ERR_BAD_DST_LEN
    assert out_req_cells == [0x1111]
    assert out_req_bytes == [0x2222]
    assert out_tile_rows == [0x3333]
    assert out_tile_cols == [0x4444]

    out_req_cells_ov = [0xAAAA]
    out_req_bytes_ov = [0xBBBB]
    out_tile_rows_ov = [0xCCCC]
    out_tile_cols_ov = [0xDDDD]
    huge = (1 << 62)
    err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(
        lhs,
        huge,
        huge,
        3,
        rhs,
        huge,
        2,
        3,
        1,
        2,
        2,
        out_cells,
        huge,
        huge,
        out_req_cells_ov,
        out_req_bytes_ov,
        out_tile_rows_ov,
        out_tile_cols_ov,
    )
    assert err == ERR_OVERFLOW
    assert out_req_cells_ov == [0xAAAA]
    assert out_req_bytes_ov == [0xBBBB]
    assert out_tile_rows_ov == [0xCCCC]
    assert out_tile_cols_ov == [0xDDDD]


def test_fuzz_adversarial_geometry_capacity_overflow_vectors() -> None:
    random.seed(20260422_994)

    for case_id in range(2600):
        row_count = random.randint(0, 11)
        col_count = random.randint(0, 11)
        k_block_count = random.randint(0, 8)
        lhs_stride = k_block_count + random.randint(0, 4)
        rhs_stride = k_block_count + random.randint(0, 4)
        out_stride = col_count + random.randint(0, 4)
        tile_rows = random.randint(1, 5)
        tile_cols = random.randint(1, 5)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        if random.random() < 0.25:
            lhs_capacity = max(0, lhs_capacity - random.randint(1, 3))
        if random.random() < 0.25:
            rhs_capacity = max(0, rhs_capacity - random.randint(1, 3))
        if random.random() < 0.25:
            out_capacity = max(0, out_capacity - random.randint(1, 3))

        rng = random.Random(20260422_994000 + case_id)
        lhs = [make_q4_block(rng) for _ in range(max(1, row_count * lhs_stride))]
        rhs = [make_q8_block(rng) for _ in range(max(1, col_count * rhs_stride))]
        out_cells = [0x1357] * max(1, row_count * max(1, out_stride))
        out_before = list(out_cells)

        out_req_cells = [0xAAAA]
        out_req_bytes = [0xBBBB]
        out_tile_rows = [0xCCCC]
        out_tile_cols = [0xDDDD]

        err = matmul_q4_0_q8_0_q32_checked_tiled_nopartial_commit_only_preflight_only(
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
            out_cells,
            out_capacity,
            out_stride,
            out_req_cells,
            out_req_bytes,
            out_tile_rows,
            out_tile_cols,
        )

        assert out_cells == out_before
        if err == OK:
            exp_cells, exp_bytes, exp_tile_rows, exp_tile_cols = explicit_expected_tuple(
                row_count,
                out_stride,
                tile_rows,
                tile_cols,
            )
            assert out_req_cells == [exp_cells]
            assert out_req_bytes == [exp_bytes]
            assert out_tile_rows == [exp_tile_rows]
            assert out_tile_cols == [exp_tile_cols]
        else:
            assert out_req_cells == [0xAAAA]
            assert out_req_bytes == [0xBBBB]
            assert out_tile_rows == [0xCCCC]
            assert out_tile_cols == [0xDDDD]
