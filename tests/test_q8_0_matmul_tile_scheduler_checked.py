#!/usr/bin/env python3
"""Parity harness for Q8_0MatMulTileSchedulerChecked (IQ-1144)."""

from __future__ import annotations

import random
from pathlib import Path

I64_MAX = (1 << 63) - 1

Q8_0_MATMUL_OK = 0
Q8_0_MATMUL_ERR_NULL_PTR = 1
Q8_0_MATMUL_ERR_BAD_DST_LEN = 2
Q8_0_MATMUL_ERR_OVERFLOW = 3


def try_add_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def q8_0_matmul_tiled_compute_span_bounds_checked(
    span_start: int,
    span_len: int,
    axis_len: int,
) -> tuple[int, int]:
    if span_start < 0 or span_len < 0 or axis_len < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN, 0
    if span_start > axis_len:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN, 0

    ok, span_end = try_add_i64_nonneg(span_start, span_len)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW, 0

    return Q8_0_MATMUL_OK, min(span_end, axis_len)


def q8_0_matmul_tiled_validate_tile_shape_checked(tile_rows: int, tile_cols: int) -> int:
    if tile_rows <= 0 or tile_cols <= 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    return Q8_0_MATMUL_OK


def q8_0_matmul_tile_scheduler_checked(
    row_tile_start: int,
    col_tile_start: int,
    row_count: int,
    col_count: int,
    tile_rows: int,
    tile_cols: int,
) -> tuple[int, int, int]:
    err = q8_0_matmul_tiled_validate_tile_shape_checked(tile_rows, tile_cols)
    if err != Q8_0_MATMUL_OK:
        return err, 0, 0

    err, row_tile_end = q8_0_matmul_tiled_compute_span_bounds_checked(
        row_tile_start,
        tile_rows,
        row_count,
    )
    if err != Q8_0_MATMUL_OK:
        return err, 0, 0

    err, col_tile_end = q8_0_matmul_tiled_compute_span_bounds_checked(
        col_tile_start,
        tile_cols,
        col_count,
    )
    if err != Q8_0_MATMUL_OK:
        return err, 0, 0

    return Q8_0_MATMUL_OK, row_tile_end, col_tile_end


def test_targeted_scheduler_cases() -> None:
    cases = [
        # happy path
        (0, 0, 16, 12, 4, 3, Q8_0_MATMUL_OK, 4, 3),
        (13, 8, 16, 12, 7, 5, Q8_0_MATMUL_OK, 16, 12),
        (16, 12, 16, 12, 1, 1, Q8_0_MATMUL_OK, 16, 12),
        # bad tile shape
        (0, 0, 16, 12, 0, 3, Q8_0_MATMUL_ERR_BAD_DST_LEN, 0, 0),
        (0, 0, 16, 12, 3, -1, Q8_0_MATMUL_ERR_BAD_DST_LEN, 0, 0),
        # bad starts/axes
        (-1, 0, 16, 12, 3, 3, Q8_0_MATMUL_ERR_BAD_DST_LEN, 0, 0),
        (0, 13, 16, 12, 3, 3, Q8_0_MATMUL_ERR_BAD_DST_LEN, 0, 0),
        (17, 0, 16, 12, 3, 3, Q8_0_MATMUL_ERR_BAD_DST_LEN, 0, 0),
        (0, 0, -1, 12, 3, 3, Q8_0_MATMUL_ERR_BAD_DST_LEN, 0, 0),
        # overflow on checked add
        (I64_MAX, 0, I64_MAX, 12, 1, 1, Q8_0_MATMUL_ERR_OVERFLOW, 0, 0),
        (0, I64_MAX, 12, I64_MAX, 1, 1, Q8_0_MATMUL_ERR_OVERFLOW, 0, 0),
    ]

    for row_start, col_start, row_count, col_count, tile_rows, tile_cols, want_err, want_row_end, want_col_end in cases:
        got_err, got_row_end, got_col_end = q8_0_matmul_tile_scheduler_checked(
            row_start,
            col_start,
            row_count,
            col_count,
            tile_rows,
            tile_cols,
        )
        assert got_err == want_err
        if got_err == Q8_0_MATMUL_OK:
            assert got_row_end == want_row_end
            assert got_col_end == want_col_end


def test_randomized_scheduler_invariants() -> None:
    rng = random.Random(2026042201)

    for _ in range(2000):
        mode = rng.choice(["ok", "bad", "overflow"])
        if mode == "ok":
            row_count = rng.randint(0, 1 << 20)
            col_count = rng.randint(0, 1 << 20)
            tile_rows = rng.randint(1, 1 << 12)
            tile_cols = rng.randint(1, 1 << 12)
            row_start = rng.randint(0, row_count)
            col_start = rng.randint(0, col_count)
        elif mode == "bad":
            row_count = rng.randint(-10, 1 << 12)
            col_count = rng.randint(-10, 1 << 12)
            tile_rows = rng.randint(-2, 2)
            tile_cols = rng.randint(-2, 2)
            row_start = rng.randint(-10, 1 << 12)
            col_start = rng.randint(-10, 1 << 12)
        else:
            row_count = I64_MAX
            col_count = I64_MAX
            tile_rows = rng.randint(1, 4096)
            tile_cols = rng.randint(1, 4096)
            row_start = I64_MAX - rng.randint(0, 2048)
            col_start = I64_MAX - rng.randint(0, 2048)

        err, row_end, col_end = q8_0_matmul_tile_scheduler_checked(
            row_start,
            col_start,
            row_count,
            col_count,
            tile_rows,
            tile_cols,
        )

        if err == Q8_0_MATMUL_OK:
            assert 0 <= row_start <= row_end <= row_count
            assert 0 <= col_start <= col_end <= col_count
            assert row_end - row_start <= tile_rows
            assert col_end - col_start <= tile_cols


def test_source_contains_scheduler_and_callsite_adoption() -> None:
    source = Path("src/matmul/q8_0_matmul.HC").read_text(encoding="utf-8")

    assert "I32 Q8_0MatMulTileSchedulerChecked(" in source

    # Three tiled kernels each call scheduler twice (outer row + inner col).
    assert source.count("Q8_0MatMulTileSchedulerChecked(") >= 7

    # Scheduler implementation should rely on the shared span helper.
    assert source.count("Q8_0MatMulTiledComputeSpanBoundsChecked(") >= 3


def run() -> None:
    test_targeted_scheduler_cases()
    test_randomized_scheduler_invariants()
    test_source_contains_scheduler_and_callsite_adoption()
    print("q8_0_matmul_tile_scheduler_checked=ok")


if __name__ == "__main__":
    run()
