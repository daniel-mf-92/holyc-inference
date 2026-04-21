#!/usr/bin/env python3
"""Parity harness for ...PreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnly (IQ-862)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    I64_MAX,
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    make_q4_block,
    make_q8_block,
    try_mul_i64_nonneg,
)
from test_q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity import (
    q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity,
)


def q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only(
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
    iter_count: int,
    out_cells_per_iter,
    out_block_dots_per_iter,
    out_total_cells,
    out_total_block_dots,
) -> int:
    if (
        out_cells_per_iter is None
        or out_block_dots_per_iter is None
        or out_total_cells is None
        or out_total_block_dots is None
    ):
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    snapshot_row_count = row_count
    snapshot_col_count = col_count
    snapshot_k_block_count = k_block_count
    snapshot_iter_count = iter_count

    parity_cells_per_iter = [0]
    parity_block_dots_per_iter = [0]
    parity_total_cells = [0]
    parity_total_block_dots = [0]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity(
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
        iter_count,
        parity_cells_per_iter,
        parity_block_dots_per_iter,
        parity_total_cells,
        parity_total_block_dots,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    if snapshot_row_count != row_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_col_count != col_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_k_block_count != k_block_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_iter_count != iter_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, canonical_cells_per_iter = try_mul_i64_nonneg(snapshot_row_count, snapshot_col_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, canonical_block_dots_per_iter = try_mul_i64_nonneg(
        canonical_cells_per_iter, snapshot_k_block_count
    )
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, canonical_total_cells = try_mul_i64_nonneg(canonical_cells_per_iter, snapshot_iter_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, canonical_total_block_dots = try_mul_i64_nonneg(
        canonical_block_dots_per_iter, snapshot_iter_count
    )
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    if parity_cells_per_iter[0] != canonical_cells_per_iter:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if parity_block_dots_per_iter[0] != canonical_block_dots_per_iter:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if parity_total_cells[0] != canonical_total_cells:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if parity_total_block_dots[0] != canonical_total_block_dots:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_cells_per_iter[0] = parity_cells_per_iter[0]
    out_block_dots_per_iter[0] = parity_block_dots_per_iter[0]
    out_total_cells[0] = parity_total_cells[0]
    out_total_block_dots[0] = parity_total_block_dots[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only(
        *args
    )


def test_source_contains_signature_and_rederived_tuple_checks() -> None:
    source = Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "Q4_0Q8_0MatMulTryMulI64NonNeg(snapshot_row_count," in body
    assert "Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_cells_per_iter," in body
    assert "Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_block_dots_per_iter," in body
    assert "if (parity_cells_per_iter != canonical_cells_per_iter)" in body
    assert "if (parity_block_dots_per_iter != canonical_block_dots_per_iter)" in body
    assert "if (parity_total_cells != canonical_total_cells)" in body
    assert "if (parity_total_block_dots != canonical_total_block_dots)" in body
    assert "*out_cells_per_iter = parity_cells_per_iter;" in body
    assert "*out_block_dots_per_iter = parity_block_dots_per_iter;" in body
    assert "*out_total_cells = parity_total_cells;" in body
    assert "*out_total_block_dots = parity_total_block_dots;" in body


def test_known_vector_and_no_publish_on_failure() -> None:
    row_count = 7
    col_count = 3
    k_block_count = 4
    iter_count = 5
    lhs_stride = 4
    rhs_stride = 4
    out_stride = 3

    rng = random.Random(20260421_8621)
    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x1212] * (row_count * out_stride)

    cpi = [0x1111]
    bdi = [0x2222]
    tc = [0x3333]
    tbd = [0x4444]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only(
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
        len(out),
        out_stride,
        iter_count,
        cpi,
        bdi,
        tc,
        tbd,
    )
    assert err == Q4_0_Q8_0_AVX2_OK
    assert cpi == [row_count * col_count]
    assert bdi == [row_count * col_count * k_block_count]
    assert tc == [row_count * col_count * iter_count]
    assert tbd == [row_count * col_count * k_block_count * iter_count]

    cpi_fail = [0xAAAA]
    bdi_fail = [0xBBBB]
    tc_fail = [0xCCCC]
    tbd_fail = [0xDDDD]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only(
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
        len(out),
        out_stride,
        -1,
        cpi_fail,
        bdi_fail,
        tc_fail,
        tbd_fail,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert cpi_fail == [0xAAAA]
    assert bdi_fail == [0xBBBB]
    assert tc_fail == [0xCCCC]
    assert tbd_fail == [0xDDDD]


def test_randomized_parity_matches_explicit_composition() -> None:
    rng = random.Random(20260421_8622)

    for _ in range(3000):
        row_count = rng.randint(0, 28)
        col_count = rng.randint(0, 28)
        k_block_count = rng.randint(0, 20)
        iter_count = rng.randint(0, 20)

        if rng.random() < 0.08:
            row_count = -rng.randint(1, 200)
        if rng.random() < 0.08:
            col_count = -rng.randint(1, 200)
        if rng.random() < 0.08:
            k_block_count = -rng.randint(1, 200)
        if rng.random() < 0.08:
            iter_count = -rng.randint(1, 200)

        lhs_stride = max(0, k_block_count + rng.randint(-4, 4))
        rhs_stride = max(0, k_block_count + rng.randint(-4, 4))
        out_stride = max(0, col_count + rng.randint(-4, 4))

        if rng.random() < 0.08:
            lhs_stride = -rng.randint(1, 40)
        if rng.random() < 0.08:
            rhs_stride = -rng.randint(1, 40)
        if rng.random() < 0.08:
            out_stride = -rng.randint(1, 40)

        lhs_needed = row_count * lhs_stride if row_count >= 0 and lhs_stride >= 0 else 0
        rhs_needed = col_count * rhs_stride if col_count >= 0 and rhs_stride >= 0 else 0
        out_needed = row_count * out_stride if row_count >= 0 and out_stride >= 0 else 0

        lhs_capacity = max(0, lhs_needed + rng.randint(-10, 10))
        rhs_capacity = max(0, rhs_needed + rng.randint(-10, 10))
        out_capacity = max(0, out_needed + rng.randint(-10, 10))

        if rng.random() < 0.08:
            lhs_capacity = -rng.randint(1, 80)
        if rng.random() < 0.08:
            rhs_capacity = -rng.randint(1, 80)
        if rng.random() < 0.08:
            out_capacity = -rng.randint(1, 80)

        if rng.random() < 0.03:
            row_count = I64_MAX
            col_count = 2
            k_block_count = 2
            iter_count = 2
            lhs_stride = k_block_count
            rhs_stride = k_block_count
            out_stride = col_count
            lhs_capacity = 1
            rhs_capacity = 1
            out_capacity = 1

        lhs = [make_q4_block(rng) for _ in range(max(1, lhs_capacity))]
        rhs = [make_q8_block(rng) for _ in range(max(1, rhs_capacity))]
        out = [0] * max(1, out_capacity)

        cpi_actual = [rng.randint(0, 1000)]
        bdi_actual = [rng.randint(0, 1000)]
        tc_actual = [rng.randint(0, 1000)]
        tbd_actual = [rng.randint(0, 1000)]

        cpi_expected = [cpi_actual[0]]
        bdi_expected = [bdi_actual[0]]
        tc_expected = [tc_actual[0]]
        tbd_expected = [tbd_actual[0]]

        err_actual = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only(
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
            iter_count,
            cpi_actual,
            bdi_actual,
            tc_actual,
            tbd_actual,
        )

        err_expected = explicit_checked_composition(
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
            iter_count,
            cpi_expected,
            bdi_expected,
            tc_expected,
            tbd_expected,
        )

        assert err_actual == err_expected
        assert cpi_actual == cpi_expected
        assert bdi_actual == bdi_expected
        assert tc_actual == tc_expected
        assert tbd_actual == tbd_expected


if __name__ == "__main__":
    test_source_contains_signature_and_rederived_tuple_checks()
    test_known_vector_and_no_publish_on_failure()
    test_randomized_parity_matches_explicit_composition()
    print("ok")
