#!/usr/bin/env python3
"""Parity harness for ...PreflightOnlyCommitOnlyParityCommitOnlyPreflightOnly (IQ-853)."""

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
from test_q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only import (
    q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only,
)


def q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only(
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

    commit_cells_per_iter = [out_cells_per_iter[0]]
    commit_block_dots_per_iter = [out_block_dots_per_iter[0]]
    commit_total_cells = [out_total_cells[0]]
    commit_total_block_dots = [out_total_block_dots[0]]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only(
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
        commit_cells_per_iter,
        commit_block_dots_per_iter,
        commit_total_cells,
        commit_total_block_dots,
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

    if commit_cells_per_iter[0] != canonical_cells_per_iter:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if commit_block_dots_per_iter[0] != canonical_block_dots_per_iter:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if commit_total_cells[0] != canonical_total_cells:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if commit_total_block_dots[0] != canonical_total_block_dots:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_cells_per_iter[0] = commit_cells_per_iter[0]
    out_block_dots_per_iter[0] = commit_block_dots_per_iter[0]
    out_total_cells[0] = commit_total_cells[0]
    out_total_block_dots[0] = commit_total_block_dots[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only(
        *args
    )


def test_source_contains_signature_and_commit_only_preflight_revalidation() -> None:
    source = Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnly(" in body
    assert "snapshot_row_count = row_count;" in body
    assert "snapshot_col_count = col_count;" in body
    assert "snapshot_k_block_count = k_block_count;" in body
    assert "snapshot_iter_count = iter_count;" in body
    assert "Q4_0Q8_0MatMulTryMulI64NonNeg(snapshot_row_count," in body
    assert "Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_cells_per_iter," in body
    assert "Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_block_dots_per_iter," in body
    assert "if (commit_cells_per_iter != canonical_cells_per_iter)" in body
    assert "if (commit_block_dots_per_iter != canonical_block_dots_per_iter)" in body
    assert "if (commit_total_cells != canonical_total_cells)" in body
    assert "if (commit_total_block_dots != canonical_total_block_dots)" in body
    assert "*out_cells_per_iter = commit_cells_per_iter;" in body
    assert "*out_block_dots_per_iter = commit_block_dots_per_iter;" in body
    assert "*out_total_cells = commit_total_cells;" in body
    assert "*out_total_block_dots = commit_total_block_dots;" in body


def test_known_vector_and_no_publish_on_failure() -> None:
    row_count = 4
    col_count = 3
    k_block_count = 5
    iter_count = 6
    lhs_stride = 5
    rhs_stride = 5
    out_stride = 3

    rng = random.Random(20260421_8531)
    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x7272] * (row_count * out_stride)

    cpi = [0x1111]
    bdi = [0x2222]
    tc = [0x3333]
    tbd = [0x4444]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only(
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

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only(
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


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260421_8532)

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

        lhs_needed = 0
        rhs_needed = 0
        out_needed = 0
        if row_count >= 0 and lhs_stride >= 0:
            lhs_needed = row_count * lhs_stride
        if col_count >= 0 and rhs_stride >= 0:
            rhs_needed = col_count * rhs_stride
        if row_count >= 0 and out_stride >= 0:
            out_needed = row_count * out_stride

        lhs_capacity = max(0, lhs_needed + rng.randint(-10, 10))
        rhs_capacity = max(0, rhs_needed + rng.randint(-10, 10))
        out_capacity = max(0, out_needed + rng.randint(-10, 10))

        if rng.random() < 0.08:
            lhs_capacity = -rng.randint(1, 80)
        if rng.random() < 0.08:
            rhs_capacity = -rng.randint(1, 80)
        if rng.random() < 0.08:
            out_capacity = -rng.randint(1, 80)

        lhs = [make_q4_block(rng) for _ in range(max(1, lhs_capacity))]
        rhs = [make_q8_block(rng) for _ in range(max(1, rhs_capacity))]
        out = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(1, out_capacity))]

        lhs_ptr = lhs if rng.random() > 0.03 else None
        rhs_ptr = rhs if rng.random() > 0.03 else None
        out_ptr = out if rng.random() > 0.03 else None

        cpi_a = [rng.randint(-100000, 100000)]
        bdi_a = [rng.randint(-100000, 100000)]
        tc_a = [rng.randint(-100000, 100000)]
        tbd_a = [rng.randint(-100000, 100000)]

        cpi_b = cpi_a.copy()
        bdi_b = bdi_a.copy()
        tc_b = tc_a.copy()
        tbd_b = tbd_a.copy()

        err_a = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only(
            lhs_ptr,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs_ptr,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            out_ptr,
            out_capacity,
            out_stride,
            iter_count,
            cpi_a,
            bdi_a,
            tc_a,
            tbd_a,
        )

        err_b = explicit_checked_composition(
            lhs_ptr,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs_ptr,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            out_ptr,
            out_capacity,
            out_stride,
            iter_count,
            cpi_b,
            bdi_b,
            tc_b,
            tbd_b,
        )

        assert err_a == err_b
        if err_a == Q4_0_Q8_0_AVX2_OK:
            assert cpi_a == cpi_b
            assert bdi_a == bdi_b
            assert tc_a == tc_b
            assert tbd_a == tbd_b

    huge = 1 << 62
    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only(
        [make_q4_block(random.Random(1))],
        I64_MAX,
        huge,
        huge,
        [make_q8_block(random.Random(2))],
        I64_MAX,
        huge,
        huge,
        huge,
        [0],
        I64_MAX,
        huge,
        huge,
        [0],
        [0],
        [0],
        [0],
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_OVERFLOW


if __name__ == "__main__":
    test_source_contains_signature_and_commit_only_preflight_revalidation()
    test_known_vector_and_no_publish_on_failure()
    test_randomized_parity_against_explicit_composition()
    print("ok")
