#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnly (IQ-836)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_OK,
    make_q4_block,
    make_q8_block,
)
from test_q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only import (
    q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only,
)
from test_q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity import (
    q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity,
)


def q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only(
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

    parity_cells_per_iter = [0xAAA1]
    parity_block_dots_per_iter = [0xAAA2]
    parity_total_cells = [0xAAA3]
    parity_total_block_dots = [0xAAA4]

    canonical_cells_per_iter = [0xBBB1]
    canonical_block_dots_per_iter = [0xBBB2]
    canonical_total_cells = [0xBBB3]
    canonical_total_block_dots = [0xBBB4]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity(
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

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only(
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
        canonical_cells_per_iter,
        canonical_block_dots_per_iter,
        canonical_total_cells,
        canonical_total_block_dots,
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

    if parity_cells_per_iter[0] != canonical_cells_per_iter[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if parity_block_dots_per_iter[0] != canonical_block_dots_per_iter[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if parity_total_cells[0] != canonical_total_cells[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if parity_total_block_dots[0] != canonical_total_block_dots[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_cells_per_iter[0] = parity_cells_per_iter[0]
    out_block_dots_per_iter[0] = parity_block_dots_per_iter[0]
    out_total_cells[0] = parity_total_cells[0]
    out_total_block_dots[0] = parity_total_block_dots[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only(*args)


def test_source_contains_signature_and_zero_write_parity_preflight_logic() -> None:
    source = Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParity(" in body
    assert "Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnly(" in body
    assert "if (parity_cells_per_iter != canonical_cells_per_iter)" in body
    assert "if (parity_block_dots_per_iter != canonical_block_dots_per_iter)" in body
    assert "if (parity_total_cells != canonical_total_cells)" in body
    assert "if (parity_total_block_dots != canonical_total_block_dots)" in body
    assert "*out_cells_per_iter = parity_cells_per_iter;" in body
    assert "*out_block_dots_per_iter = parity_block_dots_per_iter;" in body
    assert "*out_total_cells = parity_total_cells;" in body
    assert "*out_total_block_dots = parity_total_block_dots;" in body


def test_known_vector_and_failure_no_publish() -> None:
    row_count = 5
    col_count = 4
    k_block_count = 3
    iter_count = 7
    lhs_stride = 3
    rhs_stride = 3
    out_stride = 4

    rng = random.Random(20260421_8401)
    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x5555] * (row_count * out_stride)

    cpi = [0x1010]
    bdi = [0x2020]
    tc = [0x3030]
    tbd = [0x4040]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only(
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
    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only(
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


def test_fuzz_parity_against_explicit_checked_composition() -> None:
    random.seed(20260421_840)

    for i in range(2600):
        row_count = random.randint(0, 10)
        col_count = random.randint(0, 10)
        k_block_count = random.randint(0, 9)
        iter_count = random.randint(0, 20)

        lhs_stride = k_block_count + random.randint(0, 5)
        rhs_stride = k_block_count + random.randint(0, 5)
        out_stride = col_count + random.randint(0, 5)

        lhs_capacity = max(1, row_count * lhs_stride)
        rhs_capacity = max(1, col_count * rhs_stride)
        out_capacity = max(1, row_count * out_stride)

        rng = random.Random(20260421_1040000 + i)
        lhs = [make_q4_block(rng) for _ in range(lhs_capacity)]
        rhs = [make_q8_block(rng) for _ in range(rhs_capacity)]
        out_a = [0x7A7A] * out_capacity
        out_b = list(out_a)

        cpi_a = [0x5151]
        cpi_b = [0x5151]
        bdi_a = [0x6161]
        bdi_b = [0x6161]
        tc_a = [0x7171]
        tc_b = [0x7171]
        tbd_a = [0x8181]
        tbd_b = [0x8181]

        err_a = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only(
            lhs,
            lhs_capacity,
            row_count,
            lhs_stride,
            rhs,
            rhs_capacity,
            col_count,
            rhs_stride,
            k_block_count,
            out_a,
            out_capacity,
            out_stride,
            iter_count,
            cpi_a,
            bdi_a,
            tc_a,
            tbd_a,
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
            out_b,
            out_capacity,
            out_stride,
            iter_count,
            cpi_b,
            bdi_b,
            tc_b,
            tbd_b,
        )

        assert err_a == err_b
        assert out_a == out_b
        assert cpi_a == cpi_b
        assert bdi_a == bdi_b
        assert tc_a == tc_b
        assert tbd_a == tbd_b


if __name__ == "__main__":
    test_source_contains_signature_and_zero_write_parity_preflight_logic()
    test_known_vector_and_failure_no_publish()
    test_fuzz_parity_against_explicit_checked_composition()
    print("ok")
