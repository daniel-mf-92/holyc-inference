#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnly (IQ-826)."""

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


def q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only(
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
    snapshot_lhs_q4_block_capacity = lhs_q4_block_capacity
    snapshot_rhs_q8_block_capacity = rhs_q8_block_capacity
    snapshot_out_cell_capacity = out_cell_capacity
    snapshot_lhs_row_stride_blocks = lhs_row_stride_blocks
    snapshot_rhs_col_stride_blocks = rhs_col_stride_blocks
    snapshot_out_row_stride_cells = out_row_stride_cells
    snapshot_lhs_q4_blocks = lhs_q4_blocks
    snapshot_rhs_q8_col_blocks = rhs_q8_col_blocks
    snapshot_out_cells_q32 = out_cells_q32

    staged_cells_per_iter = [out_cells_per_iter[0]]
    staged_block_dots_per_iter = [out_block_dots_per_iter[0]]
    staged_total_cells = [out_total_cells[0]]
    staged_total_block_dots = [out_total_block_dots[0]]

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
        staged_cells_per_iter,
        staged_block_dots_per_iter,
        staged_total_cells,
        staged_total_block_dots,
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
    if snapshot_lhs_q4_block_capacity != lhs_q4_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_rhs_q8_block_capacity != rhs_q8_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_out_cell_capacity != out_cell_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_lhs_row_stride_blocks != lhs_row_stride_blocks:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_rhs_col_stride_blocks != rhs_col_stride_blocks:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_out_row_stride_cells != out_row_stride_cells:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_lhs_q4_blocks is not lhs_q4_blocks:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_rhs_q8_col_blocks is not rhs_q8_col_blocks:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if snapshot_out_cells_q32 is not out_cells_q32:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_cells_per_iter[0] = staged_cells_per_iter[0]
    out_block_dots_per_iter[0] = staged_block_dots_per_iter[0]
    out_total_cells[0] = staged_total_cells[0]
    out_total_block_dots[0] = staged_total_block_dots[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only(*args)


def test_source_contains_bench_preflight_only_commit_only_signature_and_atomic_publish() -> None:
    source = Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnly(" in body
    assert "if (status != Q4_0_Q8_0_MATMUL_OK)" in body
    assert "if (snapshot_row_count != row_count ||" in body
    assert "snapshot_lhs_q4_block_capacity = lhs_q4_block_capacity;" in body
    assert "snapshot_rhs_q8_block_capacity = rhs_q8_block_capacity;" in body
    assert "snapshot_out_cell_capacity = out_cell_capacity;" in body
    assert "snapshot_lhs_q4_blocks = lhs_q4_blocks;" in body
    assert "snapshot_rhs_q8_col_blocks = rhs_q8_col_blocks;" in body
    assert "snapshot_out_cells_q32 = out_cells_q32;" in body
    assert "snapshot_lhs_q4_block_capacity != lhs_q4_block_capacity ||" in body
    assert "snapshot_rhs_q8_block_capacity != rhs_q8_block_capacity ||" in body
    assert "snapshot_out_cell_capacity != out_cell_capacity ||" in body
    assert "snapshot_lhs_q4_blocks != lhs_q4_blocks ||" in body
    assert "snapshot_rhs_q8_col_blocks != rhs_q8_col_blocks ||" in body
    assert "snapshot_out_cells_q32 != out_cells_q32)" in body
    assert "*out_cells_per_iter = staged_cells_per_iter;" in body
    assert "*out_block_dots_per_iter = staged_block_dots_per_iter;" in body
    assert "*out_total_cells = staged_total_cells;" in body
    assert "*out_total_block_dots = staged_total_block_dots;" in body


def test_known_vector_and_no_publish_on_failure() -> None:
    row_count = 3
    col_count = 4
    k_block_count = 5
    iter_count = 6
    lhs_stride = 5
    rhs_stride = 5
    out_stride = 4

    rng = random.Random(20260421_8261)
    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x7777] * (row_count * out_stride)

    cpi = [0x1111]
    bdi = [0x2222]
    tc = [0x3333]
    tbd = [0x4444]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only(
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
    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only(
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
    random.seed(20260421_826)

    for i in range(2200):
        row_count = random.randint(0, 9)
        col_count = random.randint(0, 9)
        k_block_count = random.randint(0, 8)
        iter_count = random.randint(0, 16)

        lhs_stride = k_block_count + random.randint(0, 4)
        rhs_stride = k_block_count + random.randint(0, 4)
        out_stride = col_count + random.randint(0, 4)

        lhs_capacity = max(1, row_count * lhs_stride)
        rhs_capacity = max(1, col_count * rhs_stride)
        out_capacity = max(1, row_count * out_stride)

        rng = random.Random(20260421_980000 + i)
        lhs = [make_q4_block(rng) for _ in range(lhs_capacity)]
        rhs = [make_q8_block(rng) for _ in range(rhs_capacity)]
        out_a = [0x5A5A] * out_capacity
        out_b = list(out_a)

        cpi_a = [0x1010]
        cpi_b = [0x1010]
        bdi_a = [0x2020]
        bdi_b = [0x2020]
        tc_a = [0x3030]
        tc_b = [0x3030]
        tbd_a = [0x4040]
        tbd_b = [0x4040]

        err_a = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only(
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
    test_source_contains_bench_preflight_only_commit_only_signature_and_atomic_publish()
    test_known_vector_and_no_publish_on_failure()
    test_fuzz_parity_against_explicit_checked_composition()
    print("ok")
