#!/usr/bin/env python3
"""Parity harness for ...PreflightOnlyCommitOnlyParityCommitOnly (IQ-839)."""

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
)
from test_q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity import (
    q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity,
)


def q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only(
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

    staged_cells_per_iter = [out_cells_per_iter[0]]
    staged_block_dots_per_iter = [out_block_dots_per_iter[0]]
    staged_total_cells = [out_total_cells[0]]
    staged_total_block_dots = [out_total_block_dots[0]]

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

    out_cells_per_iter[0] = staged_cells_per_iter[0]
    out_block_dots_per_iter[0] = staged_block_dots_per_iter[0]
    out_total_cells[0] = staged_total_cells[0]
    out_total_block_dots[0] = staged_total_block_dots[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_commit_only_composition(*args):
    staged_cells_per_iter = [args[13][0]]
    staged_block_dots_per_iter = [args[14][0]]
    staged_total_cells = [args[15][0]]
    staged_total_block_dots = [args[16][0]]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        args[10],
        args[11],
        args[12],
        staged_cells_per_iter,
        staged_block_dots_per_iter,
        staged_total_cells,
        staged_total_block_dots,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

    args[13][0] = staged_cells_per_iter[0]
    args[14][0] = staged_block_dots_per_iter[0]
    args[15][0] = staged_total_cells[0]
    args[16][0] = staged_total_block_dots[0]
    return Q4_0_Q8_0_AVX2_OK


def test_source_contains_commit_only_wrapper_signature_and_atomic_publish() -> None:
    source = Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParity(" in body
    assert "snapshot_row_count = row_count;" in body
    assert "snapshot_col_count = col_count;" in body
    assert "snapshot_k_block_count = k_block_count;" in body
    assert "snapshot_iter_count = iter_count;" in body
    assert "*out_cells_per_iter = parity_cells_per_iter;" in body
    assert "*out_block_dots_per_iter = parity_block_dots_per_iter;" in body
    assert "*out_total_cells = parity_total_cells;" in body
    assert "*out_total_block_dots = parity_total_block_dots;" in body


def test_known_vector_and_error_preserves_outputs() -> None:
    row_count = 5
    col_count = 4
    k_block_count = 3
    iter_count = 7
    lhs_stride = 3
    rhs_stride = 3
    out_stride = 4

    rng = random.Random(20260421_8391)
    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x1234] * (row_count * out_stride)

    cpi = [0xAAAA]
    bdi = [0xBBBB]
    tc = [0xCCCC]
    tbd = [0xDDDD]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only(
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

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only(
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


def test_overflow_vectors_no_publish() -> None:
    huge = (I64_MAX // 2) + 1
    lhs = [make_q4_block(random.Random(111))]
    rhs = [make_q8_block(random.Random(222))]
    out = [0]

    cpi = [0x1111]
    bdi = [0x2222]
    tc = [0x3333]
    tbd = [0x4444]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only(
        lhs,
        len(lhs),
        huge,
        huge,
        rhs,
        len(rhs),
        3,
        huge,
        3,
        out,
        len(out),
        3,
        3,
        cpi,
        bdi,
        tc,
        tbd,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    assert cpi == [0x1111]
    assert bdi == [0x2222]
    assert tc == [0x3333]
    assert tbd == [0x4444]


def test_fuzz_commit_only_vs_explicit_staged_composition() -> None:
    random.seed(20260421_839)

    for i in range(2800):
        if i % 7 == 0:
            row_count = random.choice([0, 1, 2, 3, 7, 31, 127, (I64_MAX // 8)])
            col_count = random.choice([0, 1, 2, 5, 13, 63, 255, 4])
            k_block_count = random.choice([0, 1, 3, 11, 17, 64])
            iter_count = random.choice([0, 1, 2, 7, 9, 63, (I64_MAX // 16)])
            lhs_stride = max(k_block_count, 1)
            rhs_stride = max(k_block_count, 1)
            out_stride = max(col_count if col_count < 4096 else 1, 1)
            lhs_capacity = len([0])
            rhs_capacity = len([0])
            out_capacity = len([0])
            lhs = [make_q4_block(random.Random(700000 + i))]
            rhs = [make_q8_block(random.Random(710000 + i))]
            out_a = [0xA5A5]
            out_b = [0xA5A5]
        else:
            row_count = random.randint(0, 10)
            col_count = random.randint(0, 10)
            k_block_count = random.randint(0, 9)
            iter_count = random.randint(0, 18)

            lhs_stride = k_block_count + random.randint(0, 4)
            rhs_stride = k_block_count + random.randint(0, 4)
            out_stride = col_count + random.randint(0, 4)

            lhs_capacity = max(1, row_count * lhs_stride)
            rhs_capacity = max(1, col_count * rhs_stride)
            out_capacity = max(1, row_count * out_stride)

            rng = random.Random(20260421_939000 + i)
            lhs = [make_q4_block(rng) for _ in range(lhs_capacity)]
            rhs = [make_q8_block(rng) for _ in range(rhs_capacity)]
            out_a = [0x5A5A] * out_capacity
            out_b = list(out_a)

        cpi_a = [0x0101]
        cpi_b = [0x0101]
        bdi_a = [0x0202]
        bdi_b = [0x0202]
        tc_a = [0x0303]
        tc_b = [0x0303]
        tbd_a = [0x0404]
        tbd_b = [0x0404]

        args_a = (
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
        args_b = (
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

        err_a = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only(
            *args_a
        )
        err_b = explicit_commit_only_composition(*args_b)

        assert err_a == err_b
        assert out_a == out_b
        assert cpi_a == cpi_b
        assert bdi_a == bdi_b
        assert tc_a == tc_b
        assert tbd_a == tbd_b


if __name__ == "__main__":
    test_source_contains_commit_only_wrapper_signature_and_atomic_publish()
    test_known_vector_and_error_preserves_outputs()
    test_overflow_vectors_no_publish()
    test_fuzz_commit_only_vs_explicit_staged_composition()
    print("ok")
