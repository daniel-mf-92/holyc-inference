#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0MatMulQ32TiledAVX2BenchRows (IQ-822)."""

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
    q4_0_q8_0_matmul_tiled_avx2_q32_checked,
)


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows(
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
    out_total_cells,
    out_total_block_dots,
    out_last_status,
) -> int:
    if out_total_cells is None or out_total_block_dots is None or out_last_status is None:
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR
    if iter_count < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    ok, cells_per_iter = try_mul_i64_nonneg(row_count, col_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, block_dots_per_iter = try_mul_i64_nonneg(cells_per_iter, k_block_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, total_cells = try_mul_i64_nonneg(cells_per_iter, iter_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW
    ok, total_block_dots = try_mul_i64_nonneg(block_dots_per_iter, iter_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    staged_last_status = Q4_0_Q8_0_AVX2_OK
    for _ in range(iter_count):
        err, staged = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
            lhs_q4_blocks=lhs_q4_blocks,
            lhs_q4_block_capacity=lhs_q4_block_capacity,
            row_count=row_count,
            lhs_row_stride_blocks=lhs_row_stride_blocks,
            rhs_q8_col_blocks=rhs_q8_col_blocks,
            rhs_q8_block_capacity=rhs_q8_block_capacity,
            col_count=col_count,
            rhs_col_stride_blocks=rhs_col_stride_blocks,
            k_block_count=k_block_count,
            tile_rows=4,
            tile_cols=4,
            out_cell_capacity=out_cell_capacity,
            out_row_stride_cells=out_row_stride_cells,
        )
        if err != Q4_0_Q8_0_AVX2_OK:
            return err

        required_out_cells = row_count * out_row_stride_cells
        for i in range(required_out_cells):
            out_cells_q32[i] = staged[i]

        staged_last_status = err

    out_total_cells[0] = total_cells
    out_total_block_dots[0] = total_block_dots
    out_last_status[0] = staged_last_status
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows(*args)


def test_source_contains_bench_rows_signature_and_checked_loop() -> None:
    source = Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRows("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "Q4_0Q8_0MatMulQ32TiledAVX2Checked(" in body
    assert "Q4_0Q8_0MatMulTryMulI64NonNeg(staged_cells_per_iter," in body
    assert "*out_total_cells = staged_total_cells;" in body
    assert "*out_total_block_dots = staged_total_block_dots;" in body
    assert "*out_last_status = staged_last_status;" in body


def test_known_vector_counts_and_error_propagation() -> None:
    rng = random.Random(20260421_822)

    row_count = 3
    col_count = 2
    k_block_count = 4
    lhs_stride = 4
    rhs_stride = 4
    out_stride = 5
    iter_count = 7

    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x1234] * (row_count * out_stride)

    total_cells = [0xAAAA]
    total_block_dots = [0xBBBB]
    last_status = [0xCCCC]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows(
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
        total_cells,
        total_block_dots,
        last_status,
    )
    assert err == Q4_0_Q8_0_AVX2_OK
    assert total_cells == [row_count * col_count * iter_count]
    assert total_block_dots == [row_count * col_count * k_block_count * iter_count]
    assert last_status == [Q4_0_Q8_0_AVX2_OK]

    total_cells_bad = [0x1111]
    total_block_dots_bad = [0x2222]
    last_status_bad = [0x3333]
    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows(
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
        total_cells_bad,
        total_block_dots_bad,
        last_status_bad,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert total_cells_bad == [0x1111]
    assert total_block_dots_bad == [0x2222]
    assert last_status_bad == [0x3333]


def test_fuzz_parity() -> None:
    random.seed(20260421_1822)

    for _ in range(2200):
        row_count = random.randint(0, 8)
        col_count = random.randint(0, 8)
        k_block_count = random.randint(0, 6)

        lhs_stride = k_block_count + random.randint(0, 3)
        rhs_stride = k_block_count + random.randint(0, 3)
        out_stride = col_count + random.randint(0, 3)
        iter_count = random.randint(0, 12)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        rng = random.Random(20260421_9000 + _)
        lhs = [make_q4_block(rng) for _ in range(lhs_capacity)]
        rhs = [make_q8_block(rng) for _ in range(rhs_capacity)]
        out_a = [0x5A5A] * max(1, out_capacity)
        out_b = list(out_a)

        tc_a = [0x1111]
        tc_b = [0x1111]
        td_a = [0x2222]
        td_b = [0x2222]
        ls_a = [0x3333]
        ls_b = [0x3333]

        err_a = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows(
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
            max(1, out_capacity),
            out_stride,
            iter_count,
            tc_a,
            td_a,
            ls_a,
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
            max(1, out_capacity),
            out_stride,
            iter_count,
            tc_b,
            td_b,
            ls_b,
        )

        assert err_a == err_b
        assert tc_a == tc_b
        assert td_a == td_b
        assert ls_a == ls_b


if __name__ == "__main__":
    test_source_contains_bench_rows_signature_and_checked_loop()
    test_known_vector_counts_and_error_propagation()
    test_fuzz_parity()
    print("ok")
