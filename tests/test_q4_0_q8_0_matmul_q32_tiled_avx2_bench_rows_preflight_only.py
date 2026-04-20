#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnly (IQ-825)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_q4_0_q8_0_matmul_tiled_q32_checked_noalloc_default_tiles_preflight_only import (
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    explicit_checked_composition as default_tiles_preflight,
)
from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    make_q4_block,
    make_q8_block,
)

I64_MAX = 0x7FFFFFFFFFFFFFFF


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only(
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

    if iter_count < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    req_lhs = [0x1111]
    req_rhs = [0x2222]
    req_out = [0x3333]
    tile_m = [0x4444]
    tile_n = [0x5555]
    tile_k = [0x6666]
    err = default_tiles_preflight(
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
        req_lhs,
        req_rhs,
        req_out,
        tile_m,
        tile_n,
        tile_k,
    )
    if err != Q4_0_Q8_0_AVX2_OK:
        return err

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

    out_cells_per_iter[0] = cells_per_iter
    out_block_dots_per_iter[0] = block_dots_per_iter
    out_total_cells[0] = total_cells
    out_total_block_dots[0] = total_block_dots
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only(*args)


def test_source_contains_preflight_only_signature_and_composition() -> None:
    source = Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesPreflightOnly(" in body
    assert "Q4_0Q8_0MatMulTryMulI64NonNeg(row_count," in body
    assert "*out_cells_per_iter = staged_cells_per_iter;" in body
    assert "*out_block_dots_per_iter = staged_block_dots_per_iter;" in body
    assert "*out_total_cells = staged_total_cells;" in body
    assert "*out_total_block_dots = staged_total_block_dots;" in body


def test_known_vector_geometry_and_counts() -> None:
    row_count = 3
    col_count = 5
    k_block_count = 7
    iter_count = 11
    lhs_stride = 9
    rhs_stride = 9
    out_stride = 6

    rng = random.Random(20260421_8251)
    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x7A7A] * (row_count * out_stride)

    cpi = [0xAAAA]
    bdi = [0xBBBB]
    tc = [0xCCCC]
    tbd = [0xDDDD]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only(
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

    cpi_bad = [0xAAAA]
    bdi_bad = [0xBBBB]
    tc_bad = [0xCCCC]
    tbd_bad = [0xDDDD]
    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only(
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
        cpi_bad,
        bdi_bad,
        tc_bad,
        tbd_bad,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert cpi_bad == [0xAAAA]
    assert bdi_bad == [0xBBBB]
    assert tc_bad == [0xCCCC]
    assert tbd_bad == [0xDDDD]


def test_fuzz_parity_adversarial_shapes() -> None:
    random.seed(20260421_825)

    for _ in range(2400):
        row_count = random.randint(0, 9)
        col_count = random.randint(0, 9)
        k_block_count = random.randint(0, 8)

        lhs_stride = k_block_count + random.randint(0, 4)
        rhs_stride = k_block_count + random.randint(0, 4)
        out_stride = col_count + random.randint(0, 4)
        iter_count = random.randint(0, 15)

        lhs_capacity = row_count * lhs_stride
        rhs_capacity = col_count * rhs_stride
        out_capacity = row_count * out_stride

        rng = random.Random(20260421_900000 + _)
        lhs = [make_q4_block(rng) for _ in range(max(1, lhs_capacity))]
        rhs = [make_q8_block(rng) for _ in range(max(1, rhs_capacity))]
        out_a = [0x5A5A] * max(1, out_capacity)
        out_b = list(out_a)

        cpi_a = [0x1111]
        cpi_b = [0x1111]
        bdi_a = [0x2222]
        bdi_b = [0x2222]
        tc_a = [0x3333]
        tc_b = [0x3333]
        tbd_a = [0x4444]
        tbd_b = [0x4444]

        err_a = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only(
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
            max(1, out_capacity),
            out_stride,
            iter_count,
            cpi_b,
            bdi_b,
            tc_b,
            tbd_b,
        )

        assert err_a == err_b
        assert cpi_a == cpi_b
        assert bdi_a == bdi_b
        assert tc_a == tc_b
        assert tbd_a == tbd_b


def test_overflow_surface() -> None:
    huge = I64_MAX
    rng = random.Random(20260421_8252)
    lhs = [make_q4_block(rng)]
    rhs = [make_q8_block(rng)]
    out = [0]

    cpi = [0x1111]
    bdi = [0x2222]
    tc = [0x3333]
    tbd = [0x4444]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only(
        lhs,
        1,
        huge,
        huge,
        rhs,
        1,
        2,
        huge,
        1,
        out,
        1,
        0,
        1,
        cpi,
        bdi,
        tc,
        tbd,
    )
    assert err in (Q4_0_Q8_0_AVX2_ERR_OVERFLOW, Q4_0_Q8_0_AVX2_ERR_BAD_LEN)


if __name__ == "__main__":
    test_source_contains_preflight_only_signature_and_composition()
    test_known_vector_geometry_and_counts()
    test_fuzz_parity_adversarial_shapes()
    test_overflow_surface()
    print("ok")
