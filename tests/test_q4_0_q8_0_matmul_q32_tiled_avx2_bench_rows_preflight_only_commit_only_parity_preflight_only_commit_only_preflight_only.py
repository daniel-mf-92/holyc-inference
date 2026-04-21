#!/usr/bin/env python3
"""Parity harness for ...PreflightOnlyCommitOnlyParityPreflightOnlyCommitOnlyPreflightOnly (IQ-857)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_q4_0_q8_0_matmul_tiled_avx2_q32 import (
    Q4_0_Q8_0_AVX2_ERR_BAD_LEN,
    Q4_0_Q8_0_AVX2_ERR_NULL_PTR,
    Q4_0_Q8_0_AVX2_ERR_OVERFLOW,
    Q4_0_Q8_0_AVX2_OK,
    make_q4_block,
    make_q8_block,
)
from test_q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only import (
    q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only,
)


def count_exact_holyc_signature_occurrences(source: str, signature: str) -> int:
    count = 0
    search_start = 0

    while True:
        idx = source.find(signature, search_start)
        if idx < 0:
            break

        cursor = idx + len(signature)
        depth = 1
        while cursor < len(source) and depth > 0:
            if source[cursor] == "(":
                depth += 1
            elif source[cursor] == ")":
                depth -= 1
            cursor += 1

        while cursor < len(source) and source[cursor].isspace():
            cursor += 1

        if cursor < len(source) and source[cursor] in {";", "{"}:
            count += 1

        search_start = idx + 1

    return count


def _safe_mul_nonneg(a: int, b: int):
    if a < 0 or b < 0:
        return False, 0
    if a == 0 or b == 0:
        return True, 0
    max_i64 = 0x7FFFFFFFFFFFFFFF
    if a > max_i64 // b:
        return False, 0
    return True, a * b


def q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only_preflight_only(
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

    staged_cells_per_iter = [0]
    staged_block_dots_per_iter = [0]
    staged_total_cells = [0]
    staged_total_block_dots = [0]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only(
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

    ok, canonical_cells_per_iter = _safe_mul_nonneg(snapshot_row_count, snapshot_col_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    ok, canonical_block_dots_per_iter = _safe_mul_nonneg(canonical_cells_per_iter, snapshot_k_block_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    ok, canonical_total_cells = _safe_mul_nonneg(canonical_cells_per_iter, snapshot_iter_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    ok, canonical_total_block_dots = _safe_mul_nonneg(canonical_block_dots_per_iter, snapshot_iter_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    if staged_cells_per_iter[0] != canonical_cells_per_iter:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_block_dots_per_iter[0] != canonical_block_dots_per_iter:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_total_cells[0] != canonical_total_cells:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_total_block_dots[0] != canonical_total_block_dots:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_cells_per_iter[0] = staged_cells_per_iter[0]
    out_block_dots_per_iter[0] = staged_block_dots_per_iter[0]
    out_total_cells[0] = staged_total_cells[0]
    out_total_block_dots[0] = staged_total_block_dots[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    (
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
        out_cells_per_iter,
        out_block_dots_per_iter,
        out_total_cells,
        out_total_block_dots,
    ) = args

    if (
        out_cells_per_iter is None
        or out_block_dots_per_iter is None
        or out_total_cells is None
        or out_total_block_dots is None
    ):
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    staged_cells_per_iter = [0]
    staged_block_dots_per_iter = [0]
    staged_total_cells = [0]
    staged_total_block_dots = [0]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only(
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

    ok, canonical_cells_per_iter = _safe_mul_nonneg(row_count, col_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    ok, canonical_block_dots_per_iter = _safe_mul_nonneg(canonical_cells_per_iter, k_block_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    ok, canonical_total_cells = _safe_mul_nonneg(canonical_cells_per_iter, iter_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    ok, canonical_total_block_dots = _safe_mul_nonneg(canonical_block_dots_per_iter, iter_count)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW

    if staged_cells_per_iter[0] != canonical_cells_per_iter:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_block_dots_per_iter[0] != canonical_block_dots_per_iter:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_total_cells[0] != canonical_total_cells:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if staged_total_block_dots[0] != canonical_total_block_dots:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_cells_per_iter[0] = staged_cells_per_iter[0]
    out_block_dots_per_iter[0] = staged_block_dots_per_iter[0]
    out_total_cells[0] = staged_total_cells[0]
    out_total_block_dots[0] = staged_total_block_dots[0]
    return Q4_0_Q8_0_AVX2_OK


def test_source_contains_signature_and_no_write_parity_flow() -> None:
    source = Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnlyPreflightOnly("
    assert sig in source
    assert count_exact_holyc_signature_occurrences(source, sig) == 2
    body = source.split(sig, 1)[1]

    assert "Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnly(" in body
    assert "if (!Q4_0Q8_0MatMulTryMulI64NonNeg(snapshot_row_count," in body
    assert "snapshot_lhs_q4_block_capacity != lhs_q4_block_capacity" in body
    assert "snapshot_rhs_q8_block_capacity != rhs_q8_block_capacity" in body
    assert "snapshot_out_cell_capacity != out_cell_capacity" in body
    assert "snapshot_lhs_row_stride_blocks != lhs_row_stride_blocks" in body
    assert "snapshot_rhs_col_stride_blocks != rhs_col_stride_blocks" in body
    assert "snapshot_out_row_stride_cells != out_row_stride_cells" in body
    assert "if (staged_cells_per_iter != canonical_cells_per_iter)" in body
    assert "if (staged_block_dots_per_iter != canonical_block_dots_per_iter)" in body
    assert "if (staged_total_cells != canonical_total_cells)" in body
    assert "if (staged_total_block_dots != canonical_total_block_dots)" in body
    assert "*out_cells_per_iter = staged_cells_per_iter;" in body
    assert "*out_block_dots_per_iter = staged_block_dots_per_iter;" in body
    assert "*out_total_cells = staged_total_cells;" in body
    assert "*out_total_block_dots = staged_total_block_dots;" in body


def test_known_vector_and_no_publish_on_failure() -> None:
    row_count = 4
    col_count = 5
    k_block_count = 3
    iter_count = 6
    lhs_stride = 3
    rhs_stride = 3
    out_stride = 5

    rng = random.Random(20260421_8571)
    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x5656] * (row_count * out_stride)

    cpi = [0x1111]
    bdi = [0x2222]
    tc = [0x3333]
    tbd = [0x4444]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only_preflight_only(
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

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only_preflight_only(
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


def test_null_output_ptr_rejected() -> None:
    rng = random.Random(20260421_8572)
    lhs = [make_q4_block(rng)]
    rhs = [make_q8_block(rng)]
    out = [0]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only_preflight_only(
        lhs,
        len(lhs),
        1,
        1,
        rhs,
        len(rhs),
        1,
        1,
        1,
        out,
        len(out),
        1,
        1,
        None,
        [1],
        [1],
        [1],
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_NULL_PTR


def test_fuzz_parity_matches_explicit_checked_composition() -> None:
    random.seed(20260421_857)

    for i in range(2600):
        row_count = random.randint(0, 11)
        col_count = random.randint(0, 11)
        k_block_count = random.randint(0, 9)
        iter_count = random.randint(0, 20)

        lhs_stride = k_block_count + random.randint(0, 5)
        rhs_stride = k_block_count + random.randint(0, 5)
        out_stride = col_count + random.randint(0, 5)

        lhs_capacity = max(1, row_count * lhs_stride)
        rhs_capacity = max(1, col_count * rhs_stride)
        out_capacity = max(1, row_count * out_stride)

        rng = random.Random(20260421_1857000 + i)
        lhs = [make_q4_block(rng) for _ in range(lhs_capacity)]
        rhs = [make_q8_block(rng) for _ in range(rhs_capacity)]

        out_a = [0x6B6B] * out_capacity
        out_b = list(out_a)

        cpi_a = [0x5151]
        cpi_b = [0x5151]
        bdi_a = [0x6161]
        bdi_b = [0x6161]
        tc_a = [0x7171]
        tc_b = [0x7171]
        tbd_a = [0x8181]
        tbd_b = [0x8181]

        err_a = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_preflight_only_commit_only_preflight_only(
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
    test_source_contains_signature_and_no_write_parity_flow()
    test_known_vector_and_no_publish_on_failure()
    test_null_output_ptr_rejected()
    test_fuzz_parity_matches_explicit_checked_composition()
    print("ok")
