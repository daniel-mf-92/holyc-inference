#!/usr/bin/env python3
"""Parity harness for ...PreflightOnlyCommitOnlyParity...PreflightOnlyCommitOnlyParity (IQ-868)."""

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
from test_q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only import (
    q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only,
)
from test_q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only_commit_only import (
    q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only_commit_only,
)


def q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only_commit_only_parity(
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

    preflight_cells_per_iter = [0]
    preflight_block_dots_per_iter = [0]
    preflight_total_cells = [0]
    preflight_total_block_dots = [0]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only_commit_only(
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

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only(
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
        preflight_cells_per_iter,
        preflight_block_dots_per_iter,
        preflight_total_cells,
        preflight_total_block_dots,
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

    if commit_cells_per_iter[0] != preflight_cells_per_iter[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if commit_block_dots_per_iter[0] != preflight_block_dots_per_iter[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if commit_total_cells[0] != preflight_total_cells[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    if commit_total_block_dots[0] != preflight_total_block_dots[0]:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    out_cells_per_iter[0] = commit_cells_per_iter[0]
    out_block_dots_per_iter[0] = commit_block_dots_per_iter[0]
    out_total_cells[0] = commit_total_cells[0]
    out_total_block_dots[0] = commit_total_block_dots[0]
    return Q4_0_Q8_0_AVX2_OK


def explicit_checked_composition(*args):
    return q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only_commit_only_parity(
        *args
    )


def test_source_contains_signature_and_parity_gate() -> None:
    source = Path("src/matmul/q4_0_q8_0_matmul.HC").read_text(encoding="utf-8")
    sig = (
        "I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParity"
        "CommitOnlyPreflightOnlyParityPreflightOnlyCommitOnlyParity("
    )
    assert sig in source
    body = source.split(sig, 1)[1]

    assert (
        "Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParity"
        "CommitOnlyPreflightOnlyParityPreflightOnlyCommitOnly(" in body
    )
    assert (
        "Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParity"
        "CommitOnlyPreflightOnlyParityPreflightOnly(" in body
    )
    assert "if (commit_cells_per_iter != preflight_cells_per_iter)" in body
    assert "if (commit_block_dots_per_iter != preflight_block_dots_per_iter)" in body
    assert "if (commit_total_cells != preflight_total_cells)" in body
    assert "if (commit_total_block_dots != preflight_total_block_dots)" in body
    assert "*out_cells_per_iter = commit_cells_per_iter;" in body
    assert "*out_block_dots_per_iter = commit_block_dots_per_iter;" in body
    assert "*out_total_cells = commit_total_cells;" in body
    assert "*out_total_block_dots = commit_total_block_dots;" in body


def test_known_vector_and_no_partial_publish_on_failure() -> None:
    row_count = 5
    col_count = 6
    k_block_count = 4
    iter_count = 3
    lhs_stride = 4
    rhs_stride = 4
    out_stride = 6

    rng = random.Random(20260421_8681)
    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x5656] * (row_count * out_stride)

    cpi = [0x1111]
    bdi = [0x2222]
    tc = [0x3333]
    tbd = [0x4444]

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only_commit_only_parity(
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

    err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only_commit_only_parity(
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


def test_adversarial_geometry_capacity_vectors_match_explicit_composition() -> None:
    vectors = [
        # degenerate but valid zero-work geometry
        dict(rc=0, cc=0, kb=0, it=0, ls=0, rs=0, os=0, lc=1, rc2=1, oc=1),
        # exact-fit capacities
        dict(rc=1, cc=1, kb=1, it=1, ls=1, rs=1, os=1, lc=1, rc2=1, oc=1),
        # over-provisioned strides/capacities
        dict(rc=3, cc=2, kb=2, it=4, ls=5, rs=4, os=6, lc=15, rc2=8, oc=18),
        # under-capacity lhs
        dict(rc=4, cc=3, kb=2, it=2, ls=2, rs=2, os=3, lc=7, rc2=6, oc=12),
        # under-capacity rhs
        dict(rc=2, cc=4, kb=3, it=2, ls=3, rs=3, os=4, lc=6, rc2=11, oc=8),
        # under-capacity output
        dict(rc=3, cc=3, kb=2, it=2, ls=2, rs=2, os=3, lc=6, rc2=6, oc=8),
        # negative iter count rejected
        dict(rc=2, cc=2, kb=2, it=-1, ls=2, rs=2, os=2, lc=4, rc2=4, oc=4),
        # bad stride smaller than k_block_count
        dict(rc=2, cc=2, kb=3, it=1, ls=2, rs=3, os=2, lc=4, rc2=6, oc=4),
    ]

    for i, vec in enumerate(vectors):
        rng = random.Random(20260421_868100 + i)

        lhs = [make_q4_block(rng) for _ in range(max(1, vec["lc"]))]
        rhs = [make_q8_block(rng) for _ in range(max(1, vec["rc2"]))]
        out_a = [0x5A5A] * max(1, vec["oc"])
        out_b = list(out_a)

        cpi_a = [0x1111]
        cpi_b = [0x1111]
        bdi_a = [0x2222]
        bdi_b = [0x2222]
        tc_a = [0x3333]
        tc_b = [0x3333]
        tbd_a = [0x4444]
        tbd_b = [0x4444]

        err_a = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only_commit_only_parity(
            lhs,
            vec["lc"],
            vec["rc"],
            vec["ls"],
            rhs,
            vec["rc2"],
            vec["cc"],
            vec["rs"],
            vec["kb"],
            out_a,
            vec["oc"],
            vec["os"],
            vec["it"],
            cpi_a,
            bdi_a,
            tc_a,
            tbd_a,
        )
        err_b = explicit_checked_composition(
            lhs,
            vec["lc"],
            vec["rc"],
            vec["ls"],
            rhs,
            vec["rc2"],
            vec["cc"],
            vec["rs"],
            vec["kb"],
            out_b,
            vec["oc"],
            vec["os"],
            vec["it"],
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


def test_no_partial_publish_when_subwrappers_disagree() -> None:
    row_count = 3
    col_count = 4
    k_block_count = 2
    iter_count = 3
    lhs_stride = 2
    rhs_stride = 2
    out_stride = 4

    rng = random.Random(20260421_868200)
    lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
    rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]
    out = [0x7777] * (row_count * out_stride)

    cpi = [0xA1A1]
    bdi = [0xB2B2]
    tc = [0xC3C3]
    tbd = [0xD4D4]

    global q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only
    real_preflight = (
        q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only
    )

    def tampered_preflight(*args):
        err = real_preflight(*args)
        if err == Q4_0_Q8_0_AVX2_OK:
            args[14][0] += 1
        return err

    q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only = tampered_preflight
    try:
        err = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only_commit_only_parity(
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
    finally:
        q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only = real_preflight

    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN
    assert cpi == [0xA1A1]
    assert bdi == [0xB2B2]
    assert tc == [0xC3C3]
    assert tbd == [0xD4D4]


def test_fuzz_parity_against_explicit_checked_composition() -> None:
    random.seed(20260421_868)

    for i in range(2400):
        row_count = random.randint(0, 10)
        col_count = random.randint(0, 10)
        k_block_count = random.randint(0, 8)
        iter_count = random.randint(0, 12)

        lhs_stride = k_block_count + random.randint(0, 4)
        rhs_stride = k_block_count + random.randint(0, 4)
        out_stride = col_count + random.randint(0, 4)

        lhs_capacity = max(1, row_count * lhs_stride)
        rhs_capacity = max(1, col_count * rhs_stride)
        out_capacity = max(1, row_count * out_stride)

        rng = random.Random(20260421_868000 + i)
        lhs = [make_q4_block(rng) for _ in range(lhs_capacity)]
        rhs = [make_q8_block(rng) for _ in range(rhs_capacity)]
        out_a = [0x3A3A] * out_capacity
        out_b = list(out_a)

        cpi_a = [0x1010]
        cpi_b = [0x1010]
        bdi_a = [0x2020]
        bdi_b = [0x2020]
        tc_a = [0x3030]
        tc_b = [0x3030]
        tbd_a = [0x4040]
        tbd_b = [0x4040]

        err_a = q4_0_q8_0_matmul_q32_tiled_avx2_bench_rows_preflight_only_commit_only_parity_commit_only_preflight_only_parity_preflight_only_commit_only_parity(
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
    test_source_contains_signature_and_parity_gate()
    test_known_vector_and_no_partial_publish_on_failure()
    test_adversarial_geometry_capacity_vectors_match_explicit_composition()
    test_no_partial_publish_when_subwrappers_disagree()
    test_fuzz_parity_against_explicit_checked_composition()
    print("ok")
