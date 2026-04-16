#!/usr/bin/env python3
"""Stride/K preflight parity across scalar + AVX2 tiled Q8_0 matmul entrypoints.

Targets IQ-121 invariant centralization:
  - k_block_count must fit both source strides
  - out_row_stride must cover active output columns when rows > 0

The harness checks that scalar Q16, AVX2 Q16, and AVX2 Q32 surfaces agree on
BAD_LEN for identical stride/K-invalid inputs, and all three continue to accept
valid shape contracts.
"""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_avx2_blocks_q32 import (
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_OK,
    make_block,
)
from test_q8_0_matmul_tiled_avx2_q16 import q8_0_matmul_tiled_avx2_q16_checked
from test_q8_0_matmul_tiled_avx2_q32 import q8_0_matmul_tiled_avx2_q32_checked
from test_q8_0_matmul_tiled_checked import q8_0_matmul_q16_tiled_checked


def to_scalar_blocks(blocks):
    out = []
    for block in blocks:
        qs = bytes((int(v) + 256) & 0xFF for v in block["qs"])
        out.append((int(block["d_fp16"]) & 0xFFFF, qs))
    return out


def run_scalar_q16(case: dict[str, int], lhs, rhs) -> int:
    lhs_scalar = to_scalar_blocks(lhs)
    rhs_scalar = to_scalar_blocks(rhs)
    err, _ = q8_0_matmul_q16_tiled_checked(
        lhs_blocks=lhs_scalar,
        lhs_block_capacity=case["lhs_cap"],
        row_count=case["rows"],
        lhs_row_stride_blocks=case["lhs_stride"],
        rhs_col_blocks=rhs_scalar,
        rhs_block_capacity=case["rhs_cap"],
        col_count=case["cols"],
        rhs_col_stride_blocks=case["rhs_stride"],
        k_block_count=case["k_blocks"],
        tile_rows=case["tile_rows"],
        tile_cols=case["tile_cols"],
        out_cell_capacity=case["out_cap"],
        out_row_stride_cells=case["out_stride"],
    )
    return err


def run_avx2_q16(case: dict[str, int], lhs, rhs) -> int:
    err, _ = q8_0_matmul_tiled_avx2_q16_checked(
        lhs_blocks=lhs,
        lhs_block_capacity=case["lhs_cap"],
        row_count=case["rows"],
        lhs_row_stride_blocks=case["lhs_stride"],
        rhs_col_blocks=rhs,
        rhs_block_capacity=case["rhs_cap"],
        col_count=case["cols"],
        rhs_col_stride_blocks=case["rhs_stride"],
        k_block_count=case["k_blocks"],
        tile_rows=case["tile_rows"],
        tile_cols=case["tile_cols"],
        out_cell_capacity=case["out_cap"],
        out_row_stride_cells=case["out_stride"],
    )
    return err


def run_avx2_q32(case: dict[str, int], lhs, rhs) -> int:
    err, _ = q8_0_matmul_tiled_avx2_q32_checked(
        lhs_matrix_blocks=lhs,
        lhs_block_capacity=case["lhs_cap"],
        lhs_rows=case["rows"],
        lhs_row_stride_blocks=case["lhs_stride"],
        rhs_col_blocks=rhs,
        rhs_block_capacity=case["rhs_cap"],
        rhs_cols=case["cols"],
        rhs_col_stride_blocks=case["rhs_stride"],
        k_block_count=case["k_blocks"],
        tile_rows=case["tile_rows"],
        tile_cols=case["tile_cols"],
        out_capacity=case["out_cap"],
        out_row_stride_cols=case["out_stride"],
    )
    return err


def test_targeted_stride_k_invariants_match_all_entrypoints() -> None:
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(64)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(64)]

    scenarios = [
        # k_block_count cannot exceed either source stride.
        dict(rows=2, cols=3, lhs_stride=2, rhs_stride=3, k_blocks=4, out_stride=3, tile_rows=1, tile_cols=2, lhs_cap=64, rhs_cap=64, out_cap=64),
        dict(rows=2, cols=3, lhs_stride=4, rhs_stride=2, k_blocks=3, out_stride=3, tile_rows=2, tile_cols=1, lhs_cap=64, rhs_cap=64, out_cap=64),
        # out_row_stride must cover col_count whenever rows are active.
        dict(rows=4, cols=5, lhs_stride=3, rhs_stride=3, k_blocks=2, out_stride=4, tile_rows=2, tile_cols=2, lhs_cap=64, rhs_cap=64, out_cap=64),
        dict(rows=1, cols=2, lhs_stride=2, rhs_stride=2, k_blocks=2, out_stride=1, tile_rows=1, tile_cols=1, lhs_cap=64, rhs_cap=64, out_cap=64),
        # rows == 0 permits narrow out stride; k/stride contract still enforced.
        dict(rows=0, cols=7, lhs_stride=1, rhs_stride=1, k_blocks=2, out_stride=0, tile_rows=1, tile_cols=1, lhs_cap=64, rhs_cap=64, out_cap=64),
    ]

    for case in scenarios:
        scalar_err = run_scalar_q16(case, lhs, rhs)
        avx2_q16_err = run_avx2_q16(case, lhs, rhs)
        avx2_q32_err = run_avx2_q32(case, lhs, rhs)
        assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_ERR_BAD_LEN


def test_randomized_valid_stride_k_contract_acceptance() -> None:
    rng = random.Random(20260416121)
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(256)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(256)]

    for _ in range(300):
        rows = rng.randint(0, 8)
        cols = rng.randint(0, 8)
        lhs_stride = rng.randint(0, 8)
        rhs_stride = rng.randint(0, 8)
        k_blocks = rng.randint(0, min(lhs_stride, rhs_stride) if min(lhs_stride, rhs_stride) > 0 else 0)
        out_stride = rng.randint(cols, cols + 4) if rows > 0 else rng.randint(0, 4)

        case = dict(
            rows=rows,
            cols=cols,
            lhs_stride=lhs_stride,
            rhs_stride=rhs_stride,
            k_blocks=k_blocks,
            out_stride=out_stride,
            tile_rows=rng.randint(1, 4),
            tile_cols=rng.randint(1, 4),
            lhs_cap=rows * lhs_stride,
            rhs_cap=cols * rhs_stride,
            out_cap=rows * out_stride,
        )

        scalar_err = run_scalar_q16(case, lhs, rhs)
        avx2_q16_err = run_avx2_q16(case, lhs, rhs)
        avx2_q32_err = run_avx2_q32(case, lhs, rhs)

        assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_OK


def run() -> None:
    test_targeted_stride_k_invariants_match_all_entrypoints()
    test_randomized_valid_stride_k_contract_acceptance()
    print("q8_0_matmul_tiled_stride_k_preflight=ok")


if __name__ == "__main__":
    run()
