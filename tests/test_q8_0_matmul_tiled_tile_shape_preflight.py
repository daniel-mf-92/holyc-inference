#!/usr/bin/env python3
"""Tile-shape preflight parity across scalar + AVX2 tiled Q8_0 matmul entrypoints.

Targets IQ-122 invariant centralization:
  - tile_rows > 0
  - tile_cols > 0

The harness checks BAD_LEN parity for non-positive tile extents and confirms
all entrypoints accept valid tile shapes under identical matrix contracts.
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


def test_targeted_non_positive_tile_shapes_rejected() -> None:
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(64)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(64)]

    base = dict(
        rows=3,
        cols=4,
        lhs_stride=3,
        rhs_stride=3,
        k_blocks=2,
        out_stride=4,
        lhs_cap=64,
        rhs_cap=64,
        out_cap=64,
    )

    scenarios = [
        dict(base, tile_rows=0, tile_cols=1),
        dict(base, tile_rows=1, tile_cols=0),
        dict(base, tile_rows=-1, tile_cols=2),
        dict(base, tile_rows=2, tile_cols=-1),
        dict(base, tile_rows=0, tile_cols=0),
    ]

    for case in scenarios:
        scalar_err = run_scalar_q16(case, lhs, rhs)
        avx2_q16_err = run_avx2_q16(case, lhs, rhs)
        avx2_q32_err = run_avx2_q32(case, lhs, rhs)
        assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_ERR_BAD_LEN


def test_randomized_valid_tile_shape_acceptance_parity() -> None:
    rng = random.Random(20260416122)
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(512)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(512)]

    for _ in range(300):
        rows = rng.randint(0, 8)
        cols = rng.randint(0, 8)
        lhs_stride = rng.randint(0, 8)
        rhs_stride = rng.randint(0, 8)
        max_k = min(lhs_stride, rhs_stride)
        k_blocks = rng.randint(0, max_k)
        out_stride = rng.randint(cols, cols + 4) if rows > 0 else rng.randint(0, 4)

        case = dict(
            rows=rows,
            cols=cols,
            lhs_stride=lhs_stride,
            rhs_stride=rhs_stride,
            k_blocks=k_blocks,
            out_stride=out_stride,
            tile_rows=rng.randint(1, 5),
            tile_cols=rng.randint(1, 5),
            lhs_cap=rows * lhs_stride,
            rhs_cap=cols * rhs_stride,
            out_cap=rows * out_stride,
        )

        scalar_err = run_scalar_q16(case, lhs, rhs)
        avx2_q16_err = run_avx2_q16(case, lhs, rhs)
        avx2_q32_err = run_avx2_q32(case, lhs, rhs)
        assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_OK


def run() -> None:
    test_targeted_non_positive_tile_shapes_rejected()
    test_randomized_valid_tile_shape_acceptance_parity()
    print("q8_0_matmul_tiled_tile_shape_preflight=ok")


if __name__ == "__main__":
    run()
