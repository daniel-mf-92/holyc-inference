#!/usr/bin/env python3
"""Shared preflight error-surface parity checks for Q8_0 tiled AVX2 matmul.

This harness isolates argument/capacity validation (BAD_LEN/OVERFLOW) and
proves Q32 and Q16 entrypoints fail identically for the same invalid shape
configurations. Dot-product math is intentionally bypassed by using block sets
large enough for any valid preflight scenario in this file.
"""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_avx2_blocks_q32 import (
    I64_MAX,
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_ERR_OVERFLOW,
    Q8_0_AVX2_OK,
    make_block,
)
from test_q8_0_matmul_tiled_avx2_q16 import q8_0_matmul_tiled_avx2_q16_checked
from test_q8_0_matmul_tiled_avx2_q32 import q8_0_matmul_tiled_avx2_q32_checked


def run_scenario(case: dict[str, int], lhs, rhs) -> tuple[int, int]:
    err_q16, _ = q8_0_matmul_tiled_avx2_q16_checked(
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

    err_q32, _ = q8_0_matmul_tiled_avx2_q32_checked(
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

    return err_q16, err_q32


def test_targeted_bad_len_and_overflow_surface() -> None:
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(32)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(32)]

    scenarios = [
        # BAD_LEN: negative dimensions/strides/capacities.
        dict(rows=-1, cols=2, k_blocks=1, lhs_stride=1, rhs_stride=1, out_stride=2, tile_rows=1, tile_cols=1, lhs_cap=32, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_BAD_LEN),
        dict(rows=2, cols=-1, k_blocks=1, lhs_stride=1, rhs_stride=1, out_stride=2, tile_rows=1, tile_cols=1, lhs_cap=32, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_BAD_LEN),
        dict(rows=2, cols=2, k_blocks=1, lhs_stride=-1, rhs_stride=1, out_stride=2, tile_rows=1, tile_cols=1, lhs_cap=32, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_BAD_LEN),
        dict(rows=2, cols=2, k_blocks=1, lhs_stride=1, rhs_stride=-1, out_stride=2, tile_rows=1, tile_cols=1, lhs_cap=32, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_BAD_LEN),
        dict(rows=2, cols=2, k_blocks=1, lhs_stride=1, rhs_stride=1, out_stride=-1, tile_rows=1, tile_cols=1, lhs_cap=32, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_BAD_LEN),
        dict(rows=2, cols=2, k_blocks=1, lhs_stride=1, rhs_stride=1, out_stride=2, tile_rows=0, tile_cols=1, lhs_cap=32, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_BAD_LEN),
        dict(rows=2, cols=2, k_blocks=3, lhs_stride=2, rhs_stride=3, out_stride=2, tile_rows=1, tile_cols=1, lhs_cap=32, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_BAD_LEN),
        dict(rows=2, cols=2, k_blocks=2, lhs_stride=2, rhs_stride=2, out_stride=1, tile_rows=1, tile_cols=1, lhs_cap=32, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_BAD_LEN),
        dict(rows=3, cols=3, k_blocks=2, lhs_stride=3, rhs_stride=3, out_stride=3, tile_rows=2, tile_cols=2, lhs_cap=8, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_BAD_LEN),
        dict(rows=3, cols=3, k_blocks=2, lhs_stride=3, rhs_stride=3, out_stride=3, tile_rows=2, tile_cols=2, lhs_cap=32, rhs_cap=8, out_cap=32, expected=Q8_0_AVX2_ERR_BAD_LEN),
        dict(rows=3, cols=3, k_blocks=2, lhs_stride=3, rhs_stride=3, out_stride=3, tile_rows=2, tile_cols=2, lhs_cap=32, rhs_cap=32, out_cap=8, expected=Q8_0_AVX2_ERR_BAD_LEN),
        # OVERFLOW: non-negative products that overflow I64 capacity proofs.
        dict(rows=I64_MAX, cols=2, k_blocks=1, lhs_stride=2, rhs_stride=1, out_stride=2, tile_rows=1, tile_cols=1, lhs_cap=32, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_OVERFLOW),
        dict(rows=0, cols=I64_MAX, k_blocks=1, lhs_stride=1, rhs_stride=2, out_stride=1, tile_rows=1, tile_cols=1, lhs_cap=32, rhs_cap=32, out_cap=32, expected=Q8_0_AVX2_ERR_OVERFLOW),
    ]

    for case in scenarios:
        err_q16, err_q32 = run_scenario(case, lhs, rhs)
        assert err_q16 == err_q32
        assert err_q16 == case["expected"]


def test_randomized_preflight_parity_invalid_inputs() -> None:
    rng = random.Random(2026041609)
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(128)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(128)]

    samples = 450
    seen_bad_len = 0
    seen_overflow = 0

    for _ in range(samples):
        kind = rng.choice(["bad_len", "overflow"])

        if kind == "overflow":
            case = dict(
                rows=I64_MAX,
                cols=rng.choice([1, 2, 3, 7]),
                k_blocks=rng.choice([1, 2, 4]),
                lhs_stride=2,
                rhs_stride=rng.choice([1, 2, 4]),
                out_stride=rng.choice([1, 2, 4]),
                tile_rows=rng.choice([1, 2, 3, 5]),
                tile_cols=rng.choice([1, 2, 3, 5]),
                lhs_cap=128,
                rhs_cap=128,
                out_cap=128,
            )
        else:
            rows = rng.randint(1, 8)
            cols = rng.randint(2, 8)
            lhs_stride = rng.randint(1, 8)
            rhs_stride = rng.randint(1, 8)
            out_stride = rng.randint(1, 8)
            k_blocks = rng.randint(1, 8)

            mode = rng.choice(["k_vs_stride", "out_stride", "lhs_cap", "rhs_cap", "out_cap"])
            if mode == "k_vs_stride":
                max_stride = min(lhs_stride, rhs_stride)
                k_blocks = max_stride + rng.randint(1, 4)
            elif mode == "out_stride":
                out_stride = cols - 1
            elif mode == "lhs_cap":
                lhs_cap = max(0, rows * lhs_stride - rng.randint(1, max(1, rows * lhs_stride)))
                rhs_cap = 128
                out_cap = 128
            elif mode == "rhs_cap":
                lhs_cap = 128
                rhs_cap = max(0, cols * rhs_stride - rng.randint(1, max(1, cols * rhs_stride)))
                out_cap = 128
            else:
                lhs_cap = 128
                rhs_cap = 128
                out_cap = max(0, rows * out_stride - rng.randint(1, max(1, rows * out_stride)))

            if mode in ("k_vs_stride", "out_stride"):
                lhs_cap = 128
                rhs_cap = 128
                out_cap = 128

            case = dict(
                rows=rows,
                cols=cols,
                k_blocks=k_blocks,
                lhs_stride=lhs_stride,
                rhs_stride=rhs_stride,
                out_stride=out_stride,
                tile_rows=rng.choice([1, 2, 3, 5]),
                tile_cols=rng.choice([1, 2, 3, 5]),
                lhs_cap=lhs_cap,
                rhs_cap=rhs_cap,
                out_cap=out_cap,
            )

        err_q16, err_q32 = run_scenario(case, lhs, rhs)
        assert err_q16 == err_q32
        assert err_q16 in (Q8_0_AVX2_ERR_BAD_LEN, Q8_0_AVX2_ERR_OVERFLOW)

        if err_q16 == Q8_0_AVX2_ERR_BAD_LEN:
            seen_bad_len += 1
        else:
            seen_overflow += 1

    assert seen_bad_len > 0
    assert seen_overflow > 0


def run() -> None:
    test_targeted_bad_len_and_overflow_surface()
    test_randomized_preflight_parity_invalid_inputs()
    print("q8_0_matmul_tiled_avx2_preflight_parity=ok")


if __name__ == "__main__":
    run()
