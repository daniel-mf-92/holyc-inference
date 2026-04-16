#!/usr/bin/env python3
"""Out-index helper parity across scalar + AVX2 tiled Q8_0 matmul paths.

Targets IQ-125 centralization:
  - out_index = out_row_base + col_index
  - BAD_LEN for negative row-base/col-index
  - OVERFLOW on checked signed add

The harness checks helper-level parity directly and then asserts all three
entrypoints still agree on overflow surfaces for large non-negative index math.
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
from test_q8_0_matmul_tiled_avx2_q16 import (
    compute_out_index_checked as compute_out_index_checked_avx2_q16,
    q8_0_matmul_tiled_avx2_q16_checked,
)
from test_q8_0_matmul_tiled_avx2_q32 import (
    compute_out_index_checked as compute_out_index_checked_avx2_q32,
    q8_0_matmul_tiled_avx2_q32_checked,
)
from test_q8_0_matmul_tiled_checked import (
    compute_out_index_checked as compute_out_index_checked_scalar_q16,
    q8_0_matmul_q16_tiled_checked,
)


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


def test_helper_targeted_cases_match_all_paths() -> None:
    cases = [
        (0, 0, Q8_0_AVX2_OK),
        (7, 9, Q8_0_AVX2_OK),
        (I64_MAX - 1, 1, Q8_0_AVX2_OK),
        (-1, 0, Q8_0_AVX2_ERR_BAD_LEN),
        (0, -1, Q8_0_AVX2_ERR_BAD_LEN),
        (I64_MAX, 1, Q8_0_AVX2_ERR_OVERFLOW),
        (I64_MAX - 3, 5, Q8_0_AVX2_ERR_OVERFLOW),
    ]

    for out_row_base, col_index, expected_err in cases:
        err_scalar, out_scalar = compute_out_index_checked_scalar_q16(out_row_base, col_index)
        err_q16, out_q16 = compute_out_index_checked_avx2_q16(out_row_base, col_index)
        err_q32, out_q32 = compute_out_index_checked_avx2_q32(out_row_base, col_index)

        assert err_scalar == err_q16 == err_q32 == expected_err
        if expected_err == Q8_0_AVX2_OK:
            assert out_scalar == out_q16 == out_q32


def test_helper_randomized_parity() -> None:
    rng = random.Random(20260416125)

    for _ in range(800):
        mode = rng.choice(["ok", "bad", "overflow"])
        if mode == "ok":
            out_row_base = rng.randint(0, 1 << 20)
            col_index = rng.randint(0, 1 << 20)
        elif mode == "bad":
            out_row_base = rng.choice([-rng.randint(1, 1 << 20), rng.randint(0, 1 << 20)])
            col_index = rng.choice([-rng.randint(1, 1 << 20), rng.randint(0, 1 << 20)])
            if out_row_base >= 0 and col_index >= 0:
                out_row_base = -1
        else:
            out_row_base = I64_MAX - rng.randint(0, 64)
            col_index = rng.randint(1, 128)

        err_scalar, out_scalar = compute_out_index_checked_scalar_q16(out_row_base, col_index)
        err_q16, out_q16 = compute_out_index_checked_avx2_q16(out_row_base, col_index)
        err_q32, out_q32 = compute_out_index_checked_avx2_q32(out_row_base, col_index)

        assert err_scalar == err_q16 == err_q32
        if err_scalar == Q8_0_AVX2_OK:
            assert out_scalar == out_q16 == out_q32


def test_entrypoint_overflow_surface_still_matches() -> None:
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(128)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(128)]

    scenario = dict(
        rows=I64_MAX,
        cols=3,
        lhs_stride=2,
        rhs_stride=2,
        k_blocks=1,
        out_stride=I64_MAX,
        tile_rows=1,
        tile_cols=2,
        lhs_cap=128,
        rhs_cap=128,
        out_cap=128,
    )

    # rows * out_stride overflows first; all entrypoints must surface OVERFLOW.
    scalar_err = run_scalar_q16(scenario, lhs, rhs)
    avx2_q16_err = run_avx2_q16(scenario, lhs, rhs)
    avx2_q32_err = run_avx2_q32(scenario, lhs, rhs)

    assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_ERR_OVERFLOW


def run() -> None:
    test_helper_targeted_cases_match_all_paths()
    test_helper_randomized_parity()
    test_entrypoint_overflow_surface_still_matches()
    print("q8_0_matmul_tiled_out_index_preflight=ok")


if __name__ == "__main__":
    run()
