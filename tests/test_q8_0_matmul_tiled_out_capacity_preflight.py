#!/usr/bin/env python3
"""Out-index capacity parity for checked tiled Q8_0 matmul paths.

Targets IQ-131 centralization:
  - out_index = out_row_base + col_index
  - require out_index < out_cell_capacity
  - BAD_LEN for negative capacity / bounds overflow
  - OVERFLOW on checked signed add

This harness validates helper parity directly, then asserts scalar/AVX2
entrypoints share identical BAD_LEN/OVERFLOW surfaces for output-index writes.
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
    q8_0_matmul_tiled_avx2_q16_checked,
    validate_out_index_capacity_checked as validate_out_index_capacity_checked_avx2_q16,
)
from test_q8_0_matmul_tiled_avx2_q32 import (
    q8_0_matmul_tiled_avx2_q32_checked,
    validate_out_index_capacity_checked as validate_out_index_capacity_checked_avx2_q32,
)
from test_q8_0_matmul_tiled_checked import (
    q8_0_matmul_q16_tiled_checked,
)


def try_add_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def validate_out_index_capacity_checked_scalar_q16(
    out_row_base: int,
    col_index: int,
    out_cell_capacity: int,
) -> tuple[int, int]:
    if out_cell_capacity < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0
    if out_row_base < 0 or col_index < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    ok, out_index = try_add_i64_nonneg(out_row_base, col_index)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, 0
    if out_index >= out_cell_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    return Q8_0_AVX2_OK, out_index


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
        (0, 0, 1, Q8_0_AVX2_OK),
        (7, 9, 32, Q8_0_AVX2_OK),
        (31, 0, 32, Q8_0_AVX2_OK),
        (31, 1, 32, Q8_0_AVX2_ERR_BAD_LEN),
        (1, 1, 2, Q8_0_AVX2_ERR_BAD_LEN),
        (0, 0, -1, Q8_0_AVX2_ERR_BAD_LEN),
        (-1, 0, 8, Q8_0_AVX2_ERR_BAD_LEN),
        (0, -1, 8, Q8_0_AVX2_ERR_BAD_LEN),
        (I64_MAX, 1, I64_MAX, Q8_0_AVX2_ERR_OVERFLOW),
        (I64_MAX - 3, 5, I64_MAX, Q8_0_AVX2_ERR_OVERFLOW),
    ]

    for out_row_base, col_index, out_cap, expected_err in cases:
        err_scalar, out_scalar = validate_out_index_capacity_checked_scalar_q16(out_row_base, col_index, out_cap)
        err_q16, out_q16 = validate_out_index_capacity_checked_avx2_q16(out_row_base, col_index, out_cap)
        err_q32, out_q32 = validate_out_index_capacity_checked_avx2_q32(out_row_base, col_index, out_cap)

        assert err_scalar == err_q16 == err_q32 == expected_err
        if expected_err == Q8_0_AVX2_OK:
            assert out_scalar == out_q16 == out_q32


def test_helper_randomized_parity() -> None:
    rng = random.Random(20260416131)

    for _ in range(1000):
        mode = rng.choice(["ok", "bad", "overflow", "capacity"])
        if mode == "ok":
            out_cap = rng.randint(1, 1 << 20)
            out_row_base = rng.randint(0, out_cap - 1)
            col_index = rng.randint(0, out_cap - out_row_base - 1)
        elif mode == "bad":
            out_cap = rng.randint(0, 1 << 20)
            out_row_base = rng.choice([-rng.randint(1, 1 << 20), rng.randint(0, 1 << 20)])
            col_index = rng.choice([-rng.randint(1, 1 << 20), rng.randint(0, 1 << 20)])
            if out_row_base >= 0 and col_index >= 0 and out_cap >= 0:
                out_row_base = -1
        elif mode == "overflow":
            out_cap = I64_MAX
            out_row_base = I64_MAX - rng.randint(0, 64)
            col_index = rng.randint(1, 128)
        else:
            out_cap = rng.randint(0, 1 << 20)
            out_row_base = rng.randint(0, 1 << 20)
            col_index = rng.randint(0, 1 << 20)
            if out_row_base >= 0 and col_index >= 0 and out_cap >= 0:
                sum_index = out_row_base + col_index
                if sum_index < out_cap:
                    col_index = (out_cap - out_row_base) if out_cap >= out_row_base else col_index

        err_scalar, out_scalar = validate_out_index_capacity_checked_scalar_q16(out_row_base, col_index, out_cap)
        err_q16, out_q16 = validate_out_index_capacity_checked_avx2_q16(out_row_base, col_index, out_cap)
        err_q32, out_q32 = validate_out_index_capacity_checked_avx2_q32(out_row_base, col_index, out_cap)

        assert err_scalar == err_q16 == err_q32
        if err_scalar == Q8_0_AVX2_OK:
            assert out_scalar == out_q16 == out_q32


def test_entrypoint_capacity_surface_still_matches() -> None:
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(64)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(64)]

    # preflight requires out_cap >= rows*out_stride; inner write needs out_index < out_cap.
    # rows=2, out_stride=4 => required=8. With cols=4, last write wants index 7.
    # out_cap=7 passes nothing: should fail BAD_LEN identically at inner out-index check.
    scenario = dict(
        rows=2,
        cols=4,
        lhs_stride=2,
        rhs_stride=2,
        k_blocks=2,
        out_stride=4,
        tile_rows=1,
        tile_cols=2,
        lhs_cap=64,
        rhs_cap=64,
        out_cap=7,
    )

    scalar_err = run_scalar_q16(scenario, lhs, rhs)
    avx2_q16_err = run_avx2_q16(scenario, lhs, rhs)
    avx2_q32_err = run_avx2_q32(scenario, lhs, rhs)

    assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_ERR_BAD_LEN


def test_entrypoint_overflow_surface_still_matches() -> None:
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(64)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(64)]

    # Keep lhs/rhs required-capacity terms at zero so every path reaches the
    # shared out_required = rows * out_stride overflow site.
    scenario = dict(
        rows=I64_MAX,
        cols=0,
        lhs_stride=0,
        rhs_stride=0,
        k_blocks=0,
        out_stride=2,
        tile_rows=1,
        tile_cols=1,
        lhs_cap=64,
        rhs_cap=64,
        out_cap=64,
    )

    scalar_err = run_scalar_q16(scenario, lhs, rhs)
    avx2_q16_err = run_avx2_q16(scenario, lhs, rhs)
    avx2_q32_err = run_avx2_q32(scenario, lhs, rhs)

    assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_ERR_OVERFLOW


def run() -> None:
    test_helper_targeted_cases_match_all_paths()
    test_helper_randomized_parity()
    test_entrypoint_capacity_surface_still_matches()
    test_entrypoint_overflow_surface_still_matches()
    print("q8_0_matmul_tiled_out_capacity_preflight=ok")


if __name__ == "__main__":
    run()
