#!/usr/bin/env python3
"""Row-slice preflight parity for checked tiled Q8_0 matmul paths.

Targets IQ-132 and `Q8_0MatMulTiledValidateRowSliceChecked` semantics:
  - row_slice_end = row_base + k_block_count
  - BAD_LEN for negative operands and bounds overflow (`row_slice_end > capacity`)
  - OVERFLOW on checked signed add

The harness validates helper-level parity across scalar/AVX2 reference helpers,
then checks scalar + AVX2 tiled entrypoints keep aligned BAD_LEN/OVERFLOW
surfaces on row-slice-adjacent capacity invariants.
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
from test_q8_0_matmul_tiled_avx2_q32 import (
    q8_0_matmul_tiled_avx2_q32_checked,
    q8_0_try_add_i64,
)
from test_q8_0_matmul_tiled_checked import (
    q8_0_matmul_q16_tiled_checked,
    validate_row_slice_checked,
)


def validate_row_slice_checked_avx2_q16(
    row_base: int,
    k_block_count: int,
    block_capacity: int,
) -> tuple[int, int]:
    # Semantics mirror HolyC helper contract and the AVX2 callers.
    if row_base < 0 or k_block_count < 0 or block_capacity < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    ok, row_slice_end = q8_0_try_add_i64(row_base, k_block_count)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, 0

    if row_slice_end > block_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    return Q8_0_AVX2_OK, row_slice_end


def validate_row_slice_checked_avx2_q32(
    row_base: int,
    k_block_count: int,
    block_capacity: int,
) -> tuple[int, int]:
    if row_base < 0 or k_block_count < 0 or block_capacity < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    ok, row_slice_end = q8_0_try_add_i64(row_base, k_block_count)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, 0

    if row_slice_end > block_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    return Q8_0_AVX2_OK, row_slice_end


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
        (0, 0, 0, Q8_0_AVX2_OK),
        (5, 3, 8, Q8_0_AVX2_OK),
        (7, 2, 8, Q8_0_AVX2_ERR_BAD_LEN),
        (1, 0, 0, Q8_0_AVX2_ERR_BAD_LEN),
        (-1, 2, 8, Q8_0_AVX2_ERR_BAD_LEN),
        (2, -1, 8, Q8_0_AVX2_ERR_BAD_LEN),
        (2, 1, -1, Q8_0_AVX2_ERR_BAD_LEN),
        (I64_MAX, 1, I64_MAX, Q8_0_AVX2_ERR_OVERFLOW),
        (I64_MAX - 4, 9, I64_MAX, Q8_0_AVX2_ERR_OVERFLOW),
    ]

    for row_base, k_blocks, capacity, expected_err in cases:
        err_scalar, end_scalar = validate_row_slice_checked(row_base, k_blocks, capacity)
        err_q16, end_q16 = validate_row_slice_checked_avx2_q16(row_base, k_blocks, capacity)
        err_q32, end_q32 = validate_row_slice_checked_avx2_q32(row_base, k_blocks, capacity)

        assert err_scalar == err_q16 == err_q32 == expected_err
        if expected_err == Q8_0_AVX2_OK:
            assert end_scalar == end_q16 == end_q32


def test_helper_randomized_parity() -> None:
    rng = random.Random(20260416132)

    for _ in range(1200):
        mode = rng.choice(["ok", "bad", "overflow", "bounds"])

        if mode == "ok":
            capacity = rng.randint(0, 1 << 20)
            row_base = rng.randint(0, capacity)
            k_blocks = rng.randint(0, capacity - row_base)
        elif mode == "bad":
            row_base = rng.choice([-rng.randint(1, 1 << 20), rng.randint(0, 1 << 20)])
            k_blocks = rng.choice([-rng.randint(1, 1 << 20), rng.randint(0, 1 << 20)])
            capacity = rng.choice([-rng.randint(1, 1 << 20), rng.randint(0, 1 << 20)])
            if row_base >= 0 and k_blocks >= 0 and capacity >= 0:
                row_base = -1
        elif mode == "overflow":
            capacity = I64_MAX
            row_base = I64_MAX - rng.randint(0, 256)
            k_blocks = rng.randint(1, 512)
        else:
            capacity = rng.randint(0, 1 << 20)
            row_base = rng.randint(0, 1 << 20)
            k_blocks = rng.randint(0, 1 << 20)
            if row_base >= 0 and k_blocks >= 0 and capacity >= 0 and row_base + k_blocks <= capacity:
                k_blocks = (capacity - row_base) + 1 if capacity >= row_base else k_blocks

        err_scalar, end_scalar = validate_row_slice_checked(row_base, k_blocks, capacity)
        err_q16, end_q16 = validate_row_slice_checked_avx2_q16(row_base, k_blocks, capacity)
        err_q32, end_q32 = validate_row_slice_checked_avx2_q32(row_base, k_blocks, capacity)

        assert err_scalar == err_q16 == err_q32
        if err_scalar == Q8_0_AVX2_OK:
            assert end_scalar == end_q16 == end_q32


def test_entrypoint_row_slice_boundary_ok_matches() -> None:
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(16)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(16)]

    # Last row/col slices end exactly at capacity: row_base + k == capacity.
    scenario = dict(
        rows=2,
        cols=2,
        lhs_stride=3,
        rhs_stride=3,
        k_blocks=3,
        out_stride=2,
        tile_rows=1,
        tile_cols=1,
        lhs_cap=6,
        rhs_cap=6,
        out_cap=4,
    )

    scalar_err = run_scalar_q16(scenario, lhs, rhs)
    avx2_q16_err = run_avx2_q16(scenario, lhs, rhs)
    avx2_q32_err = run_avx2_q32(scenario, lhs, rhs)

    assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_OK


def test_entrypoint_row_slice_bad_len_surface_matches() -> None:
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(16)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(16)]

    # rows*lhs_stride = 6 but lhs_cap is 5 -> BAD_LEN at shared row-slice-adjacent capacity guard.
    scenario = dict(
        rows=2,
        cols=2,
        lhs_stride=3,
        rhs_stride=3,
        k_blocks=3,
        out_stride=2,
        tile_rows=1,
        tile_cols=1,
        lhs_cap=5,
        rhs_cap=6,
        out_cap=4,
    )

    scalar_err = run_scalar_q16(scenario, lhs, rhs)
    avx2_q16_err = run_avx2_q16(scenario, lhs, rhs)
    avx2_q32_err = run_avx2_q32(scenario, lhs, rhs)

    assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_ERR_BAD_LEN


def test_entrypoint_row_slice_overflow_surface_matches() -> None:
    lhs = [make_block(0x3C00, [1] * 32)]
    rhs = [make_block(0x3C00, [1] * 32)]

    # Keep rhs/out required products zero so all paths hit lhs required overflow first.
    scenario = dict(
        rows=I64_MAX,
        cols=0,
        lhs_stride=2,
        rhs_stride=0,
        k_blocks=0,
        out_stride=0,
        tile_rows=1,
        tile_cols=1,
        lhs_cap=1,
        rhs_cap=1,
        out_cap=1,
    )

    scalar_err = run_scalar_q16(scenario, lhs, rhs)
    avx2_q16_err = run_avx2_q16(scenario, lhs, rhs)
    avx2_q32_err = run_avx2_q32(scenario, lhs, rhs)

    assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_ERR_OVERFLOW


def run() -> None:
    test_helper_targeted_cases_match_all_paths()
    test_helper_randomized_parity()
    test_entrypoint_row_slice_boundary_ok_matches()
    test_entrypoint_row_slice_bad_len_surface_matches()
    test_entrypoint_row_slice_overflow_surface_matches()
    print("q8_0_matmul_tiled_row_slice_preflight=ok")


if __name__ == "__main__":
    run()
