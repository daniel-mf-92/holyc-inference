#!/usr/bin/env python3
"""Row-base preflight parity across scalar + AVX2 tiled Q8_0 matmul.

Targets IQ-129 and `Q8_0MatMulTiledComputeRowBaseChecked` semantics:
  - lhs_row_base = row_index * lhs_row_stride_blocks
  - out_row_base = row_index * out_row_stride_cells
  - BAD_LEN on negative operands
  - OVERFLOW on checked signed multiply

The harness verifies helper-level parity and confirms all three tiled
entrypoints surface identical errors for row/stride overflow contracts.
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
    try_mul_i64_nonneg as try_mul_i64_nonneg_avx2_q16,
)
from test_q8_0_matmul_tiled_avx2_q32 import (
    q8_0_matmul_tiled_avx2_q32_checked,
    q8_0_try_mul_i64 as try_mul_i64_nonneg_avx2_q32,
)
from test_q8_0_matmul_tiled_checked import (
    q8_0_matmul_q16_tiled_checked,
    try_mul_i64_nonneg as try_mul_i64_nonneg_scalar_q16,
)


def compute_row_base_checked_scalar_q16(
    row_index: int,
    lhs_row_stride_blocks: int,
    out_row_stride_cells: int,
) -> tuple[int, tuple[int, int]]:
    if row_index < 0 or lhs_row_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, (0, 0)

    ok, lhs_row_base = try_mul_i64_nonneg_scalar_q16(row_index, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, (0, 0)

    ok, out_row_base = try_mul_i64_nonneg_scalar_q16(row_index, out_row_stride_cells)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, (0, 0)

    return Q8_0_AVX2_OK, (lhs_row_base, out_row_base)


def compute_row_base_checked_avx2_q16(
    row_index: int,
    lhs_row_stride_blocks: int,
    out_row_stride_cells: int,
) -> tuple[int, tuple[int, int]]:
    if row_index < 0 or lhs_row_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, (0, 0)

    ok, lhs_row_base = try_mul_i64_nonneg_avx2_q16(row_index, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, (0, 0)

    ok, out_row_base = try_mul_i64_nonneg_avx2_q16(row_index, out_row_stride_cells)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, (0, 0)

    return Q8_0_AVX2_OK, (lhs_row_base, out_row_base)


def compute_row_base_checked_avx2_q32(
    row_index: int,
    lhs_row_stride_blocks: int,
    out_row_stride_cells: int,
) -> tuple[int, tuple[int, int]]:
    if row_index < 0 or lhs_row_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, (0, 0)

    ok, lhs_row_base = try_mul_i64_nonneg_avx2_q32(row_index, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, (0, 0)

    ok, out_row_base = try_mul_i64_nonneg_avx2_q32(row_index, out_row_stride_cells)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, (0, 0)

    return Q8_0_AVX2_OK, (lhs_row_base, out_row_base)


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
        (9, 11, 13, Q8_0_AVX2_OK),
        (I64_MAX, 1, 1, Q8_0_AVX2_OK),
        (I64_MAX // 3 + 1, 3, 1, Q8_0_AVX2_ERR_OVERFLOW),
        (I64_MAX // 5 + 1, 1, 5, Q8_0_AVX2_ERR_OVERFLOW),
        (-1, 1, 1, Q8_0_AVX2_ERR_BAD_LEN),
        (1, -1, 1, Q8_0_AVX2_ERR_BAD_LEN),
        (1, 1, -1, Q8_0_AVX2_ERR_BAD_LEN),
    ]

    for row_index, lhs_stride, out_stride, expected_err in cases:
        err_scalar, out_scalar = compute_row_base_checked_scalar_q16(row_index, lhs_stride, out_stride)
        err_q16, out_q16 = compute_row_base_checked_avx2_q16(row_index, lhs_stride, out_stride)
        err_q32, out_q32 = compute_row_base_checked_avx2_q32(row_index, lhs_stride, out_stride)

        assert err_scalar == err_q16 == err_q32 == expected_err
        if expected_err == Q8_0_AVX2_OK:
            assert out_scalar == out_q16 == out_q32


def test_helper_randomized_parity() -> None:
    rng = random.Random(20260416129)

    for _ in range(1000):
        mode = rng.choice(["ok", "bad", "overflow_lhs", "overflow_out"])
        if mode == "ok":
            row_index = rng.randint(0, 1 << 20)
            lhs_stride = rng.randint(0, 1 << 20)
            out_stride = rng.randint(0, 1 << 20)
        elif mode == "bad":
            row_index = rng.choice([-rng.randint(1, 1 << 20), rng.randint(0, 1 << 20)])
            lhs_stride = rng.choice([-rng.randint(1, 1 << 20), rng.randint(0, 1 << 20)])
            out_stride = rng.choice([-rng.randint(1, 1 << 20), rng.randint(0, 1 << 20)])
            if row_index >= 0 and lhs_stride >= 0 and out_stride >= 0:
                row_index = -1
        elif mode == "overflow_lhs":
            row_index = I64_MAX - rng.randint(0, 1024)
            lhs_stride = rng.randint(2, 2048)
            out_stride = 1
        else:
            row_index = I64_MAX - rng.randint(0, 1024)
            lhs_stride = 1
            out_stride = rng.randint(2, 2048)

        err_scalar, out_scalar = compute_row_base_checked_scalar_q16(row_index, lhs_stride, out_stride)
        err_q16, out_q16 = compute_row_base_checked_avx2_q16(row_index, lhs_stride, out_stride)
        err_q32, out_q32 = compute_row_base_checked_avx2_q32(row_index, lhs_stride, out_stride)

        assert err_scalar == err_q16 == err_q32
        if err_scalar == Q8_0_AVX2_OK:
            assert out_scalar == out_q16 == out_q32


def test_entrypoint_row_stride_overflow_surface_matches() -> None:
    lhs = [make_block(0x3C00, [1] * 32) for _ in range(4)]
    rhs = [make_block(0x3C00, [1] * 32) for _ in range(4)]

    scenario = dict(
        rows=I64_MAX,
        cols=1,
        lhs_stride=2,
        rhs_stride=1,
        k_blocks=1,
        out_stride=1,
        tile_rows=1,
        tile_cols=1,
        lhs_cap=4,
        rhs_cap=4,
        out_cap=4,
    )

    # Preflight row*stride checked multiplies must report OVERFLOW identically.
    scalar_err = run_scalar_q16(scenario, lhs, rhs)
    avx2_q16_err = run_avx2_q16(scenario, lhs, rhs)
    avx2_q32_err = run_avx2_q32(scenario, lhs, rhs)

    assert scalar_err == avx2_q16_err == avx2_q32_err == Q8_0_AVX2_ERR_OVERFLOW


def run() -> None:
    test_helper_targeted_cases_match_all_paths()
    test_helper_randomized_parity()
    test_entrypoint_row_stride_overflow_surface_matches()
    print("q8_0_matmul_tiled_row_base_preflight=ok")


if __name__ == "__main__":
    run()
