#!/usr/bin/env python3
"""Parity harness for IQ-1199.

Models HolyC helper:
Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnlyPreflightOnly
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

Q8_0_OK = 0
Q8_0_ERR_NULL_PTR = 1
Q8_0_ERR_BAD_DST_LEN = 2
Q8_0_ERR_OVERFLOW = 3
Q8_0_I64_MAX = 0x7FFFFFFFFFFFFFFF


@dataclass(frozen=True)
class BlockQ80:
    d_fp16: int
    qs: bytes


def i64_mul_nonneg_checked(lhs: int, rhs: int) -> Tuple[int, int]:
    if lhs < 0 or rhs < 0:
        return Q8_0_ERR_BAD_DST_LEN, 0
    if lhs == 0 or rhs == 0:
        return Q8_0_OK, 0
    if lhs > Q8_0_I64_MAX // rhs:
        return Q8_0_ERR_OVERFLOW, 0
    return Q8_0_OK, lhs * rhs


def i64_add_nonneg_checked(lhs: int, rhs: int) -> Tuple[int, int]:
    if lhs < 0 or rhs < 0:
        return Q8_0_ERR_BAD_DST_LEN, 0
    if lhs > Q8_0_I64_MAX - rhs:
        return Q8_0_ERR_OVERFLOW, 0
    return Q8_0_OK, lhs + rhs


def compute_required_capacities_checked(
    row_count: int,
    lhs_row_stride_blocks: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    out_row_stride_cells: int,
) -> Tuple[int, int, int, int]:
    err, lhs_required = i64_mul_nonneg_checked(row_count, lhs_row_stride_blocks)
    if err != Q8_0_OK:
        return err, 0, 0, 0

    err, rhs_required = i64_mul_nonneg_checked(col_count, rhs_col_stride_blocks)
    if err != Q8_0_OK:
        return err, 0, 0, 0

    err, out_required = i64_mul_nonneg_checked(row_count, out_row_stride_cells)
    if err != Q8_0_OK:
        return err, 0, 0, 0

    return Q8_0_OK, lhs_required, rhs_required, out_required


def validate_strides_and_k_checked(
    row_count: int,
    col_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_row_stride_cells: int,
) -> int:
    if row_count < 0 or col_count < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if k_block_count < 0 or out_row_stride_cells < 0:
        return Q8_0_ERR_BAD_DST_LEN

    if lhs_row_stride_blocks < k_block_count:
        return Q8_0_ERR_BAD_DST_LEN
    if rhs_col_stride_blocks < k_block_count:
        return Q8_0_ERR_BAD_DST_LEN
    if out_row_stride_cells < col_count:
        return Q8_0_ERR_BAD_DST_LEN

    return Q8_0_OK


def q8_0_matmul_q16_nopartial_commit_only(
    lhs_blocks: List[BlockQ80],
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks: List[BlockQ80],
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells_q16: List[int],
    out_cell_capacity: int,
    out_row_stride_cells: int,
) -> int:
    if lhs_blocks is None or rhs_col_blocks is None or out_cells_q16 is None:
        return Q8_0_ERR_NULL_PTR
    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_cell_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN

    err = validate_strides_and_k_checked(
        row_count,
        col_count,
        lhs_row_stride_blocks,
        rhs_col_stride_blocks,
        k_block_count,
        out_row_stride_cells,
    )
    if err != Q8_0_OK:
        return err

    err, lhs_required, rhs_required, out_required = compute_required_capacities_checked(
        row_count,
        lhs_row_stride_blocks,
        col_count,
        rhs_col_stride_blocks,
        out_row_stride_cells,
    )
    if err != Q8_0_OK:
        return err

    if lhs_required > lhs_block_capacity or rhs_required > rhs_block_capacity or out_required > out_cell_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    if out_required == 0:
        return Q8_0_OK

    # This preflight model intentionally omits full kernel math; IQ-1199 validates
    # capacity diagnostics parity, not dot-product parity.
    return Q8_0_OK


def q8_0_matmul_q16_nopartial_commit_only_preflight_only(
    lhs_blocks: List[BlockQ80] | None,
    lhs_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_blocks: List[BlockQ80] | None,
    rhs_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_cells_q16: List[int] | None,
    out_cell_capacity: int,
    out_row_stride_cells: int,
    out_lhs_required_blocks: List[int] | None,
    out_rhs_required_blocks: List[int] | None,
    out_out_required_cells: List[int] | None,
) -> int:
    if out_lhs_required_blocks is None or out_rhs_required_blocks is None or out_out_required_cells is None:
        return Q8_0_ERR_NULL_PTR

    snapshot = (
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16,
        out_cell_capacity,
        out_row_stride_cells,
        out_lhs_required_blocks,
        out_rhs_required_blocks,
        out_out_required_cells,
    )

    canonical_status = q8_0_matmul_q16_nopartial_commit_only(
        lhs_blocks if lhs_blocks is not None else [],
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks if rhs_col_blocks is not None else [],
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16 if out_cells_q16 is not None else [],
        out_cell_capacity,
        out_row_stride_cells,
    )
    if canonical_status != Q8_0_OK:
        return canonical_status

    err, staged_lhs_required, staged_rhs_required, staged_out_required = compute_required_capacities_checked(
        row_count,
        lhs_row_stride_blocks,
        col_count,
        rhs_col_stride_blocks,
        out_row_stride_cells,
    )
    if err != Q8_0_OK:
        return err

    if snapshot != (
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16,
        out_cell_capacity,
        out_row_stride_cells,
        out_lhs_required_blocks,
        out_rhs_required_blocks,
        out_out_required_cells,
    ):
        return Q8_0_ERR_BAD_DST_LEN

    out_lhs_required_blocks[0] = staged_lhs_required
    out_rhs_required_blocks[0] = staged_rhs_required
    out_out_required_cells[0] = staged_out_required
    return Q8_0_OK


def make_block(rng: random.Random) -> BlockQ80:
    d = rng.randrange(0, 0x10000)
    qs = bytes(rng.randrange(0, 256) for _ in range(32))
    return BlockQ80(d_fp16=d, qs=qs)


def test_preflight_derives_capacity_tuple_randomized() -> None:
    rng = random.Random(2026042301)

    for _ in range(500):
        row_count = rng.randint(0, 16)
        col_count = rng.randint(0, 16)
        k_block_count = rng.randint(0, 8)
        lhs_row_stride_blocks = k_block_count + rng.randint(0, 6)
        rhs_col_stride_blocks = k_block_count + rng.randint(0, 6)
        out_row_stride_cells = col_count + rng.randint(0, 6)

        err, lhs_required, rhs_required, out_required = compute_required_capacities_checked(
            row_count,
            lhs_row_stride_blocks,
            col_count,
            rhs_col_stride_blocks,
            out_row_stride_cells,
        )
        assert err == Q8_0_OK

        lhs_capacity = lhs_required + rng.randint(0, 10)
        rhs_capacity = rhs_required + rng.randint(0, 10)
        out_capacity = out_required + rng.randint(0, 10)

        lhs_blocks = [make_block(rng) for _ in range(lhs_capacity)]
        rhs_blocks = [make_block(rng) for _ in range(rhs_capacity)]
        out_cells = [0x66 for _ in range(max(1, out_capacity))]

        diag_lhs = [0x11]
        diag_rhs = [0x22]
        diag_out = [0x33]

        status = q8_0_matmul_q16_nopartial_commit_only_preflight_only(
            lhs_blocks,
            lhs_capacity,
            row_count,
            lhs_row_stride_blocks,
            rhs_blocks,
            rhs_capacity,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            out_cells,
            out_capacity,
            out_row_stride_cells,
            diag_lhs,
            diag_rhs,
            diag_out,
        )
        assert status == Q8_0_OK
        assert diag_lhs[0] == lhs_required
        assert diag_rhs[0] == rhs_required
        assert diag_out[0] == out_required


def test_preflight_propagates_bad_len_from_stride_contract() -> None:
    rng = random.Random(2026042302)
    lhs_blocks = [make_block(rng) for _ in range(8)]
    rhs_blocks = [make_block(rng) for _ in range(8)]
    out_cells = [0 for _ in range(8)]

    diag_lhs = [123]
    diag_rhs = [456]
    diag_out = [789]

    status = q8_0_matmul_q16_nopartial_commit_only_preflight_only(
        lhs_blocks,
        8,
        2,
        1,
        rhs_blocks,
        8,
        2,
        2,
        2,
        out_cells,
        8,
        2,
        diag_lhs,
        diag_rhs,
        diag_out,
    )
    assert status == Q8_0_ERR_BAD_DST_LEN
    assert diag_lhs[0] == 123
    assert diag_rhs[0] == 456
    assert diag_out[0] == 789


def test_preflight_propagates_overflow_from_capacity_math() -> None:
    rng = random.Random(2026042303)
    lhs_blocks = [make_block(rng)]
    rhs_blocks = [make_block(rng)]
    out_cells = [0]

    diag_lhs = [9]
    diag_rhs = [9]
    diag_out = [9]

    huge = Q8_0_I64_MAX
    status = q8_0_matmul_q16_nopartial_commit_only_preflight_only(
        lhs_blocks,
        huge,
        huge,
        2,
        rhs_blocks,
        huge,
        1,
        1,
        1,
        out_cells,
        huge,
        1,
        diag_lhs,
        diag_rhs,
        diag_out,
    )
    assert status == Q8_0_ERR_OVERFLOW
    assert diag_lhs[0] == 9
    assert diag_rhs[0] == 9
    assert diag_out[0] == 9


def test_preflight_rejects_null_diagnostics_outputs() -> None:
    rng = random.Random(2026042304)
    lhs_blocks = [make_block(rng) for _ in range(4)]
    rhs_blocks = [make_block(rng) for _ in range(4)]
    out_cells = [0 for _ in range(4)]

    diag_lhs = [0]
    diag_rhs = [0]
    diag_out = [0]

    status = q8_0_matmul_q16_nopartial_commit_only_preflight_only(
        lhs_blocks,
        4,
        1,
        1,
        rhs_blocks,
        4,
        1,
        1,
        1,
        out_cells,
        4,
        1,
        None,
        diag_rhs,
        diag_out,
    )
    assert status == Q8_0_ERR_NULL_PTR

    status = q8_0_matmul_q16_nopartial_commit_only_preflight_only(
        lhs_blocks,
        4,
        1,
        1,
        rhs_blocks,
        4,
        1,
        1,
        1,
        out_cells,
        4,
        1,
        diag_lhs,
        None,
        diag_out,
    )
    assert status == Q8_0_ERR_NULL_PTR

    status = q8_0_matmul_q16_nopartial_commit_only_preflight_only(
        lhs_blocks,
        4,
        1,
        1,
        rhs_blocks,
        4,
        1,
        1,
        1,
        out_cells,
        4,
        1,
        diag_lhs,
        diag_rhs,
        None,
    )
    assert status == Q8_0_ERR_NULL_PTR
