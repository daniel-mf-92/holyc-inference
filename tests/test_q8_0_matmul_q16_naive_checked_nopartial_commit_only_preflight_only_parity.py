#!/usr/bin/env python3
"""Host-side parity harness for IQ-1276.

Validates the control-flow/math contract for:
  Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnlyPreflightOnlyParity

The harness mirrors HolyC integer checks and verifies:
- strict tuple parity on required capacities
- preflight path is zero-write (diagnostics only)
- error/status parity against commit-only canonical path
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

Q8_0_MATMUL_OK = 0
Q8_0_MATMUL_ERR_NULL_PTR = 1
Q8_0_MATMUL_ERR_BAD_DST_LEN = 2
Q8_0_MATMUL_ERR_OVERFLOW = 3
Q8_0_MATMUL_I64_MAX = (1 << 63) - 1


@dataclass
class DiagOut:
    lhs_required_blocks: int
    rhs_required_blocks: int
    out_required_cells: int


@dataclass
class MatrixCase:
    row_count: int
    col_count: int
    lhs_row_stride_blocks: int
    rhs_col_stride_blocks: int
    k_block_count: int
    out_row_stride_cells: int
    lhs_block_capacity: int
    rhs_block_capacity: int
    out_cell_capacity: int


class Ptr:
    def __init__(self, value: Optional[int] = None):
        self.value = value


def _try_mul_nonneg(lhs: int, rhs: int) -> Tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > Q8_0_MATMUL_I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def _try_add_nonneg(lhs: int, rhs: int) -> Tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > Q8_0_MATMUL_I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def _validate_strides_and_k(
    row_count: int,
    col_count: int,
    lhs_row_stride_blocks: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_row_stride_cells: int,
) -> int:
    if row_count < 0 or col_count < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if k_block_count < 0 or out_row_stride_cells < 0:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if lhs_row_stride_blocks < k_block_count:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if rhs_col_stride_blocks < k_block_count:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if out_row_stride_cells < col_count:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    return Q8_0_MATMUL_OK


def _compute_required_capacities(
    row_count: int,
    lhs_row_stride_blocks: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    out_row_stride_cells: int,
) -> Tuple[int, Optional[DiagOut]]:
    ok, lhs_required = _try_mul_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW, None
    ok, rhs_required = _try_mul_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW, None
    ok, out_required = _try_mul_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW, None
    return Q8_0_MATMUL_OK, DiagOut(lhs_required, rhs_required, out_required)


def q8_0_matmul_q16_naive_checked_nopartial_commit_only(
    case: MatrixCase,
) -> Tuple[int, Optional[DiagOut]]:
    status = _validate_strides_and_k(
        case.row_count,
        case.col_count,
        case.lhs_row_stride_blocks,
        case.rhs_col_stride_blocks,
        case.k_block_count,
        case.out_row_stride_cells,
    )
    if status != Q8_0_MATMUL_OK:
        return status, None

    status, diag = _compute_required_capacities(
        case.row_count,
        case.lhs_row_stride_blocks,
        case.col_count,
        case.rhs_col_stride_blocks,
        case.out_row_stride_cells,
    )
    if status != Q8_0_MATMUL_OK:
        return status, None

    if diag.lhs_required_blocks > case.lhs_block_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN, None
    if diag.rhs_required_blocks > case.rhs_block_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN, None
    if diag.out_required_cells > case.out_cell_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN, None

    return Q8_0_MATMUL_OK, diag


def q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(
    case: MatrixCase,
    out_lhs_required_blocks: Optional[Ptr],
    out_rhs_required_blocks: Optional[Ptr],
    out_out_required_cells: Optional[Ptr],
) -> int:
    if out_lhs_required_blocks is None or out_rhs_required_blocks is None or out_out_required_cells is None:
        return Q8_0_MATMUL_ERR_NULL_PTR

    status = _validate_strides_and_k(
        case.row_count,
        case.col_count,
        case.lhs_row_stride_blocks,
        case.rhs_col_stride_blocks,
        case.k_block_count,
        case.out_row_stride_cells,
    )
    if status != Q8_0_MATMUL_OK:
        return status

    status, diag = _compute_required_capacities(
        case.row_count,
        case.lhs_row_stride_blocks,
        case.col_count,
        case.rhs_col_stride_blocks,
        case.out_row_stride_cells,
    )
    if status != Q8_0_MATMUL_OK:
        return status

    if diag.lhs_required_blocks > case.lhs_block_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if diag.rhs_required_blocks > case.rhs_block_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if diag.out_required_cells > case.out_cell_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    out_lhs_required_blocks.value = diag.lhs_required_blocks
    out_rhs_required_blocks.value = diag.rhs_required_blocks
    out_out_required_cells.value = diag.out_required_cells
    return Q8_0_MATMUL_OK


def q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only_parity(
    case: MatrixCase,
    out_lhs_required_blocks: Optional[Ptr],
    out_rhs_required_blocks: Optional[Ptr],
    out_out_required_cells: Optional[Ptr],
) -> int:
    if out_lhs_required_blocks is None or out_rhs_required_blocks is None or out_out_required_cells is None:
        return Q8_0_MATMUL_ERR_NULL_PTR

    snapshot = (
        case.row_count,
        case.col_count,
        case.lhs_row_stride_blocks,
        case.rhs_col_stride_blocks,
        case.k_block_count,
        case.out_row_stride_cells,
        case.lhs_block_capacity,
        case.rhs_block_capacity,
        case.out_cell_capacity,
        out_lhs_required_blocks,
        out_rhs_required_blocks,
        out_out_required_cells,
    )

    pre_lhs = Ptr()
    pre_rhs = Ptr()
    pre_out = Ptr()
    preflight_status = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(
        case,
        pre_lhs,
        pre_rhs,
        pre_out,
    )

    commit_status, commit_diag = q8_0_matmul_q16_naive_checked_nopartial_commit_only(case)

    if snapshot != (
        case.row_count,
        case.col_count,
        case.lhs_row_stride_blocks,
        case.rhs_col_stride_blocks,
        case.k_block_count,
        case.out_row_stride_cells,
        case.lhs_block_capacity,
        case.rhs_block_capacity,
        case.out_cell_capacity,
        out_lhs_required_blocks,
        out_rhs_required_blocks,
        out_out_required_cells,
    ):
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if preflight_status != commit_status:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if preflight_status != Q8_0_MATMUL_OK:
        return preflight_status

    if commit_diag is None:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    if pre_lhs.value != commit_diag.lhs_required_blocks:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if pre_rhs.value != commit_diag.rhs_required_blocks:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if pre_out.value != commit_diag.out_required_cells:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    out_lhs_required_blocks.value = pre_lhs.value
    out_rhs_required_blocks.value = pre_rhs.value
    out_out_required_cells.value = pre_out.value
    return Q8_0_MATMUL_OK


def _run_cases() -> List[str]:
    notes: List[str] = []

    sentinel_a = Ptr(111)
    sentinel_b = Ptr(222)
    sentinel_c = Ptr(333)
    ok_case = MatrixCase(
        row_count=4,
        col_count=3,
        lhs_row_stride_blocks=9,
        rhs_col_stride_blocks=8,
        k_block_count=8,
        out_row_stride_cells=5,
        lhs_block_capacity=64,
        rhs_block_capacity=64,
        out_cell_capacity=64,
    )
    status = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only_parity(
        ok_case,
        sentinel_a,
        sentinel_b,
        sentinel_c,
    )
    assert status == Q8_0_MATMUL_OK
    assert (sentinel_a.value, sentinel_b.value, sentinel_c.value) == (36, 24, 20)
    notes.append("ok-path parity tuple matches")

    unchanged_a = Ptr(9001)
    unchanged_b = Ptr(9002)
    unchanged_c = Ptr(9003)
    bad_case = MatrixCase(
        row_count=2,
        col_count=4,
        lhs_row_stride_blocks=3,
        rhs_col_stride_blocks=4,
        k_block_count=4,
        out_row_stride_cells=4,
        lhs_block_capacity=64,
        rhs_block_capacity=64,
        out_cell_capacity=64,
    )
    status = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only_parity(
        bad_case,
        unchanged_a,
        unchanged_b,
        unchanged_c,
    )
    assert status == Q8_0_MATMUL_ERR_BAD_DST_LEN
    assert (unchanged_a.value, unchanged_b.value, unchanged_c.value) == (9001, 9002, 9003)
    notes.append("bad-stride preserves diagnostics outputs")

    ov_a = Ptr(7)
    ov_b = Ptr(8)
    ov_c = Ptr(9)
    overflow_case = MatrixCase(
        row_count=1 << 62,
        col_count=3,
        lhs_row_stride_blocks=8,
        rhs_col_stride_blocks=8,
        k_block_count=8,
        out_row_stride_cells=8,
        lhs_block_capacity=Q8_0_MATMUL_I64_MAX,
        rhs_block_capacity=Q8_0_MATMUL_I64_MAX,
        out_cell_capacity=Q8_0_MATMUL_I64_MAX,
    )
    status = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only_parity(
        overflow_case,
        ov_a,
        ov_b,
        ov_c,
    )
    assert status == Q8_0_MATMUL_ERR_OVERFLOW
    assert (ov_a.value, ov_b.value, ov_c.value) == (7, 8, 9)
    notes.append("overflow parity and zero-write diagnostics")

    null_case = ok_case
    status = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only_parity(
        null_case,
        None,
        Ptr(0),
        Ptr(0),
    )
    assert status == Q8_0_MATMUL_ERR_NULL_PTR
    notes.append("null diagnostics pointer rejected")

    return notes


def main() -> None:
    notes = _run_cases()
    print("ok:", "; ".join(notes))


if __name__ == "__main__":
    main()
