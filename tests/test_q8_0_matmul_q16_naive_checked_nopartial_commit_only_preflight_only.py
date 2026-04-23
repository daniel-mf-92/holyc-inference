#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

Q8_0_MATMUL_OK = 0
Q8_0_MATMUL_ERR_NULL_PTR = 1
Q8_0_MATMUL_ERR_BAD_DST_LEN = 2
Q8_0_MATMUL_ERR_OVERFLOW = 3
I64_MAX = (1 << 63) - 1


@dataclass
class Ptr:
    value: int


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


def _try_mul_nonneg(lhs: int, rhs: int) -> Tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


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
) -> Tuple[int, int, int, int]:
    ok, lhs_required = _try_mul_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW, 0, 0, 0
    ok, rhs_required = _try_mul_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW, 0, 0, 0
    ok, out_required = _try_mul_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q8_0_MATMUL_ERR_OVERFLOW, 0, 0, 0
    return Q8_0_MATMUL_OK, lhs_required, rhs_required, out_required


def q8_0_matmul_q16_naive_checked_nopartial_commit_only(
    case: MatrixCase,
    out_lhs_required_blocks: Optional[Ptr],
    out_rhs_required_blocks: Optional[Ptr],
    out_out_required_cells: Optional[Ptr],
) -> int:
    if out_lhs_required_blocks is None or out_rhs_required_blocks is None or out_out_required_cells is None:
        return Q8_0_MATMUL_ERR_NULL_PTR
    if (
        out_lhs_required_blocks is out_rhs_required_blocks
        or out_lhs_required_blocks is out_out_required_cells
        or out_rhs_required_blocks is out_out_required_cells
    ):
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

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

    status, lhs_required, rhs_required, out_required = _compute_required_capacities(
        case.row_count,
        case.lhs_row_stride_blocks,
        case.col_count,
        case.rhs_col_stride_blocks,
        case.out_row_stride_cells,
    )
    if status != Q8_0_MATMUL_OK:
        return status

    if lhs_required > case.lhs_block_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if rhs_required > case.rhs_block_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if out_required > case.out_cell_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    out_lhs_required_blocks.value = lhs_required
    out_rhs_required_blocks.value = rhs_required
    out_out_required_cells.value = out_required
    return Q8_0_MATMUL_OK


def q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(
    case: MatrixCase,
    out_lhs_required_blocks: Optional[Ptr],
    out_rhs_required_blocks: Optional[Ptr],
    out_out_required_cells: Optional[Ptr],
) -> int:
    if out_lhs_required_blocks is None or out_rhs_required_blocks is None or out_out_required_cells is None:
        return Q8_0_MATMUL_ERR_NULL_PTR
    if (
        out_lhs_required_blocks is out_rhs_required_blocks
        or out_lhs_required_blocks is out_out_required_cells
        or out_rhs_required_blocks is out_out_required_cells
    ):
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

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

    status, lhs_required, rhs_required, out_required = _compute_required_capacities(
        case.row_count,
        case.lhs_row_stride_blocks,
        case.col_count,
        case.rhs_col_stride_blocks,
        case.out_row_stride_cells,
    )
    if status != Q8_0_MATMUL_OK:
        return status

    if lhs_required > case.lhs_block_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if rhs_required > case.rhs_block_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN
    if out_required > case.out_cell_capacity:
        return Q8_0_MATMUL_ERR_BAD_DST_LEN

    # preflight-only: zero write to outputs
    return Q8_0_MATMUL_OK


def test_source_contains_iq1199_preflight_only_signature_and_guards() -> None:
    source = Path("src/matmul/q8_0_matmul.HC").read_text(encoding="utf-8")

    sig = "I32 Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnlyPreflightOnlyParity(", 1)[0]

    assert "if (!out_lhs_required_blocks ||" in body
    assert "!out_rhs_required_blocks ||" in body
    assert "!out_out_required_cells)" in body
    assert "if (out_lhs_required_blocks == out_rhs_required_blocks ||" in body
    assert "out_lhs_required_blocks == out_out_required_cells ||" in body
    assert "out_rhs_required_blocks == out_out_required_cells)" in body
    assert "snapshot_row_count = row_count;" in body
    assert "snapshot_col_count = col_count;" in body
    assert "snapshot_lhs_row_stride_blocks = lhs_row_stride_blocks;" in body
    assert "snapshot_rhs_col_stride_blocks = rhs_col_stride_blocks;" in body
    assert "snapshot_k_block_count = k_block_count;" in body
    assert "snapshot_out_row_stride_cells = out_row_stride_cells;" in body
    assert "snapshot_lhs_block_capacity = lhs_block_capacity;" in body
    assert "snapshot_rhs_block_capacity = rhs_block_capacity;" in body
    assert "snapshot_out_cell_capacity = out_cell_capacity;" in body
    assert "status = Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnly(" in body
    assert "status = Q8_0MatMulQ16NaiveCheckedNoPartial(" in body
    assert "if (canonical_lhs_required_blocks != staged_lhs_required_blocks ||" in body
    assert "canonical_rhs_required_blocks != staged_rhs_required_blocks ||" in body
    assert "canonical_out_required_cells != staged_out_required_cells)" in body
    assert "return Q8_0_MATMUL_OK;" in body


def test_preflight_only_null_and_alias_rejected_without_writes() -> None:
    case = MatrixCase(1, 1, 1, 1, 1, 1, 1, 1, 1)
    a = Ptr(11)
    b = Ptr(22)
    c = Ptr(33)

    err = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(case, None, b, c)
    assert err == Q8_0_MATMUL_ERR_NULL_PTR
    assert (a.value, b.value, c.value) == (11, 22, 33)

    err = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(case, a, a, c)
    assert err == Q8_0_MATMUL_ERR_BAD_DST_LEN
    assert (a.value, c.value) == (11, 33)


def test_preflight_only_bad_stride_and_capacity_paths() -> None:
    out_lhs = Ptr(101)
    out_rhs = Ptr(202)
    out_req = Ptr(303)

    bad_stride = MatrixCase(2, 3, 2, 3, 3, 3, 8, 12, 6)
    err = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(bad_stride, out_lhs, out_rhs, out_req)
    assert err == Q8_0_MATMUL_ERR_BAD_DST_LEN
    assert (out_lhs.value, out_rhs.value, out_req.value) == (101, 202, 303)

    bad_capacity = MatrixCase(3, 2, 4, 3, 2, 4, 11, 10, 12)
    err = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(bad_capacity, out_lhs, out_rhs, out_req)
    assert err == Q8_0_MATMUL_ERR_BAD_DST_LEN
    assert (out_lhs.value, out_rhs.value, out_req.value) == (101, 202, 303)


def test_preflight_only_overflow_path_no_writes() -> None:
    out_lhs = Ptr(700)
    out_rhs = Ptr(800)
    out_req = Ptr(900)

    overflow_case = MatrixCase((1 << 62), 1, 4, 1, 1, 1, I64_MAX, I64_MAX, I64_MAX)
    err = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(overflow_case, out_lhs, out_rhs, out_req)
    assert err == Q8_0_MATMUL_ERR_OVERFLOW
    assert (out_lhs.value, out_rhs.value, out_req.value) == (700, 800, 900)


def test_preflight_only_success_no_publish_with_commit_parity() -> None:
    case = MatrixCase(4, 3, 5, 4, 3, 6, 20, 12, 24)

    out_lhs = Ptr(17)
    out_rhs = Ptr(19)
    out_req = Ptr(23)

    err = q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only(case, out_lhs, out_rhs, out_req)
    assert err == Q8_0_MATMUL_OK
    assert (out_lhs.value, out_rhs.value, out_req.value) == (17, 19, 23)

    commit_lhs = Ptr(-1)
    commit_rhs = Ptr(-1)
    commit_req = Ptr(-1)
    commit_status = q8_0_matmul_q16_naive_checked_nopartial_commit_only(
        case,
        commit_lhs,
        commit_rhs,
        commit_req,
    )
    assert commit_status == Q8_0_MATMUL_OK
    assert (commit_lhs.value, commit_rhs.value, commit_req.value) == (20, 12, 24)


if __name__ == "__main__":
    test_source_contains_iq1199_preflight_only_signature_and_guards()
    test_preflight_only_null_and_alias_rejected_without_writes()
    test_preflight_only_bad_stride_and_capacity_paths()
    test_preflight_only_overflow_path_no_writes()
    test_preflight_only_success_no_publish_with_commit_parity()
    print("q8_0_matmul_q16_naive_checked_nopartial_commit_only_preflight_only=ok")
