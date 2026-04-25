#!/usr/bin/env python3
"""Harness for IQ-1503 latency-window no-partial commit-only hardening wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_NULL_PTR = 1
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3
GPU_SEC_PERF_ERR_CAPACITY = 4
GPU_SEC_PERF_ERR_OVERFLOW = 5

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = 1
GPU_SEC_PERF_I64_MAX = 0x7FFFFFFFFFFFFFFF


@dataclass(frozen=True)
class RowOutput:
    tok_per_sec_q16: int
    audit_overhead_delta_q16: int
    secure_cycles_per_token_q16: int


def _add_checked(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs > GPU_SEC_PERF_I64_MAX - rhs:
        return None
    return lhs + rhs


def _mul_checked(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs == 0 or rhs == 0:
        return 0
    if lhs > GPU_SEC_PERF_I64_MAX // rhs:
        return None
    return lhs * rhs


def _select_percentile_nearest_rank_q16_checked(rows: list[RowOutput], percentile_num: int, percentile_den: int) -> tuple[int, int]:
    if not rows or percentile_num <= 0 or percentile_den <= 0 or percentile_num > percentile_den:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    scaled_rank_num = _mul_checked(percentile_num, len(rows))
    if scaled_rank_num is None:
        return GPU_SEC_PERF_ERR_OVERFLOW, 0

    rank = scaled_rank_num // percentile_den
    if scaled_rank_num % percentile_den:
        rank_plus = _add_checked(rank, 1)
        if rank_plus is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0
        rank = rank_plus

    rank = max(1, min(rank, len(rows)))
    ordered = sorted(row.secure_cycles_per_token_q16 for row in rows)
    if ordered[0] < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    return GPU_SEC_PERF_OK, ordered[rank - 1]


def _latency_window_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, int, int, int, int]:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0
    if iommu_active != 1 or book_of_truth_gpu_hooks != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0

    p05_status, p05_q16 = _select_percentile_nearest_rank_q16_checked(rows, 5, 100)
    p25_status, p25_q16 = _select_percentile_nearest_rank_q16_checked(rows, 25, 100)
    p75_status, p75_q16 = _select_percentile_nearest_rank_q16_checked(rows, 75, 100)
    p99_status, p99_q16 = _select_percentile_nearest_rank_q16_checked(rows, 99, 100)
    status = p05_status or p25_status or p75_status or p99_status
    if status != GPU_SEC_PERF_OK:
        return status, 0, 0, 0, 0

    return GPU_SEC_PERF_OK, p05_q16, p25_q16, p75_q16, p99_q16


def _latency_window_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, int, int, int, int]:
    if out_capacity < 4:
        return GPU_SEC_PERF_ERR_CAPACITY, 0, 0, 0, 0

    first = _latency_window_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    second = _latency_window_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if first[0] != second[0] or first[1:] != second[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    return first


def _latency_window_nopartial(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, int, int, int, int]:
    first = _latency_window_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    second = _latency_window_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if first[0] != second[0] or first[1:] != second[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    return first


def _latency_window_commit_only_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int, int],
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    staged = _latency_window_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if staged[0] != GPU_SEC_PERF_OK:
        return staged[0], caller_outputs, (0, 0, 0, 0)

    canonical = _latency_window_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if canonical[0] != GPU_SEC_PERF_OK:
        return canonical[0], caller_outputs, (0, 0, 0, 0)

    if staged[1:] != canonical[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, caller_outputs, canonical[1:]


def gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_commit_only(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int, int],
    has_null_output: bool = False,
    outputs_alias: bool = False,
    inject_parity_drift: bool = False,
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    if rows is None or has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0, 0)
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    parity = _latency_window_nopartial(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    parity_status, parity_p05, parity_p25, parity_p75, parity_p99 = parity

    preflight = _latency_window_commit_only_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        caller_outputs=(parity_p05, parity_p25, parity_p75, parity_p99),
    )
    preflight_status, _seed_preserved, staged = preflight

    if inject_parity_drift:
        staged = (staged[0], staged[1], staged[2] + 1, staged[3])

    if parity_status != preflight_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if staged != (parity_p05, parity_p25, parity_p75, parity_p99):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    if parity_status != GPU_SEC_PERF_OK:
        return parity_status, caller_outputs, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, (parity_p05, parity_p25, parity_p75, parity_p99), (parity_p05, parity_p25, parity_p75, parity_p99)


def test_source_contains_iq1503_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartialCommitOnly(" in src
    assert "status_nopartial = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartial(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnlyPreflightOnly(" in src
    assert "saved_p99_q16" in src
    assert "if (status_nopartial != status_preflight_only)" in src


def test_null_alias_capacity_and_gate_vectors() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(11, 12, 13, 14),
        has_null_output=True,
    )
    assert status == GPU_SEC_PERF_ERR_NULL_PTR
    assert outputs_after == (11, 12, 13, 14)
    assert diag == (0, 0, 0, 0)

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(21, 22, 23, 24),
        outputs_alias=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs_after == (21, 22, 23, 24)
    assert diag == (0, 0, 0, 0)

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(31, 32, 33, 34),
    )
    assert status == GPU_SEC_PERF_ERR_CAPACITY
    assert outputs_after == (31, 32, 33, 34)
    assert diag == (0, 0, 0, 0)

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(41, 42, 43, 44),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs_after == (41, 42, 43, 44)
    assert diag == (0, 0, 0, 0)


def test_duplicate_latency_and_parity_drift_vectors() -> None:
    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(101_000, 2_100, 10_000),
        RowOutput(102_000, 2_200, 30_000),
        RowOutput(103_000, 2_300, 40_000),
        RowOutput(104_000, 2_400, 80_000),
    ]

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_commit_only(
        rows,
        rows_capacity=5,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3, 4),
    )
    assert status == GPU_SEC_PERF_OK
    assert outputs_after == (10_000, 10_000, 40_000, 80_000)
    assert diag == (10_000, 10_000, 40_000, 80_000)

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_commit_only(
        rows,
        rows_capacity=5,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(5, 6, 7, 8),
        inject_parity_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs_after == (5, 6, 7, 8)
    assert diag == (0, 0, 0, 0)


if __name__ == "__main__":
    test_source_contains_iq1503_symbols()
    test_null_alias_capacity_and_gate_vectors()
    test_duplicate_latency_and_parity_drift_vectors()
    print("ok")
