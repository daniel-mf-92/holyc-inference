#!/usr/bin/env python3
"""Harness for IQ-1573 secure-local overhead-envelope budget gate."""

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

GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK = 0
GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS = 1
GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH = 2


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


def _select_audit_overhead_percentile_nearest_rank_q16_checked(
    rows: list[RowOutput],
    percentile_num: int,
    percentile_den: int,
) -> tuple[int, int]:
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0
    if percentile_num <= 0 or percentile_den <= 0 or percentile_num > percentile_den:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    scaled_rank_num = _mul_checked(percentile_num, len(rows))
    if scaled_rank_num is None:
        return GPU_SEC_PERF_ERR_OVERFLOW, 0

    rank = scaled_rank_num // percentile_den
    if scaled_rank_num % percentile_den != 0:
        rank = _add_checked(rank, 1)
        if rank is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0

    rank = max(1, min(rank, len(rows)))

    found = False
    best = 0
    for candidate_row in rows:
        candidate = candidate_row.audit_overhead_delta_q16
        if candidate < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0

        less_count = 0
        less_equal_count = 0
        for observed_row in rows:
            observed = observed_row.audit_overhead_delta_q16
            if observed < 0:
                return GPU_SEC_PERF_ERR_BAD_PARAM, 0
            if observed < candidate:
                less_count += 1
            if observed <= candidate:
                less_equal_count += 1

        if less_count < rank <= less_equal_count:
            if not found or candidate < best:
                best = candidate
                found = True

    if not found:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    return GPU_SEC_PERF_OK, best


def _budget_gate_status_is_valid(status: int) -> bool:
    return status in {
        GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH,
    }


def _secure_local_budget_gate_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    force_status_domain_drift: bool = False,
) -> tuple[int, int, int, int]:
    p50 = 0
    p95 = 0
    gate_status = GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK

    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, p50, p95, gate_status
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, p50, p95, gate_status
    if max_p50_overhead_q16 < 0 or max_p95_overhead_q16 < 0 or max_p95_overhead_q16 < max_p50_overhead_q16:
        return GPU_SEC_PERF_ERR_BAD_PARAM, p50, p95, gate_status

    snapshot_secure_local_mode = secure_local_mode
    snapshot_iommu_active = iommu_active
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks
    snapshot_policy_digest_parity = policy_digest_parity

    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, p50, p95, gate_status
    if iommu_active != 1 or book_of_truth_gpu_hooks != 1 or policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, p50, p95, gate_status

    st_p50, p50 = _select_audit_overhead_percentile_nearest_rank_q16_checked(rows, 50, 100)
    if st_p50 != GPU_SEC_PERF_OK:
        return st_p50, 0, 0, gate_status

    st_p95, p95 = _select_audit_overhead_percentile_nearest_rank_q16_checked(rows, 95, 100)
    if st_p95 != GPU_SEC_PERF_OK:
        return st_p95, 0, 0, gate_status

    if (
        snapshot_secure_local_mode != secure_local_mode
        or snapshot_iommu_active != iommu_active
        or snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks
        or snapshot_policy_digest_parity != policy_digest_parity
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, gate_status

    gate_status = GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS
    if p50 > max_p50_overhead_q16 or p95 > max_p95_overhead_q16:
        gate_status = GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH

    if force_status_domain_drift:
        gate_status = 77

    if not _budget_gate_status_is_valid(gate_status):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK

    if gate_status != GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, p50, p95, gate_status

    return GPU_SEC_PERF_OK, p50, p95, gate_status


def test_source_contains_iq1573_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH" in src
    assert "Bool GPUSecurityPerfBudgetGateStatusIsValid(I64 status_code)" in src
    assert "I32 GPUSecurityPerfMatrixSelectAuditOverheadPercentileNearestRankQ16Checked(" in src
    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGate(" in src
    assert "status_p50 = GPUSecurityPerfMatrixSelectAuditOverheadPercentileNearestRankQ16Checked(" in src
    assert "status_p95 = GPUSecurityPerfMatrixSelectAuditOverheadPercentileNearestRankQ16Checked(" in src
    assert "if (!GPUSecurityPerfBudgetGateStatusIsValid(staged_budget_gate_status))" in src


def test_gate_missing_vectors_fail_closed() -> None:
    rows = [
        RowOutput(90_000, 5_500, 48_000),
        RowOutput(88_000, 6_200, 49_000),
        RowOutput(86_000, 6_800, 50_000),
    ]

    status, p50, p95, gate = _secure_local_budget_gate_checked(
        rows,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=8_000,
        max_p95_overhead_q16=9_000,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert (p50, p95, gate) == (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    status, _, _, gate = _secure_local_budget_gate_checked(
        rows,
        rows_capacity=3,
        secure_local_mode=0,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=8_000,
        max_p95_overhead_q16=9_000,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert gate == GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK


def test_threshold_breach_vectors() -> None:
    rows = [
        RowOutput(100_000, 3_000, 40_000),
        RowOutput(100_000, 5_000, 45_000),
        RowOutput(100_000, 8_000, 50_000),
        RowOutput(100_000, 11_000, 55_000),
        RowOutput(100_000, 14_000, 60_000),
    ]

    status, p50, p95, gate = _secure_local_budget_gate_checked(
        rows,
        rows_capacity=5,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=7_000,
        max_p95_overhead_q16=12_000,
    )

    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert p50 == 8_000
    assert p95 == 14_000
    assert gate == GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH


def test_status_domain_drift_rejected() -> None:
    rows = [
        RowOutput(100_000, 3_000, 40_000),
        RowOutput(100_000, 4_000, 42_000),
        RowOutput(100_000, 5_000, 45_000),
    ]

    status, p50, p95, gate = _secure_local_budget_gate_checked(
        rows,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=6_000,
        max_p95_overhead_q16=6_500,
        force_status_domain_drift=True,
    )

    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert p50 == 0
    assert p95 == 0
    assert gate == GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK


def test_secure_local_pass_vector() -> None:
    rows = [
        RowOutput(110_000, 2_000, 38_000),
        RowOutput(109_000, 3_500, 42_000),
        RowOutput(108_000, 4_200, 44_000),
        RowOutput(107_000, 5_000, 47_000),
        RowOutput(106_000, 6_000, 49_000),
    ]

    status, p50, p95, gate = _secure_local_budget_gate_checked(
        rows,
        rows_capacity=5,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=5_000,
        max_p95_overhead_q16=6_000,
    )

    assert status == GPU_SEC_PERF_OK
    assert p50 == 4_200
    assert p95 == 6_000
    assert gate == GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS
