#!/usr/bin/env python3
"""Harness for IQ-1589 secure-local overhead budget gate preflight-only parity wrapper."""

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


def _status_is_valid(status_code: int) -> bool:
    return status_code in {
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ERR_NULL_PTR,
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ERR_CAPACITY,
        GPU_SEC_PERF_ERR_OVERFLOW,
    }


def _budget_gate_status_is_valid(status: int) -> bool:
    return status in {
        GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH,
    }


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


def _rows_snapshot_digest_q64(rows: list[RowOutput], rows_capacity: int) -> tuple[int, int]:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    digest = 1469598103934665603
    for i, row in enumerate(rows):
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0

        factor = i + 1
        term = _mul_checked(row.tok_per_sec_q16, factor)
        if term is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0
        digest = _add_checked(digest, term)
        if digest is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0

        term = _mul_checked(row.audit_overhead_delta_q16, factor + len(rows))
        if term is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0
        digest = _add_checked(digest, term)
        if digest is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0

        term = _mul_checked(row.secure_cycles_per_token_q16, factor + (len(rows) << 1))
        if term is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0
        digest = _add_checked(digest, term)
        if digest is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0

    return GPU_SEC_PERF_OK, digest


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


def _secure_local_budget_gate_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
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

    gate_status = GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS
    if p50 > max_p50_overhead_q16 or p95 > max_p95_overhead_q16:
        gate_status = GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH

    if not _budget_gate_status_is_valid(gate_status):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK

    if force_status_domain_drift:
        return 77, p50, p95, gate_status

    if gate_status != GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, p50, p95, gate_status

    return GPU_SEC_PERF_OK, p50, p95, gate_status


def _secure_local_budget_gate_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    caller_outputs: tuple[int, int, int],
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, caller_outputs, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    status_primary, staged_p50, staged_p95, staged_gate = _secure_local_budget_gate_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
    )

    status_parity, parity_p50, parity_p95, parity_gate = _secure_local_budget_gate_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
    )

    status_after, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_after != GPU_SEC_PERF_OK or digest_before != digest_after:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    if not _status_is_valid(status_primary) or not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    if (status_primary, staged_p50, staged_p95, staged_gate) != (status_parity, parity_p50, parity_p95, parity_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    return status_primary, caller_outputs, (staged_p50, staged_p95, staged_gate)


def _secure_local_budget_gate_preflight_only_parity(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    caller_outputs: tuple[int, int, int],
    *,
    force_status_domain_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    status_preflight, preserved, staged = _secure_local_budget_gate_preflight_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        caller_outputs,
    )
    status_canonical, can_p50, can_p95, can_gate = _secure_local_budget_gate_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        force_status_domain_drift=force_status_domain_drift,
    )

    if not _status_is_valid(status_preflight) or not _status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, preserved
    if not _budget_gate_status_is_valid(staged[2]) or not _budget_gate_status_is_valid(can_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, preserved

    if status_preflight != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, preserved
    if staged != (can_p50, can_p95, can_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, preserved

    return status_preflight, preserved


def test_source_contains_iq1589_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGatePreflightOnlyParity(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGatePreflightOnly(" in src
    assert "status_canonical = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGate(" in src
    assert "if (status_preflight_only != status_canonical)" in src
    assert "return status_preflight_only;" in src


def test_gate_missing_vectors_fail_closed_and_preserve_outputs() -> None:
    rows = [
        RowOutput(90_000, 5_000, 48_000),
        RowOutput(88_000, 6_000, 49_000),
        RowOutput(86_000, 7_000, 50_000),
    ]
    saved = (444, 555, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS)

    status, outputs = _secure_local_budget_gate_preflight_only_parity(
        rows,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=8_000,
        max_p95_overhead_q16=9_000,
        caller_outputs=saved,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs == saved


def test_threshold_breach_vectors_fail_closed_and_preserve_outputs() -> None:
    rows = [
        RowOutput(100_000, 3_000, 40_000),
        RowOutput(100_000, 5_000, 45_000),
        RowOutput(100_000, 8_000, 50_000),
        RowOutput(100_000, 11_000, 55_000),
        RowOutput(100_000, 14_000, 60_000),
    ]
    saved = (701, 702, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    status, outputs = _secure_local_budget_gate_preflight_only_parity(
        rows,
        rows_capacity=5,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=7_000,
        max_p95_overhead_q16=12_000,
        caller_outputs=saved,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs == saved


def test_secure_on_pass_preserves_outputs() -> None:
    rows = [
        RowOutput(90_000, 3_000, 41_000),
        RowOutput(88_000, 4_000, 42_000),
        RowOutput(86_000, 5_000, 43_000),
        RowOutput(84_000, 6_000, 44_000),
    ]
    saved = (1234, 5678, GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH)

    status, outputs = _secure_local_budget_gate_preflight_only_parity(
        rows,
        rows_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=6_000,
        max_p95_overhead_q16=7_000,
        caller_outputs=saved,
    )
    assert status == GPU_SEC_PERF_OK
    assert outputs == saved


def test_status_domain_drift_rejected() -> None:
    rows = [
        RowOutput(90_000, 3_000, 41_000),
        RowOutput(88_000, 4_000, 42_000),
        RowOutput(86_000, 5_000, 43_000),
        RowOutput(84_000, 6_000, 44_000),
    ]
    saved = (222, 333, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    status, outputs = _secure_local_budget_gate_preflight_only_parity(
        rows,
        rows_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=6_000,
        max_p95_overhead_q16=7_000,
        caller_outputs=saved,
        force_status_domain_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs == saved
