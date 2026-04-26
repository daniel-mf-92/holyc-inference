#!/usr/bin/env python3
"""Harness for IQ-1611 commit-only hardening wrapper over IQ-1596 and IQ-1595."""

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


def _secure_local_budget_gate_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
) -> tuple[int, int, int, int]:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK
    if max_p50_overhead_q16 < 0 or max_p95_overhead_q16 < 0 or max_p95_overhead_q16 < max_p50_overhead_q16:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK

    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK
    if iommu_active != 1 or book_of_truth_gpu_hooks != 1 or policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK

    vals = sorted(r.audit_overhead_delta_q16 for r in rows)
    if vals[0] < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK

    def nearest_rank(pct: int) -> int:
        rank = (pct * len(vals) + 99) // 100
        rank = max(1, min(rank, len(vals)))
        return vals[rank - 1]

    p50 = nearest_rank(50)
    p95 = nearest_rank(95)
    gate = GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS
    if p50 > max_p50_overhead_q16 or p95 > max_p95_overhead_q16:
        gate = GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH
        return GPU_SEC_PERF_ERR_POLICY_GUARD, p50, p95, gate

    return GPU_SEC_PERF_OK, p50, p95, gate


def _iq1595_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    caller: tuple[int, int, int],
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    status, p50, p95, gate = _secure_local_budget_gate_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
    )
    return status, caller, (p50, p95, gate)


def _iq1596_preflight_only_parity(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
) -> tuple[int, int, int, int]:
    return _secure_local_budget_gate_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
    )


def secure_local_budget_gate_iq1611(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    current_p50: int,
    current_p95: int,
    current_gate: int,
    *,
    has_null_output: bool = False,
    outputs_alias: bool = False,
    force_status_domain_drift: bool = False,
    force_tuple_parity_drift: bool = False,
) -> tuple[int, int, int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_p50, current_p95, current_gate
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_p50, current_p95, current_gate

    status_parity, parity_p50, parity_p95, parity_gate = _iq1596_preflight_only_parity(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
    )

    status_preflight, _, staged = _iq1595_preflight_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        (parity_p50, parity_p95, parity_gate),
    )
    staged_p50, staged_p95, staged_gate = staged

    if force_status_domain_drift:
        status_preflight = 77
    if force_tuple_parity_drift:
        staged_gate = GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK

    if not _status_is_valid(status_parity) or not _status_is_valid(status_preflight):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_p50, current_p95, current_gate
    if not _budget_gate_status_is_valid(parity_gate) or not _budget_gate_status_is_valid(staged_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_p50, current_p95, current_gate

    if status_parity != status_preflight:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_p50, current_p95, current_gate
    if (parity_p50, parity_p95, parity_gate) != (staged_p50, staged_p95, staged_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_p50, current_p95, current_gate

    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, current_p50, current_p95, current_gate

    return GPU_SEC_PERF_OK, parity_p50, parity_p95, parity_gate


def test_source_contains_iq1611_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGatePreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    assert "status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGatePreflightOnlyParityCommitOnlyPreflightOnlyParity(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGatePreflightOnlyParityCommitOnlyPreflightOnly(" in src
    assert "if (status_parity != GPU_SEC_PERF_OK)" in src
    assert "if (*out_p50_overhead_q16 != saved_p50_overhead_q16" in src


def test_gate_missing_and_threshold_breach_vectors() -> None:
    rows = [
        RowOutput(100 << 16, 8 << 16, 2 << 16),
        RowOutput(100 << 16, 10 << 16, 2 << 16),
        RowOutput(100 << 16, 12 << 16, 2 << 16),
    ]

    status, p50, p95, gate = secure_local_budget_gate_iq1611(
        rows,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=20 << 16,
        max_p95_overhead_q16=20 << 16,
        current_p50=11,
        current_p95=12,
        current_gate=13,
    )
    assert (status, p50, p95, gate) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 11, 12, 13)

    status, p50, p95, gate = secure_local_budget_gate_iq1611(
        rows,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=5 << 16,
        max_p95_overhead_q16=6 << 16,
        current_p50=21,
        current_p95=22,
        current_gate=23,
    )
    assert (status, p50, p95, gate) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 21, 22, 23)


def test_status_domain_drift_and_deterministic_tuple_parity_vectors() -> None:
    rows = [
        RowOutput(100 << 16, 2 << 16, 2 << 16),
        RowOutput(100 << 16, 3 << 16, 2 << 16),
        RowOutput(100 << 16, 4 << 16, 2 << 16),
        RowOutput(100 << 16, 5 << 16, 2 << 16),
    ]

    status, p50, p95, gate = secure_local_budget_gate_iq1611(
        rows,
        rows_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=8 << 16,
        max_p95_overhead_q16=8 << 16,
        current_p50=31,
        current_p95=32,
        current_gate=33,
        force_status_domain_drift=True,
    )
    assert (status, p50, p95, gate) == (GPU_SEC_PERF_ERR_BAD_PARAM, 31, 32, 33)

    first = secure_local_budget_gate_iq1611(
        rows,
        rows_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=8 << 16,
        max_p95_overhead_q16=8 << 16,
        current_p50=41,
        current_p95=42,
        current_gate=43,
    )
    second = secure_local_budget_gate_iq1611(
        rows,
        rows_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=8 << 16,
        max_p95_overhead_q16=8 << 16,
        current_p50=51,
        current_p95=52,
        current_gate=53,
    )

    assert first == (GPU_SEC_PERF_OK, 3 << 16, 5 << 16, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS)
    assert second == (GPU_SEC_PERF_OK, 3 << 16, 5 << 16, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS)
