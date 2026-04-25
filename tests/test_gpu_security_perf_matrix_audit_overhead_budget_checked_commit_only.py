#!/usr/bin/env python3
"""Harness for IQ-1463 secure-on audit-overhead commit-only wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = 1
GPU_SEC_PERF_PROFILE_DEV_LOCAL = 2


@dataclass(frozen=True)
class RowOutput:
    tok_per_sec_q16: int
    audit_overhead_delta_q16: int
    secure_cycles_per_token_q16: int


def _policy_allow_dispatch_checked(
    secure_local_mode: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> bool:
    if secure_local_mode not in (GPU_SEC_PERF_PROFILE_SECURE_LOCAL, GPU_SEC_PERF_PROFILE_DEV_LOCAL):
        return False
    if iommu_enabled != 1:
        return False
    if bot_dma_log_enabled != 1:
        return False
    if bot_mmio_log_enabled != 1:
        return False
    if bot_dispatch_log_enabled != 1:
        return False
    return True


def _audit_overhead_budget_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
    max_audit_overhead_delta_q16: int,
) -> tuple[int, int, int, int]:
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    if rows_capacity < len(rows) or max_audit_overhead_delta_q16 < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, -1, 0
    if not _policy_allow_dispatch_checked(
        secure_local_mode,
        iommu_enabled,
        bot_dma_log_enabled,
        bot_mmio_log_enabled,
        bot_dispatch_log_enabled,
    ):
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, -1, 0

    checked_rows = 0
    first_violation_index = -1
    max_observed_overhead_q16 = 0

    for i, row in enumerate(rows):
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0

        checked_rows = i + 1
        if i == 0 or row.audit_overhead_delta_q16 > max_observed_overhead_q16:
            max_observed_overhead_q16 = row.audit_overhead_delta_q16

        if row.audit_overhead_delta_q16 > max_audit_overhead_delta_q16:
            first_violation_index = i
            return GPU_SEC_PERF_ERR_POLICY_GUARD, checked_rows, first_violation_index, max_observed_overhead_q16

    return GPU_SEC_PERF_OK, checked_rows, first_violation_index, max_observed_overhead_q16


def gpu_security_perf_matrix_audit_overhead_budget_checked_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_enabled: int,
    book_hooks: int,
    max_audit_overhead_delta_q16: int,
) -> tuple[int, int, int, int]:
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    if rows_capacity < len(rows) or max_audit_overhead_delta_q16 < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    if iommu_enabled not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    if book_hooks not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, -1, 0

    first = _audit_overhead_budget_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        book_hooks,
        book_hooks,
        max_audit_overhead_delta_q16,
    )
    second = _audit_overhead_budget_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        book_hooks,
        book_hooks,
        max_audit_overhead_delta_q16,
    )
    if first[0] != second[0]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    if first[1:] != second[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    return first


def test_source_contains_iq1463_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixAuditOverheadBudgetCheckedCommitOnly(" in src
    assert "snapshot_max_audit_overhead_delta_q16" in src
    assert "snapshot_book_hooks" in src
    assert src.count("GPUSecurityPerfMatrixAuditOverheadBudgetChecked(") >= 3


def test_commit_only_boundary_and_over_budget_vectors() -> None:
    rows = [
        RowOutput(100_000, 4000, 9_000),
        RowOutput(120_000, 4500, 10_000),
        RowOutput(130_000, 4501, 11_000),
    ]

    status, checked, first_bad, max_observed = gpu_security_perf_matrix_audit_overhead_budget_checked_commit_only(
        rows,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=4500,
    )

    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert checked == 3
    assert first_bad == 2
    assert max_observed == 4501

    status, checked, first_bad, max_observed = gpu_security_perf_matrix_audit_overhead_budget_checked_commit_only(
        rows[:2],
        rows_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=4500,
    )

    assert status == GPU_SEC_PERF_OK
    assert checked == 2
    assert first_bad == -1
    assert max_observed == 4500


def test_commit_only_gate_missing_and_bad_param_vectors() -> None:
    rows = [RowOutput(50_000, 2000, 7000)]

    status, *_ = gpu_security_perf_matrix_audit_overhead_budget_checked_commit_only(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_DEV_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=3000,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, *_ = gpu_security_perf_matrix_audit_overhead_budget_checked_commit_only(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=0,
        book_hooks=1,
        max_audit_overhead_delta_q16=3000,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, *_ = gpu_security_perf_matrix_audit_overhead_budget_checked_commit_only(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=0,
        max_audit_overhead_delta_q16=3000,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, *_ = gpu_security_perf_matrix_audit_overhead_budget_checked_commit_only(
        rows,
        rows_capacity=0,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=3000,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM

    status, *_ = gpu_security_perf_matrix_audit_overhead_budget_checked_commit_only(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=-1,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM


if __name__ == "__main__":
    test_source_contains_iq1463_symbols()
    test_commit_only_boundary_and_over_budget_vectors()
    test_commit_only_gate_missing_and_bad_param_vectors()
    print("ok")
