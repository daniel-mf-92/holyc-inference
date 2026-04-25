#!/usr/bin/env python3
"""Harness for IQ-1464 secure-on audit-overhead no-partial wrapper."""

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


def _rows_snapshot_digest_q64(rows: list[RowOutput], rows_capacity: int) -> tuple[int, int]:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    digest = 1_469_598_103_934_665_603
    row_count = len(rows)
    for i, row in enumerate(rows):
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0
        factor = i + 1
        digest += row.tok_per_sec_q16 * factor
        digest += row.audit_overhead_delta_q16 * (factor + row_count)
        digest += row.secure_cycles_per_token_q16 * (factor + (row_count << 1))
    return GPU_SEC_PERF_OK, digest


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


def _audit_overhead_budget_checked_commit_only(
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
    if iommu_enabled not in (0, 1) or book_hooks not in (0, 1):
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
    if first[0] != second[0] or first[1:] != second[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    return first


def gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_enabled: int,
    book_hooks: int,
    max_audit_overhead_delta_q16: int,
    current_checked_rows: int,
    current_first_violation_index: int,
    current_max_observed_overhead_q16: int,
) -> tuple[int, int, int, int]:
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_checked_rows, current_first_violation_index, current_max_observed_overhead_q16
    if rows_capacity < len(rows) or max_audit_overhead_delta_q16 < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_checked_rows, current_first_violation_index, current_max_observed_overhead_q16
    if iommu_enabled not in (0, 1) or book_hooks not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_checked_rows, current_first_violation_index, current_max_observed_overhead_q16
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, current_checked_rows, current_first_violation_index, current_max_observed_overhead_q16

    digest_status, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if digest_status != GPU_SEC_PERF_OK:
        return digest_status, current_checked_rows, current_first_violation_index, current_max_observed_overhead_q16

    preflight = _audit_overhead_budget_checked_commit_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
    )
    digest_status, digest_mid = _rows_snapshot_digest_q64(rows, rows_capacity)
    if digest_status != GPU_SEC_PERF_OK or digest_mid != digest_before:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_checked_rows, current_first_violation_index, current_max_observed_overhead_q16

    commit = _audit_overhead_budget_checked_commit_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
    )
    digest_status, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if digest_status != GPU_SEC_PERF_OK or digest_after != digest_before:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_checked_rows, current_first_violation_index, current_max_observed_overhead_q16

    if preflight[0] != commit[0] or preflight[1:] != commit[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_checked_rows, current_first_violation_index, current_max_observed_overhead_q16

    if preflight[0] != GPU_SEC_PERF_OK:
        return preflight[0], current_checked_rows, current_first_violation_index, current_max_observed_overhead_q16

    return preflight


def test_source_contains_iq1464_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixAuditOverheadBudgetCheckedNoPartial(" in src
    assert "status_preflight = GPUSecurityPerfMatrixAuditOverheadBudgetCheckedCommitOnly(" in src
    assert "status_commit = GPUSecurityPerfMatrixAuditOverheadBudgetCheckedCommitOnly(" in src
    assert "snapshot_before_digest_q64" in src


def test_nopartial_success_and_over_budget_publish_vectors() -> None:
    rows = [
        RowOutput(100_000, 4000, 9_000),
        RowOutput(120_000, 4500, 10_000),
        RowOutput(130_000, 4501, 11_000),
    ]

    status, checked, first_bad, max_observed = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial(
        rows[:2],
        rows_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=4500,
        current_checked_rows=77,
        current_first_violation_index=88,
        current_max_observed_overhead_q16=99,
    )
    assert status == GPU_SEC_PERF_OK
    assert checked == 2
    assert first_bad == -1
    assert max_observed == 4500

    status, checked, first_bad, max_observed = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial(
        rows,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=4500,
        current_checked_rows=11,
        current_first_violation_index=12,
        current_max_observed_overhead_q16=13,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert checked == 11
    assert first_bad == 12
    assert max_observed == 13


def test_nopartial_no_write_on_failure_vectors() -> None:
    rows = [RowOutput(100_000, 1000, 9_000)]

    status, checked, first_bad, max_observed = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=0,
        book_hooks=1,
        max_audit_overhead_delta_q16=3000,
        current_checked_rows=222,
        current_first_violation_index=333,
        current_max_observed_overhead_q16=444,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert checked == 222
    assert first_bad == 333
    assert max_observed == 444

    status, checked, first_bad, max_observed = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial(
        rows,
        rows_capacity=0,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=3000,
        current_checked_rows=555,
        current_first_violation_index=666,
        current_max_observed_overhead_q16=777,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert checked == 555
    assert first_bad == 666
    assert max_observed == 777


if __name__ == "__main__":
    test_source_contains_iq1464_symbols()
    test_nopartial_success_and_over_budget_publish_vectors()
    test_nopartial_no_write_on_failure_vectors()
    print("ok")
