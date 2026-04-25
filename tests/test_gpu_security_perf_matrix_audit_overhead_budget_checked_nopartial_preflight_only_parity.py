#!/usr/bin/env python3
"""Harness for IQ-1466 audit-overhead preflight-only parity gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_NULL_PTR = 1
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3
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


def _policy_allow_dispatch_checked(
    secure_local_mode: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> bool:
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
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
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0
    if rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    digest = 1_469_598_103_934_665_603
    row_count = len(rows)
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

        term = _mul_checked(row.audit_overhead_delta_q16, factor + row_count)
        if term is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0
        digest = _add_checked(digest, term)
        if digest is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0

        term = _mul_checked(row.secure_cycles_per_token_q16, factor + (row_count << 1))
        if term is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0
        digest = _add_checked(digest, term)
        if digest is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0

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


def _audit_overhead_budget_checked_nopartial(
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

    before_status, before_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if before_status != GPU_SEC_PERF_OK:
        return before_status, 0, -1, 0

    preflight = _audit_overhead_budget_checked_commit_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
    )

    mid_status, mid_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if mid_status != GPU_SEC_PERF_OK or mid_digest != before_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0

    commit = _audit_overhead_budget_checked_commit_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
    )

    after_status, after_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if after_status != GPU_SEC_PERF_OK or after_digest != before_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    if preflight[0] != commit[0] or preflight[1:] != commit[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, -1, 0
    if preflight[0] != GPU_SEC_PERF_OK:
        return preflight
    return preflight


def _audit_overhead_budget_checked_nopartial_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_enabled: int,
    book_hooks: int,
    max_audit_overhead_delta_q16: int,
    caller_outputs: tuple[int, int, int],
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    saved_outputs = caller_outputs

    before_status, before_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if before_status != GPU_SEC_PERF_OK:
        return before_status, caller_outputs, (0, -1, 0)

    staged = _audit_overhead_budget_checked_nopartial(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
    )
    canonical = _audit_overhead_budget_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        book_hooks,
        book_hooks,
        max_audit_overhead_delta_q16,
    )

    after_status, after_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if after_status != GPU_SEC_PERF_OK or after_digest != before_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, -1, 0)
    if staged[0] != canonical[0] or staged[1:] != canonical[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, -1, 0)
    return staged[0], saved_outputs, canonical[1:]


def gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial_preflight_only_parity(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    secure_local_mode: int,
    iommu_enabled: int,
    book_hooks: int,
    max_audit_overhead_delta_q16: int,
    caller_outputs: tuple[int, int, int],
    *,
    alias_outputs: bool = False,
    simulate_parity_drift: bool = False,
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    if rows is None:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, -1, 0)
    if alias_outputs:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, -1, 0)
    if not rows or rows_capacity < len(rows) or max_audit_overhead_delta_q16 < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, -1, 0)
    if iommu_enabled not in (0, 1) or book_hooks not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, -1, 0)
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, caller_outputs, (0, -1, 0)

    saved_outputs = caller_outputs

    before_status, before_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if before_status != GPU_SEC_PERF_OK:
        return before_status, caller_outputs, (0, -1, 0)

    preflight_only = _audit_overhead_budget_checked_nopartial_preflight_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
        caller_outputs,
    )
    nopartial = _audit_overhead_budget_checked_nopartial(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
    )

    after_status, after_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if after_status != GPU_SEC_PERF_OK or after_digest != before_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, -1, 0)
    if preflight_only[1] != saved_outputs:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, -1, 0)
    if preflight_only[0] != nopartial[0]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, -1, 0)

    preflight_tuple = preflight_only[2]
    nopartial_tuple = nopartial[1:]
    if simulate_parity_drift:
        nopartial_tuple = (nopartial_tuple[0] + 1, nopartial_tuple[1], nopartial_tuple[2])

    if preflight_tuple != nopartial_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, -1, 0)
    return preflight_only[0], saved_outputs, preflight_tuple


def test_source_contains_iq1466_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixAuditOverheadBudgetCheckedNoPartialPreflightOnlyParity(" in src
    assert "status_preflight_only" in src
    assert "status_nopartial" in src
    assert "snapshot_after_digest_q64" in src
    assert "saved_checked_rows" in src


def test_preflight_only_parity_success_and_zero_write() -> None:
    rows = [
        RowOutput(100_000, 1_000, 9_000),
        RowOutput(110_000, 2_000, 8_000),
        RowOutput(120_000, 3_000, 7_000),
    ]
    caller_outputs = (44, 55, 66)

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial_preflight_only_parity(
        rows,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=3_000,
        caller_outputs=caller_outputs,
    )
    assert status == GPU_SEC_PERF_OK
    assert outputs_after == caller_outputs
    assert diag_tuple == (3, -1, 3_000)


def test_preflight_only_parity_null_alias_capacity_and_policy_vectors() -> None:
    rows = [RowOutput(100_000, 1_000, 9_000)]

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial_preflight_only_parity(
        None,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=1_000,
        caller_outputs=(1, 2, 3),
    )
    assert status == GPU_SEC_PERF_ERR_NULL_PTR
    assert outputs_after == (1, 2, 3)
    assert diag_tuple == (0, -1, 0)

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial_preflight_only_parity(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=1_000,
        caller_outputs=(1, 2, 3),
        alias_outputs=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs_after == (1, 2, 3)
    assert diag_tuple == (0, -1, 0)

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial_preflight_only_parity(
        rows,
        rows_capacity=0,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=1_000,
        caller_outputs=(1, 2, 3),
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs_after == (1, 2, 3)
    assert diag_tuple == (0, -1, 0)

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial_preflight_only_parity(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=0,
        book_hooks=1,
        max_audit_overhead_delta_q16=1_000,
        caller_outputs=(1, 2, 3),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs_after == (1, 2, 3)
    assert diag_tuple == (0, -1, 0)


def test_preflight_only_parity_over_budget_and_drift_vectors() -> None:
    rows = [
        RowOutput(100_000, 1_000, 9_000),
        RowOutput(120_000, 2_500, 8_000),
        RowOutput(130_000, 4_500, 7_000),
    ]

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial_preflight_only_parity(
        rows,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=3_000,
        caller_outputs=(7, 8, 9),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs_after == (7, 8, 9)
    assert diag_tuple == (3, 2, 4_500)

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_audit_overhead_budget_checked_nopartial_preflight_only_parity(
        rows,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        book_hooks=1,
        max_audit_overhead_delta_q16=10_000,
        caller_outputs=(7, 8, 9),
        simulate_parity_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs_after == (7, 8, 9)
    assert diag_tuple == (0, -1, 0)


if __name__ == "__main__":
    test_source_contains_iq1466_symbols()
    test_preflight_only_parity_success_and_zero_write()
    test_preflight_only_parity_null_alias_capacity_and_policy_vectors()
    test_preflight_only_parity_over_budget_and_drift_vectors()
    print("ok")
