#!/usr/bin/env python3
"""Harness for IQ-1508 overhead-envelope no-partial commit-only preflight-only wrapper."""

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


def _summary_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> int:
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM
    if rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD
    if iommu_enabled != 1 or bot_dma_log_enabled != 1 or bot_mmio_log_enabled != 1 or bot_dispatch_log_enabled != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD

    tok_total = 0
    for row in rows:
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM
        tok_total_next = _add_checked(tok_total, row.tok_per_sec_q16)
        if tok_total_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW
        tok_total = tok_total_next

    return GPU_SEC_PERF_OK


def _overhead_envelope_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, int, int, int]:
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    if rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0

    baseline = _summary_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
    )
    if baseline != GPU_SEC_PERF_OK:
        return baseline, 0, 0, 0

    min_overhead = 0
    max_overhead = 0
    overhead_sum = 0
    prev_overhead = 0
    prev_secure_cycles = 0

    for i, row in enumerate(rows):
        if i > 0 and row.secure_cycles_per_token_q16 > prev_secure_cycles and row.audit_overhead_delta_q16 < prev_overhead:
            return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0
        if i == 0 or row.audit_overhead_delta_q16 < min_overhead:
            min_overhead = row.audit_overhead_delta_q16
        if i == 0 or row.audit_overhead_delta_q16 > max_overhead:
            max_overhead = row.audit_overhead_delta_q16
        next_sum = _add_checked(overhead_sum, row.audit_overhead_delta_q16)
        if next_sum is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0
        overhead_sum = next_sum
        prev_overhead = row.audit_overhead_delta_q16
        prev_secure_cycles = row.secure_cycles_per_token_q16

    return GPU_SEC_PERF_OK, min_overhead, max_overhead, overhead_sum // len(rows)


def _overhead_envelope_nopartial(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, int, int, int]:
    if out_capacity < 3:
        return GPU_SEC_PERF_ERR_CAPACITY, 0, 0, 0
    first = _overhead_envelope_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    second = _overhead_envelope_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if first[0] != second[0] or first[1:] != second[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    return first


def _overhead_envelope_nopartial_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int],
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, caller_outputs, (0, 0, 0)

    staged = _overhead_envelope_nopartial(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    canonical = _overhead_envelope_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    status_after, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_after != GPU_SEC_PERF_OK or digest_before != digest_after:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    if staged[0] != canonical[0]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if staged[0] != GPU_SEC_PERF_OK:
        return staged[0], caller_outputs, (0, 0, 0)
    if staged[1:] != canonical[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    return GPU_SEC_PERF_OK, caller_outputs, staged[1:]


def _overhead_envelope_nopartial_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, int, int, int]:
    parity = _overhead_envelope_nopartial(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    preflight = _overhead_envelope_nopartial_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        caller_outputs=(parity[1], parity[2], parity[3]),
    )
    if parity[0] != preflight[0]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    if parity[0] != GPU_SEC_PERF_OK:
        return parity[0], 0, 0, 0
    if (parity[1], parity[2], parity[3]) != preflight[2]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    return parity


def gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int] | None,
    outputs_are_distinct: bool,
) -> tuple[int, tuple[int, int, int] | None, tuple[int, int, int]]:
    if rows is None or caller_outputs is None:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0)
    if not outputs_are_distinct:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if out_capacity < 3:
        return GPU_SEC_PERF_ERR_CAPACITY, caller_outputs, (0, 0, 0)
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, caller_outputs, (0, 0, 0)
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, caller_outputs, (0, 0, 0)

    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, caller_outputs, (0, 0, 0)

    commit_only = _overhead_envelope_nopartial_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    canonical = _overhead_envelope_nopartial(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    status_after, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_after != GPU_SEC_PERF_OK or digest_before != digest_after:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    if commit_only[0] != canonical[0]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if commit_only[0] != GPU_SEC_PERF_OK:
        return commit_only[0], caller_outputs, (0, 0, 0)

    if commit_only[1:] != canonical[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    return GPU_SEC_PERF_OK, caller_outputs, canonical[1:]


def test_source_contains_iq1508_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialCommitOnlyPreflightOnly(" in src
    assert "status_commit_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialCommitOnly(" in src
    assert "status_nopartial = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartial(" in src
    assert "if (status_commit_only != status_nopartial)" in src
    assert "snapshot_after_digest_q64" in src


def test_null_alias_capacity_vectors() -> None:
    rows = [RowOutput(100_000, 1_000, 10_000)]

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only(
        None,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3),
        outputs_are_distinct=True,
    )
    assert status == GPU_SEC_PERF_ERR_NULL_PTR
    assert outputs_after == (1, 2, 3)
    assert diag == (0, 0, 0)

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3),
        outputs_are_distinct=False,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs_after == (1, 2, 3)
    assert diag == (0, 0, 0)

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only(
        rows,
        rows_capacity=1,
        out_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(7, 8, 9),
        outputs_are_distinct=True,
    )
    assert status == GPU_SEC_PERF_ERR_CAPACITY
    assert outputs_after == (7, 8, 9)
    assert diag == (0, 0, 0)


def test_gate_missing_and_monotonicity_breach_vectors() -> None:
    base_rows = [RowOutput(100_000, 2_000, 10_000)]
    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only(
        base_rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(3, 4, 5),
        outputs_are_distinct=True,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs_after == (3, 4, 5)
    assert diag == (0, 0, 0)

    monotonicity_breach_rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(95_000, 1_000, 11_000),
    ]
    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only(
        monotonicity_breach_rows,
        rows_capacity=2,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(11, 12, 13),
        outputs_are_distinct=True,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs_after == (11, 12, 13)
    assert diag == (0, 0, 0)


def test_overflow_parity_and_secure_on_measurement_plan_vectors() -> None:
    overflow_rows = [
        RowOutput(1, GPU_SEC_PERF_I64_MAX, 1),
        RowOutput(1, 1, 2),
    ]
    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only(
        overflow_rows,
        rows_capacity=2,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(21, 22, 23),
        outputs_are_distinct=True,
    )
    assert status == GPU_SEC_PERF_ERR_OVERFLOW
    assert outputs_after == (21, 22, 23)
    assert diag == (0, 0, 0)

    # secure-on overhead measurement plan vector set: low/median/high overhead rows.
    secure_plan_rows = [
        RowOutput(120_000, 800, 9_000),
        RowOutput(110_000, 1_600, 10_000),
        RowOutput(100_000, 3_200, 12_000),
    ]
    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only(
        secure_plan_rows,
        rows_capacity=3,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(31, 32, 33),
        outputs_are_distinct=True,
    )
    assert status == GPU_SEC_PERF_OK
    assert outputs_after == (31, 32, 33)
    assert diag == (800, 3_200, 1_866)


if __name__ == "__main__":
    test_source_contains_iq1508_symbols()
    test_null_alias_capacity_vectors()
    test_gate_missing_and_monotonicity_breach_vectors()
    test_overflow_parity_and_secure_on_measurement_plan_vectors()
    print("ok")
