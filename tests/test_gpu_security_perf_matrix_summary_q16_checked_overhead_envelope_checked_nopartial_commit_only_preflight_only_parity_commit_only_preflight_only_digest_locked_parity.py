#!/usr/bin/env python3
"""Harness for IQ-1538 overhead-envelope digest-locked strict diagnostics parity gate."""

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


def _status_is_valid(status: int) -> bool:
    return status in (
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ERR_NULL_PTR,
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ERR_CAPACITY,
        GPU_SEC_PERF_ERR_OVERFLOW,
    )


def _summary_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> int:
    if not rows or rows_capacity < len(rows):
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
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL or policy_digest_parity != 1:
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


def _overhead_envelope_nopartial_commit_only(
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


def _overhead_envelope_nopartial_commit_only_preflight_only_parity_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int]]:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if out_capacity < 3:
        return GPU_SEC_PERF_ERR_CAPACITY, (0, 0, 0)
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (0, 0, 0)
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (0, 0, 0)

    parity = _overhead_envelope_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    commit = _overhead_envelope_nopartial_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    if parity[0] != commit[0]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if parity[0] != GPU_SEC_PERF_OK:
        return parity[0], (0, 0, 0)
    if parity[1:] != commit[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    return GPU_SEC_PERF_OK, commit[1:]


def _overhead_envelope_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int]]:
    commit_status, staged_tuple = _overhead_envelope_nopartial_commit_only_preflight_only_parity_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    parity = _overhead_envelope_nopartial_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    parity_status = parity[0]
    parity_tuple = parity[1:]

    if commit_status != parity_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if commit_status != GPU_SEC_PERF_OK:
        return commit_status, (0, 0, 0)
    if staged_tuple != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)

    return GPU_SEC_PERF_OK, parity_tuple


def gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int],
    outputs_alias: bool = False,
    has_null_output: bool = False,
    inject_digest_drift: bool = False,
    inject_status_drift: bool = False,
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    if rows is None or has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0)
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, caller_outputs, (0, 0, 0)

    preflight_status, staged_tuple = _overhead_envelope_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    canonical_status, canonical_tuple = _overhead_envelope_nopartial_commit_only_preflight_only_parity_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    status_after, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_after != GPU_SEC_PERF_OK:
        return status_after, caller_outputs, (0, 0, 0)

    if inject_digest_drift:
        digest_after += 1
    if inject_status_drift:
        preflight_status = 77

    if not _status_is_valid(preflight_status) or not _status_is_valid(canonical_status):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if digest_before != digest_after:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    if preflight_status != canonical_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if preflight_status != GPU_SEC_PERF_OK:
        return preflight_status, caller_outputs, (0, 0, 0)
    if staged_tuple != canonical_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    return GPU_SEC_PERF_OK, caller_outputs, canonical_tuple


def test_source_contains_iq1538_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert (
        "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyDigestLockedParity(" in src
    )
    assert (
        "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in src
    )
    assert (
        "status_canonical = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in src
    )
    assert "snapshot_before_digest_q64" in src
    assert "snapshot_after_digest_q64" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_preflight_only))" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_canonical))" in src


def test_success_preserves_outputs_and_reports_overhead_tuple() -> None:
    rows = [
        RowOutput(100_000, 1_500, 10_000),
        RowOutput(99_000, 1_700, 10_500),
        RowOutput(98_000, 1_900, 11_000),
    ]

    status, preserved, staged = (
        gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
            rows,
            rows_capacity=4,
            out_capacity=3,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            caller_outputs=(7, 8, 9),
        )
    )

    assert status == GPU_SEC_PERF_OK
    assert preserved == (7, 8, 9)
    assert staged == (1_500, 1_900, 1_700)


def test_null_alias_capacity_vectors() -> None:
    rows = [RowOutput(100_000, 1_500, 10_000)]

    status, preserved, staged = (
        gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
            None,
            rows_capacity=1,
            out_capacity=3,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            caller_outputs=(1, 2, 3),
        )
    )
    assert (status, preserved, staged) == (GPU_SEC_PERF_ERR_NULL_PTR, (1, 2, 3), (0, 0, 0))

    status, _, _ = (
        gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
            rows,
            rows_capacity=1,
            out_capacity=3,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            caller_outputs=(1, 2, 3),
            outputs_alias=True,
        )
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM

    status, _, _ = (
        gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
            rows,
            rows_capacity=1,
            out_capacity=2,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            caller_outputs=(1, 2, 3),
        )
    )
    assert status == GPU_SEC_PERF_ERR_CAPACITY


def test_gate_missing_monotonicity_breach_and_digest_drift_vectors() -> None:
    guarded_rows = [
        RowOutput(100_000, 1_500, 10_000),
        RowOutput(99_000, 1_400, 10_500),
    ]

    status, _, _ = (
        gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
            guarded_rows,
            rows_capacity=2,
            out_capacity=3,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            caller_outputs=(11, 12, 13),
        )
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, _, _ = (
        gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
            [RowOutput(100_000, 1_500, 10_000)],
            rows_capacity=1,
            out_capacity=3,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=0,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            caller_outputs=(11, 12, 13),
        )
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    parity_rows = [
        RowOutput(100_000, 1_500, 10_000),
        RowOutput(99_000, 1_700, 10_500),
        RowOutput(98_000, 1_900, 11_000),
    ]
    status, preserved, staged = (
        gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
            parity_rows,
            rows_capacity=3,
            out_capacity=3,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            caller_outputs=(21, 22, 23),
            inject_digest_drift=True,
        )
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert preserved == (21, 22, 23)
    assert staged == (0, 0, 0)


def test_invalid_status_and_overflow_vectors_preserve_outputs() -> None:
    rows = [
        RowOutput(100_000, 1_500, 10_000),
        RowOutput(99_000, 1_700, 10_500),
    ]

    status, preserved, staged = (
        gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
            rows,
            rows_capacity=2,
            out_capacity=3,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            caller_outputs=(31, 32, 33),
            inject_status_drift=True,
        )
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert preserved == (31, 32, 33)
    assert staged == (0, 0, 0)

    overflow_rows = [RowOutput(GPU_SEC_PERF_I64_MAX, 1, 1)]
    status, preserved, staged = (
        gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
            overflow_rows,
            rows_capacity=1,
            out_capacity=3,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            caller_outputs=(41, 42, 43),
        )
    )
    assert status == GPU_SEC_PERF_ERR_OVERFLOW
    assert preserved == (41, 42, 43)
    assert staged == (0, 0, 0)
