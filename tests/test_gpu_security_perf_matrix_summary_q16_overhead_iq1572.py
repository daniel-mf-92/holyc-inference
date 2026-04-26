#!/usr/bin/env python3
"""Harness for IQ-1572 digest-locked overhead-envelope zero-write preflight companion."""

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
        for term_value, stride in (
            (row.tok_per_sec_q16, factor),
            (row.audit_overhead_delta_q16, factor + len(rows)),
            (row.secure_cycles_per_token_q16, factor + (len(rows) << 1)),
        ):
            term = _mul_checked(term_value, stride)
            if term is None:
                return GPU_SEC_PERF_ERR_OVERFLOW, 0
            digest = _add_checked(digest, term)
            if digest is None:
                return GPU_SEC_PERF_ERR_OVERFLOW, 0

    return GPU_SEC_PERF_OK, digest


def _status_is_valid(status: int) -> bool:
    return status in {
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ERR_NULL_PTR,
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ERR_CAPACITY,
        GPU_SEC_PERF_ERR_OVERFLOW,
    }


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
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0
    if iommu_active != 1 or book_of_truth_gpu_hooks != 1 or policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0

    min_overhead = 0
    max_overhead = 0
    overhead_sum = 0
    prev_overhead = 0
    prev_secure_cycles = 0
    for i, row in enumerate(rows):
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
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


def _overhead_envelope_preflight_only_parity_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int]]:
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


def _overhead_envelope_preflight_only_parity_commit_only_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int]]:
    commit_status, staged_tuple = _overhead_envelope_preflight_only_parity_commit_only(
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
    if commit_status != parity[0]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if commit_status != GPU_SEC_PERF_OK:
        return commit_status, (0, 0, 0)
    if staged_tuple != parity[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    return GPU_SEC_PERF_OK, parity[1:]


def _overhead_envelope_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int]]:
    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, (0, 0, 0)
    preflight_status, staged_tuple = _overhead_envelope_preflight_only_parity_commit_only_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    canonical_status, canonical_tuple = _overhead_envelope_preflight_only_parity_commit_only(
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
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if preflight_status != canonical_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if preflight_status != GPU_SEC_PERF_OK:
        return preflight_status, (0, 0, 0)
    if staged_tuple != canonical_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    return GPU_SEC_PERF_OK, canonical_tuple


def _overhead_envelope_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int]]:
    status_parity, parity_tuple = _overhead_envelope_preflight_only_parity_commit_only_preflight_only_digest_locked_parity(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    status_preflight, _, staged_tuple = _overhead_envelope_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        caller_outputs=parity_tuple,
    )
    if status_parity != status_preflight:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, (0, 0, 0)
    if parity_tuple != staged_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    return GPU_SEC_PERF_OK, parity_tuple


def _overhead_envelope_preflight_only(
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
    staged = _overhead_envelope_nopartial_commit_only(
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
    return GPU_SEC_PERF_OK, caller_outputs, canonical[1:]


def _overhead_envelope_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int]]:
    status, _, tuple_out = _overhead_envelope_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        caller_outputs=(0, 0, 0),
    )
    return status, tuple_out


def _overhead_envelope_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int],
    outputs_alias: bool = False,
    inject_digest_drift: bool = False,
    inject_invalid_status: bool = False,
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    if rows is None:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0)
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, caller_outputs, (0, 0, 0)

    status_preflight_only, staged_tuple = _overhead_envelope_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    status_commit_only, canonical_tuple = _overhead_envelope_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only(
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

    if inject_invalid_status:
        status_preflight_only = 77
    if inject_digest_drift:
        digest_after += 1

    if not _status_is_valid(status_preflight_only) or not _status_is_valid(status_commit_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if digest_before != digest_after:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if status_preflight_only != status_commit_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if staged_tuple != canonical_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if status_preflight_only != GPU_SEC_PERF_OK:
        return status_preflight_only, caller_outputs, (0, 0, 0)
    return GPU_SEC_PERF_OK, caller_outputs, canonical_tuple


def gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int],
    outputs_alias: bool = False,
    inject_digest_drift: bool = False,
    inject_invalid_status: bool = False,
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    if rows is None:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0)
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, caller_outputs, (0, 0, 0)

    status_commit_only, commit_only_tuple = _overhead_envelope_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    status_parity, _, parity_tuple = _overhead_envelope_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        caller_outputs=commit_only_tuple,
    )

    status_after, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_after != GPU_SEC_PERF_OK:
        return status_after, caller_outputs, (0, 0, 0)

    if inject_invalid_status:
        status_commit_only = 77
    if inject_digest_drift:
        digest_after += 1

    if not _status_is_valid(status_commit_only) or not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if digest_before != digest_after:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if status_commit_only != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if commit_only_tuple != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if status_commit_only != GPU_SEC_PERF_OK:
        return status_commit_only, caller_outputs, (0, 0, 0)
    return GPU_SEC_PERF_OK, caller_outputs, commit_only_tuple


def test_source_contains_iq1572_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert (
        "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyDigestLockedParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
        in src
    )
    assert (
        "status_commit_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyDigestLockedParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
        in src
    )
    assert (
        "status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyDigestLockedParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
        in src
    )
    assert "snapshot_before_digest_q64" in src
    assert "snapshot_after_digest_q64" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_commit_only))" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_parity))" in src


def test_secure_on_success_atomic_publish() -> None:
    rows = [
        RowOutput(100_000, 1_500, 10_000),
        RowOutput(99_000, 1_700, 10_500),
        RowOutput(98_000, 1_900, 11_000),
    ]
    status, preserved, published = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        rows,
        rows_capacity=3,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(9, 8, 7),
    )
    assert status == GPU_SEC_PERF_OK
    assert preserved == (9, 8, 7)
    assert published == (1_500, 1_900, 1_700)


def test_null_alias_capacity_gate_and_monotonicity_vectors() -> None:
    rows = [RowOutput(100_000, 1_500, 10_000)]
    status, preserved, published = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        None,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3),
    )
    assert (status, preserved, published) == (GPU_SEC_PERF_ERR_NULL_PTR, (1, 2, 3), (0, 0, 0))

    status, _, _ = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM

    status, _, _ = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        rows,
        rows_capacity=1,
        out_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3),
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM

    for iommu_active, hooks, parity in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        status, preserved, published = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
            rows,
            rows_capacity=1,
            out_capacity=3,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=hooks,
            policy_digest_parity=parity,
            caller_outputs=(11, 12, 13),
        )
        assert (status, preserved, published) == (GPU_SEC_PERF_ERR_POLICY_GUARD, (11, 12, 13), (0, 0, 0))

    monotonicity_breach_rows = [
        RowOutput(100_000, 1_500, 10_000),
        RowOutput(99_000, 1_400, 11_000),
    ]
    status, _, _ = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        monotonicity_breach_rows,
        rows_capacity=2,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(21, 22, 23),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD


def test_digest_drift_and_invalid_status_vectors() -> None:
    rows = [
        RowOutput(100_000, 1_500, 10_000),
        RowOutput(99_000, 1_700, 10_500),
        RowOutput(98_000, 1_900, 11_000),
    ]
    status, preserved, published = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        rows,
        rows_capacity=3,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(31, 32, 33),
        inject_digest_drift=True,
    )
    assert (status, preserved, published) == (GPU_SEC_PERF_ERR_BAD_PARAM, (31, 32, 33), (0, 0, 0))

    status, preserved, published = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        rows,
        rows_capacity=3,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(41, 42, 43),
        inject_invalid_status=True,
    )
    assert (status, preserved, published) == (GPU_SEC_PERF_ERR_BAD_PARAM, (41, 42, 43), (0, 0, 0))
