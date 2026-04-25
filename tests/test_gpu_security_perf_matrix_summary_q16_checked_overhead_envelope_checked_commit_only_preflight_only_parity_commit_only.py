#!/usr/bin/env python3
"""Harness for IQ-1507 overhead-envelope commit-only parity commit-only wrapper."""

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


def _overhead_envelope_commit_only(
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


def _overhead_envelope_commit_only_preflight_only(
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

    staged = _overhead_envelope_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if staged[0] != GPU_SEC_PERF_OK:
        return staged[0], caller_outputs, (0, 0, 0)

    canonical = _overhead_envelope_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if canonical[0] != GPU_SEC_PERF_OK:
        return canonical[0], caller_outputs, (0, 0, 0)

    status_after, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_after != GPU_SEC_PERF_OK or digest_after != digest_before:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if staged[1:] != canonical[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    return GPU_SEC_PERF_OK, caller_outputs, canonical[1:]


def _overhead_envelope_commit_only_preflight_only_parity(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int]]:
    preflight = _overhead_envelope_commit_only_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        (0, 0, 0),
    )
    commit_only = _overhead_envelope_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    if preflight[0] != commit_only[0]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if preflight[0] != GPU_SEC_PERF_OK:
        return preflight[0], (0, 0, 0)
    if preflight[2] != commit_only[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)

    return GPU_SEC_PERF_OK, preflight[2]


def gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only_preflight_only_parity_commit_only(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int],
    outputs_alias: bool = False,
    inject_parity_drift: bool = False,
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    if rows is None:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0)
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    parity_status, parity_tuple = _overhead_envelope_commit_only_preflight_only_parity(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    preflight = _overhead_envelope_commit_only_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        parity_tuple,
    )
    preflight_status, preserved_tuple, staged_tuple = preflight

    if inject_parity_drift:
        staged_tuple = (staged_tuple[0], staged_tuple[1], staged_tuple[2] + 1)

    if parity_status != preflight_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if parity_status != GPU_SEC_PERF_OK:
        return parity_status, caller_outputs, (0, 0, 0)
    if preserved_tuple != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)
    if staged_tuple != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0)

    return GPU_SEC_PERF_OK, parity_tuple, staged_tuple


def test_source_contains_iq1507_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedCommitOnlyPreflightOnlyParityCommitOnly(" in src
    assert "status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedCommitOnlyPreflightOnlyParity(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedCommitOnlyPreflightOnly(" in src
    assert "saved_min_overhead_q16" in src
    assert "saved_max_overhead_q16" in src
    assert "saved_mean_overhead_q16" in src


def test_commit_only_success_publishes_tuple() -> None:
    rows = [
        RowOutput(100_000, 1_600, 10_000),
        RowOutput(98_000, 1_800, 10_400),
        RowOutput(96_000, 2_000, 10_800),
    ]
    status, persisted, staged = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only_preflight_only_parity_commit_only(
        rows,
        rows_capacity=3,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(7, 8, 9),
    )

    assert status == GPU_SEC_PERF_OK
    assert persisted == (1_600, 2_000, 1_800)
    assert staged == (1_600, 2_000, 1_800)


def test_null_alias_capacity_gate_monotonicity_and_parity_vectors() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, persisted, staged = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only_preflight_only_parity_commit_only(
        None,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3),
    )
    assert (status, persisted, staged) == (GPU_SEC_PERF_ERR_NULL_PTR, (1, 2, 3), (0, 0, 0))

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only_preflight_only_parity_commit_only(
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

    status, persisted, staged = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only_preflight_only_parity_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(10, 11, 12),
    )
    assert (status, persisted, staged) == (GPU_SEC_PERF_ERR_CAPACITY, (10, 11, 12), (0, 0, 0))

    status, persisted, staged = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only_preflight_only_parity_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=0,
        caller_outputs=(10, 11, 12),
    )
    assert (status, persisted, staged) == (GPU_SEC_PERF_ERR_POLICY_GUARD, (10, 11, 12), (0, 0, 0))

    bad_monotonic_rows = [
        RowOutput(100_000, 1_900, 10_000),
        RowOutput(101_000, 1_800, 11_000),
    ]
    status, persisted, staged = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only_preflight_only_parity_commit_only(
        bad_monotonic_rows,
        rows_capacity=2,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(10, 11, 12),
    )
    assert (status, persisted, staged) == (GPU_SEC_PERF_ERR_POLICY_GUARD, (10, 11, 12), (0, 0, 0))

    parity_rows = [
        RowOutput(100_000, 2_100, 10_000),
        RowOutput(100_100, 2_200, 11_000),
        RowOutput(100_200, 2_300, 12_000),
    ]
    status, persisted, staged = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only_preflight_only_parity_commit_only(
        parity_rows,
        rows_capacity=3,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(5, 6, 7),
        inject_parity_drift=True,
    )
    assert (status, persisted, staged) == (GPU_SEC_PERF_ERR_BAD_PARAM, (5, 6, 7), (0, 0, 0))
