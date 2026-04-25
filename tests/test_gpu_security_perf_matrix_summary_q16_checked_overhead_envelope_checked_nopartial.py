#!/usr/bin/env python3
"""Harness for IQ-1478 overhead-envelope no-partial wrapper."""

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
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0

        if i > 0 and row.secure_cycles_per_token_q16 > prev_secure_cycles and row.audit_overhead_delta_q16 < prev_overhead:
            return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0

        if i == 0 or row.audit_overhead_delta_q16 < min_overhead:
            min_overhead = row.audit_overhead_delta_q16
        if i == 0 or row.audit_overhead_delta_q16 > max_overhead:
            max_overhead = row.audit_overhead_delta_q16

        overhead_sum_next = _add_checked(overhead_sum, row.audit_overhead_delta_q16)
        if overhead_sum_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0
        overhead_sum = overhead_sum_next

        prev_overhead = row.audit_overhead_delta_q16
        prev_secure_cycles = row.secure_cycles_per_token_q16

    return GPU_SEC_PERF_OK, min_overhead, max_overhead, overhead_sum // len(rows)


def gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, int, int, int]:
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    if rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    if out_capacity < 3:
        return GPU_SEC_PERF_ERR_CAPACITY, 0, 0, 0
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0

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
    if first[0] != GPU_SEC_PERF_OK:
        return first
    return first


def gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    current_min_overhead_q16: int,
    current_max_overhead_q16: int,
    current_mean_overhead_q16: int,
    outputs_alias: bool = False,
    has_null_output: bool = False,
) -> tuple[int, int, int, int]:
    if has_null_output:
        return (
            GPU_SEC_PERF_ERR_NULL_PTR,
            current_min_overhead_q16,
            current_max_overhead_q16,
            current_mean_overhead_q16,
        )
    if outputs_alias:
        return (
            GPU_SEC_PERF_ERR_BAD_PARAM,
            current_min_overhead_q16,
            current_max_overhead_q16,
            current_mean_overhead_q16,
        )

    preflight = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    commit = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    if preflight[0] != commit[0] or preflight[1:] != commit[1:]:
        return (
            GPU_SEC_PERF_ERR_BAD_PARAM,
            current_min_overhead_q16,
            current_max_overhead_q16,
            current_mean_overhead_q16,
        )
    if preflight[0] != GPU_SEC_PERF_OK:
        return (
            preflight[0],
            current_min_overhead_q16,
            current_max_overhead_q16,
            current_mean_overhead_q16,
        )
    return preflight


def test_source_contains_iq1478_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartial(" in src
    assert "status_preflight = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedCommitOnly(" in src
    assert "status_commit = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedCommitOnly(" in src
    assert "if (out_min_overhead_q16 == out_max_overhead_q16" in src
    assert "snapshot_out_capacity" in src
    assert "staged_mean_overhead_q16" in src
    assert "commit_mean_overhead_q16" in src


def test_nopartial_null_alias_capacity_vectors() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, min_q16, max_q16, mean_q16 = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_min_overhead_q16=11,
        current_max_overhead_q16=12,
        current_mean_overhead_q16=13,
        has_null_output=True,
    )
    assert (status, min_q16, max_q16, mean_q16) == (GPU_SEC_PERF_ERR_NULL_PTR, 11, 12, 13)

    status, min_q16, max_q16, mean_q16 = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_min_overhead_q16=21,
        current_max_overhead_q16=22,
        current_mean_overhead_q16=23,
        outputs_alias=True,
    )
    assert (status, min_q16, max_q16, mean_q16) == (GPU_SEC_PERF_ERR_BAD_PARAM, 21, 22, 23)

    status, min_q16, max_q16, mean_q16 = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial(
        rows,
        rows_capacity=1,
        out_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_min_overhead_q16=31,
        current_max_overhead_q16=32,
        current_mean_overhead_q16=33,
    )
    assert (status, min_q16, max_q16, mean_q16) == (GPU_SEC_PERF_ERR_CAPACITY, 31, 32, 33)


def test_gate_missing_and_monotonicity_breach_vectors() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, min_q16, max_q16, mean_q16 = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_min_overhead_q16=41,
        current_max_overhead_q16=42,
        current_mean_overhead_q16=43,
    )
    assert (status, min_q16, max_q16, mean_q16) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 41, 42, 43)

    monotonicity_breach_rows = [
        RowOutput(100_000, 1_800, 10_000),
        RowOutput(100_000, 1_700, 11_000),
    ]
    status, min_q16, max_q16, mean_q16 = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial(
        monotonicity_breach_rows,
        rows_capacity=2,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_min_overhead_q16=51,
        current_max_overhead_q16=52,
        current_mean_overhead_q16=53,
    )
    assert (status, min_q16, max_q16, mean_q16) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 51, 52, 53)


def test_overflow_and_success_publish_vectors() -> None:
    overflow_rows = [
        RowOutput(100_000, GPU_SEC_PERF_I64_MAX, 10_000),
        RowOutput(100_000, 1, 10_000),
    ]
    status, min_q16, max_q16, mean_q16 = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial(
        overflow_rows,
        rows_capacity=2,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_min_overhead_q16=61,
        current_max_overhead_q16=62,
        current_mean_overhead_q16=63,
    )
    assert (status, min_q16, max_q16, mean_q16) == (GPU_SEC_PERF_ERR_OVERFLOW, 61, 62, 63)

    rows = [
        RowOutput(100_000, 1_800, 10_000),
        RowOutput(110_000, 2_200, 11_000),
        RowOutput(120_000, 2_600, 12_000),
    ]
    status, min_q16, max_q16, mean_q16 = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked_nopartial(
        rows,
        rows_capacity=3,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_min_overhead_q16=71,
        current_max_overhead_q16=72,
        current_mean_overhead_q16=73,
    )
    assert status == GPU_SEC_PERF_OK
    assert (min_q16, max_q16, mean_q16) == (1_800, 2_600, 2_200)


if __name__ == "__main__":
    test_source_contains_iq1478_symbols()
    test_nopartial_null_alias_capacity_vectors()
    test_gate_missing_and_monotonicity_breach_vectors()
    test_overflow_and_success_publish_vectors()
    print("ok")
