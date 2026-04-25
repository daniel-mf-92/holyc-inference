#!/usr/bin/env python3
"""Harness for IQ-1462 secure-on overhead-envelope summary helper."""

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


def gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    simulate_null_output: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    if rows is None or simulate_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, (0, 0, 0)
    if rows_capacity < len(rows) or len(rows) <= 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (0, 0, 0)
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (0, 0, 0)

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
        return baseline, (0, 0, 0)

    min_overhead = 0
    max_overhead = 0
    overhead_sum = 0
    prev_overhead = 0
    prev_secure_cycles = 0

    for i, row in enumerate(rows):
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0)

        if i > 0 and row.secure_cycles_per_token_q16 > prev_secure_cycles and row.audit_overhead_delta_q16 < prev_overhead:
            return GPU_SEC_PERF_ERR_POLICY_GUARD, (0, 0, 0)

        if i == 0 or row.audit_overhead_delta_q16 < min_overhead:
            min_overhead = row.audit_overhead_delta_q16
        if i == 0 or row.audit_overhead_delta_q16 > max_overhead:
            max_overhead = row.audit_overhead_delta_q16

        overhead_sum_next = _add_checked(overhead_sum, row.audit_overhead_delta_q16)
        if overhead_sum_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, (0, 0, 0)
        overhead_sum = overhead_sum_next

        prev_overhead = row.audit_overhead_delta_q16
        prev_secure_cycles = row.secure_cycles_per_token_q16

    return GPU_SEC_PERF_OK, (min_overhead, max_overhead, overhead_sum // len(rows))


def test_source_contains_iq1462_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeChecked(" in src
    assert "out_min_overhead_q16" in src
    assert "out_max_overhead_q16" in src
    assert "out_mean_overhead_q16" in src


def test_overhead_envelope_secure_on_success() -> None:
    rows = [
        RowOutput(100_000, 1800, 10_000),
        RowOutput(110_000, 2200, 11_000),
        RowOutput(120_000, 2600, 12_000),
        RowOutput(130_000, 2600, 12_000),
    ]

    status, out_tuple = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked(
        rows,
        rows_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )

    assert status == GPU_SEC_PERF_OK
    assert out_tuple == (1800, 2600, 2300)


def test_overhead_envelope_gate_missing_and_budget_breach() -> None:
    rows = [
        RowOutput(100_000, 1800, 10_000),
        RowOutput(110_000, 1700, 11_000),
    ]

    status, _ = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked(
        rows,
        rows_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, _ = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked(
        rows,
        rows_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD


def test_overhead_envelope_overflow_and_bad_param() -> None:
    overflow_rows = [
        RowOutput(100_000, GPU_SEC_PERF_I64_MAX, 10_000),
        RowOutput(100_000, 1, 10_000),
    ]

    status, _ = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked(
        overflow_rows,
        rows_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_OVERFLOW

    status, _ = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked(
        None,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_NULL_PTR

    bad_rows = [RowOutput(100_000, -1, 10_000)]
    status, _ = gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_checked(
        bad_rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM


if __name__ == "__main__":
    test_source_contains_iq1462_symbols()
    test_overhead_envelope_secure_on_success()
    test_overhead_envelope_gate_missing_and_budget_breach()
    test_overhead_envelope_overflow_and_bad_param()
    print("ok")
