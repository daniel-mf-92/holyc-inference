#!/usr/bin/env python3
"""Harness for IQ-1474 latency-window commit-only wrapper."""

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
GPU_SEC_PERF_Q16_ONE = 65536
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


def _select_percentile_nearest_rank_q16_checked(
    rows: list[RowOutput],
    percentile_num: int,
    percentile_den: int,
) -> tuple[int, int]:
    if not rows or percentile_num <= 0 or percentile_den <= 0 or percentile_num > percentile_den:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    scaled_rank_num = _mul_checked(percentile_num, len(rows))
    if scaled_rank_num is None:
        return GPU_SEC_PERF_ERR_OVERFLOW, 0

    rank = scaled_rank_num // percentile_den
    if scaled_rank_num % percentile_den:
        rank_next = _add_checked(rank, 1)
        if rank_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0
        rank = rank_next

    rank = max(1, min(rank, len(rows)))
    ordered = sorted(row.secure_cycles_per_token_q16 for row in rows)
    if ordered[0] < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0
    return GPU_SEC_PERF_OK, ordered[rank - 1]


def _summary_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> tuple[int, int, int, int, int]:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0
    if iommu_enabled != 1 or bot_dma_log_enabled != 1 or bot_mmio_log_enabled != 1 or bot_dispatch_log_enabled != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0

    tok_total = 0
    overhead_total = 0

    for row in rows:
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
        tok_total_next = _add_checked(tok_total, row.tok_per_sec_q16)
        overhead_total_next = _add_checked(overhead_total, row.audit_overhead_delta_q16)
        if tok_total_next is None or overhead_total_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, 0
        tok_total = tok_total_next
        overhead_total = overhead_total_next

    row_count_q16 = _mul_checked(len(rows), GPU_SEC_PERF_Q16_ONE)
    overhead_total_q16 = _mul_checked(overhead_total, GPU_SEC_PERF_Q16_ONE)
    if row_count_q16 in (None, 0) or overhead_total_q16 is None:
        return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, 0

    p50_status, p50_q16 = _select_percentile_nearest_rank_q16_checked(rows, 50, 100)
    if p50_status != GPU_SEC_PERF_OK:
        return p50_status, 0, 0, 0, 0

    p95_status, p95_q16 = _select_percentile_nearest_rank_q16_checked(rows, 95, 100)
    if p95_status != GPU_SEC_PERF_OK:
        return p95_status, 0, 0, 0, 0

    return GPU_SEC_PERF_OK, tok_total, overhead_total_q16 // row_count_q16, p50_q16, p95_q16


def _latency_window_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, int, int, int, int]:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0

    baseline = _summary_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
    )
    if baseline[0] != GPU_SEC_PERF_OK:
        return baseline[0], 0, 0, 0, 0

    p05_status, p05_q16 = _select_percentile_nearest_rank_q16_checked(rows, 5, 100)
    if p05_status != GPU_SEC_PERF_OK:
        return p05_status, 0, 0, 0, 0

    p25_status, p25_q16 = _select_percentile_nearest_rank_q16_checked(rows, 25, 100)
    if p25_status != GPU_SEC_PERF_OK:
        return p25_status, 0, 0, 0, 0

    p75_status, p75_q16 = _select_percentile_nearest_rank_q16_checked(rows, 75, 100)
    if p75_status != GPU_SEC_PERF_OK:
        return p75_status, 0, 0, 0, 0

    p99_status, p99_q16 = _select_percentile_nearest_rank_q16_checked(rows, 99, 100)
    if p99_status != GPU_SEC_PERF_OK:
        return p99_status, 0, 0, 0, 0

    return GPU_SEC_PERF_OK, p05_q16, p25_q16, p75_q16, p99_q16


def gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, int, int, int, int]:
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if out_capacity < 4:
        return GPU_SEC_PERF_ERR_CAPACITY, 0, 0, 0, 0
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0

    first = _latency_window_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    second = _latency_window_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if first[0] != second[0]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if first[1:] != second[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if first[0] != GPU_SEC_PERF_OK:
        return first
    return first


def test_source_contains_iq1474_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnly(" in src
    assert "snapshot_out_capacity" in src
    assert "status_primary = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowChecked(" in src
    assert "status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowChecked(" in src
    assert "staged_p99_q16" in src


def test_gate_missing_vectors_fail_closed() -> None:
    rows = [RowOutput(100_000, 2_000, 11_000)]

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=0,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=0,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD


def test_duplicate_latency_ordering_vectors() -> None:
    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(101_000, 2_100, 10_000),
        RowOutput(102_000, 2_200, 20_000),
        RowOutput(103_000, 2_300, 40_000),
        RowOutput(104_000, 2_400, 40_000),
        RowOutput(105_000, 2_500, 80_000),
    ]

    status, p05_q16, p25_q16, p75_q16, p99_q16 = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only(
        rows,
        rows_capacity=6,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_OK
    assert (p05_q16, p25_q16, p75_q16, p99_q16) == (10_000, 10_000, 40_000, 80_000)


def test_overflow_boundary_vectors() -> None:
    overflow_tok_rows = [
        RowOutput(GPU_SEC_PERF_I64_MAX, 0, 10_000),
        RowOutput(1, 0, 20_000),
    ]
    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only(
        overflow_tok_rows,
        rows_capacity=2,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_OVERFLOW

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only(
        [RowOutput(100_000, 2_000, 10_000)],
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_CAPACITY


if __name__ == "__main__":
    test_source_contains_iq1474_symbols()
    test_gate_missing_vectors_fail_closed()
    test_duplicate_latency_ordering_vectors()
    test_overflow_boundary_vectors()
    print("ok")
