#!/usr/bin/env python3
"""Harness for IQ-1476 latency-window no-partial wrapper."""

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

    found = False
    best_value = 0
    for candidate_row in rows:
        candidate = candidate_row.secure_cycles_per_token_q16
        if candidate < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0

        less_count = 0
        less_equal_count = 0
        for observed_row in rows:
            observed = observed_row.secure_cycles_per_token_q16
            if observed < 0:
                return GPU_SEC_PERF_ERR_BAD_PARAM, 0
            if observed < candidate:
                less_count += 1
            if observed <= candidate:
                less_equal_count += 1

        if less_count < rank <= less_equal_count:
            if not found or candidate < best_value:
                found = True
                best_value = candidate

    if not found:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0
    return GPU_SEC_PERF_OK, best_value


def _summary_checked(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> tuple[int, int, int, int, int]:
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if rows_capacity < len(rows):
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
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if rows_capacity < len(rows):
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
    if first[0] != second[0] or first[1:] != second[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if first[0] != GPU_SEC_PERF_OK:
        return first
    return first


def gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    current_p05_q16: int,
    current_p25_q16: int,
    current_p75_q16: int,
    current_p99_q16: int,
    outputs_alias: bool = False,
    has_null_output: bool = False,
) -> tuple[int, int, int, int, int]:
    if has_null_output:
        return (
            GPU_SEC_PERF_ERR_NULL_PTR,
            current_p05_q16,
            current_p25_q16,
            current_p75_q16,
            current_p99_q16,
        )
    if outputs_alias:
        return (
            GPU_SEC_PERF_ERR_BAD_PARAM,
            current_p05_q16,
            current_p25_q16,
            current_p75_q16,
            current_p99_q16,
        )

    preflight = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    commit = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only(
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
            current_p05_q16,
            current_p25_q16,
            current_p75_q16,
            current_p99_q16,
        )
    if preflight[0] != GPU_SEC_PERF_OK:
        return (
            preflight[0],
            current_p05_q16,
            current_p25_q16,
            current_p75_q16,
            current_p99_q16,
        )

    return preflight


def test_source_contains_iq1476_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartial(" in src
    assert "status_preflight = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnly(" in src
    assert "status_commit = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnly(" in src
    assert "if (out_p05_q16 == out_p25_q16" in src
    assert "snapshot_out_capacity" in src


def test_nopartial_null_alias_capacity_vectors() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, p05, p25, p75, p99 = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_p05_q16=7,
        current_p25_q16=8,
        current_p75_q16=9,
        current_p99_q16=10,
        has_null_output=True,
    )
    assert (status, p05, p25, p75, p99) == (GPU_SEC_PERF_ERR_NULL_PTR, 7, 8, 9, 10)

    status, p05, p25, p75, p99 = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_p05_q16=11,
        current_p25_q16=12,
        current_p75_q16=13,
        current_p99_q16=14,
        outputs_alias=True,
    )
    assert (status, p05, p25, p75, p99) == (GPU_SEC_PERF_ERR_BAD_PARAM, 11, 12, 13, 14)

    status, p05, p25, p75, p99 = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_p05_q16=20,
        current_p25_q16=21,
        current_p75_q16=22,
        current_p99_q16=23,
    )
    assert (status, p05, p25, p75, p99) == (GPU_SEC_PERF_ERR_CAPACITY, 20, 21, 22, 23)


def test_nopartial_gate_missing_and_duplicate_latency_vectors() -> None:
    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(101_000, 2_100, 10_000),
        RowOutput(102_000, 2_200, 30_000),
        RowOutput(103_000, 2_300, 30_000),
        RowOutput(104_000, 2_400, 50_000),
    ]

    status, p05, p25, p75, p99 = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial(
        rows,
        rows_capacity=5,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_p05_q16=31,
        current_p25_q16=32,
        current_p75_q16=33,
        current_p99_q16=34,
    )
    assert (status, p05, p25, p75, p99) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 31, 32, 33, 34)

    status, p05, p25, p75, p99 = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial(
        rows,
        rows_capacity=5,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_p05_q16=41,
        current_p25_q16=42,
        current_p75_q16=43,
        current_p99_q16=44,
    )
    assert status == GPU_SEC_PERF_OK
    assert (p05, p25, p75, p99) == (10_000, 10_000, 30_000, 50_000)


def test_nopartial_overflow_no_partial_publish() -> None:
    rows = [
        RowOutput(GPU_SEC_PERF_I64_MAX - 7, 2_000, 10_000),
        RowOutput(100, 2_100, 11_000),
    ]

    status, p05, p25, p75, p99 = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial(
        rows,
        rows_capacity=3,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        current_p05_q16=51,
        current_p25_q16=52,
        current_p75_q16=53,
        current_p99_q16=54,
    )

    assert (status, p05, p25, p75, p99) == (GPU_SEC_PERF_ERR_OVERFLOW, 51, 52, 53, 54)


if __name__ == "__main__":
    test_source_contains_iq1476_symbols()
    test_nopartial_null_alias_capacity_vectors()
    test_nopartial_gate_missing_and_duplicate_latency_vectors()
    test_nopartial_overflow_no_partial_publish()
    print("ok")
