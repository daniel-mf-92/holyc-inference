#!/usr/bin/env python3
"""Harness for IQ-1461 latency-window summary helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_NULL_PTR = 1
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3
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

    if rank <= 0:
        rank = 1
    if rank > len(rows):
        rank = len(rows)

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


def gpu_security_perf_matrix_summary_q16_checked_latency_window_checked(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    simulate_null_output: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    if rows is None or simulate_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, (0, 0, 0, 0)
    if rows_capacity < len(rows) or len(rows) <= 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (0, 0, 0, 0)
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (0, 0, 0, 0)

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
        return baseline[0], (0, 0, 0, 0)

    p05_status, p05_q16 = _select_percentile_nearest_rank_q16_checked(rows, 5, 100)
    if p05_status != GPU_SEC_PERF_OK:
        return p05_status, (0, 0, 0, 0)

    p25_status, p25_q16 = _select_percentile_nearest_rank_q16_checked(rows, 25, 100)
    if p25_status != GPU_SEC_PERF_OK:
        return p25_status, (0, 0, 0, 0)

    p75_status, p75_q16 = _select_percentile_nearest_rank_q16_checked(rows, 75, 100)
    if p75_status != GPU_SEC_PERF_OK:
        return p75_status, (0, 0, 0, 0)

    p99_status, p99_q16 = _select_percentile_nearest_rank_q16_checked(rows, 99, 100)
    if p99_status != GPU_SEC_PERF_OK:
        return p99_status, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, (p05_q16, p25_q16, p75_q16, p99_q16)


def test_source_contains_iq1461_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowChecked(" in src
    assert "status_baseline" in src
    assert "staged_p05_q16" in src
    assert "99," in src


def test_latency_window_secure_on_success_with_duplicate_latencies() -> None:
    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(101_000, 2_100, 10_000),
        RowOutput(102_000, 2_200, 30_000),
        RowOutput(103_000, 2_300, 30_000),
        RowOutput(104_000, 2_400, 50_000),
    ]

    status, out_tuple = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked(
        rows,
        rows_capacity=5,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )

    assert status == GPU_SEC_PERF_OK
    assert out_tuple == (10_000, 10_000, 30_000, 50_000)


def test_latency_window_fail_closed_policy_and_bad_inputs() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, _ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, _ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=0,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, _ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked(
        None,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_NULL_PTR


def test_latency_window_rejects_negative_latency_samples() -> None:
    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(100_000, 2_000, -1),
    ]

    status, _ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked(
        rows,
        rows_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )

    assert status == GPU_SEC_PERF_ERR_BAD_PARAM


if __name__ == "__main__":
    test_source_contains_iq1461_symbols()
    test_latency_window_secure_on_success_with_duplicate_latencies()
    test_latency_window_fail_closed_policy_and_bad_inputs()
    test_latency_window_rejects_negative_latency_samples()
    print("ok")
