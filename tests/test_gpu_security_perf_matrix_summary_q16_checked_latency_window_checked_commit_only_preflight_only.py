#!/usr/bin/env python3
"""Harness for IQ-1489 latency-window commit-only preflight-only wrapper."""

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


@dataclass
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
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0
    if rows_capacity < len(rows):
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


def _select_percentile_nearest_rank_q16_checked(rows: list[RowOutput], percentile_num: int, percentile_den: int) -> tuple[int, int]:
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
) -> int:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD
    if iommu_enabled != 1 or bot_dma_log_enabled != 1 or bot_mmio_log_enabled != 1 or bot_dispatch_log_enabled != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD

    tok_total = 0
    overhead_total = 0
    for row in rows:
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM
        tok_total_next = _add_checked(tok_total, row.tok_per_sec_q16)
        overhead_total_next = _add_checked(overhead_total, row.audit_overhead_delta_q16)
        if tok_total_next is None or overhead_total_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW
        tok_total = tok_total_next
        overhead_total = overhead_total_next

    row_count_q16 = _mul_checked(len(rows), GPU_SEC_PERF_Q16_ONE)
    overhead_total_q16 = _mul_checked(overhead_total, GPU_SEC_PERF_Q16_ONE)
    if row_count_q16 in (None, 0) or overhead_total_q16 is None:
        return GPU_SEC_PERF_ERR_OVERFLOW
    return GPU_SEC_PERF_OK


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

    baseline_status = _summary_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
    )
    if baseline_status != GPU_SEC_PERF_OK:
        return baseline_status, 0, 0, 0, 0

    p05_status, p05_q16 = _select_percentile_nearest_rank_q16_checked(rows, 5, 100)
    p25_status, p25_q16 = _select_percentile_nearest_rank_q16_checked(rows, 25, 100)
    p75_status, p75_q16 = _select_percentile_nearest_rank_q16_checked(rows, 75, 100)
    p99_status, p99_q16 = _select_percentile_nearest_rank_q16_checked(rows, 99, 100)
    status = p05_status or p25_status or p75_status or p99_status
    if status != GPU_SEC_PERF_OK:
        return status, 0, 0, 0, 0

    return GPU_SEC_PERF_OK, p05_q16, p25_q16, p75_q16, p99_q16


def _latency_window_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, int, int, int, int]:
    if out_capacity < 4:
        return GPU_SEC_PERF_ERR_CAPACITY, 0, 0, 0, 0

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
    return first


def gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int, int],
    force_alias_outputs: bool = False,
    inject_tuple_drift: bool = False,
    inject_row_mutation: bool = False,
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    if rows is None:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0, 0)
    if force_alias_outputs:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if rows_capacity < len(rows) or len(rows) <= 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if out_capacity < 4:
        return GPU_SEC_PERF_ERR_CAPACITY, caller_outputs, (0, 0, 0, 0)
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, caller_outputs, (0, 0, 0, 0)
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, caller_outputs, (0, 0, 0, 0)

    digest_status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if digest_status_before != GPU_SEC_PERF_OK:
        return digest_status_before, caller_outputs, (0, 0, 0, 0)

    staged = _latency_window_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if staged[0] != GPU_SEC_PERF_OK:
        return staged[0], caller_outputs, (0, 0, 0, 0)

    canonical = _latency_window_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if canonical[0] != GPU_SEC_PERF_OK:
        return canonical[0], caller_outputs, (0, 0, 0, 0)

    if inject_tuple_drift:
        staged = (staged[0], staged[1], staged[2], staged[3], staged[4] + 1)
    if inject_row_mutation:
        rows[0].secure_cycles_per_token_q16 += 1

    digest_status_after, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if digest_status_after != GPU_SEC_PERF_OK or digest_after != digest_before:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    if staged[1:] != canonical[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, caller_outputs, canonical[1:]


def test_source_contains_iq1489_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnlyPreflightOnly(" in src
    assert "saved_p05_q16" in src
    assert "status_commit_only = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnly(" in src
    assert "status_canonical = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowChecked(" in src
    assert "snapshot_after_digest_q64" in src


def test_duplicate_latency_vectors_preserve_outputs_and_return_diag_tuple() -> None:
    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(101_000, 2_100, 10_000),
        RowOutput(102_000, 2_200, 20_000),
        RowOutput(103_000, 2_300, 40_000),
        RowOutput(104_000, 2_400, 40_000),
        RowOutput(105_000, 2_500, 80_000),
    ]
    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only(
        rows,
        rows_capacity=6,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(111, 222, 333, 444),
    )
    assert status == GPU_SEC_PERF_OK
    assert outputs_after == (111, 222, 333, 444)
    assert diag == (10_000, 10_000, 40_000, 80_000)


def test_null_alias_capacity_and_gate_missing_vectors() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only(
        None,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3, 4),
    )
    assert (status, outputs_after, diag) == (GPU_SEC_PERF_ERR_NULL_PTR, (1, 2, 3, 4), (0, 0, 0, 0))

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3, 4),
        force_alias_outputs=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3, 4),
    )
    assert status == GPU_SEC_PERF_ERR_CAPACITY

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3, 4),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD


def test_overflow_and_parity_drift_vectors() -> None:
    overflow_rows = [
        RowOutput(GPU_SEC_PERF_I64_MAX, 2_000, 10_000),
        RowOutput(1, 2_000, 20_000),
    ]
    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only(
        overflow_rows,
        rows_capacity=2,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(9, 8, 7, 6),
    )
    assert (status, outputs_after, diag) == (GPU_SEC_PERF_ERR_OVERFLOW, (9, 8, 7, 6), (0, 0, 0, 0))

    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(100_001, 2_000, 20_000),
        RowOutput(100_002, 2_000, 30_000),
    ]
    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only(
        rows,
        rows_capacity=3,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(5, 6, 7, 8),
        inject_tuple_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM

    rows_for_digest_drift = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(100_001, 2_000, 20_000),
        RowOutput(100_002, 2_000, 30_000),
    ]
    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only(
        rows_for_digest_drift,
        rows_capacity=3,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(5, 6, 7, 8),
        inject_row_mutation=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM


if __name__ == "__main__":
    test_source_contains_iq1489_symbols()
    test_duplicate_latency_vectors_preserve_outputs_and_return_diag_tuple()
    test_null_alias_capacity_and_gate_missing_vectors()
    test_overflow_and_parity_drift_vectors()
    print("ok")
