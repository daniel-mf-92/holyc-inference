#!/usr/bin/env python3
"""Harness for IQ-1504 latency-window no-partial preflight-only wrapper."""

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
    if iommu_active != 1 or book_of_truth_gpu_hooks != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0

    for row in rows:
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0

    p05_status, p05_q16 = _select_percentile_nearest_rank_q16_checked(rows, 5, 100)
    p25_status, p25_q16 = _select_percentile_nearest_rank_q16_checked(rows, 25, 100)
    p75_status, p75_q16 = _select_percentile_nearest_rank_q16_checked(rows, 75, 100)
    p99_status, p99_q16 = _select_percentile_nearest_rank_q16_checked(rows, 99, 100)
    status = p05_status or p25_status or p75_status or p99_status
    if status != GPU_SEC_PERF_OK:
        return status, 0, 0, 0, 0

    return GPU_SEC_PERF_OK, p05_q16, p25_q16, p75_q16, p99_q16


def _latency_window_nopartial(
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


def gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int, int],
    has_null_output: bool = False,
    outputs_alias: bool = False,
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    if rows is None or has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0, 0)
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, caller_outputs, (0, 0, 0, 0)

    staged = _latency_window_nopartial(
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

    status_after, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_after != GPU_SEC_PERF_OK or digest_before != digest_after:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    if staged[1:] != canonical[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, caller_outputs, canonical[1:]


def test_source_contains_iq1504_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartialPreflightOnly(" in src
    assert "status_nopartial" in src
    assert "status_canonical" in src
    assert "snapshot_after_digest_q64" in src


def test_preflight_only_preserves_outputs_and_matches_canonical_on_duplicate_latencies() -> None:
    rows = [
        RowOutput(100_000, 1_000, 10_000),
        RowOutput(99_000, 1_100, 10_000),
        RowOutput(98_000, 1_200, 30_000),
        RowOutput(97_000, 1_300, 30_000),
        RowOutput(96_000, 1_400, 50_000),
    ]

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only(
        rows,
        rows_capacity=5,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(11, 22, 33, 44),
    )

    assert status == GPU_SEC_PERF_OK
    assert outputs_after == (11, 22, 33, 44)
    assert diag_tuple == (10_000, 10_000, 30_000, 50_000)


def test_preflight_only_fail_closed_null_alias_capacity_and_policy_vectors() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only(
        None,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3, 4),
    )
    assert status == GPU_SEC_PERF_ERR_NULL_PTR
    assert outputs_after == (1, 2, 3, 4)
    assert diag_tuple == (0, 0, 0, 0)

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3, 4),
        outputs_alias=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs_after == (1, 2, 3, 4)
    assert diag_tuple == (0, 0, 0, 0)

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(5, 6, 7, 8),
    )
    assert status == GPU_SEC_PERF_ERR_CAPACITY
    assert outputs_after == (5, 6, 7, 8)
    assert diag_tuple == (0, 0, 0, 0)

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(9, 10, 11, 12),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs_after == (9, 10, 11, 12)
    assert diag_tuple == (0, 0, 0, 0)


def test_preflight_only_overflow_parity_vector_preserves_outputs() -> None:
    rows = [RowOutput(GPU_SEC_PERF_I64_MAX, 1, 1)]

    status, outputs_after, diag_tuple = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(13, 14, 15, 16),
    )

    assert status == GPU_SEC_PERF_ERR_OVERFLOW
    assert outputs_after == (13, 14, 15, 16)
    assert diag_tuple == (0, 0, 0, 0)
