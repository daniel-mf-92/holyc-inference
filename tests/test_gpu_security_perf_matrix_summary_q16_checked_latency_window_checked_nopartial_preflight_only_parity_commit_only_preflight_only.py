#!/usr/bin/env python3
"""Harness for IQ-1521 latency-window no-partial parity-commit-only preflight-only wrapper."""

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


def _latency_window_nopartial_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int, int],
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
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


def _latency_window_nopartial_preflight_only_parity(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int, int]]:
    preflight = _latency_window_nopartial_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        caller_outputs=(0, 0, 0, 0),
    )
    parity = _latency_window_nopartial(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    if preflight[0] != parity[0]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if preflight[0] != GPU_SEC_PERF_OK:
        return preflight[0], (0, 0, 0, 0)
    if preflight[2] != parity[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, preflight[2]


def _latency_window_nopartial_preflight_only_parity_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int, int]]:
    parity_status, parity_tuple = _latency_window_nopartial_preflight_only_parity(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    preflight = _latency_window_nopartial_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        caller_outputs=parity_tuple,
    )
    preflight_status, preserved_tuple, staged_tuple = preflight

    if parity_status != preflight_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if parity_status != GPU_SEC_PERF_OK:
        return parity_status, (0, 0, 0, 0)
    if preserved_tuple != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if staged_tuple != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, parity_tuple


def gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int, int],
    outputs_alias: bool = False,
    inject_parity_drift: bool = False,
    inject_digest_drift: bool = False,
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    if rows is None:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0, 0)
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    digest_status, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if digest_status != GPU_SEC_PERF_OK:
        return digest_status, caller_outputs, (0, 0, 0, 0)

    commit_status, staged_tuple = _latency_window_nopartial_preflight_only_parity_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    parity_status, canonical_tuple = _latency_window_nopartial_preflight_only_parity(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    digest_status, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if digest_status != GPU_SEC_PERF_OK:
        return digest_status, caller_outputs, (0, 0, 0, 0)
    if inject_digest_drift:
        digest_after += 1
    if digest_after != digest_before:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    if inject_parity_drift:
        staged_tuple = (staged_tuple[0], staged_tuple[1], staged_tuple[2] + 1, staged_tuple[3])

    if commit_status != parity_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if commit_status != GPU_SEC_PERF_OK:
        return commit_status, caller_outputs, (0, 0, 0, 0)
    if staged_tuple != canonical_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, caller_outputs, canonical_tuple


def test_source_contains_iq1521_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartialPreflightOnlyParityCommitOnlyPreflightOnly(" in src
    assert "status_commit_only = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartialPreflightOnlyParityCommitOnly(" in src
    assert "status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartialPreflightOnlyParity(" in src
    assert "status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(" in src
    assert "snapshot_before_digest_q64" in src
    assert "snapshot_after_digest_q64" in src
    assert "saved_p05_q16" in src
    assert "saved_p99_q16" in src


def test_preflight_only_success_preserves_outputs_duplicate_latency_tuple() -> None:
    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(100_100, 2_100, 10_000),
        RowOutput(100_200, 2_200, 30_000),
        RowOutput(100_300, 2_300, 30_000),
        RowOutput(100_400, 2_400, 80_000),
    ]
    status, preserved, staged = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only(
        rows,
        rows_capacity=5,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(7, 8, 9, 10),
    )

    assert status == GPU_SEC_PERF_OK
    assert preserved == (7, 8, 9, 10)
    assert staged == (10_000, 10_000, 30_000, 80_000)


def test_null_alias_capacity_gate_missing_and_parity_drift_vectors() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, preserved, staged = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only(
        None,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3, 4),
    )
    assert (status, preserved, staged) == (GPU_SEC_PERF_ERR_NULL_PTR, (1, 2, 3, 4), (0, 0, 0, 0))

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only(
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

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only(
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

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only(
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

    parity_rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(100_001, 2_000, 20_000),
        RowOutput(100_002, 2_000, 30_000),
    ]
    status, preserved, staged = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only(
        parity_rows,
        rows_capacity=3,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(31, 32, 33, 34),
        inject_parity_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert preserved == (31, 32, 33, 34)
    assert staged == (0, 0, 0, 0)


def test_overflow_vector_returns_bad_param_and_preserves_outputs() -> None:
    rows = [RowOutput(GPU_SEC_PERF_I64_MAX, 1, 1)]

    status, preserved, staged = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only(
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
    assert preserved == (13, 14, 15, 16)
    assert staged == (0, 0, 0, 0)


def test_digest_drift_returns_bad_param_and_preserves_outputs() -> None:
    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(100_100, 2_100, 20_000),
        RowOutput(100_200, 2_200, 30_000),
    ]

    status, preserved, staged = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only(
        rows,
        rows_capacity=4,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(17, 18, 19, 20),
        inject_digest_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert preserved == (17, 18, 19, 20)
    assert staged == (0, 0, 0, 0)
