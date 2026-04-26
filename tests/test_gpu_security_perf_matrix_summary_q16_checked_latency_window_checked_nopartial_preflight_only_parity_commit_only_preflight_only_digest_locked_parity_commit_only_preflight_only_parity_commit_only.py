#!/usr/bin/env python3
"""Harness for IQ-1567 digest-locked latency-window commit-only hardening wrapper."""

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
VALID_STATUS = {
    GPU_SEC_PERF_OK,
    GPU_SEC_PERF_ERR_NULL_PTR,
    GPU_SEC_PERF_ERR_BAD_PARAM,
    GPU_SEC_PERF_ERR_POLICY_GUARD,
    GPU_SEC_PERF_ERR_CAPACITY,
    GPU_SEC_PERF_ERR_OVERFLOW,
}


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
    for idx, row in enumerate(rows):
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0

        factor = idx + 1
        for value, stride in (
            (row.tok_per_sec_q16, factor),
            (row.audit_overhead_delta_q16, factor + len(rows)),
            (row.secure_cycles_per_token_q16, factor + (len(rows) << 1)),
        ):
            term = _mul_checked(value, stride)
            if term is None:
                return GPU_SEC_PERF_ERR_OVERFLOW, 0
            digest = _add_checked(digest, term)
            if digest is None:
                return GPU_SEC_PERF_ERR_OVERFLOW, 0

    return GPU_SEC_PERF_OK, digest


def _select_percentile_nearest_rank(rows: list[RowOutput], pct_num: int, pct_den: int) -> tuple[int, int]:
    if not rows or pct_num <= 0 or pct_den <= 0 or pct_num > pct_den:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    scaled = _mul_checked(pct_num, len(rows))
    if scaled is None:
        return GPU_SEC_PERF_ERR_OVERFLOW, 0

    rank = scaled // pct_den
    if scaled % pct_den:
        nxt = _add_checked(rank, 1)
        if nxt is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0
        rank = nxt

    rank = max(1, min(rank, len(rows)))
    ordered = sorted(r.secure_cycles_per_token_q16 for r in rows)
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
) -> tuple[int, tuple[int, int, int, int]]:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (0, 0, 0, 0)
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if iommu_active != 1 or book_of_truth_gpu_hooks != 1 or policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (0, 0, 0, 0)

    p05_status, p05 = _select_percentile_nearest_rank(rows, 5, 100)
    p25_status, p25 = _select_percentile_nearest_rank(rows, 25, 100)
    p75_status, p75 = _select_percentile_nearest_rank(rows, 75, 100)
    p99_status, p99 = _select_percentile_nearest_rank(rows, 99, 100)
    status = p05_status or p25_status or p75_status or p99_status
    if status != GPU_SEC_PERF_OK:
        return status, (0, 0, 0, 0)
    return GPU_SEC_PERF_OK, (p05, p25, p75, p99)


def _latency_window_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int, int]]:
    if out_capacity < 4:
        return GPU_SEC_PERF_ERR_CAPACITY, (0, 0, 0, 0)

    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, (0, 0, 0, 0)

    status_commit, staged = _latency_window_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    status_canon, canon = _latency_window_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    status_after, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_after != GPU_SEC_PERF_OK or digest_before != digest_after:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if status_commit != status_canon:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if status_commit != GPU_SEC_PERF_OK:
        return status_commit, (0, 0, 0, 0)
    if staged != canon:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    return GPU_SEC_PERF_OK, canon


def _latency_window_parity(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
) -> tuple[int, tuple[int, int, int, int]]:
    status_preflight, staged = _latency_window_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    status_commit, canon = _latency_window_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )

    if status_preflight != status_commit:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if status_preflight != GPU_SEC_PERF_OK:
        return status_preflight, (0, 0, 0, 0)
    if staged != canon:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    return GPU_SEC_PERF_OK, canon


def gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only(
    rows: list[RowOutput] | None,
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int, int],
    outputs_alias: bool = False,
    inject_digest_drift: bool = False,
    inject_invalid_status: bool = False,
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    if rows is None:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0, 0)
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, caller_outputs, (0, 0, 0, 0)

    status_parity, parity_tuple = _latency_window_parity(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    status_preflight_only, staged_tuple = _latency_window_preflight_only(
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
        return status_after, caller_outputs, (0, 0, 0, 0)

    if inject_invalid_status:
        status_parity = 77
    if inject_digest_drift:
        digest_after += 1

    if status_parity not in VALID_STATUS or status_preflight_only not in VALID_STATUS:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if digest_before != digest_after:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if status_parity != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if parity_tuple != staged_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, caller_outputs, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, parity_tuple, parity_tuple


def test_source_contains_iq1567_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartialPreflightOnlyParityCommitOnlyPreflightOnlyDigestLockedParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    assert "status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartialPreflightOnlyParityCommitOnlyPreflightOnlyDigestLockedParityCommitOnlyPreflightOnlyParity(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartialPreflightOnlyParityCommitOnlyPreflightOnlyDigestLockedParityCommitOnlyPreflightOnly(" in src
    assert "snapshot_before_digest_q64" in src
    assert "snapshot_after_digest_q64" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_parity))" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_preflight_only))" in src


def test_duplicate_latency_success_commit_publish() -> None:
    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(100_100, 2_100, 10_000),
        RowOutput(100_200, 2_200, 30_000),
        RowOutput(100_300, 2_300, 30_000),
        RowOutput(100_400, 2_400, 80_000),
    ]

    status, final_outputs, published = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only(
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
    assert final_outputs == (10_000, 10_000, 30_000, 80_000)
    assert published == (10_000, 10_000, 30_000, 80_000)


def test_null_alias_capacity_gate_missing_digest_drift_vectors() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, final_outputs, published = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only(
        None,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(1, 2, 3, 4),
    )
    assert (status, final_outputs, published) == (GPU_SEC_PERF_ERR_NULL_PTR, (1, 2, 3, 4), (0, 0, 0, 0))

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only(
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

    status, final_outputs, published = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(11, 12, 13, 14),
    )
    assert (status, final_outputs, published) == (GPU_SEC_PERF_ERR_CAPACITY, (11, 12, 13, 14), (0, 0, 0, 0))

    for iommu_active, hooks, parity in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        status, final_outputs, published = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only(
            rows,
            rows_capacity=1,
            out_capacity=4,
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=hooks,
            policy_digest_parity=parity,
            caller_outputs=(21, 22, 23, 24),
        )
        assert (status, final_outputs, published) == (GPU_SEC_PERF_ERR_POLICY_GUARD, (21, 22, 23, 24), (0, 0, 0, 0))

    parity_rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(100_001, 2_000, 20_000),
        RowOutput(100_002, 2_000, 30_000),
    ]
    status, final_outputs, published = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only(
        parity_rows,
        rows_capacity=3,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(31, 32, 33, 34),
        inject_digest_drift=True,
    )
    assert (status, final_outputs, published) == (GPU_SEC_PERF_ERR_BAD_PARAM, (31, 32, 33, 34), (0, 0, 0, 0))

    status, final_outputs, published = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_nopartial_preflight_only_parity_commit_only_preflight_only_digest_locked_parity_commit_only_preflight_only_parity_commit_only(
        parity_rows,
        rows_capacity=3,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(41, 42, 43, 44),
        inject_invalid_status=True,
    )
    assert (status, final_outputs, published) == (GPU_SEC_PERF_ERR_BAD_PARAM, (41, 42, 43, 44), (0, 0, 0, 0))


if __name__ == "__main__":
    test_source_contains_iq1567_symbols()
    test_duplicate_latency_success_commit_publish()
    test_null_alias_capacity_gate_missing_digest_drift_vectors()
    print("ok")
