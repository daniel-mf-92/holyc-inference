#!/usr/bin/env python3
"""Harness for IQ-1490 latency-window commit-only preflight-only parity gate."""

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


def _latency_window_checked(rows: list[RowOutput], rows_capacity: int) -> tuple[int, int, int, int, int]:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    ordered = sorted(row.secure_cycles_per_token_q16 for row in rows)
    if ordered[0] < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    return (
        GPU_SEC_PERF_OK,
        ordered[((5 * len(rows) + 99) // 100) - 1],
        ordered[((25 * len(rows) + 99) // 100) - 1],
        ordered[((75 * len(rows) + 99) // 100) - 1],
        ordered[((99 * len(rows) + 99) // 100) - 1],
    )


def _latency_window_commit_only(
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
    if out_capacity < 4:
        return GPU_SEC_PERF_ERR_CAPACITY, 0, 0, 0, 0
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL or policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0
    if iommu_active != 1 or book_of_truth_gpu_hooks != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0

    first = _latency_window_checked(rows, rows_capacity)
    second = _latency_window_checked(rows, rows_capacity)
    if first[0] != second[0] or first[1:] != second[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    return first


def _latency_window_commit_only_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    caller_outputs: tuple[int, int, int, int],
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    before_status, before_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if before_status != GPU_SEC_PERF_OK:
        return before_status, caller_outputs, (0, 0, 0, 0)

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

    canonical = _latency_window_checked(rows, rows_capacity)
    if canonical[0] != GPU_SEC_PERF_OK:
        return canonical[0], caller_outputs, (0, 0, 0, 0)

    after_status, after_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if after_status != GPU_SEC_PERF_OK or after_digest != before_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if staged[1:] != canonical[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    return GPU_SEC_PERF_OK, caller_outputs, canonical[1:]


def gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only_parity(
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
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    if rows is None:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs, (0, 0, 0, 0)
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if out_capacity < 4:
        return GPU_SEC_PERF_ERR_CAPACITY, caller_outputs, (0, 0, 0, 0)
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL or policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, caller_outputs, (0, 0, 0, 0)

    before_status, before_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if before_status != GPU_SEC_PERF_OK:
        return before_status, caller_outputs, (0, 0, 0, 0)

    preflight = _latency_window_commit_only_preflight_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        caller_outputs,
    )
    if preflight[0] != GPU_SEC_PERF_OK:
        return preflight[0], caller_outputs, (0, 0, 0, 0)

    parity = _latency_window_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if parity[0] != GPU_SEC_PERF_OK:
        return parity[0], caller_outputs, (0, 0, 0, 0)

    parity_tuple = parity[1:]
    if inject_parity_drift:
        parity_tuple = (parity_tuple[0], parity_tuple[1], parity_tuple[2] + 1, parity_tuple[3])

    after_status, after_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if after_status != GPU_SEC_PERF_OK or after_digest != before_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    if preflight[2] != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, caller_outputs, preflight[2]


def test_source_contains_iq1490_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnlyPreflightOnlyParity(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnlyPreflightOnly(" in src
    assert "status_commit_only = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnly(" in src
    assert "snapshot_after_digest_q64" in src
    assert "saved_p99_q16" in src


def test_parity_ok_zero_write_outputs() -> None:
    rows = [
        RowOutput(100_000, 2_000, 10_000),
        RowOutput(101_000, 2_200, 10_000),
        RowOutput(102_000, 2_300, 30_000),
        RowOutput(103_000, 2_400, 40_000),
        RowOutput(104_000, 2_500, 80_000),
    ]
    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only_parity(
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
    assert outputs_after == (7, 8, 9, 10)
    assert diag == (10_000, 10_000, 40_000, 80_000)


def test_null_alias_capacity_gate_and_parity_drift_vectors() -> None:
    rows = [RowOutput(100_000, 2_000, 10_000)]

    status, outputs_after, diag = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only_parity(
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

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only_parity(
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

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only_parity(
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

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only_parity(
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
    status, *_ = gpu_security_perf_matrix_summary_q16_checked_latency_window_checked_commit_only_preflight_only_parity(
        parity_rows,
        rows_capacity=3,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        caller_outputs=(3, 4, 5, 6),
        inject_parity_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM


if __name__ == "__main__":
    test_source_contains_iq1490_symbols()
    test_parity_ok_zero_write_outputs()
    test_null_alias_capacity_gate_and_parity_drift_vectors()
    print("ok")
