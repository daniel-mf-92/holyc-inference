#!/usr/bin/env python3
"""Harness for IQ-1457 secure-on summary commit-only wrapper."""

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


def _policy_allow_dispatch_checked(
    secure_local_mode: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> bool:
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return False
    if iommu_enabled != 1:
        return False
    if bot_dma_log_enabled != 1:
        return False
    if bot_mmio_log_enabled != 1:
        return False
    if bot_dispatch_log_enabled != 1:
        return False
    return True


def _select_percentile_nearest_rank(values_q16: list[int], percentile_num: int, percentile_den: int) -> int:
    if not values_q16:
        raise ValueError("empty")
    if percentile_num <= 0 or percentile_den <= 0 or percentile_num > percentile_den:
        raise ValueError("bad-percentile")
    if any(v < 0 for v in values_q16):
        raise ValueError("negative")

    scaled = percentile_num * len(values_q16)
    rank = scaled // percentile_den
    if scaled % percentile_den:
        rank += 1
    rank = max(1, min(rank, len(values_q16)))

    ordered = sorted(values_q16)
    return ordered[rank - 1]


def gpu_security_perf_matrix_summary_q16_checked(
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
    if not _policy_allow_dispatch_checked(
        secure_local_mode,
        iommu_enabled,
        bot_dma_log_enabled,
        bot_mmio_log_enabled,
        bot_dispatch_log_enabled,
    ):
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0

    tok_total = 0
    overhead_total = 0
    latencies: list[int] = []
    for row in rows:
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0

        next_tok = _add_checked(tok_total, row.tok_per_sec_q16)
        next_overhead = _add_checked(overhead_total, row.audit_overhead_delta_q16)
        if next_tok is None or next_overhead is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, 0

        tok_total = next_tok
        overhead_total = next_overhead
        latencies.append(row.secure_cycles_per_token_q16)

    row_count_q16 = _mul_checked(len(rows), GPU_SEC_PERF_Q16_ONE)
    overhead_total_q16 = _mul_checked(overhead_total, GPU_SEC_PERF_Q16_ONE)
    if row_count_q16 in (None, 0) or overhead_total_q16 is None:
        return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, 0

    p50_q16 = _select_percentile_nearest_rank(latencies, 50, 100)
    p95_q16 = _select_percentile_nearest_rank(latencies, 95, 100)
    return GPU_SEC_PERF_OK, tok_total, overhead_total_q16 // row_count_q16, p50_q16, p95_q16


def gpu_security_perf_matrix_summary_q16_checked_commit_only(
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
    if iommu_active not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if book_of_truth_gpu_hooks not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, 0

    first = gpu_security_perf_matrix_summary_q16_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
    )
    if first[0] != GPU_SEC_PERF_OK:
        return first

    second = gpu_security_perf_matrix_summary_q16_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
    )
    if second[0] != GPU_SEC_PERF_OK:
        return second

    if first[1:] != second[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    return first


def test_source_contains_iq1457_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedCommitOnly(" in src
    assert "snapshot_row_count" in src
    assert "snapshot_out_capacity" in src
    assert "snapshot_policy_digest_parity" in src
    assert src.count("GPUSecurityPerfMatrixSummaryQ16Checked(") >= 3


def test_commit_only_secure_on_success_and_boundary() -> None:
    rows = [
        RowOutput(100_000, 1_000, 10_000),
        RowOutput(120_000, 3_000, 20_000),
        RowOutput(110_000, 2_000, 30_000),
    ]

    status, tok_total, overhead_q16, p50_q16, p95_q16 = gpu_security_perf_matrix_summary_q16_checked_commit_only(
        rows,
        rows_capacity=3,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_OK
    assert tok_total == 330_000
    assert overhead_q16 == 2_000
    assert p50_q16 == 20_000
    assert p95_q16 == 30_000


def test_commit_only_gate_missing_vectors_fail_closed() -> None:
    rows = [RowOutput(100_000, 1_000, 10_000)]

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=0,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_commit_only(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=0,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD


def test_commit_only_overflow_and_capacity_vectors() -> None:
    overflow_rows = [
        RowOutput(GPU_SEC_PERF_I64_MAX, GPU_SEC_PERF_I64_MAX, 10_000),
        RowOutput(1, 1, 20_000),
    ]

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_commit_only(
        overflow_rows,
        rows_capacity=2,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_OVERFLOW

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_commit_only(
        [RowOutput(10_000, 100, 1_000)],
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_CAPACITY


if __name__ == "__main__":
    test_source_contains_iq1457_symbols()
    test_commit_only_secure_on_success_and_boundary()
    test_commit_only_gate_missing_vectors_fail_closed()
    test_commit_only_overflow_and_capacity_vectors()
    print("ok")
