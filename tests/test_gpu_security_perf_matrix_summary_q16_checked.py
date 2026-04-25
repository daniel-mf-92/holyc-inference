#!/usr/bin/env python3
"""Harness for IQ-1446 secure-on GPU matrix summary aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_NULL_PTR = 1
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3
GPU_SEC_PERF_ERR_OVERFLOW = 5

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = 1
GPU_SEC_PERF_PROFILE_DEV_LOCAL = 2

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
    if secure_local_mode not in (GPU_SEC_PERF_PROFILE_SECURE_LOCAL, GPU_SEC_PERF_PROFILE_DEV_LOCAL):
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
        raise ValueError("bad percentile")
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
    latency_values: list[int] = []
    for row in rows:
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
        tok_total_next = _add_checked(tok_total, row.tok_per_sec_q16)
        overhead_total_next = _add_checked(overhead_total, row.audit_overhead_delta_q16)
        if tok_total_next is None or overhead_total_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, 0
        tok_total = tok_total_next
        overhead_total = overhead_total_next
        latency_values.append(row.secure_cycles_per_token_q16)

    row_count_q16 = _mul_checked(len(rows), GPU_SEC_PERF_Q16_ONE)
    if row_count_q16 is None or row_count_q16 <= 0:
        return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, 0
    overhead_total_q16 = _mul_checked(overhead_total, GPU_SEC_PERF_Q16_ONE)
    if overhead_total_q16 is None:
        return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, 0

    p50_q16 = _select_percentile_nearest_rank(latency_values, 50, 100)
    p95_q16 = _select_percentile_nearest_rank(latency_values, 95, 100)
    audit_overhead_q16 = overhead_total_q16 // row_count_q16
    return GPU_SEC_PERF_OK, tok_total, audit_overhead_q16, p50_q16, p95_q16


def test_source_contains_iq1446_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16Checked(" in src
    assert "GPUSecurityPerfMatrixSelectPercentileNearestRankQ16Checked" in src
    assert "GPUPolicyAllowDispatchChecked" in src
    assert "out_p50_q16" in src
    assert "out_p95_q16" in src


def test_secure_on_summary_aggregates_totals_and_percentiles() -> None:
    rows = [
        RowOutput(100_000, 3000, 10_000),
        RowOutput(120_000, 2000, 20_000),
        RowOutput(110_000, 1000, 30_000),
        RowOutput(130_000, 4000, 40_000),
        RowOutput(140_000, 3000, 50_000),
    ]
    status, tok_total, overhead_q16, p50_q16, p95_q16 = gpu_security_perf_matrix_summary_q16_checked(
        rows,
        rows_capacity=5,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )
    assert status == GPU_SEC_PERF_OK
    assert tok_total == sum(r.tok_per_sec_q16 for r in rows)
    expected_overhead = (sum(r.audit_overhead_delta_q16 for r in rows) * GPU_SEC_PERF_Q16_ONE) // (
        len(rows) * GPU_SEC_PERF_Q16_ONE
    )
    assert overhead_q16 == expected_overhead
    assert p50_q16 == 30_000
    assert p95_q16 == 50_000


def test_fail_closed_policy_vectors() -> None:
    rows = [RowOutput(10_000, 500, 1000)]

    status, *_ = gpu_security_perf_matrix_summary_q16_checked(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_DEV_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, *_ = gpu_security_perf_matrix_summary_q16_checked(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=0,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD


def test_bad_param_and_overflow_vectors() -> None:
    bad_rows = [RowOutput(-1, 100, 100)]
    status, *_ = gpu_security_perf_matrix_summary_q16_checked(
        bad_rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM

    overflow_rows = [
        RowOutput(GPU_SEC_PERF_I64_MAX, GPU_SEC_PERF_I64_MAX, 100),
        RowOutput(1, 1, 200),
    ]
    status, *_ = gpu_security_perf_matrix_summary_q16_checked(
        overflow_rows,
        rows_capacity=2,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )
    assert status == GPU_SEC_PERF_ERR_OVERFLOW


if __name__ == "__main__":
    test_source_contains_iq1446_symbols()
    test_secure_on_summary_aggregates_totals_and_percentiles()
    test_fail_closed_policy_vectors()
    test_bad_param_and_overflow_vectors()
    print("ok")
