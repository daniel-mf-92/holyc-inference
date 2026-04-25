#!/usr/bin/env python3
"""Harness for IQ-1458 secure-on summary no-partial wrapper."""

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

    ordered = sorted(latencies)
    p50_q16 = ordered[((50 * len(rows) + 99) // 100) - 1]
    p95_q16 = ordered[((95 * len(rows) + 99) // 100) - 1]
    return GPU_SEC_PERF_OK, tok_total, overhead_total_q16 // row_count_q16, p50_q16, p95_q16


def _summary_commit_only(
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

    first = _summary_checked(
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

    second = _summary_checked(
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


def gpu_security_perf_matrix_summary_q16_checked_nopartial(
    rows: list[RowOutput],
    rows_capacity: int,
    out_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    alias_outputs: bool = False,
) -> tuple[int, int, int, int, int]:
    if alias_outputs:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0

    digest_status, before_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if digest_status != GPU_SEC_PERF_OK:
        return digest_status, 0, 0, 0, 0

    preflight = _summary_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if preflight[0] != GPU_SEC_PERF_OK:
        return preflight

    digest_status, mid_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if digest_status != GPU_SEC_PERF_OK:
        return digest_status, 0, 0, 0, 0
    if mid_digest != before_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0

    commit = _summary_commit_only(
        rows,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
    )
    if commit[0] != GPU_SEC_PERF_OK:
        return commit

    digest_status, after_digest = _rows_snapshot_digest_q64(rows, rows_capacity)
    if digest_status != GPU_SEC_PERF_OK:
        return digest_status, 0, 0, 0, 0
    if after_digest != before_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0

    if preflight[1:] != commit[1:]:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, 0
    return preflight


def test_source_contains_iq1458_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedNoPartial(" in src
    assert "I32 GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(" in src
    assert "snapshot_before_digest_q64" in src
    assert "snapshot_mid_digest_q64" in src
    assert "snapshot_after_digest_q64" in src


def test_nopartial_secure_on_parity_success() -> None:
    rows = [
        RowOutput(100_000, 1_000, 10_000),
        RowOutput(110_000, 3_000, 20_000),
        RowOutput(120_000, 2_000, 30_000),
    ]
    status, tok_total, overhead_q16, p50_q16, p95_q16 = gpu_security_perf_matrix_summary_q16_checked_nopartial(
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


def test_nopartial_null_alias_capacity_and_gate_vectors() -> None:
    rows = [RowOutput(100_000, 1_000, 10_000)]

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_nopartial(
        rows,
        rows_capacity=1,
        out_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_CAPACITY

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_nopartial(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        alias_outputs=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM

    status, *_ = gpu_security_perf_matrix_summary_q16_checked_nopartial(
        rows,
        rows_capacity=1,
        out_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD


if __name__ == "__main__":
    test_source_contains_iq1458_symbols()
    test_nopartial_secure_on_parity_success()
    test_nopartial_null_alias_capacity_and_gate_vectors()
    print("ok")
