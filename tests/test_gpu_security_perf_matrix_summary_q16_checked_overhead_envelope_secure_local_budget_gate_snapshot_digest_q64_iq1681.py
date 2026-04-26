#!/usr/bin/env python3
"""Harness for IQ-1681 secure-local budget-gate snapshot digest guard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = 1

GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK = 0
GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS = 1
GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH = 2


@dataclass(frozen=True)
class RowOutput:
    audit_overhead_delta_q16: int


def _status_is_valid(status_code: int) -> bool:
    return status_code in {0, 1, 2, 3, 4, 5}


def _budget_gate_status_is_valid(status_code: int) -> bool:
    return status_code in {
        GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH,
    }


def _rows_digest(rows: list[RowOutput]) -> int:
    digest = 1469598103934665603
    stride = 104729
    for i, row in enumerate(rows):
        if row.audit_overhead_delta_q16 < 0:
            return -1
        digest += (row.audit_overhead_delta_q16 + i + 1) * stride
        stride += 104729
    return digest


def _nearest_rank(values: list[int], percentile_num: int, percentile_den: int) -> int:
    rank = (percentile_num * len(values) + percentile_den - 1) // percentile_den
    rank = max(1, min(rank, len(values)))
    return sorted(values)[rank - 1]


def _canonical_budget_gate(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
) -> tuple[int, tuple[int, int, int]]:
    out = (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, out
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, out
    if max_p50_overhead_q16 < 0 or max_p95_overhead_q16 < 0 or max_p95_overhead_q16 < max_p50_overhead_q16:
        return GPU_SEC_PERF_ERR_BAD_PARAM, out

    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, out
    if iommu_active == 0 or book_of_truth_gpu_hooks == 0 or policy_digest_parity == 0:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, out

    values = [r.audit_overhead_delta_q16 for r in rows]
    if any(v < 0 for v in values):
        return GPU_SEC_PERF_ERR_BAD_PARAM, out

    p50 = _nearest_rank(values, 50, 100)
    p95 = _nearest_rank(values, 95, 100)
    gate = GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS
    if p50 > max_p50_overhead_q16 or p95 > max_p95_overhead_q16:
        gate = GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH

    if gate != GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (p50, p95, gate)
    return GPU_SEC_PERF_OK, (p50, p95, gate)


def _tuple_digest(values: list[int]) -> int:
    digest = 1469598103934665603
    stride = 104729
    for i, value in enumerate(values):
        digest += (value + i + 1) * stride
        stride += 104729
    return digest


def _iq1681_snapshot_digest_guard(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    force_mutation_tamper: bool = False,
    force_replay_tamper: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    out = (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)
    digest_before = _rows_digest(rows)
    if digest_before < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, out

    status, (p50, p95, gate) = _canonical_budget_gate(
        rows,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
    )

    digest_after = _rows_digest(rows)
    if digest_after != digest_before:
        return GPU_SEC_PERF_ERR_BAD_PARAM, out

    if not _status_is_valid(status):
        return GPU_SEC_PERF_ERR_BAD_PARAM, out
    if not _budget_gate_status_is_valid(gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, out

    if status != GPU_SEC_PERF_OK:
        return status, (p50, p95, gate, 0)

    primary_tuple = [
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        p50,
        p95,
        gate,
    ]
    parity_tuple = list(primary_tuple)
    if force_mutation_tamper:
        parity_tuple[7] += 1

    digest_primary = _tuple_digest(primary_tuple)
    digest_parity = _tuple_digest(parity_tuple)
    if force_replay_tamper:
        digest_parity += 1

    if digest_primary != digest_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, out

    return GPU_SEC_PERF_OK, (p50, p95, gate, digest_primary)


def test_source_contains_iq1681_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked(" in src
    assert "status_primary = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGate(" in src
    assert "status_row_snapshot = GPUSecurityPerfMatrixRowOutputsSnapshotDigestQ64(" in src
    assert "if (digest_primary != digest_parity)" in src


def test_gate_missing_vector() -> None:
    status, out = _iq1681_snapshot_digest_guard(
        rows=[RowOutput(10), RowOutput(12), RowOutput(14), RowOutput(16), RowOutput(18)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert out == (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)


def test_mutation_replay_tamper_vectors_fail_closed() -> None:
    status_mutation, out_mutation = _iq1681_snapshot_digest_guard(
        rows=[RowOutput(11), RowOutput(13), RowOutput(15), RowOutput(17), RowOutput(19)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=128,
        max_p95_overhead_q16=128,
        force_mutation_tamper=True,
    )
    assert status_mutation == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_mutation == (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)

    status_replay, out_replay = _iq1681_snapshot_digest_guard(
        rows=[RowOutput(11), RowOutput(13), RowOutput(15), RowOutput(17), RowOutput(19)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=128,
        max_p95_overhead_q16=128,
        force_replay_tamper=True,
    )
    assert status_replay == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_replay == (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)


def test_deterministic_secure_on_parity_vector() -> None:
    rows = [RowOutput(7), RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15)]
    first = _iq1681_snapshot_digest_guard(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    second = _iq1681_snapshot_digest_guard(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )

    assert first[0] == GPU_SEC_PERF_OK
    assert first[1][0:3] == (11, 15, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS)
    assert first[1][3] > 0
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1681_symbols()
    test_gate_missing_vector()
    test_mutation_replay_tamper_vectors_fail_closed()
    test_deterministic_secure_on_parity_vector()
    print("ok")
