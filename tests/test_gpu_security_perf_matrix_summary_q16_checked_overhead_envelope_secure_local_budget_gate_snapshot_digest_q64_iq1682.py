#!/usr/bin/env python3
"""Harness for IQ-1682 snapshot-digest commit-only wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BASE_PATH = Path(
    "tests/test_gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_secure_local_budget_gate_snapshot_digest_q64_iq1681.py"
)
_BASE_SPEC = importlib.util.spec_from_file_location("iq1681_models", _BASE_PATH)
assert _BASE_SPEC is not None and _BASE_SPEC.loader is not None
_BASE_MOD = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = _BASE_MOD
_BASE_SPEC.loader.exec_module(_BASE_MOD)

GPU_SEC_PERF_OK = _BASE_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_NULL_PTR = 1
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = _BASE_MOD.GPU_SEC_PERF_ERR_POLICY_GUARD
GPU_SEC_PERF_ERR_CAPACITY = 4
GPU_SEC_PERF_ERR_OVERFLOW = 5

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL
GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK = _BASE_MOD.GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK
GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS = _BASE_MOD.GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS
GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH = _BASE_MOD.GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH

RowOutput = _BASE_MOD.RowOutput


def _status_is_valid(status_code: int) -> bool:
    return status_code in {
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ERR_NULL_PTR,
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ERR_CAPACITY,
        GPU_SEC_PERF_ERR_OVERFLOW,
    }


def _budget_gate_status_is_valid(status_code: int) -> bool:
    return status_code in {
        GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH,
    }


def matrix_summary_snapshot_digest_q64_checked_commit_only_iq1682(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
) -> tuple[int, int, int, int, int]:
    status_primary, primary_out = _BASE_MOD._iq1681_snapshot_digest_guard(
        rows=rows,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        max_p50_overhead_q16=max_p50_overhead_q16,
        max_p95_overhead_q16=max_p95_overhead_q16,
    )
    status_parity, parity_out = _BASE_MOD._iq1681_snapshot_digest_guard(
        rows=rows,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        max_p50_overhead_q16=max_p50_overhead_q16,
        max_p95_overhead_q16=max_p95_overhead_q16,
    )

    staged_p50, staged_p95, staged_gate, staged_digest_q64 = primary_out
    parity_p50, parity_p95, parity_gate, parity_digest_q64 = parity_out

    if force_status_domain_drift:
        status_parity = 99
    if force_digest_drift:
        parity_digest_q64 += 1

    if not _status_is_valid(status_primary) or not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0
    if not _budget_gate_status_is_valid(staged_gate) or not _budget_gate_status_is_valid(parity_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0

    if min(staged_p50, staged_p95, parity_p50, parity_p95) < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0

    if status_primary == GPU_SEC_PERF_OK and (staged_digest_q64 <= 0 or parity_digest_q64 <= 0):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0

    if status_primary != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0
    if staged_p50 != parity_p50 or staged_p95 != parity_p95:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0
    if staged_gate != parity_gate:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0
    if staged_digest_q64 != parity_digest_q64:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0

    return status_primary, staged_p50, staged_p95, staged_gate, staged_digest_q64


def test_source_contains_iq1682_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert (
        "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64CheckedCommitOnly("
        in src
    )
    assert (
        "status_primary = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked("
        in src
    )
    assert (
        "status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked("
        in src
    )
    assert "if (staged_snapshot_digest_q64 != parity_snapshot_digest_q64)" in src


def test_gate_missing_vector() -> None:
    status, p50, p95, gate, digest_q64 = matrix_summary_snapshot_digest_q64_checked_commit_only_iq1682(
        rows=[RowOutput(8), RowOutput(10), RowOutput(12), RowOutput(14), RowOutput(16)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    assert (status, p50, p95, gate, digest_q64) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        0,
        0,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        0,
    )


def test_digest_drift_vector() -> None:
    status, p50, p95, gate, digest_q64 = matrix_summary_snapshot_digest_q64_checked_commit_only_iq1682(
        rows=[RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15), RowOutput(17)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
        force_digest_drift=True,
    )
    assert (status, p50, p95, gate, digest_q64) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        0,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        0,
    )


def test_status_domain_drift_vector() -> None:
    status, p50, p95, gate, digest_q64 = matrix_summary_snapshot_digest_q64_checked_commit_only_iq1682(
        rows=[RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15), RowOutput(17)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
        force_status_domain_drift=True,
    )
    assert (status, p50, p95, gate, digest_q64) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        0,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        0,
    )


def test_deterministic_tuple_parity_vector() -> None:
    rows = [RowOutput(7), RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15)]

    first = matrix_summary_snapshot_digest_q64_checked_commit_only_iq1682(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    second = matrix_summary_snapshot_digest_q64_checked_commit_only_iq1682(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )

    assert first[0:4] == (GPU_SEC_PERF_OK, 11, 15, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS)
    assert first[4] > 0
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1682_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
