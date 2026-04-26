#!/usr/bin/env python3
"""Harness for IQ-1693 snapshot-digest commit-only hardening wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BASE_IQ1683_PATH = Path(
    "tests/test_gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_secure_local_budget_gate_snapshot_digest_q64_iq1683.py"
)
_BASE_IQ1683_SPEC = importlib.util.spec_from_file_location("iq1683_models", _BASE_IQ1683_PATH)
assert _BASE_IQ1683_SPEC is not None and _BASE_IQ1683_SPEC.loader is not None
_BASE_IQ1683_MOD = importlib.util.module_from_spec(_BASE_IQ1683_SPEC)
sys.modules[_BASE_IQ1683_SPEC.name] = _BASE_IQ1683_MOD
_BASE_IQ1683_SPEC.loader.exec_module(_BASE_IQ1683_MOD)

_BASE_IQ1692_PATH = Path(
    "tests/test_gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_secure_local_budget_gate_snapshot_digest_q64_iq1692.py"
)
_BASE_IQ1692_SPEC = importlib.util.spec_from_file_location("iq1692_models", _BASE_IQ1692_PATH)
assert _BASE_IQ1692_SPEC is not None and _BASE_IQ1692_SPEC.loader is not None
_BASE_IQ1692_MOD = importlib.util.module_from_spec(_BASE_IQ1692_SPEC)
sys.modules[_BASE_IQ1692_SPEC.name] = _BASE_IQ1692_MOD
_BASE_IQ1692_SPEC.loader.exec_module(_BASE_IQ1692_MOD)

GPU_SEC_PERF_OK = _BASE_IQ1692_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_IQ1692_MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = _BASE_IQ1692_MOD.GPU_SEC_PERF_ERR_POLICY_GUARD

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_IQ1692_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK = _BASE_IQ1692_MOD.GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK
GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS = _BASE_IQ1692_MOD.GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS

RowOutput = _BASE_IQ1692_MOD.RowOutput


def matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_iq1693(
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
) -> tuple[int, tuple[int, int, int, int]]:
    status_parity, _ = (
        _BASE_IQ1692_MOD.matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_iq1692(
            rows=rows,
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
            initial_out=(0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0),
        )
    )
    status_tuple, parity_p50, parity_p95, parity_gate, parity_digest_q64 = (
        _BASE_IQ1683_MOD._BASE_IQ1682_MOD.matrix_summary_snapshot_digest_q64_checked_commit_only_iq1682(
            rows=rows,
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
        )
    )
    if status_parity == GPU_SEC_PERF_OK and status_tuple != GPU_SEC_PERF_OK:
        status_parity = status_tuple

    status_preflight_only, _ = (
        _BASE_IQ1683_MOD.matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_iq1683(
            rows=rows,
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
            initial_out=(parity_p50, parity_p95, parity_gate, parity_digest_q64),
        )
    )

    staged_status, staged_p50, staged_p95, staged_gate, staged_digest_q64 = (
        _BASE_IQ1683_MOD._BASE_IQ1682_MOD.matrix_summary_snapshot_digest_q64_checked_commit_only_iq1682(
            rows=rows,
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
        )
    )
    if status_preflight_only == GPU_SEC_PERF_OK and staged_status != GPU_SEC_PERF_OK:
        status_preflight_only = staged_status

    if force_status_domain_drift:
        status_preflight_only = 99
    if force_digest_drift:
        staged_digest_q64 += 1

    if not _BASE_IQ1683_MOD._status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)
    if not _BASE_IQ1683_MOD._status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)
    if not _BASE_IQ1683_MOD._budget_gate_status_is_valid(parity_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)
    if not _BASE_IQ1683_MOD._budget_gate_status_is_valid(staged_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)

    if min(parity_p50, parity_p95, staged_p50, staged_p95) < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)

    if status_parity == GPU_SEC_PERF_OK and (parity_digest_q64 <= 0 or staged_digest_q64 <= 0):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)

    if status_parity != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)
    if parity_p50 != staged_p50 or parity_p95 != staged_p95:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)
    if parity_gate != staged_gate:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)
    if parity_digest_q64 != staged_digest_q64:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)

    return status_parity, (parity_p50, parity_p95, parity_gate, parity_digest_q64)


def test_source_contains_iq1693_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert (
        "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParityCommitOnly("
        in src
    )
    assert (
        "status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParity("
        in src
    )
    assert (
        "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnly("
        in src
    )
    assert "if (parity_snapshot_digest_q64 != staged_snapshot_digest_q64)" in src


def test_gate_missing_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_iq1693(
        rows=[RowOutput(8), RowOutput(10), RowOutput(12), RowOutput(14), RowOutput(16)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert out_values == (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)


def test_digest_drift_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_iq1693(
        rows=[RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15), RowOutput(17)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
        force_digest_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)


def test_status_domain_drift_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_iq1693(
        rows=[RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15), RowOutput(17)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
        force_status_domain_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK, 0)


def test_deterministic_tuple_parity_vector() -> None:
    rows = [RowOutput(7), RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15)]

    first = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_iq1693(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    second = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_iq1693(
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
    test_source_contains_iq1693_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
