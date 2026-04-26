#!/usr/bin/env python3
"""Harness for IQ-1683 snapshot-digest commit-only preflight-only wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BASE_IQ1681_PATH = Path(
    "tests/test_gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_secure_local_budget_gate_snapshot_digest_q64_iq1681.py"
)
_BASE_IQ1681_SPEC = importlib.util.spec_from_file_location("iq1681_models", _BASE_IQ1681_PATH)
assert _BASE_IQ1681_SPEC is not None and _BASE_IQ1681_SPEC.loader is not None
_BASE_IQ1681_MOD = importlib.util.module_from_spec(_BASE_IQ1681_SPEC)
sys.modules[_BASE_IQ1681_SPEC.name] = _BASE_IQ1681_MOD
_BASE_IQ1681_SPEC.loader.exec_module(_BASE_IQ1681_MOD)

_BASE_IQ1682_PATH = Path(
    "tests/test_gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_secure_local_budget_gate_snapshot_digest_q64_iq1682.py"
)
_BASE_IQ1682_SPEC = importlib.util.spec_from_file_location("iq1682_models", _BASE_IQ1682_PATH)
assert _BASE_IQ1682_SPEC is not None and _BASE_IQ1682_SPEC.loader is not None
_BASE_IQ1682_MOD = importlib.util.module_from_spec(_BASE_IQ1682_SPEC)
sys.modules[_BASE_IQ1682_SPEC.name] = _BASE_IQ1682_MOD
_BASE_IQ1682_SPEC.loader.exec_module(_BASE_IQ1682_MOD)

GPU_SEC_PERF_OK = _BASE_IQ1681_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_IQ1681_MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = _BASE_IQ1681_MOD.GPU_SEC_PERF_ERR_POLICY_GUARD

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_IQ1681_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK = _BASE_IQ1681_MOD.GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK
GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS = _BASE_IQ1681_MOD.GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS

RowOutput = _BASE_IQ1681_MOD.RowOutput


def _status_is_valid(status_code: int) -> bool:
    return status_code in {0, 1, 2, 3, 4, 5}


def _budget_gate_status_is_valid(status_code: int) -> bool:
    return status_code in {
        _BASE_IQ1681_MOD.GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        _BASE_IQ1681_MOD.GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS,
        _BASE_IQ1681_MOD.GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH,
    }


def matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_iq1683(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    initial_out: tuple[int, int, int, int] = (111, 222, 7, 999),
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
    force_no_write_drift: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    out_p50, out_p95, out_gate, out_digest = initial_out
    saved_out = (out_p50, out_p95, out_gate, out_digest)

    status_preflight, staged_p50, staged_p95, staged_gate, staged_digest_q64 = (
        _BASE_IQ1682_MOD.matrix_summary_snapshot_digest_q64_checked_commit_only_iq1682(
            rows=rows,
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
        )
    )

    status_canonical, canonical_out = _BASE_IQ1681_MOD._iq1681_snapshot_digest_guard(
        rows=rows,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        max_p50_overhead_q16=max_p50_overhead_q16,
        max_p95_overhead_q16=max_p95_overhead_q16,
    )
    canonical_p50, canonical_p95, canonical_gate, canonical_digest_q64 = canonical_out

    if force_status_domain_drift:
        status_canonical = 99
    if force_digest_drift:
        canonical_digest_q64 += 1
    if force_no_write_drift:
        out_digest += 1

    if not _status_is_valid(status_preflight) or not _status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _budget_gate_status_is_valid(staged_gate) or not _budget_gate_status_is_valid(canonical_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if min(staged_p50, staged_p95, canonical_p50, canonical_p95) < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_preflight == GPU_SEC_PERF_OK and (staged_digest_q64 <= 0 or canonical_digest_q64 <= 0):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_preflight != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_p50 != canonical_p50 or staged_p95 != canonical_p95:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_gate != canonical_gate:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_digest_q64 != canonical_digest_q64:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if (out_p50, out_p95, out_gate, out_digest) != saved_out:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    return status_preflight, saved_out


def test_source_contains_iq1683_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert (
        "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnly("
        in src
    )
    assert (
        "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64CheckedCommitOnly("
        in src
    )
    assert (
        "status_canonical = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked("
        in src
    )
    assert "if (staged_snapshot_digest_q64 != canonical_snapshot_digest_q64)" in src


def test_gate_missing_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_iq1683(
        rows=[RowOutput(8), RowOutput(10), RowOutput(12), RowOutput(14), RowOutput(16)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert out_values == (111, 222, 7, 999)


def test_digest_drift_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_iq1683(
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
    assert out_values == (111, 222, 7, 999)


def test_status_domain_drift_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_iq1683(
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
    assert out_values == (111, 222, 7, 999)


def test_no_write_parity_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_iq1683(
        rows=[RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15), RowOutput(17)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
        force_no_write_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (111, 222, 7, 999)


def test_deterministic_secure_on_vector() -> None:
    rows = [RowOutput(7), RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15)]

    first = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_iq1683(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    second = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_iq1683(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )

    assert first[0] == GPU_SEC_PERF_OK
    assert first[1] == (111, 222, 7, 999)
    assert second == first


def test_underlying_tuple_reference_remains_consistent() -> None:
    status, p50, p95, gate, digest_q64 = _BASE_IQ1682_MOD.matrix_summary_snapshot_digest_q64_checked_commit_only_iq1682(
        rows=[RowOutput(7), RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    assert (status, p50, p95, gate) == (
        GPU_SEC_PERF_OK,
        11,
        15,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS,
    )
    assert digest_q64 > 0


if __name__ == "__main__":
    test_source_contains_iq1683_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vector()
    test_deterministic_secure_on_vector()
    test_underlying_tuple_reference_remains_consistent()
    print("ok")
