#!/usr/bin/env python3
"""Harness for IQ-1699 snapshot-digest strict diagnostics parity gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BASE_IQ1698_PATH = Path(
    "tests/test_gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_secure_local_budget_gate_snapshot_digest_q64_iq1698.py"
)
_BASE_IQ1698_SPEC = importlib.util.spec_from_file_location("iq1698_models", _BASE_IQ1698_PATH)
assert _BASE_IQ1698_SPEC is not None and _BASE_IQ1698_SPEC.loader is not None
_BASE_IQ1698_MOD = importlib.util.module_from_spec(_BASE_IQ1698_SPEC)
sys.modules[_BASE_IQ1698_SPEC.name] = _BASE_IQ1698_MOD
_BASE_IQ1698_SPEC.loader.exec_module(_BASE_IQ1698_MOD)

_BASE_IQ1697_PATH = Path(
    "tests/test_gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_secure_local_budget_gate_snapshot_digest_q64_iq1697.py"
)
_BASE_IQ1697_SPEC = importlib.util.spec_from_file_location("iq1697_models", _BASE_IQ1697_PATH)
assert _BASE_IQ1697_SPEC is not None and _BASE_IQ1697_SPEC.loader is not None
_BASE_IQ1697_MOD = importlib.util.module_from_spec(_BASE_IQ1697_SPEC)
sys.modules[_BASE_IQ1697_SPEC.name] = _BASE_IQ1697_MOD
_BASE_IQ1697_SPEC.loader.exec_module(_BASE_IQ1697_MOD)

GPU_SEC_PERF_OK = _BASE_IQ1698_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_IQ1698_MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = _BASE_IQ1698_MOD.GPU_SEC_PERF_ERR_POLICY_GUARD

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_IQ1698_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

RowOutput = _BASE_IQ1698_MOD.RowOutput


def matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1699(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    initial_out: tuple[int, int, int, int] = (703, 901, 0, 7777),
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
    force_no_write_drift: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    out_p50, out_p95, out_gate, out_digest = initial_out
    saved_out = (out_p50, out_p95, out_gate, out_digest)

    status_preflight_only, staged_tuple = (
        _BASE_IQ1698_MOD.matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_iq1698(
            rows=rows,
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
            initial_out=(out_p50, out_p95, out_gate, out_digest),
        )
    )
    staged_p50, staged_p95, staged_gate, staged_digest_q64 = staged_tuple

    status_canonical, canonical_tuple = (
        _BASE_IQ1697_MOD.matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_iq1697(
            rows=rows,
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
            initial_out=(staged_p50, staged_p95, staged_gate, staged_digest_q64),
        )
    )
    canonical_p50, canonical_p95, canonical_gate, canonical_digest_q64 = canonical_tuple

    if force_status_domain_drift:
        status_canonical = 99
    if force_digest_drift:
        canonical_digest_q64 += 1
    if force_no_write_drift:
        out_digest += 1

    if not _BASE_IQ1698_MOD._BASE_IQ1692_MOD._BASE_IQ1683_MOD._status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _BASE_IQ1698_MOD._BASE_IQ1692_MOD._BASE_IQ1683_MOD._status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _BASE_IQ1698_MOD._BASE_IQ1692_MOD._BASE_IQ1683_MOD._budget_gate_status_is_valid(staged_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _BASE_IQ1698_MOD._BASE_IQ1692_MOD._BASE_IQ1683_MOD._budget_gate_status_is_valid(canonical_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if min(staged_p50, staged_p95, canonical_p50, canonical_p95) < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_preflight_only == GPU_SEC_PERF_OK and (staged_digest_q64 <= 0 or canonical_digest_q64 <= 0):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_preflight_only != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_p50 != canonical_p50 or staged_p95 != canonical_p95:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_gate != canonical_gate:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_digest_q64 != canonical_digest_q64:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if (out_p50, out_p95, out_gate, out_digest) != saved_out:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    return status_preflight_only, saved_out


def test_source_contains_iq1699_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert (
        "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
        in src
    )
    assert (
        "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
        in src
    )
    assert (
        "status_canonical = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParityCommitOnly("
        in src
    )
    assert "if (staged_snapshot_digest_q64 != canonical_snapshot_digest_q64)" in src
    assert "// IQ-1699 strict diagnostics parity gate:" in src


def test_gate_missing_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1699(
        rows=[RowOutput(8), RowOutput(10), RowOutput(12), RowOutput(14), RowOutput(16)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert out_values == (703, 901, 0, 7777)


def test_digest_drift_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1699(
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
    assert out_values == (703, 901, 0, 7777)


def test_status_domain_drift_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1699(
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
    assert out_values == (703, 901, 0, 7777)


def test_no_write_parity_vector() -> None:
    status, out_values = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1699(
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
    assert out_values == (703, 901, 0, 7777)


def test_deterministic_secure_on_parity_vector() -> None:
    rows = [RowOutput(7), RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15)]

    first = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1699(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    second = matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1699(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )

    assert first[0] == GPU_SEC_PERF_OK
    assert first[1] == (703, 901, 0, 7777)
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1699_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vector()
    test_deterministic_secure_on_parity_vector()
    print("ok")
