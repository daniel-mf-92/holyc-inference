#!/usr/bin/env python3
"""Harness for IQ-1706 snapshot-digest zero-write diagnostics companion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BASE_IQ1702_PATH = Path(
    "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1702.py"
)
_BASE_IQ1702_SPEC = importlib.util.spec_from_file_location("iq1702_models", _BASE_IQ1702_PATH)
assert _BASE_IQ1702_SPEC is not None and _BASE_IQ1702_SPEC.loader is not None
_BASE_IQ1702_MOD = importlib.util.module_from_spec(_BASE_IQ1702_SPEC)
sys.modules[_BASE_IQ1702_SPEC.name] = _BASE_IQ1702_MOD
_BASE_IQ1702_SPEC.loader.exec_module(_BASE_IQ1702_MOD)

_BASE_IQ1704_PATH = Path(
    "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1704.py"
)
_BASE_IQ1704_SPEC = importlib.util.spec_from_file_location("iq1704_models", _BASE_IQ1704_PATH)
assert _BASE_IQ1704_SPEC is not None and _BASE_IQ1704_SPEC.loader is not None
_BASE_IQ1704_MOD = importlib.util.module_from_spec(_BASE_IQ1704_SPEC)
sys.modules[_BASE_IQ1704_SPEC.name] = _BASE_IQ1704_MOD
_BASE_IQ1704_SPEC.loader.exec_module(_BASE_IQ1704_MOD)

_BASE_IQ1705_PATH = Path(
    "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1705.py"
)
_BASE_IQ1705_SPEC = importlib.util.spec_from_file_location("iq1705_models", _BASE_IQ1705_PATH)
assert _BASE_IQ1705_SPEC is not None and _BASE_IQ1705_SPEC.loader is not None
_BASE_IQ1705_MOD = importlib.util.module_from_spec(_BASE_IQ1705_SPEC)
sys.modules[_BASE_IQ1705_SPEC.name] = _BASE_IQ1705_MOD
_BASE_IQ1705_SPEC.loader.exec_module(_BASE_IQ1705_MOD)

GPU_SEC_PERF_OK = _BASE_IQ1702_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_IQ1702_MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = _BASE_IQ1702_MOD.GPU_SEC_PERF_ERR_POLICY_GUARD

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_IQ1702_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL


def fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1706(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    p50_overhead_q16: int,
    p95_overhead_q16: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    initial_out: tuple[int, int, int] = (313, 11, 1901),
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
    force_no_write_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    out_enabled, out_reason, out_digest = initial_out
    saved_out = (out_enabled, out_reason, out_digest)

    status_commit_only, staged_enabled, staged_reason, staged_digest_q64 = (
        _BASE_IQ1705_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_iq1705(
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            dispatch_transcript_parity=dispatch_transcript_parity,
            p50_overhead_q16=p50_overhead_q16,
            p95_overhead_q16=p95_overhead_q16,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
        )
    )

    status_canonical, canonical_out = (
        _BASE_IQ1704_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1704(
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            dispatch_transcript_parity=dispatch_transcript_parity,
            p50_overhead_q16=p50_overhead_q16,
            p95_overhead_q16=p95_overhead_q16,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
            initial_out=(staged_enabled, staged_reason, staged_digest_q64),
        )
    )
    canonical_enabled, canonical_reason, canonical_digest_q64 = canonical_out

    if force_status_domain_drift:
        status_canonical = 99
    if force_digest_drift:
        canonical_digest_q64 += 1
    if force_no_write_drift:
        out_digest += 1

    if not _BASE_IQ1702_MOD._status_is_valid(status_commit_only) or not _BASE_IQ1702_MOD._status_is_valid(
        status_canonical
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _BASE_IQ1702_MOD._flag_is_binary(staged_enabled) or not _BASE_IQ1702_MOD._flag_is_binary(
        canonical_enabled
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _BASE_IQ1702_MOD._disable_reason_is_valid(staged_reason) or not _BASE_IQ1702_MOD._disable_reason_is_valid(
        canonical_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_commit_only == GPU_SEC_PERF_OK and (staged_digest_q64 <= 0 or canonical_digest_q64 <= 0):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_commit_only != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_enabled != canonical_enabled or staged_reason != canonical_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_digest_q64 != canonical_digest_q64:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if (out_enabled, out_reason, out_digest) != saved_out:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    return status_commit_only, saved_out


def test_source_contains_iq1706_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert (
        "I32 GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
        in src
    )
    assert (
        "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
        in src
    )
    assert (
        "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
        in src
    )
    assert "if (*out_fast_path_enabled != saved_fast_path_enabled ||" in src
    assert "// IQ-1706 zero-write diagnostics companion:" in src


def test_gate_missing_vector() -> None:
    status, out_values = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1706(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=0,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=100,
            p95_overhead_q16=200,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
        )
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert out_values == (313, 11, 1901)


def test_digest_drift_vector() -> None:
    status, out_values = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1706(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=120,
            p95_overhead_q16=240,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
            force_digest_drift=True,
        )
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (313, 11, 1901)


def test_status_domain_drift_vector() -> None:
    status, out_values = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1706(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=120,
            p95_overhead_q16=240,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
            force_status_domain_drift=True,
        )
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (313, 11, 1901)


def test_no_write_parity_vector() -> None:
    status, out_values = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1706(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=120,
            p95_overhead_q16=240,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
            force_no_write_drift=True,
        )
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (313, 11, 1901)


def test_deterministic_secure_on_vector() -> None:
    first = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1706(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=128,
            p95_overhead_q16=256,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
        )
    )
    second = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1706(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=128,
            p95_overhead_q16=256,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
        )
    )

    assert first[0] == GPU_SEC_PERF_OK
    assert first[1] == (313, 11, 1901)
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1706_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vector()
    test_deterministic_secure_on_vector()
    print("ok")
