#!/usr/bin/env python3
"""Harness for IQ-1721 snapshot-digest commit-only hardening wrapper."""

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

_BASE_IQ1720_PATH = Path(
    "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1720.py"
)
_BASE_IQ1720_SPEC = importlib.util.spec_from_file_location("iq1720_models", _BASE_IQ1720_PATH)
assert _BASE_IQ1720_SPEC is not None and _BASE_IQ1720_SPEC.loader is not None
_BASE_IQ1720_MOD = importlib.util.module_from_spec(_BASE_IQ1720_SPEC)
sys.modules[_BASE_IQ1720_SPEC.name] = _BASE_IQ1720_MOD
_BASE_IQ1720_SPEC.loader.exec_module(_BASE_IQ1720_MOD)

_BASE_IQ1719_PATH = Path(
    "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1719.py"
)
_BASE_IQ1719_SPEC = importlib.util.spec_from_file_location("iq1719_models", _BASE_IQ1719_PATH)
assert _BASE_IQ1719_SPEC is not None and _BASE_IQ1719_SPEC.loader is not None
_BASE_IQ1719_MOD = importlib.util.module_from_spec(_BASE_IQ1719_SPEC)
sys.modules[_BASE_IQ1719_SPEC.name] = _BASE_IQ1719_MOD
_BASE_IQ1719_SPEC.loader.exec_module(_BASE_IQ1719_MOD)

GPU_SEC_PERF_OK = _BASE_IQ1702_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_IQ1702_MOD.GPU_SEC_PERF_ERR_BAD_PARAM

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_IQ1702_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD = 1


def fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_iq1721(
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
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
) -> tuple[int, int, int, int]:
    status_primary, staged_out = (
        _BASE_IQ1720_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1720(
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            dispatch_transcript_parity=dispatch_transcript_parity,
            p50_overhead_q16=p50_overhead_q16,
            p95_overhead_q16=p95_overhead_q16,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
            initial_out=(0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0),
        )
    )
    staged_enabled, staged_reason, staged_digest_q64 = staged_out

    status_parity, parity_out = (
        _BASE_IQ1719_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1719(
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
    parity_enabled, parity_reason, parity_digest_q64 = parity_out

    if force_status_domain_drift:
        status_parity = 99
    if force_digest_drift:
        parity_digest_q64 += 1

    if not _BASE_IQ1702_MOD._status_is_valid(status_primary) or not _BASE_IQ1702_MOD._status_is_valid(
        status_parity
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0
    if not _BASE_IQ1702_MOD._flag_is_binary(staged_enabled) or not _BASE_IQ1702_MOD._flag_is_binary(
        parity_enabled
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0
    if not _BASE_IQ1702_MOD._disable_reason_is_valid(staged_reason) or not _BASE_IQ1702_MOD._disable_reason_is_valid(
        parity_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0

    if status_primary == GPU_SEC_PERF_OK and (staged_digest_q64 <= 0 or parity_digest_q64 <= 0):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0

    if status_primary != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0
    if staged_enabled != parity_enabled or staged_reason != parity_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0
    if staged_digest_q64 != parity_digest_q64:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0

    return status_primary, staged_enabled, staged_reason, staged_digest_q64


def test_source_contains_iq1721_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert (
        "I32 GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
        in src
    )
    assert (
        "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
        in src
    )
    assert (
        "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
        in src
    )
    assert "if (staged_snapshot_digest_q64 != parity_snapshot_digest_q64)" in src
    assert "// IQ-1721 commit-only hardening wrapper:" in src


def test_gate_missing_vector() -> None:
    status, enabled, reason, digest_q64 = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_iq1721(
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
    assert (status, enabled, reason, digest_q64) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        0,
    )


def test_digest_drift_vector() -> None:
    status, enabled, reason, digest_q64 = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_iq1721(
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
    assert (status, enabled, reason, digest_q64) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        0,
    )


def test_status_domain_drift_vector() -> None:
    status, enabled, reason, digest_q64 = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_iq1721(
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
    assert (status, enabled, reason, digest_q64) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        0,
    )


def test_deterministic_tuple_parity_vector() -> None:
    first = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_iq1721(
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
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_iq1721(
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

    assert first == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        0,
    )
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1721_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
