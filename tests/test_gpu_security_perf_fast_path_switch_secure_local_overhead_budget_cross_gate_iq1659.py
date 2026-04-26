#!/usr/bin/env python3
"""Harness for IQ-1659 zero-write diagnostics companion over IQ-1658 commit-only + IQ-1657 parity."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_DEF_PATH = Path("tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_iq1654.py")
_SPEC = importlib.util.spec_from_file_location("iq1654_models", _DEF_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

_IQ1657_PATH = Path("tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_iq1657.py")
_IQ1657_SPEC = importlib.util.spec_from_file_location("iq1657_models", _IQ1657_PATH)
assert _IQ1657_SPEC is not None and _IQ1657_SPEC.loader is not None
_IQ1657_MOD = importlib.util.module_from_spec(_IQ1657_SPEC)
_IQ1657_SPEC.loader.exec_module(_IQ1657_MOD)

_IQ1658_PATH = Path("tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_iq1658.py")
_IQ1658_SPEC = importlib.util.spec_from_file_location("iq1658_models", _IQ1658_PATH)
assert _IQ1658_SPEC is not None and _IQ1658_SPEC.loader is not None
_IQ1658_MOD = importlib.util.module_from_spec(_IQ1658_SPEC)
_IQ1658_SPEC.loader.exec_module(_IQ1658_MOD)

GPU_SEC_PERF_OK = _MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = _MOD.GPU_SEC_PERF_ERR_POLICY_GUARD
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD = _MOD.GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD


def fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1659(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    p50_overhead_q16: int,
    p95_overhead_q16: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    caller_fast_path_enabled: int,
    caller_disable_reason: int,
    *,
    force_commit_status_domain_drift: bool = False,
    force_canonical_status_domain_drift: bool = False,
) -> tuple[int, int, int]:
    status_commit_only, staged_enabled, staged_reason = (
        _IQ1658_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_iq1658(
            secure_local_mode,
            iommu_active,
            book_of_truth_gpu_hooks,
            policy_digest_parity,
            dispatch_transcript_parity,
            p50_overhead_q16,
            p95_overhead_q16,
            max_p50_overhead_q16,
            max_p95_overhead_q16,
            force_parity_status_domain_drift=force_commit_status_domain_drift,
        )
    )

    status_canonical, canonical_enabled, canonical_reason = (
        _IQ1657_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1657(
            secure_local_mode,
            iommu_active,
            book_of_truth_gpu_hooks,
            policy_digest_parity,
            dispatch_transcript_parity,
            p50_overhead_q16,
            p95_overhead_q16,
            max_p50_overhead_q16,
            max_p95_overhead_q16,
            caller_fast_path_enabled=staged_enabled,
            caller_disable_reason=staged_reason,
            force_diagnostics_status_domain_drift=force_canonical_status_domain_drift,
        )
    )

    if not _MOD._status_is_valid(status_commit_only) or not _MOD._status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    if not _MOD._flag_is_binary(staged_enabled) or not _MOD._flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    if not _MOD._disable_reason_is_valid(staged_reason) or not _MOD._disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    if status_commit_only != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    if staged_enabled != canonical_enabled or staged_reason != canonical_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    return status_commit_only, caller_fast_path_enabled, caller_disable_reason


def test_source_contains_iq1659_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert (
        "I32 GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateCheckedParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in src
    )
    assert (
        "status_commit_only =\n        GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateCheckedParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    )
    assert (
        "status_canonical =\n        GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateCheckedParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in src
    )


def test_gate_missing_vector() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1659(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=100,
        p95_overhead_q16=200,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=1,
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    assert (status, out_enabled, out_reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        1,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )


def test_budget_breach_vector() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1659(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=900,
        p95_overhead_q16=1200,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=1,
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    assert (status, out_enabled, out_reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        1,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )


def test_status_domain_drift_vector() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1659(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=120,
        p95_overhead_q16=240,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=1,
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        force_commit_status_domain_drift=True,
    )
    assert (status, out_enabled, out_reason) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        1,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )


def test_no_write_parity_vectors() -> None:
    first = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1659(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=120,
        p95_overhead_q16=240,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=1,
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    second = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1659(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=120,
        p95_overhead_q16=240,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=1,
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )

    assert first == (GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD)
    assert second == first


def test_secure_on_vectors() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1659(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=120,
        p95_overhead_q16=240,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=0,
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    assert (status, out_enabled, out_reason) == (
        GPU_SEC_PERF_OK,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )


if __name__ == "__main__":
    test_source_contains_iq1659_symbols()
    test_gate_missing_vector()
    test_budget_breach_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vectors()
    test_secure_on_vectors()
    print("ok")
