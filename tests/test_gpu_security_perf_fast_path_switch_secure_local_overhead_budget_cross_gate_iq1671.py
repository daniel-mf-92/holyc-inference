#!/usr/bin/env python3
"""Harness for IQ-1671 zero-write diagnostics companion over IQ-1670 commit-only + IQ-1669 canonical parity."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_DEF_PATH = Path("tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_iq1654.py")
_SPEC = importlib.util.spec_from_file_location("iq1654_models", _DEF_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

_IQ1670_PATH = Path("tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_iq1670.py")
_IQ1670_SPEC = importlib.util.spec_from_file_location("iq1670_models", _IQ1670_PATH)
assert _IQ1670_SPEC is not None and _IQ1670_SPEC.loader is not None
_IQ1670_MOD = importlib.util.module_from_spec(_IQ1670_SPEC)
_IQ1670_SPEC.loader.exec_module(_IQ1670_MOD)

_IQ1669_PATH = Path("tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_iq1669.py")
_IQ1669_SPEC = importlib.util.spec_from_file_location("iq1669_models", _IQ1669_PATH)
assert _IQ1669_SPEC is not None and _IQ1669_SPEC.loader is not None
_IQ1669_MOD = importlib.util.module_from_spec(_IQ1669_SPEC)
_IQ1669_SPEC.loader.exec_module(_IQ1669_MOD)

GPU_SEC_PERF_OK = _MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = _MOD.GPU_SEC_PERF_ERR_POLICY_GUARD
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD = _MOD.GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD


def fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1671(
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
    force_diagnostics_status_domain_drift: bool = False,
    force_canonical_status_domain_drift: bool = False,
) -> tuple[int, int, int]:
    status_diagnostics, diagnostics_enabled, diagnostics_reason = (
        _IQ1670_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_iq1670(
            secure_local_mode,
            iommu_active,
            book_of_truth_gpu_hooks,
            policy_digest_parity,
            dispatch_transcript_parity,
            p50_overhead_q16,
            p95_overhead_q16,
            max_p50_overhead_q16,
            max_p95_overhead_q16,
            force_parity_status_domain_drift=force_diagnostics_status_domain_drift,
        )
    )

    status_canonical, canonical_enabled, canonical_reason = (
        _IQ1669_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1669(
            secure_local_mode,
            iommu_active,
            book_of_truth_gpu_hooks,
            policy_digest_parity,
            dispatch_transcript_parity,
            p50_overhead_q16,
            p95_overhead_q16,
            max_p50_overhead_q16,
            max_p95_overhead_q16,
            caller_fast_path_enabled,
            caller_disable_reason,
            force_diagnostics_status_domain_drift=force_canonical_status_domain_drift,
        )
    )

    if not _MOD._status_is_valid(status_diagnostics) or not _MOD._status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    if not _MOD._flag_is_binary(diagnostics_enabled) or not _MOD._flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    if not _MOD._disable_reason_is_valid(diagnostics_reason) or not _MOD._disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    if status_diagnostics != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    if diagnostics_enabled != canonical_enabled or diagnostics_reason != canonical_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    return status_diagnostics, caller_fast_path_enabled, caller_disable_reason


def test_source_contains_iq1671_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert (
        "I32 GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateCheckedParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in src
    )
    assert (
        "status_diagnostics =\n        GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateCheckedParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    )
    assert (
        "status_canonical =\n        GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateCheckedParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in src
    )
    assert "// IQ-1671 zero-write diagnostics companion over IQ-1670 commit-only + IQ-1669 canonical parity:" in src


def test_gate_missing_vector() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1671(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=100,
        p95_overhead_q16=200,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=0,
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    assert (status, out_enabled, out_reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )


def test_budget_breach_vector() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1671(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=900,
        p95_overhead_q16=1200,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=0,
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    assert (status, out_enabled, out_reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )


def test_status_domain_drift_vector() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1671(
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
        force_diagnostics_status_domain_drift=True,
    )
    assert (status, out_enabled, out_reason) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )


def test_no_write_parity_vectors() -> None:
    first = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1671(
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
    second = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1671(
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

    assert first == (GPU_SEC_PERF_OK, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD)
    assert second == first


def test_secure_on_vectors() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1671(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=100,
        p95_overhead_q16=200,
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
    test_source_contains_iq1671_symbols()
    test_gate_missing_vector()
    test_budget_breach_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vectors()
    test_secure_on_vectors()
    print("ok")
