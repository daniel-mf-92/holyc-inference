#!/usr/bin/env python3
"""Harness for IQ-1652 strict diagnostics parity gate on secure-local overhead budget cross-gate path."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_DEF_PATH = Path("tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_iq1645.py")
_SPEC = importlib.util.spec_from_file_location("iq1645_models", _DEF_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

GPU_SEC_PERF_OK = _MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = _MOD.GPU_SEC_PERF_ERR_POLICY_GUARD
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD = _MOD.GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH = _MOD.GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH


def fast_path_switch_secure_local_overhead_budget_cross_gate_iq1652(
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
    caller_fast_path_enabled: int,
    caller_disable_reason: int,
    force_status_domain_drift: bool = False,
) -> tuple[int, int, int]:
    return _MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_checked_parity_commit_only_preflight_only_parity_iq1645(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        p50_overhead_q16,
        p95_overhead_q16,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        caller_fast_path_enabled=caller_fast_path_enabled,
        caller_disable_reason=caller_disable_reason,
        force_status_domain_drift=force_status_domain_drift,
    )


def test_source_contains_iq1652_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert (
        "I32 GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateCheckedParityCommitOnlyPreflightOnlyParity(" in src
    )
    assert (
        "status_preflight_only =\n        GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateCheckedParityCommitOnlyPreflightOnly(" in src
    )
    assert (
        "status_canonical =\n        GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateCheckedParityCommitOnly(" in src
    )


def test_gate_missing_vector() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1652(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=120,
        p95_overhead_q16=240,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=1,
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
    )
    assert (status, out_enabled, out_reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        1,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
    )


def test_budget_breach_vector() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1652(
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
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
    )
    assert (status, out_enabled, out_reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        1,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
    )


def test_status_domain_drift_vector() -> None:
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1652(
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
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
        force_status_domain_drift=True,
    )
    assert (status, out_enabled, out_reason) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        1,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
    )


def test_no_write_parity_vectors() -> None:
    caller_fast_path_enabled = 1
    caller_disable_reason = GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH
    status, out_enabled, out_reason = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1652(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=120,
        p95_overhead_q16=240,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=caller_fast_path_enabled,
        caller_disable_reason=caller_disable_reason,
    )
    assert status == GPU_SEC_PERF_OK
    assert out_enabled == caller_fast_path_enabled
    assert out_reason == caller_disable_reason


def test_deterministic_secure_on_parity_vectors() -> None:
    first = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1652(
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
    second = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1652(
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


if __name__ == "__main__":
    test_source_contains_iq1652_symbols()
    test_gate_missing_vector()
    test_budget_breach_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vectors()
    test_deterministic_secure_on_parity_vectors()
    print("ok")
