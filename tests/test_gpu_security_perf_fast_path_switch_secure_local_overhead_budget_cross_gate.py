#!/usr/bin/env python3
"""Harness for IQ-1639 secure-local overhead-budget cross gate."""

from __future__ import annotations

from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_NULL_PTR = 1
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3
GPU_SEC_PERF_ERR_CAPACITY = 4
GPU_SEC_PERF_ERR_OVERFLOW = 5

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = 1

GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW = 0
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD = 1
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_IOMMU_GUARD = 2
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_BOOK_GUARD = 3
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_POLICY_DIGEST_MISMATCH = 4
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH = 5
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH = 6


def _flag_is_binary(value: int) -> bool:
    return value in (0, 1)


def _status_is_valid(status_code: int) -> bool:
    return status_code in {
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ERR_NULL_PTR,
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ERR_CAPACITY,
        GPU_SEC_PERF_ERR_OVERFLOW,
    }


def _disable_reason_is_valid(reason_code: int) -> bool:
    return (
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW
        <= reason_code
        <= GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH
    )


def _fast_path_switch_checked_audit_parity(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
) -> tuple[int, int, int]:
    fast_path_enabled = 0
    disable_reason = GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD

    if not (
        _flag_is_binary(iommu_active)
        and _flag_is_binary(book_of_truth_gpu_hooks)
        and _flag_is_binary(policy_digest_parity)
        and _flag_is_binary(dispatch_transcript_parity)
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, fast_path_enabled, disable_reason

    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, fast_path_enabled, disable_reason
    if iommu_active != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, fast_path_enabled, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_IOMMU_GUARD
    if book_of_truth_gpu_hooks != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, fast_path_enabled, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_BOOK_GUARD
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, fast_path_enabled, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_POLICY_DIGEST_MISMATCH
    if dispatch_transcript_parity != 1:
        return (
            GPU_SEC_PERF_ERR_POLICY_GUARD,
            fast_path_enabled,
            GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH,
        )

    return GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW


def fast_path_switch_secure_local_overhead_budget_cross_gate_iq1639(
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
    force_status_domain_drift: bool = False,
) -> tuple[int, int, int]:
    if not (
        _flag_is_binary(iommu_active)
        and _flag_is_binary(book_of_truth_gpu_hooks)
        and _flag_is_binary(policy_digest_parity)
        and _flag_is_binary(dispatch_transcript_parity)
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD

    if min(
        p50_overhead_q16,
        p95_overhead_q16,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
    ) < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
    if p95_overhead_q16 < p50_overhead_q16 or max_p95_overhead_q16 < max_p50_overhead_q16:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD

    status, staged_enabled, staged_reason = _fast_path_switch_checked_audit_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
    )
    if force_status_domain_drift:
        status = 99

    if not _status_is_valid(status):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason
    if not _flag_is_binary(staged_enabled) or not _disable_reason_is_valid(staged_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_fast_path_enabled, caller_disable_reason

    if status != GPU_SEC_PERF_OK:
        return status, staged_enabled, staged_reason

    if p50_overhead_q16 > max_p50_overhead_q16 or p95_overhead_q16 > max_p95_overhead_q16:
        return (
            GPU_SEC_PERF_ERR_POLICY_GUARD,
            0,
            GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
        )

    return GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW


def test_source_contains_iq1639_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateChecked(" in src
    assert "staged_status = GPUSecurityPerfFastPathSwitchCheckedAuditParity(" in src
    assert "GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH" in src


def test_gate_missing_vectors() -> None:
    status, enabled, reason = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1639(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=10,
        p95_overhead_q16=20,
        max_p50_overhead_q16=30,
        max_p95_overhead_q16=40,
        caller_fast_path_enabled=555,
        caller_disable_reason=666,
    )
    assert (status, enabled, reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_IOMMU_GUARD,
    )

    status, enabled, reason = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1639(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=0,
        p50_overhead_q16=10,
        p95_overhead_q16=20,
        max_p50_overhead_q16=30,
        max_p95_overhead_q16=40,
        caller_fast_path_enabled=777,
        caller_disable_reason=888,
    )
    assert (status, enabled, reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH,
    )


def test_budget_breach_vector() -> None:
    status, enabled, reason = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1639(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=400,
        p95_overhead_q16=900,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=800,
        caller_fast_path_enabled=901,
        caller_disable_reason=902,
    )
    assert (status, enabled, reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
    )


def test_status_domain_drift_vector() -> None:
    status, enabled, reason = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1639(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=100,
        p95_overhead_q16=200,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=333,
        caller_disable_reason=444,
        force_status_domain_drift=True,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_BAD_PARAM, 333, 444)


def test_deterministic_secure_on_vectors() -> None:
    first = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1639(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=200,
        p95_overhead_q16=400,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=111,
        caller_disable_reason=222,
    )
    second = fast_path_switch_secure_local_overhead_budget_cross_gate_iq1639(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=200,
        p95_overhead_q16=400,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        caller_fast_path_enabled=111,
        caller_disable_reason=222,
    )

    assert first == (GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW)
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1639_symbols()
    test_gate_missing_vectors()
    test_budget_breach_vector()
    test_status_domain_drift_vector()
    test_deterministic_secure_on_vectors()
    print("ok")
