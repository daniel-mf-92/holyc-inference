#!/usr/bin/env python3
"""Harness for IQ-1640 secure-local overhead-budget cross-gate commit-only wrapper."""

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


def _fast_path_switch_secure_local_overhead_budget_cross_gate_checked_iq1639(
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
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
    if not _flag_is_binary(staged_enabled) or not _disable_reason_is_valid(staged_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD

    if status != GPU_SEC_PERF_OK:
        return status, staged_enabled, staged_reason

    if p50_overhead_q16 > max_p50_overhead_q16 or p95_overhead_q16 > max_p95_overhead_q16:
        return (
            GPU_SEC_PERF_ERR_POLICY_GUARD,
            0,
            GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
        )

    return GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW


def fast_path_switch_secure_local_overhead_budget_cross_gate_commit_only_iq1640(
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
    force_status_domain_drift: bool = False,
) -> tuple[int, int, int]:
    status_primary, staged_enabled, staged_reason = (
        _fast_path_switch_secure_local_overhead_budget_cross_gate_checked_iq1639(
            secure_local_mode,
            iommu_active,
            book_of_truth_gpu_hooks,
            policy_digest_parity,
            dispatch_transcript_parity,
            p50_overhead_q16,
            p95_overhead_q16,
            max_p50_overhead_q16,
            max_p95_overhead_q16,
        )
    )
    status_parity, parity_enabled, parity_reason = (
        _fast_path_switch_secure_local_overhead_budget_cross_gate_checked_iq1639(
            secure_local_mode,
            iommu_active,
            book_of_truth_gpu_hooks,
            policy_digest_parity,
            dispatch_transcript_parity,
            p50_overhead_q16,
            p95_overhead_q16,
            max_p50_overhead_q16,
            max_p95_overhead_q16,
            force_status_domain_drift=force_status_domain_drift,
        )
    )

    if not _status_is_valid(status_primary) or not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
    if not _flag_is_binary(staged_enabled) or not _flag_is_binary(parity_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
    if not _disable_reason_is_valid(staged_reason) or not _disable_reason_is_valid(parity_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD

    if status_primary != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
    if staged_enabled != parity_enabled or staged_reason != parity_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD

    return status_primary, staged_enabled, staged_reason


def test_source_contains_iq1640_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateCheckedCommitOnly(" in src
    assert "status_primary = GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateChecked(" in src
    assert "status_parity = GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateChecked(" in src


def test_gate_missing_vector() -> None:
    status, enabled, reason = fast_path_switch_secure_local_overhead_budget_cross_gate_commit_only_iq1640(
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
    assert (status, enabled, reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_IOMMU_GUARD,
    )


def test_budget_breach_vector() -> None:
    status, enabled, reason = fast_path_switch_secure_local_overhead_budget_cross_gate_commit_only_iq1640(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=600,
        p95_overhead_q16=800,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
    )
    assert (status, enabled, reason) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
    )


def test_status_domain_drift_vector() -> None:
    status, enabled, reason = fast_path_switch_secure_local_overhead_budget_cross_gate_commit_only_iq1640(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=100,
        p95_overhead_q16=200,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
        force_status_domain_drift=True,
    )
    assert (status, enabled, reason) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )


def test_deterministic_tuple_parity_vector() -> None:
    first = fast_path_switch_secure_local_overhead_budget_cross_gate_commit_only_iq1640(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=100,
        p95_overhead_q16=200,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
    )
    second = fast_path_switch_secure_local_overhead_budget_cross_gate_commit_only_iq1640(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=100,
        p95_overhead_q16=200,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
    )

    assert first == (GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW)
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1640_symbols()
    test_gate_missing_vector()
    test_budget_breach_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
