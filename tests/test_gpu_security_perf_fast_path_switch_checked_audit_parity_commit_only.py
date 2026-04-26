#!/usr/bin/env python3
"""Harness for IQ-1577 fast-path switch commit-only gate."""

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
    return GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW <= reason_code <= GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH


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
        return (
            GPU_SEC_PERF_ERR_POLICY_GUARD,
            fast_path_enabled,
            GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_POLICY_DIGEST_MISMATCH,
        )

    if dispatch_transcript_parity != 1:
        return (
            GPU_SEC_PERF_ERR_POLICY_GUARD,
            fast_path_enabled,
            GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH,
        )

    return GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW


def _fast_path_switch_checked_audit_parity_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    *,
    force_status_domain_drift: bool = False,
) -> tuple[int, int, int]:
    status_primary, staged_enabled, staged_reason = _fast_path_switch_checked_audit_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
    )
    status_parity, parity_enabled, parity_reason = _fast_path_switch_checked_audit_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
    )

    if force_status_domain_drift:
        status_parity = 77

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


def test_source_contains_iq1577_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH" in src
    assert "Bool GPUSecurityPerfFastPathDisableReasonIsValid(I64 reason_code)" in src
    assert "I32 GPUSecurityPerfFastPathSwitchCheckedAuditParity(" in src
    assert "I32 GPUSecurityPerfFastPathSwitchCheckedAuditParityCommitOnly(" in src
    assert "status_primary = GPUSecurityPerfFastPathSwitchCheckedAuditParity(" in src
    assert "status_parity = GPUSecurityPerfFastPathSwitchCheckedAuditParity(" in src


def test_secure_local_pass_vector() -> None:
    status, enabled, reason = _fast_path_switch_checked_audit_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )

    assert status == GPU_SEC_PERF_OK
    assert enabled == 1
    assert reason == GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW


def test_audit_parity_breach_vector() -> None:
    status, enabled, reason = _fast_path_switch_checked_audit_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=0,
    )

    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert enabled == 0
    assert reason == GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH


def test_hook_missing_vector() -> None:
    status, enabled, reason = _fast_path_switch_checked_audit_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=0,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )

    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert enabled == 0
    assert reason == GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_BOOK_GUARD


def test_status_domain_drift_vector() -> None:
    status, enabled, reason = _fast_path_switch_checked_audit_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        force_status_domain_drift=True,
    )

    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert enabled == 0
    assert reason == GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD


if __name__ == "__main__":
    test_source_contains_iq1577_symbols()
    test_secure_local_pass_vector()
    test_audit_parity_breach_vector()
    test_hook_missing_vector()
    test_status_domain_drift_vector()
    print("ok")
