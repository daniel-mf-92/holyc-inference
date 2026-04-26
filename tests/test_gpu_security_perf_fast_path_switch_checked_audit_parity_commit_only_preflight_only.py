#!/usr/bin/env python3
"""Harness for IQ-1578 zero-write diagnostics companion over fast-path audit-parity gate."""

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


def fast_path_switch_checked_audit_parity_commit_only_preflight_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
    *,
    outputs_alias: bool = False,
    has_null_output: bool = False,
    force_commit_only_status_domain_drift: bool = False,
) -> tuple[int, int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_fast_path_enabled, current_disable_reason_code
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    status_commit_only, staged_enabled, staged_reason = _fast_path_switch_checked_audit_parity_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        force_status_domain_drift=force_commit_only_status_domain_drift,
    )
    status_preflight_only, canonical_enabled, canonical_reason = _fast_path_switch_checked_audit_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
    )

    if not _status_is_valid(status_commit_only) or not _status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(staged_enabled) or not _flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(staged_reason) or not _disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if status_commit_only != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if staged_enabled != canonical_enabled or staged_reason != canonical_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status_commit_only, current_fast_path_enabled, current_disable_reason_code


def test_source_contains_iq1578_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfFastPathSwitchCheckedAuditParityCommitOnlyPreflightOnly(" in src
    assert "status_commit_only = GPUSecurityPerfFastPathSwitchCheckedAuditParityCommitOnly(" in src
    assert "status_preflight_only = GPUSecurityPerfFastPathSwitchCheckedAuditParity(" in src
    assert "saved_fast_path_enabled" in src
    assert "saved_disable_reason_code" in src
    assert "if (*out_fast_path_enabled != saved_fast_path_enabled" in src


def test_gate_missing_and_audit_parity_breach_vectors() -> None:
    status, enabled, reason = fast_path_switch_checked_audit_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=44,
        current_disable_reason_code=55,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 44, 55)

    status, enabled, reason = fast_path_switch_checked_audit_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=0,
        current_fast_path_enabled=66,
        current_disable_reason_code=77,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 66, 77)


def test_status_domain_drift_and_no_write_parity_vectors() -> None:
    status, enabled, reason = fast_path_switch_checked_audit_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=88,
        current_disable_reason_code=99,
        force_commit_only_status_domain_drift=True,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_BAD_PARAM, 88, 99)

    status, enabled, reason = fast_path_switch_checked_audit_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=123,
        current_disable_reason_code=124,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_OK, 123, 124)


def test_null_and_alias_vectors() -> None:
    status, enabled, reason = fast_path_switch_checked_audit_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=7,
        current_disable_reason_code=8,
        has_null_output=True,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_NULL_PTR, 7, 8)

    status, enabled, reason = fast_path_switch_checked_audit_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=9,
        current_disable_reason_code=10,
        outputs_alias=True,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_BAD_PARAM, 9, 10)


if __name__ == "__main__":
    test_source_contains_iq1578_symbols()
    test_gate_missing_and_audit_parity_breach_vectors()
    test_status_domain_drift_and_no_write_parity_vectors()
    test_null_and_alias_vectors()
    print("ok")
