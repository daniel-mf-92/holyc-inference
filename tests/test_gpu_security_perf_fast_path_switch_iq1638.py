#!/usr/bin/env python3
"""Harness for IQ-1638 commit-only hardening wrapper over IQ-1637 and IQ-1635."""

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
    return (
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW
        <= reason_code
        <= GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH
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
        return GPU_SEC_PERF_ERR_POLICY_GUARD, fast_path_enabled, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH

    return GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW


def _target_parity(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status, staged_enabled, staged_reason = _fast_path_switch_checked_audit_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
    )
    if not _status_is_valid(status):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(staged_enabled) or not _disable_reason_is_valid(staged_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status, current_fast_path_enabled, current_disable_reason_code


def _target_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status_parity, parity_enabled, parity_reason = _target_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )

    status_preflight_only, staged_enabled, staged_reason = _target_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=parity_enabled,
        current_disable_reason_code=parity_reason,
    )

    if not _status_is_valid(status_parity) or not _status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(parity_enabled) or not _flag_is_binary(staged_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(parity_reason) or not _disable_reason_is_valid(staged_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if status_parity != status_preflight_only or parity_enabled != staged_enabled or parity_reason != staged_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, current_fast_path_enabled, current_disable_reason_code
    return GPU_SEC_PERF_OK, parity_enabled, parity_reason


def _target_iq1616_preflight_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status_commit_only, staged_enabled, staged_reason = _target_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    status_parity, canonical_enabled, canonical_reason = _target_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=staged_enabled,
        current_disable_reason_code=staged_reason,
    )

    if not _status_is_valid(status_commit_only) or not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(staged_enabled) or not _flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(staged_reason) or not _disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if (
        status_commit_only != status_parity
        or staged_enabled != canonical_enabled
        or staged_reason != canonical_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status_commit_only, current_fast_path_enabled, current_disable_reason_code


def _target_iq1619_parity(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status_preflight_only, staged_enabled, staged_reason = _target_iq1616_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    status_commit_only, canonical_enabled, canonical_reason = _target_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=staged_enabled,
        current_disable_reason_code=staged_reason,
    )

    if not _status_is_valid(status_preflight_only) or not _status_is_valid(status_commit_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(staged_enabled) or not _flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(staged_reason) or not _disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if (
        status_preflight_only != status_commit_only
        or staged_enabled != canonical_enabled
        or staged_reason != canonical_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status_preflight_only, current_fast_path_enabled, current_disable_reason_code


def _target_iq1620_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status_parity, parity_enabled, parity_reason = _target_iq1619_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    status_preflight_only, staged_enabled, staged_reason = _target_iq1616_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=parity_enabled,
        current_disable_reason_code=parity_reason,
    )

    if not _status_is_valid(status_parity) or not _status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(parity_enabled) or not _flag_is_binary(staged_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(parity_reason) or not _disable_reason_is_valid(staged_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if (
        status_parity != status_preflight_only
        or parity_enabled != staged_enabled
        or parity_reason != staged_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, current_fast_path_enabled, current_disable_reason_code
    return GPU_SEC_PERF_OK, parity_enabled, parity_reason


def _target_iq1622_preflight_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status_commit_only, staged_enabled, staged_reason = _target_iq1620_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    status_parity, canonical_enabled, canonical_reason = _target_iq1619_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=staged_enabled,
        current_disable_reason_code=staged_reason,
    )

    if not _status_is_valid(status_commit_only) or not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(staged_enabled) or not _flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(staged_reason) or not _disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if (
        status_commit_only != status_parity
        or staged_enabled != canonical_enabled
        or staged_reason != canonical_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status_commit_only, current_fast_path_enabled, current_disable_reason_code


def _target_iq1623_parity(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status_preflight_only, staged_enabled, staged_reason = _target_iq1622_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    status_commit_only, canonical_enabled, canonical_reason = _target_iq1620_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=staged_enabled,
        current_disable_reason_code=staged_reason,
    )

    if not _status_is_valid(status_preflight_only) or not _status_is_valid(status_commit_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(staged_enabled) or not _flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(staged_reason) or not _disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if (
        status_preflight_only != status_commit_only
        or staged_enabled != canonical_enabled
        or staged_reason != canonical_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status_preflight_only, current_fast_path_enabled, current_disable_reason_code


def _target_iq1634_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status_parity, parity_enabled, parity_reason = _target_iq1623_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    status_preflight_only, staged_enabled, staged_reason = _target_iq1622_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=parity_enabled,
        current_disable_reason_code=parity_reason,
    )

    if not _status_is_valid(status_parity) or not _status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(parity_enabled) or not _flag_is_binary(staged_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(parity_reason) or not _disable_reason_is_valid(staged_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if (
        status_parity != status_preflight_only
        or parity_enabled != staged_enabled
        or parity_reason != staged_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, current_fast_path_enabled, current_disable_reason_code
    return GPU_SEC_PERF_OK, parity_enabled, parity_reason


def _target_iq1635_preflight_only(
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
    force_status_domain_drift: bool = False,
    force_tuple_drift: bool = False,
) -> tuple[int, int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_fast_path_enabled, current_disable_reason_code
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    status_commit_only, staged_enabled, staged_reason = _target_iq1634_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    status_parity, canonical_enabled, canonical_reason = _target_iq1623_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=staged_enabled,
        current_disable_reason_code=staged_reason,
    )

    if force_status_domain_drift:
        status_parity = 99
    if force_tuple_drift:
        canonical_reason = GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
        if staged_reason == canonical_reason:
            canonical_reason = GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH

    if not _status_is_valid(status_commit_only) or not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(staged_enabled) or not _flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(staged_reason) or not _disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if (
        status_commit_only != status_parity
        or staged_enabled != canonical_enabled
        or staged_reason != canonical_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status_commit_only, current_fast_path_enabled, current_disable_reason_code


def _target_iq1637_parity(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status_preflight_only, staged_enabled, staged_reason = _target_iq1635_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    status_commit_only, canonical_enabled, canonical_reason = _target_iq1634_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=staged_enabled,
        current_disable_reason_code=staged_reason,
    )

    if not _status_is_valid(status_preflight_only) or not _status_is_valid(status_commit_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(staged_enabled) or not _flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(staged_reason) or not _disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if (
        status_preflight_only != status_commit_only
        or staged_enabled != canonical_enabled
        or staged_reason != canonical_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status_preflight_only, current_fast_path_enabled, current_disable_reason_code


def fast_path_switch_checked_audit_parity_iq1638(
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
    force_status_domain_drift: bool = False,
    force_tuple_drift: bool = False,
) -> tuple[int, int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_fast_path_enabled, current_disable_reason_code
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    status_parity, parity_enabled, parity_reason = _target_iq1637_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    status_preflight_only, staged_enabled, staged_reason = _target_iq1635_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=parity_enabled,
        current_disable_reason_code=parity_reason,
    )

    if force_status_domain_drift:
        status_preflight_only = 99
    if force_tuple_drift:
        staged_reason = GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
        if parity_reason == staged_reason:
            staged_reason = GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH

    if not _status_is_valid(status_parity) or not _status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(parity_enabled) or not _flag_is_binary(staged_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(parity_reason) or not _disable_reason_is_valid(staged_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if (
        status_parity != status_preflight_only
        or parity_enabled != staged_enabled
        or parity_reason != staged_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, current_fast_path_enabled, current_disable_reason_code

    return GPU_SEC_PERF_OK, parity_enabled, parity_reason


def test_source_contains_iq1638_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfFastPathSwitchCheckedAuditParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    assert "status_parity = GPUSecurityPerfFastPathSwitchCheckedAuditParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in src
    assert "status_preflight_only = GPUSecurityPerfFastPathSwitchCheckedAuditParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in src


def test_gate_missing_and_audit_parity_breach_vectors() -> None:
    status, enabled, reason = fast_path_switch_checked_audit_parity_iq1638(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=501,
        current_disable_reason_code=502,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 501, 502)

    status, enabled, reason = fast_path_switch_checked_audit_parity_iq1638(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=0,
        current_fast_path_enabled=503,
        current_disable_reason_code=504,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 503, 504)


def test_status_domain_drift_and_tuple_parity_vectors() -> None:
    status, enabled, reason = fast_path_switch_checked_audit_parity_iq1638(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=511,
        current_disable_reason_code=512,
        force_status_domain_drift=True,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_BAD_PARAM, 511, 512)

    status, enabled, reason = fast_path_switch_checked_audit_parity_iq1638(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=513,
        current_disable_reason_code=514,
        force_tuple_drift=True,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_BAD_PARAM, 513, 514)


def test_deterministic_tuple_parity_vectors() -> None:
    first = fast_path_switch_checked_audit_parity_iq1638(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=521,
        current_disable_reason_code=522,
    )
    second = fast_path_switch_checked_audit_parity_iq1638(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=521,
        current_disable_reason_code=522,
    )
    assert first == (
        GPU_SEC_PERF_OK,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    assert second == first


def test_secure_on_overhead_budget_vector() -> None:
    status, enabled, reason = fast_path_switch_checked_audit_parity_iq1638(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=0x5A5A,
        current_disable_reason_code=0x0B0B,
    )
    assert (status, enabled, reason) == (
        GPU_SEC_PERF_OK,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )


if __name__ == "__main__":
    test_source_contains_iq1638_symbols()
    test_gate_missing_and_audit_parity_breach_vectors()
    test_status_domain_drift_and_tuple_parity_vectors()
    test_deterministic_tuple_parity_vectors()
    test_secure_on_overhead_budget_vector()
    print("ok")
