#!/usr/bin/env python3
"""Harness for IQ-1609 strict diagnostics parity gate over IQ-1608 and IQ-1593."""

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
        return GPU_SEC_PERF_ERR_POLICY_GUARD, fast_path_enabled, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_POLICY_DIGEST_MISMATCH
    if dispatch_transcript_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, fast_path_enabled, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH

    return GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW


def _fast_path_switch_checked_audit_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status, parity_enabled, parity_reason = _fast_path_switch_checked_audit_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
    )
    if status == GPU_SEC_PERF_OK:
        return GPU_SEC_PERF_OK, parity_enabled, parity_reason
    return status, current_fast_path_enabled, current_disable_reason_code


def _fast_path_switch_checked_audit_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status, staged_enabled, staged_reason = _fast_path_switch_checked_audit_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    if not _status_is_valid(status):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(staged_enabled) or not _disable_reason_is_valid(staged_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    return status, current_fast_path_enabled, current_disable_reason_code


def _fast_path_switch_checked_audit_parity_iq1592(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    staged_enabled = 0
    staged_reason = GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
    status_preflight_only, staged_enabled, staged_reason = _fast_path_switch_checked_audit_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=staged_enabled,
        current_disable_reason_code=staged_reason,
    )

    status_commit_only, canonical_enabled, canonical_reason = _fast_path_switch_checked_audit_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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

    if status_preflight_only != status_commit_only or staged_enabled != canonical_enabled or staged_reason != canonical_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status_preflight_only, current_fast_path_enabled, current_disable_reason_code


def _fast_path_switch_checked_audit_parity_iq1593(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    parity_enabled = 0
    parity_reason = GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
    status_parity, parity_enabled, parity_reason = _fast_path_switch_checked_audit_parity_iq1592(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=parity_enabled,
        current_disable_reason_code=parity_reason,
    )

    status_preflight_only, staged_enabled, staged_reason = _fast_path_switch_checked_audit_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
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


def _fast_path_switch_checked_audit_parity_iq1608(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    current_fast_path_enabled: int,
    current_disable_reason_code: int,
) -> tuple[int, int, int]:
    status_commit_only, staged_enabled, staged_reason = _fast_path_switch_checked_audit_parity_iq1593(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )

    status_parity, canonical_enabled, canonical_reason = _fast_path_switch_checked_audit_parity_iq1592(
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

    if status_commit_only != status_parity or staged_enabled != canonical_enabled or staged_reason != canonical_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status_commit_only, current_fast_path_enabled, current_disable_reason_code


def fast_path_switch_checked_audit_parity_iq1609(
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
    force_commit_status_domain_drift: bool = False,
) -> tuple[int, int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_fast_path_enabled, current_disable_reason_code
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    status_preflight_only, staged_enabled, staged_reason = _fast_path_switch_checked_audit_parity_iq1608(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=0,
        current_disable_reason_code=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )

    status_commit_only, canonical_enabled, canonical_reason = _fast_path_switch_checked_audit_parity_iq1593(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        current_fast_path_enabled=staged_enabled,
        current_disable_reason_code=staged_reason,
    )

    if force_commit_status_domain_drift:
        status_commit_only = 97

    if not _status_is_valid(status_preflight_only) or not _status_is_valid(status_commit_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _flag_is_binary(staged_enabled) or not _flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code
    if not _disable_reason_is_valid(staged_reason) or not _disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    if status_preflight_only != status_commit_only or staged_enabled != canonical_enabled or staged_reason != canonical_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_fast_path_enabled, current_disable_reason_code

    return status_preflight_only, current_fast_path_enabled, current_disable_reason_code


def test_source_contains_iq1609_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfFastPathSwitchCheckedAuditParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in src
    assert "status_preflight_only = GPUSecurityPerfFastPathSwitchCheckedAuditParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in src
    assert "status_commit_only = GPUSecurityPerfFastPathSwitchCheckedAuditParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    assert "if (*out_fast_path_enabled != saved_fast_path_enabled" in src


def test_gate_missing_and_audit_parity_breach_vectors() -> None:
    status, enabled, reason = fast_path_switch_checked_audit_parity_iq1609(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=31,
        current_disable_reason_code=32,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 31, 32)

    status, enabled, reason = fast_path_switch_checked_audit_parity_iq1609(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=0,
        current_fast_path_enabled=41,
        current_disable_reason_code=42,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 41, 42)


def test_status_domain_drift_no_write_parity_and_deterministic_vectors() -> None:
    status, enabled, reason = fast_path_switch_checked_audit_parity_iq1609(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=51,
        current_disable_reason_code=52,
        force_commit_status_domain_drift=True,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_BAD_PARAM, 51, 52)

    # No-write parity vector: IQ-1608 and IQ-1593 both fail-closed under secure-pass tuple,
    # and IQ-1609 must preserve caller slots exactly.
    status, enabled, reason = fast_path_switch_checked_audit_parity_iq1609(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=61,
        current_disable_reason_code=62,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_BAD_PARAM, 61, 62)

    # Deterministic parity vector: same secure tuple yields same fail-closed status and no-write outputs.
    status, enabled, reason = fast_path_switch_checked_audit_parity_iq1609(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        current_fast_path_enabled=71,
        current_disable_reason_code=72,
    )
    assert (status, enabled, reason) == (GPU_SEC_PERF_ERR_BAD_PARAM, 71, 72)
