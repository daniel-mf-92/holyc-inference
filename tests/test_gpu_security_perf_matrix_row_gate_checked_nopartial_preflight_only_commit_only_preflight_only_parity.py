#!/usr/bin/env python3
"""Harness for IQ-1512 row-gate strict diagnostics parity gate."""

from __future__ import annotations

from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_NULL_PTR = 1
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = 1
GPU_SEC_PERF_PROFILE_DEV_LOCAL = 2

GPU_SEC_PERF_QUANT_Q4_0 = 40
GPU_SEC_PERF_QUANT_Q8_0 = 80

GPU_SEC_PERF_ROW_GATE_REASON_ALLOW = 0
GPU_SEC_PERF_ROW_GATE_REASON_PROFILE_GUARD = 1
GPU_SEC_PERF_ROW_GATE_REASON_IOMMU_GUARD = 2
GPU_SEC_PERF_ROW_GATE_REASON_BOOK_GUARD = 3
GPU_SEC_PERF_ROW_GATE_REASON_POLICY_DIGEST_MISMATCH = 4
GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY = 5
GPU_SEC_PERF_ROW_GATE_REASON_BAD_QUANT_PROFILE = 6


def _is_binary(value: int) -> bool:
    return value in (0, 1)


def _is_supported_quant(quant_level: int) -> bool:
    return quant_level in (GPU_SEC_PERF_QUANT_Q4_0, GPU_SEC_PERF_QUANT_Q8_0)


def _row_gate_checked(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
) -> tuple[int, int, int]:
    gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY
    row_allowed = 0

    if not _is_binary(iommu_active) or not _is_binary(book_of_truth_gpu_hooks) or not _is_binary(policy_digest_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, gate_reason_code, row_allowed
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, GPU_SEC_PERF_ROW_GATE_REASON_PROFILE_GUARD, row_allowed
    if row_prompt_tokens < 0 or row_batch_size <= 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY, row_allowed
    if not _is_supported_quant(row_quant_profile):
        return GPU_SEC_PERF_ERR_BAD_PARAM, GPU_SEC_PERF_ROW_GATE_REASON_BAD_QUANT_PROFILE, row_allowed
    if iommu_active == 0:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, GPU_SEC_PERF_ROW_GATE_REASON_IOMMU_GUARD, row_allowed
    if book_of_truth_gpu_hooks == 0:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, GPU_SEC_PERF_ROW_GATE_REASON_BOOK_GUARD, row_allowed
    if policy_digest_parity == 0:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, GPU_SEC_PERF_ROW_GATE_REASON_POLICY_DIGEST_MISMATCH, row_allowed

    return GPU_SEC_PERF_OK, GPU_SEC_PERF_ROW_GATE_REASON_ALLOW, 1


def _row_gate_checked_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
) -> tuple[int, int, int]:
    status_primary, reason_primary, allowed_primary = _row_gate_checked(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    status_parity, reason_parity, allowed_parity = _row_gate_checked(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )

    if status_primary != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY, 0
    if reason_primary != reason_parity or allowed_primary != allowed_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY, 0
    if status_primary != GPU_SEC_PERF_OK:
        return status_primary, reason_primary, allowed_primary
    return GPU_SEC_PERF_OK, reason_primary, allowed_primary


def _row_gate_checked_nopartial(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
) -> tuple[int, int, int]:
    status_preflight, reason_preflight, allowed_preflight = _row_gate_checked_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    status_commit, reason_commit, allowed_commit = _row_gate_checked_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )

    if status_preflight != status_commit:
        return GPU_SEC_PERF_ERR_BAD_PARAM, GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY, 0
    if reason_preflight != reason_commit or allowed_preflight != allowed_commit:
        return GPU_SEC_PERF_ERR_BAD_PARAM, GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY, 0
    if status_preflight != GPU_SEC_PERF_OK:
        return status_preflight, reason_preflight, allowed_preflight
    return GPU_SEC_PERF_OK, reason_preflight, allowed_preflight


def _row_gate_checked_nopartial_preflight_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_gate_reason_code: int,
    current_row_allowed: int,
    outputs_alias: bool = False,
    has_null_output: bool = False,
) -> tuple[int, int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_gate_reason_code, current_row_allowed
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed

    staged_status, staged_reason, staged_allowed = _row_gate_checked_nopartial(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    canonical_status, canonical_reason, canonical_allowed = _row_gate_checked(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )

    if staged_status != canonical_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if staged_reason != canonical_reason or staged_allowed != canonical_allowed:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    return staged_status, current_gate_reason_code, current_row_allowed


def _row_gate_checked_nopartial_preflight_only_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_gate_reason_code: int,
    current_row_allowed: int,
    outputs_alias: bool = False,
    has_null_output: bool = False,
) -> tuple[int, int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_gate_reason_code, current_row_allowed
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed

    status_nopartial, reason_nopartial, allowed_nopartial = _row_gate_checked_nopartial(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    status_preflight_only, reason_preflight_only, allowed_preflight_only = _row_gate_checked_nopartial_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        reason_nopartial,
        allowed_nopartial,
    )

    if status_nopartial != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if reason_nopartial != reason_preflight_only or allowed_nopartial != allowed_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if status_nopartial != GPU_SEC_PERF_OK:
        return status_nopartial, current_gate_reason_code, current_row_allowed
    return GPU_SEC_PERF_OK, reason_nopartial, allowed_nopartial


def _row_gate_checked_nopartial_preflight_only_commit_only_preflight_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_gate_reason_code: int,
    current_row_allowed: int,
    outputs_alias: bool = False,
    has_null_output: bool = False,
) -> tuple[int, int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_gate_reason_code, current_row_allowed
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed

    status_commit_only, reason_commit_only, allowed_commit_only = _row_gate_checked_nopartial_preflight_only_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        current_gate_reason_code,
        current_row_allowed,
    )
    status_preflight_only, reason_preflight_only, allowed_preflight_only = _row_gate_checked_nopartial_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        reason_commit_only,
        allowed_commit_only,
    )

    if status_commit_only != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if reason_commit_only != reason_preflight_only or allowed_commit_only != allowed_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    return status_commit_only, current_gate_reason_code, current_row_allowed


def row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_gate_reason_code: int,
    current_row_allowed: int,
    outputs_alias: bool = False,
    has_null_output: bool = False,
    inject_reason_drift: bool = False,
) -> tuple[int, int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_gate_reason_code, current_row_allowed
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed

    status_commit_only, canonical_reason, canonical_allowed = _row_gate_checked_nopartial_preflight_only_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY,
        0,
    )
    status_preflight_only, staged_reason, staged_allowed = _row_gate_checked_nopartial_preflight_only_commit_only_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        canonical_reason,
        canonical_allowed,
    )

    if inject_reason_drift:
        canonical_reason += 1

    if status_preflight_only != status_commit_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if staged_reason != canonical_reason or staged_allowed != canonical_allowed:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    return status_preflight_only, current_gate_reason_code, current_row_allowed


def test_source_contains_iq1512_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnlyPreflightOnlyParity(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnlyPreflightOnly(" in src
    assert "status_commit_only = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnly(" in src
    assert "saved_gate_reason_code" in src
    assert "saved_row_allowed" in src
    assert "if (status_preflight_only != status_commit_only)" in src


def test_null_alias_capacity_vectors() -> None:
    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=24,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=701,
        current_row_allowed=702,
        has_null_output=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_NULL_PTR, 701, 702)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=24,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=703,
        current_row_allowed=704,
        outputs_alias=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 703, 704)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=24,
        row_batch_size=0,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=705,
        current_row_allowed=706,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 705, 706)


def test_gate_missing_and_reason_parity_vectors() -> None:
    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=48,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=711,
        current_row_allowed=712,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 711, 712)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=0,
        policy_digest_parity=1,
        row_prompt_tokens=48,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=713,
        current_row_allowed=714,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 713, 714)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=48,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=715,
        current_row_allowed=716,
        inject_reason_drift=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 715, 716)


def test_success_and_profile_guard_zero_write() -> None:
    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=96,
        row_batch_size=4,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=721,
        current_row_allowed=722,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_OK, 721, 722)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_DEV_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=12,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=723,
        current_row_allowed=724,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 723, 724)
