#!/usr/bin/env python3
"""Harness for IQ-1484 commit-only hardening over row-gate preflight-only parity wrappers."""

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


def gpu_security_perf_matrix_row_gate_checked(
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


def gpu_security_perf_matrix_row_gate_checked_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
) -> tuple[int, int, int]:
    staged_status, staged_reason, staged_allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    parity_status, parity_reason, parity_allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )

    if staged_status != parity_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY, 0
    if staged_reason != parity_reason or staged_allowed != parity_allowed:
        return GPU_SEC_PERF_ERR_BAD_PARAM, GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY, 0
    if staged_status != GPU_SEC_PERF_OK:
        return staged_status, staged_reason, staged_allowed
    return GPU_SEC_PERF_OK, staged_reason, staged_allowed


def gpu_security_perf_matrix_row_gate_checked_nopartial(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
) -> tuple[int, int, int]:
    preflight_status, preflight_reason, preflight_allowed = gpu_security_perf_matrix_row_gate_checked_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    commit_status, commit_reason, commit_allowed = gpu_security_perf_matrix_row_gate_checked_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )

    if preflight_status != commit_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY, 0
    if preflight_reason != commit_reason or preflight_allowed != commit_allowed:
        return GPU_SEC_PERF_ERR_BAD_PARAM, GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY, 0
    if preflight_status != GPU_SEC_PERF_OK:
        return preflight_status, preflight_reason, preflight_allowed
    return GPU_SEC_PERF_OK, preflight_reason, preflight_allowed


def gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only(
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

    staged_status, staged_reason, staged_allowed = gpu_security_perf_matrix_row_gate_checked_nopartial(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    canonical_status, canonical_reason, canonical_allowed = gpu_security_perf_matrix_row_gate_checked(
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


def gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_parity(
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

    parity_status, parity_reason, parity_allowed = gpu_security_perf_matrix_row_gate_checked_nopartial(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    staged_status, staged_reason, staged_allowed = gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        parity_reason,
        parity_allowed,
    )

    if staged_status != parity_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if staged_reason != parity_reason or staged_allowed != parity_allowed:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed

    return staged_status, current_gate_reason_code, current_row_allowed


def gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_parity_commit_only(
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

    parity_status, parity_reason, parity_allowed = gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_parity(
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
    staged_status, staged_reason, staged_allowed = gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        parity_reason,
        parity_allowed,
    )

    if inject_reason_drift:
        staged_reason += 1

    if staged_status != parity_status:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if staged_reason != parity_reason or staged_allowed != parity_allowed:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if parity_status != GPU_SEC_PERF_OK:
        return parity_status, current_gate_reason_code, current_row_allowed

    return GPU_SEC_PERF_OK, parity_reason, parity_allowed


def test_source_contains_iq1484_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyParityCommitOnly(" in src
    assert "status_parity = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyParity(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnly(" in src
    assert "saved_gate_reason_code" in src
    assert "saved_row_allowed" in src
    assert "if (status_parity != status_preflight_only)" in src


def test_null_alias_capacity_vectors() -> None:
    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=64,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=301,
        current_row_allowed=302,
        has_null_output=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_NULL_PTR, 301, 302)

    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=64,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=303,
        current_row_allowed=304,
        outputs_alias=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 303, 304)

    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=64,
        row_batch_size=0,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=305,
        current_row_allowed=306,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 305, 306)


def test_gate_missing_vector() -> None:
    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=32,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=401,
        current_row_allowed=402,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 401, 402)


def test_reason_code_parity_vectors() -> None:
    vectors = [
        (GPU_SEC_PERF_PROFILE_SECURE_LOCAL, 1, 1, 1, 16, 1, GPU_SEC_PERF_QUANT_Q4_0, GPU_SEC_PERF_OK, True),
        (GPU_SEC_PERF_PROFILE_SECURE_LOCAL, 1, 1, 0, 16, 1, GPU_SEC_PERF_QUANT_Q8_0, GPU_SEC_PERF_ERR_POLICY_GUARD, False),
        (GPU_SEC_PERF_PROFILE_DEV_LOCAL, 1, 1, 1, 16, 1, GPU_SEC_PERF_QUANT_Q4_0, GPU_SEC_PERF_ERR_POLICY_GUARD, False),
        (GPU_SEC_PERF_PROFILE_SECURE_LOCAL, 1, 1, 1, 16, 1, 99, GPU_SEC_PERF_ERR_BAD_PARAM, False),
    ]

    for idx, vector in enumerate(vectors):
        secure_local_mode, iommu_active, hooks, parity, tokens, batch, quant, expected_status, should_publish = vector
        prior_reason = 500 + idx
        prior_allowed = 600 + idx

        status, reason, allowed = gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_parity_commit_only(
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=hooks,
            policy_digest_parity=parity,
            row_prompt_tokens=tokens,
            row_batch_size=batch,
            row_quant_profile=quant,
            current_gate_reason_code=prior_reason,
            current_row_allowed=prior_allowed,
        )
        assert status == expected_status

        parity_status, parity_reason, parity_allowed = gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_parity(
            secure_local_mode,
            iommu_active,
            hooks,
            parity,
            tokens,
            batch,
            quant,
            GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY,
            0,
        )
        assert parity_status == expected_status

        if should_publish:
            assert (reason, allowed) == (parity_reason, parity_allowed)
        else:
            assert (reason, allowed) == (prior_reason, prior_allowed)


def test_reason_code_parity_drift_vector() -> None:
    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=64,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=777,
        current_row_allowed=778,
        inject_reason_drift=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 777, 778)
