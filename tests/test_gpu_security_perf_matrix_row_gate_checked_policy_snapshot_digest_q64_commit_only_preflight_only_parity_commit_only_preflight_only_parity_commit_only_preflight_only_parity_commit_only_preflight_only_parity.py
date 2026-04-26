#!/usr/bin/env python3
"""Harness for IQ-1562 strict diagnostics parity gate over digest commit-only and zero-write preflight wrappers."""

from __future__ import annotations

from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_NULL_PTR = 1
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3
GPU_SEC_PERF_ERR_OVERFLOW = 5

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = 1

GPU_SEC_PERF_QUANT_Q4_0 = 40
GPU_SEC_PERF_QUANT_Q8_0 = 80

GPU_SEC_PERF_ROW_GATE_REASON_ALLOW = 0
GPU_SEC_PERF_ROW_GATE_REASON_IOMMU_GUARD = 2
GPU_SEC_PERF_ROW_GATE_REASON_BOOK_GUARD = 3
GPU_SEC_PERF_ROW_GATE_REASON_POLICY_DIGEST_MISMATCH = 4
GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY = 5
GPU_SEC_PERF_ROW_GATE_REASON_BAD_QUANT_PROFILE = 6

GPU_SEC_PERF_I64_MAX = 0x7FFFFFFFFFFFFFFF


def _add_checked(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs > GPU_SEC_PERF_I64_MAX - rhs:
        return None
    return lhs + rhs


def _mul_checked(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs == 0 or rhs == 0:
        return 0
    if lhs > GPU_SEC_PERF_I64_MAX // rhs:
        return None
    return lhs * rhs


def _is_binary(value: int) -> bool:
    return value in (0, 1)


def _is_supported_quant(quant_level: int) -> bool:
    return quant_level in (GPU_SEC_PERF_QUANT_Q4_0, GPU_SEC_PERF_QUANT_Q8_0)


def _status_is_valid(status_code: int) -> bool:
    return status_code in (
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ERR_NULL_PTR,
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ERR_OVERFLOW,
    )


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
        return GPU_SEC_PERF_ERR_POLICY_GUARD, GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY, row_allowed

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


def _row_gate_policy_snapshot_digest_q64(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
) -> tuple[int, int]:
    status_gate, gate_reason_code, row_allowed = _row_gate_checked(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    if status_gate != GPU_SEC_PERF_OK:
        return status_gate, 0

    tuple_values = [
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        gate_reason_code,
        row_allowed,
    ]

    digest_q64 = 1469598103934665603
    stride = 104729
    for idx, value in enumerate(tuple_values):
        offset_value = _add_checked(value, idx + 1)
        if offset_value is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0

        mixed_term = _mul_checked(offset_value, stride)
        if mixed_term is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0

        digest_next = _add_checked(digest_q64, mixed_term)
        if digest_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0
        digest_q64 = digest_next

        stride_next = _add_checked(stride, 104729)
        if stride_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0
        stride = stride_next

    return GPU_SEC_PERF_OK, digest_q64


def _row_gate_policy_snapshot_digest_q64_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
) -> tuple[int, int]:
    status_primary, digest_primary = _row_gate_policy_snapshot_digest_q64(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    status_parity, digest_parity = _row_gate_policy_snapshot_digest_q64(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )

    if status_primary != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0
    if digest_primary != digest_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0
    if status_primary != GPU_SEC_PERF_OK:
        return status_primary, 0
    return GPU_SEC_PERF_OK, digest_primary


def _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_snapshot_digest_q64: int,
) -> tuple[int, int]:
    status_commit_only, staged_digest = _row_gate_policy_snapshot_digest_q64_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    status_canonical, canonical_digest = _row_gate_policy_snapshot_digest_q64(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )

    if status_commit_only != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if staged_digest != canonical_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    return status_commit_only, current_snapshot_digest_q64


def _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_snapshot_digest_q64: int,
) -> tuple[int, int]:
    status_commit_only, parity_digest = _row_gate_policy_snapshot_digest_q64_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    status_preflight_only, staged_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        parity_digest,
    )

    if status_commit_only != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if parity_digest != staged_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    return status_commit_only, current_snapshot_digest_q64


def _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_snapshot_digest_q64: int,
) -> tuple[int, int]:
    status_parity, parity_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        0,
    )
    status_preflight_only, staged_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        parity_digest,
    )

    if status_parity != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if parity_digest != staged_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, current_snapshot_digest_q64
    return GPU_SEC_PERF_OK, parity_digest


def _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_snapshot_digest_q64: int,
) -> tuple[int, int]:
    status_parity, parity_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        0,
    )
    status_commit_only, staged_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        parity_digest,
    )

    if status_parity != status_commit_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if parity_digest != staged_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    return status_commit_only, current_snapshot_digest_q64


def _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_snapshot_digest_q64: int,
) -> tuple[int, int]:
    status_preflight_only, staged_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        0,
    )

    status_commit_only, canonical_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        staged_digest,
    )

    if status_preflight_only != status_commit_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if staged_digest != canonical_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    return status_preflight_only, current_snapshot_digest_q64


def _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_snapshot_digest_q64: int,
) -> tuple[int, int]:
    status_parity, parity_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        0,
    )

    status_preflight_only, staged_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        parity_digest,
    )

    if status_parity != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if parity_digest != staged_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, current_snapshot_digest_q64
    return GPU_SEC_PERF_OK, parity_digest


def gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_snapshot_digest_q64: int,
    has_null_output: bool = False,
    inject_digest_drift: bool = False,
    inject_status_invalid_both: bool = False,
) -> tuple[int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_snapshot_digest_q64

    status_commit_only, commit_only_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        0,
    )

    status_preflight_only, staged_digest = _row_gate_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        commit_only_digest,
    )

    if inject_digest_drift:
        staged_digest += 1

    if inject_status_invalid_both:
        status_commit_only = 99
        status_preflight_only = 99

    if not _status_is_valid(status_commit_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if not _status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64

    if status_preflight_only != status_commit_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if staged_digest != commit_only_digest:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    return status_preflight_only, current_snapshot_digest_q64


def test_source_contains_iq1562_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in src
    assert "status_commit_only = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in src
    assert "// IQ-1562 strict diagnostics parity gate:" in src
    assert "saved_snapshot_digest_q64" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_commit_only))" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_preflight_only))" in src
    assert "if (status_preflight_only != status_commit_only)" in src
    assert "*out_snapshot_digest_q64 = saved_snapshot_digest_q64;" in src


def test_secure_on_overhead_budget_vectors() -> None:
    vectors = [
        (32, 1, GPU_SEC_PERF_QUANT_Q4_0),
        (128, 2, GPU_SEC_PERF_QUANT_Q8_0),
        (512, 4, GPU_SEC_PERF_QUANT_Q4_0),
        (1024, 8, GPU_SEC_PERF_QUANT_Q8_0),
    ]

    canonical_digests: list[int] = []
    for prompt_tokens, batch_size, quant_profile in vectors:
        canonical_status, canonical_digest = _row_gate_policy_snapshot_digest_q64_commit_only(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            row_prompt_tokens=prompt_tokens,
            row_batch_size=batch_size,
            row_quant_profile=quant_profile,
        )
        assert canonical_status == GPU_SEC_PERF_OK
        canonical_digests.append(canonical_digest)

        status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            row_prompt_tokens=prompt_tokens,
            row_batch_size=batch_size,
            row_quant_profile=quant_profile,
            current_snapshot_digest_q64=7000,
        )
        assert (status, digest) == (GPU_SEC_PERF_OK, 7000)

    assert len(set(canonical_digests)) == len(vectors)


def test_gate_missing_vectors() -> None:
    vectors = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for idx, vector in enumerate(vectors):
        iommu_active, hooks, parity = vector
        seed = 9200 + idx
        status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=hooks,
            policy_digest_parity=parity,
            row_prompt_tokens=64,
            row_batch_size=1,
            row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
            current_snapshot_digest_q64=seed,
        )
        assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
        assert digest == seed


def test_reason_code_parity_vectors() -> None:
    vectors = [
        (0, 1, 1, GPU_SEC_PERF_ROW_GATE_REASON_IOMMU_GUARD),
        (1, 0, 1, GPU_SEC_PERF_ROW_GATE_REASON_BOOK_GUARD),
        (1, 1, 0, GPU_SEC_PERF_ROW_GATE_REASON_POLICY_DIGEST_MISMATCH),
    ]
    for idx, (iommu_active, hooks, parity, expected_reason) in enumerate(vectors):
        gate_status, gate_reason, row_allowed = _row_gate_checked(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=hooks,
            policy_digest_parity=parity,
            row_prompt_tokens=96,
            row_batch_size=2,
            row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        )
        assert gate_status == GPU_SEC_PERF_ERR_POLICY_GUARD
        assert gate_reason == expected_reason
        assert row_allowed == 0

        seed = 9500 + idx
        status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=hooks,
            policy_digest_parity=parity,
            row_prompt_tokens=96,
            row_batch_size=2,
            row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
            current_snapshot_digest_q64=seed,
        )
        assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
        assert digest == seed


def test_digest_bit_flip_sensitivity() -> None:
    base_status, base_digest = _row_gate_policy_snapshot_digest_q64_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=256,
        row_batch_size=8,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
    )
    flip_status, flip_digest = _row_gate_policy_snapshot_digest_q64_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=257,
        row_batch_size=8,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
    )
    assert base_status == GPU_SEC_PERF_OK
    assert flip_status == GPU_SEC_PERF_OK
    assert base_digest != flip_digest

    status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=256,
        row_batch_size=8,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_snapshot_digest_q64=9000,
    )
    assert (status, digest) == (GPU_SEC_PERF_OK, 9000)

    status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=257,
        row_batch_size=8,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_snapshot_digest_q64=9000,
    )
    assert (status, digest) == (GPU_SEC_PERF_OK, 9000)


def test_overflow_parity_and_null_vectors() -> None:
    status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=GPU_SEC_PERF_I64_MAX,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_snapshot_digest_q64=7400,
    )
    assert (status, digest) == (GPU_SEC_PERF_ERR_OVERFLOW, 7400)

    status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=64,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_snapshot_digest_q64=7300,
        has_null_output=True,
    )
    assert (status, digest) == (GPU_SEC_PERF_ERR_NULL_PTR, 7300)


def test_drift_rejected() -> None:
    status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=192,
        row_batch_size=4,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_snapshot_digest_q64=9900,
        inject_digest_drift=True,
    )
    assert (status, digest) == (GPU_SEC_PERF_ERR_BAD_PARAM, 9900)


def test_invalid_status_domain_rejected() -> None:
    status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=192,
        row_batch_size=4,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_snapshot_digest_q64=8800,
        inject_status_invalid_both=True,
    )
    assert (status, digest) == (GPU_SEC_PERF_ERR_BAD_PARAM, 8800)


if __name__ == "__main__":
    test_source_contains_iq1562_symbols()
    test_secure_on_overhead_budget_vectors()
    test_gate_missing_vectors()
    test_reason_code_parity_vectors()
    test_digest_bit_flip_sensitivity()
    test_overflow_parity_and_null_vectors()
    test_drift_rejected()
    test_invalid_status_domain_rejected()
    print("ok")
