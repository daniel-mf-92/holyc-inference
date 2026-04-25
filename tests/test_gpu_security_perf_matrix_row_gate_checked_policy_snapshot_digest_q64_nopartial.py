#!/usr/bin/env python3
"""Harness for IQ-1473 no-partial row-gate policy snapshot digest wrapper."""

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


def gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_nopartial(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_snapshot_digest_q64: int,
    has_null_output: bool = False,
) -> tuple[int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_snapshot_digest_q64

    status_preflight, digest_preflight = _row_gate_policy_snapshot_digest_q64_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )
    status_commit, digest_commit = _row_gate_policy_snapshot_digest_q64_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
    )

    if status_preflight != status_commit:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if digest_preflight != digest_commit:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_snapshot_digest_q64
    if status_preflight != GPU_SEC_PERF_OK:
        return status_preflight, current_snapshot_digest_q64
    return GPU_SEC_PERF_OK, digest_preflight


def test_source_contains_iq1473_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64NoPartial(" in src
    assert "status_preflight = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64CommitOnly(" in src
    assert "status_commit = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64CommitOnly(" in src
    assert "saved_snapshot_digest_q64" in src
    assert "staged_snapshot_digest_q64" in src
    assert "commit_snapshot_digest_q64" in src


def test_null_and_capacity_like_vectors() -> None:
    status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_nopartial(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=32,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_snapshot_digest_q64=777,
        has_null_output=True,
    )
    assert (status, digest) == (GPU_SEC_PERF_ERR_NULL_PTR, 777)

    status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_nopartial(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=32,
        row_batch_size=0,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_snapshot_digest_q64=888,
    )
    assert (status, digest) == (GPU_SEC_PERF_ERR_BAD_PARAM, 888)


def test_gate_missing_vectors_preserve_output() -> None:
    status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_nopartial(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=64,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_snapshot_digest_q64=191,
    )
    assert (status, digest) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 191)

    status, digest = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_nopartial(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=0,
        policy_digest_parity=1,
        row_prompt_tokens=64,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_snapshot_digest_q64=192,
    )
    assert (status, digest) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 192)


def test_bit_flip_sensitivity_vectors() -> None:
    status_base, digest_base = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_nopartial(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=256,
        row_batch_size=8,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_snapshot_digest_q64=0,
    )
    assert status_base == GPU_SEC_PERF_OK

    status_prompt, digest_prompt = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_nopartial(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=257,
        row_batch_size=8,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_snapshot_digest_q64=0,
    )
    status_quant, digest_quant = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64_nopartial(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=256,
        row_batch_size=8,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_snapshot_digest_q64=0,
    )

    assert status_prompt == GPU_SEC_PERF_OK
    assert status_quant == GPU_SEC_PERF_OK
    assert digest_base != digest_prompt
    assert digest_base != digest_quant


if __name__ == "__main__":
    test_source_contains_iq1473_symbols()
    test_null_and_capacity_like_vectors()
    test_gate_missing_vectors_preserve_output()
    test_bit_flip_sensitivity_vectors()
