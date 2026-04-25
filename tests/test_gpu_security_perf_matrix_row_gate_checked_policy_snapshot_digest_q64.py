#!/usr/bin/env python3
"""Harness for IQ-1471 row-gate policy snapshot digest helper."""

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
GPU_SEC_PERF_ROW_GATE_REASON_PROFILE_GUARD = 1
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


def gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64(
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


def test_source_contains_iq1471_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64(" in src
    assert "tuple_values[9]" in src
    assert "out_snapshot_digest_q64" in src
    assert "GPUSecurityPerfMatrixRowGateChecked(" in src


def test_deterministic_digest_vectors() -> None:
    status_a, digest_a = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=128,
        row_batch_size=4,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
    )
    status_b, digest_b = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=128,
        row_batch_size=4,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
    )

    assert status_a == GPU_SEC_PERF_OK
    assert status_b == GPU_SEC_PERF_OK
    assert digest_a == digest_b


def test_bit_flip_sensitivity_vectors() -> None:
    status_base, digest_base = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=256,
        row_batch_size=8,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
    )
    assert status_base == GPU_SEC_PERF_OK

    status_prompt_flip, digest_prompt_flip = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=257,
        row_batch_size=8,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
    )
    status_batch_flip, digest_batch_flip = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=256,
        row_batch_size=9,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
    )
    status_quant_flip, digest_quant_flip = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=256,
        row_batch_size=8,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
    )

    assert status_prompt_flip == GPU_SEC_PERF_OK
    assert status_batch_flip == GPU_SEC_PERF_OK
    assert status_quant_flip == GPU_SEC_PERF_OK
    assert digest_base != digest_prompt_flip
    assert digest_base != digest_batch_flip
    assert digest_base != digest_quant_flip


def test_overflow_and_gate_boundary_vectors() -> None:
    status_overflow, digest_overflow = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=GPU_SEC_PERF_I64_MAX,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
    )
    assert status_overflow == GPU_SEC_PERF_ERR_OVERFLOW
    assert digest_overflow == 0

    status_gate, digest_gate = gpu_security_perf_matrix_row_gate_checked_policy_snapshot_digest_q64(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=32,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
    )
    assert status_gate == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert digest_gate == 0


if __name__ == "__main__":
    test_source_contains_iq1471_symbols()
    test_deterministic_digest_vectors()
    test_bit_flip_sensitivity_vectors()
    test_overflow_and_gate_boundary_vectors()
    print("ok")
