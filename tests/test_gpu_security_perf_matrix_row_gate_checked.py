#!/usr/bin/env python3
"""Harness for IQ-1456 secure-on GPU per-row gate checks."""

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


def test_source_contains_iq1456_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixRowGateChecked(" in src
    assert "GPU_SEC_PERF_ROW_GATE_REASON_POLICY_DIGEST_MISMATCH" in src
    assert "book_of_truth_gpu_hooks" in src
    assert "policy_digest_parity" in src
    assert "out_gate_reason_code" in src
    assert "out_row_allowed" in src


def test_deterministic_reason_code_vectors() -> None:
    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode=GPU_SEC_PERF_PROFILE_DEV_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=32,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
    )
    assert (status, reason, allowed) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ROW_GATE_REASON_PROFILE_GUARD,
        0,
    )

    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=32,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
    )
    assert (status, reason, allowed) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ROW_GATE_REASON_IOMMU_GUARD,
        0,
    )

    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=0,
        policy_digest_parity=1,
        row_prompt_tokens=32,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
    )
    assert (status, reason, allowed) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ROW_GATE_REASON_BOOK_GUARD,
        0,
    )

    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=0,
        row_prompt_tokens=32,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
    )
    assert (status, reason, allowed) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ROW_GATE_REASON_POLICY_DIGEST_MISMATCH,
        0,
    )


def test_hardening_boundary_vectors() -> None:
    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=0,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
    )
    assert (status, reason, allowed) == (
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ROW_GATE_REASON_ALLOW,
        1,
    )

    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=-1,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
    )
    assert (status, reason, allowed) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY,
        0,
    )

    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=16,
        row_batch_size=0,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
    )
    assert (status, reason, allowed) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY,
        0,
    )

    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=16,
        row_batch_size=1,
        row_quant_profile=13,
    )
    assert (status, reason, allowed) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ROW_GATE_REASON_BAD_QUANT_PROFILE,
        0,
    )

    status, reason, allowed = gpu_security_perf_matrix_row_gate_checked(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=2,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=16,
        row_batch_size=1,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert reason == GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY
    assert allowed == 0


if __name__ == "__main__":
    test_source_contains_iq1456_symbols()
    test_deterministic_reason_code_vectors()
    test_hardening_boundary_vectors()
    print("ok")
