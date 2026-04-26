#!/usr/bin/env python3
"""Harness for IQ-1679 secure-local fast-path snapshot digest gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = Path("tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate.py")
_BASE_SPEC = importlib.util.spec_from_file_location("iq1639_models", _BASE_PATH)
assert _BASE_SPEC is not None and _BASE_SPEC.loader is not None
_BASE_MOD = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE_MOD)

GPU_SEC_PERF_OK = _BASE_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = _BASE_MOD.GPU_SEC_PERF_ERR_POLICY_GUARD
GPU_SEC_PERF_ERR_OVERFLOW = _BASE_MOD.GPU_SEC_PERF_ERR_OVERFLOW
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW = _BASE_MOD.GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD = _BASE_MOD.GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
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


def _snapshot_digest_q64(tuple_values: list[int]) -> int | None:
    digest_q64 = 1469598103934665603
    stride = 104729
    for idx, value in enumerate(tuple_values):
        offset_value = _add_checked(value, idx + 1)
        if offset_value is None:
            return None
        mixed_term = _mul_checked(offset_value, stride)
        if mixed_term is None:
            return None
        digest_next = _add_checked(digest_q64, mixed_term)
        if digest_next is None:
            return None
        digest_q64 = digest_next
        stride_next = _add_checked(stride, 104729)
        if stride_next is None:
            return None
        stride = stride_next
    return digest_q64


def fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_iq1679(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    p50_overhead_q16: int,
    p95_overhead_q16: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    mutate_snapshot_tamper: bool = False,
    replay_digest_tamper: bool = False,
) -> tuple[int, int, int, int]:
    status_primary, staged_enabled, staged_reason = _BASE_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_iq1639(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        p50_overhead_q16,
        p95_overhead_q16,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        caller_fast_path_enabled=0,
        caller_disable_reason=GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
    )
    if status_primary != GPU_SEC_PERF_OK:
        return status_primary, staged_enabled, staged_reason, 0

    tuple_values = [
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        dispatch_transcript_parity,
        p50_overhead_q16,
        p95_overhead_q16,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
    ]
    digest_primary = _snapshot_digest_q64(tuple_values)
    if digest_primary is None:
        return GPU_SEC_PERF_ERR_OVERFLOW, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0

    parity_tuple = list(tuple_values)
    if mutate_snapshot_tamper:
        parity_tuple[4] += 1
    digest_parity = _snapshot_digest_q64(parity_tuple)
    if digest_parity is None:
        return GPU_SEC_PERF_ERR_OVERFLOW, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0
    if replay_digest_tamper:
        digest_parity += 1

    if digest_primary != digest_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0
    return GPU_SEC_PERF_OK, staged_enabled, staged_reason, digest_primary


def test_source_contains_iq1679_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64Checked(" in src
    assert "tuple_values[9]" in src
    assert "status_primary = GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateChecked(" in src
    assert "if (digest_primary != digest_parity)" in src
    assert "out_snapshot_digest_q64" in src


def test_gate_missing_vectors() -> None:
    status, enabled, reason, digest_q64 = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_iq1679(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=0,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=120,
            p95_overhead_q16=240,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
        )
    )
    assert (status, enabled, reason, digest_q64) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        0,
        _BASE_MOD.GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_IOMMU_GUARD,
        0,
    )

    status, enabled, reason, digest_q64 = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_iq1679(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=0,
            p50_overhead_q16=120,
            p95_overhead_q16=240,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
        )
    )
    assert (status, enabled, reason, digest_q64) == (
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        0,
        _BASE_MOD.GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH,
        0,
    )


def test_mutation_and_replay_tamper_vectors() -> None:
    status, enabled, reason, digest_q64 = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_iq1679(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=120,
            p95_overhead_q16=240,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
            mutate_snapshot_tamper=True,
        )
    )
    assert (status, enabled, reason, digest_q64) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        0,
    )

    status, enabled, reason, digest_q64 = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_iq1679(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=120,
            p95_overhead_q16=240,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
            replay_digest_tamper=True,
        )
    )
    assert (status, enabled, reason, digest_q64) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        0,
    )


def test_deterministic_secure_on_parity_vectors() -> None:
    first = fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_iq1679(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=128,
        p95_overhead_q16=256,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
    )
    second = fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_iq1679(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        p50_overhead_q16=128,
        p95_overhead_q16=256,
        max_p50_overhead_q16=500,
        max_p95_overhead_q16=700,
    )

    assert first[0:3] == (GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW)
    assert first[3] > 0
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1679_symbols()
    test_gate_missing_vectors()
    test_mutation_and_replay_tamper_vectors()
    test_deterministic_secure_on_parity_vectors()
    print("ok")
