#!/usr/bin/env python3
"""Harness for IQ-1784 fast-path batch audit commit-only hardening wrapper."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1781 as iq1781
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1782 as iq1782

GPU_SEC_PERF_OK = iq1782.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = iq1782.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = iq1782.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

Scenario = iq1782.Scenario

HC_FN_IQ1784 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly"
HC_FN_PRIMARY = iq1781.HC_FN_IQ1781
HC_FN_PARITY = "GPUSecurityPerfFastPathSwitchBatchAuditQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly"


def _status_is_valid(status: int) -> bool:
    return status in (GPU_SEC_PERF_OK, GPU_SEC_PERF_ERR_BAD_PARAM)


def iq1784_model(
    scenarios: list[Scenario],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    *,
    force_digest_drift_seed: bool = False,
    force_status_domain_drift_seed: bool = False,
    force_digest_drift_primary: bool = False,
    force_status_domain_drift_primary: bool = False,
    force_digest_drift_parity: bool = False,
    force_status_domain_drift_parity: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    if not scenarios:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    status_seed, staged_tuple = iq1782.iq1782_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        force_digest_drift_parity=force_digest_drift_seed,
        force_status_domain_drift_primary=force_status_domain_drift_seed,
    )
    seed_tuple = staged_tuple
    if force_status_domain_drift_seed:
        status_seed = 77

    status_primary, staged_tuple = iq1781.iq1781_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        initial_out=seed_tuple,
        force_digest_drift_companion=force_digest_drift_primary,
        force_status_domain_drift_companion=force_status_domain_drift_primary,
    )
    if force_status_domain_drift_primary:
        status_primary = 88

    status_parity, parity_tuple = iq1782.iq1782_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        force_digest_drift_parity=force_digest_drift_parity,
        force_status_domain_drift_primary=force_status_domain_drift_parity,
    )
    if force_status_domain_drift_parity:
        status_parity = 99

    if not _status_is_valid(status_seed):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if not _status_is_valid(status_primary):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_seed == GPU_SEC_PERF_OK:
        if seed_tuple[0] < 0 or seed_tuple[1] < 0 or seed_tuple[2] < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if seed_tuple[0] + seed_tuple[1] + seed_tuple[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if seed_tuple[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_primary == GPU_SEC_PERF_OK:
        if staged_tuple[0] < 0 or staged_tuple[1] < 0 or staged_tuple[2] < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if staged_tuple[0] + staged_tuple[1] + staged_tuple[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if staged_tuple[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_parity == GPU_SEC_PERF_OK:
        if parity_tuple[0] < 0 or parity_tuple[1] < 0 or parity_tuple[2] < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if parity_tuple[0] + parity_tuple[1] + parity_tuple[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if parity_tuple[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_seed != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if seed_tuple != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if status_primary != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if staged_tuple != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_primary != GPU_SEC_PERF_OK:
        return status_primary, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, staged_tuple


def test_source_contains_iq1784_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert f"I32 {HC_FN_IQ1784}(" in src
    assert f"status_seed = {HC_FN_PARITY}(" in src
    assert f"status_primary = {HC_FN_PRIMARY}(" in src
    assert f"status_parity = {HC_FN_PARITY}(" in src
    assert "if (status_seed != status_parity)" in src
    assert "if (status_seed == GPU_SEC_PERF_OK)" in src
    assert "// IQ-1784 commit-only hardening wrapper:" in src


def test_gate_missing_vector() -> None:
    status, out_values = iq1784_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (0, 0, 0, 0)


def test_digest_drift_vector() -> None:
    status, out_values = iq1784_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80), Scenario(12, 24, 48, 96)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        force_digest_drift_parity=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (0, 0, 0, 0)


def test_status_domain_drift_vector() -> None:
    status, out_values = iq1784_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        force_status_domain_drift_primary=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (0, 0, 0, 0)


def test_deterministic_tuple_parity_vector() -> None:
    scenarios = [
        Scenario(7, 14, 20, 40),
        Scenario(9, 18, 27, 45),
        Scenario(11, 22, 33, 55),
    ]

    first = iq1784_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )
    second = iq1784_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )

    assert first[0] == GPU_SEC_PERF_ERR_BAD_PARAM
    assert first == second


if __name__ == "__main__":
    test_source_contains_iq1784_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
