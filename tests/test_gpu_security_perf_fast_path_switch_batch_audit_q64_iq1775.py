#!/usr/bin/env python3
"""Harness for IQ-1775 batch-audit commit-only hardening wrapper."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1761 as iq1761
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1774 as iq1774

GPU_SEC_PERF_OK = iq1761.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = iq1761.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = iq1761.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

Scenario = iq1761.Scenario

HC_FN_IQ1775 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly"
HC_FN_PRIMARY = iq1774.HC_FN_IQ1774
HC_FN_PARITY = iq1761.HC_FN_IQ1761
HC_FN_SEED = iq1761.HC_FN_IQ1761


def iq1775_model(
    scenarios: list[Scenario],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    *,
    force_digest_drift_primary: bool = False,
    force_status_domain_drift_parity: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    if not scenarios:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    status_seed, seeded_tuple = iq1761.iq1761_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        initial_out=(0, 0, 0, 0),
    )

    if not iq1774._status_is_valid(status_seed):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    status_primary, primary_out = iq1774.iq1774_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        initial_out=seeded_tuple,
        force_digest_drift_primary=force_digest_drift_primary,
    )

    status_parity, parity_out = iq1761.iq1761_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        initial_out=primary_out,
        force_status_domain_drift_companion=force_status_domain_drift_parity,
    )

    if not iq1774._status_is_valid(status_primary):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if not iq1774._status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_primary == GPU_SEC_PERF_OK:
        if primary_out[0] < 0 or primary_out[1] < 0 or primary_out[2] < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if primary_out[0] + primary_out[1] + primary_out[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if primary_out[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_parity == GPU_SEC_PERF_OK:
        if parity_out[0] < 0 or parity_out[1] < 0 or parity_out[2] < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if parity_out[0] + parity_out[1] + parity_out[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if parity_out[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_primary != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if primary_out != parity_out:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_primary != GPU_SEC_PERF_OK:
        return status_primary, (0, 0, 0, 0)
    return GPU_SEC_PERF_OK, primary_out


def test_source_contains_iq1775_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert f"I32 {HC_FN_IQ1775}(" in src
    assert f"status_seed = {HC_FN_SEED}(" in src
    assert f"status_primary = {HC_FN_PRIMARY}(" in src
    assert f"status_parity = {HC_FN_PARITY}(" in src
    assert "// IQ-1775 commit-only hardening wrapper:" in src


def test_gate_missing_vector() -> None:
    status, out_values = iq1775_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80), Scenario(12, 24, 48, 96)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (0, 0, 0, 0)


def test_digest_drift_vector() -> None:
    status, out_values = iq1775_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        force_digest_drift_primary=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (0, 0, 0, 0)


def test_status_domain_drift_vector() -> None:
    status, out_values = iq1775_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        force_status_domain_drift_parity=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (0, 0, 0, 0)


def test_deterministic_tuple_parity_vector() -> None:
    scenarios = [
        Scenario(8, 16, 32, 64),
        Scenario(12, 24, 10, 22),
        Scenario(-1, 2, 3, 4),
        Scenario(14, 28, 56, 96),
    ]

    first = iq1775_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )
    second = iq1775_model(
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
    test_source_contains_iq1775_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
