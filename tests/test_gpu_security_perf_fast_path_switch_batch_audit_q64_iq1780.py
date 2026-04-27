#!/usr/bin/env python3
"""Harness for IQ-1780 fast-path batch audit commit-only hardening wrapper."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1759 as iq1759

GPU_SEC_PERF_OK = iq1759.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = iq1759.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = iq1759.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

Scenario = iq1759.Scenario

HC_FN_IQ1780 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly"
HC_FN_PRIMARY = iq1759.HC_FN_IQ1759
HC_FN_PARITY = iq1759.HC_FN_IQ1758
HC_FN_SEED = iq1759.iq1758.iq1757.HC_FN_IQ1757


def iq1780_model(
    scenarios: list[Scenario],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    *,
    force_digest_drift_parity: bool = False,
    force_status_domain_drift_primary: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    if not scenarios:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    status_seed, seeded_tuple = iq1759.iq1758.iq1757.iq1757_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
    )
    if not iq1759.iq1758.iq1757.iq1756.iq1755.iq1754.iq1753.iq1752.iq1751.iq1750.iq1749.iq1748.iq1747.iq1746._status_is_valid(status_seed):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    status_primary, staged_tuple = iq1759.iq1759_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        initial_out=seeded_tuple,
        force_status_domain_drift_primary=force_status_domain_drift_primary,
    )

    status_parity, parity_tuple = iq1759.iq1758.iq1758_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        initial_out=staged_tuple,
        force_digest_drift_canonical=force_digest_drift_parity,
    )

    if not iq1759.iq1758.iq1757.iq1756.iq1755.iq1754.iq1753.iq1752.iq1751.iq1750.iq1749.iq1748.iq1747.iq1746._status_is_valid(status_primary):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if not iq1759.iq1758.iq1757.iq1756.iq1755.iq1754.iq1753.iq1752.iq1751.iq1750.iq1749.iq1748.iq1747.iq1746._status_is_valid(status_parity):
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

    if status_primary != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if staged_tuple != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_primary != GPU_SEC_PERF_OK:
        return status_primary, (0, 0, 0, 0)

    return GPU_SEC_PERF_OK, staged_tuple


def test_source_contains_iq1780_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert f"I32 {HC_FN_IQ1780}(" in src
    assert f"status_seed = {HC_FN_SEED}(" in src
    assert f"status_primary = {HC_FN_PRIMARY}(" in src
    assert f"status_parity = {HC_FN_PARITY}(" in src
    assert "// IQ-1780 commit-only hardening wrapper:" in src


def test_gate_missing_vector() -> None:
    status, out_values = iq1780_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80), Scenario(12, 24, 48, 96)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )
    assert status == GPU_SEC_PERF_OK
    assert out_values[0] == 0
    assert out_values[1] == 3
    assert out_values[2] == 0
    assert out_values[3] > 0


def test_digest_drift_vector() -> None:
    status, out_values = iq1780_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80)],
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
    status, out_values = iq1780_model(
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
        Scenario(8, 16, 32, 64),
        Scenario(12, 24, 10, 22),
        Scenario(-1, 2, 3, 4),
        Scenario(14, 28, 56, 96),
    ]

    first = iq1780_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )
    second = iq1780_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )

    assert first[0] == GPU_SEC_PERF_OK
    assert first[1] == second[1]


if __name__ == "__main__":
    test_source_contains_iq1780_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
