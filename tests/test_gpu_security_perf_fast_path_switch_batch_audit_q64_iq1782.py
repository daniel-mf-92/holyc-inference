#!/usr/bin/env python3
"""Harness for IQ-1782 fast-path batch audit commit-only hardening refresh."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1780 as iq1780

GPU_SEC_PERF_OK = iq1780.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = iq1780.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = iq1780.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

Scenario = iq1780.Scenario

HC_FN_IQ1782 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly"


def iq1782_model(
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
    # IQ-1782 is a hardening refresh over the same HolyC symbol as IQ-1780.
    return iq1780.iq1780_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        force_digest_drift_parity=force_digest_drift_parity,
        force_status_domain_drift_primary=force_status_domain_drift_primary,
    )


def test_source_contains_iq1782_hardening_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert f"I32 {HC_FN_IQ1782}(" in src
    assert "// IQ-1780/IQ-1782 commit-only hardening wrapper:" in src
    assert "if (status_seed != status_parity)" in src


def test_gate_missing_vector() -> None:
    status, out_values = iq1782_model(
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
    status, out_values = iq1782_model(
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
    status, out_values = iq1782_model(
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

    first = iq1782_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )
    second = iq1782_model(
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
    test_source_contains_iq1782_hardening_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
