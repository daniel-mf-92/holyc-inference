#!/usr/bin/env python3
"""Harness for IQ-1781 batch-audit zero-write diagnostics companion."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1780 as iq1780
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1779 as iq1779

GPU_SEC_PERF_OK = iq1780.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = iq1780.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = iq1780.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

Scenario = iq1780.Scenario

HC_FN_IQ1781 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
HC_FN_COMPANION = iq1780.HC_FN_IQ1780
HC_FN_CANONICAL = iq1779.HC_FN_IQ1779


def _status_is_valid(status: int) -> bool:
    return status in (GPU_SEC_PERF_OK, GPU_SEC_PERF_ERR_BAD_PARAM)


def iq1781_model(
    scenarios: list[Scenario],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    *,
    initial_out: tuple[int, int, int, int],
    force_digest_drift_companion: bool = False,
    force_status_domain_drift_companion: bool = False,
    force_digest_drift_canonical: bool = False,
    force_status_domain_drift_canonical: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    saved_out = initial_out

    if not scenarios:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    status_companion, staged_tuple = iq1780.iq1780_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        force_digest_drift_parity=force_digest_drift_companion,
        force_status_domain_drift_primary=force_status_domain_drift_companion,
    )

    status_canonical, canonical_tuple = iq1779.iq1779_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        initial_out=staged_tuple,
        force_digest_drift_primary=force_digest_drift_canonical,
        force_status_domain_drift_canonical=force_status_domain_drift_canonical,
    )

    if not _status_is_valid(status_companion):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_companion == GPU_SEC_PERF_OK:
        if staged_tuple[0] < 0 or staged_tuple[1] < 0 or staged_tuple[2] < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
        if staged_tuple[0] + staged_tuple[1] + staged_tuple[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
        if staged_tuple[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_canonical == GPU_SEC_PERF_OK:
        if canonical_tuple[0] < 0 or canonical_tuple[1] < 0 or canonical_tuple[2] < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
        if canonical_tuple[0] + canonical_tuple[1] + canonical_tuple[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
        if canonical_tuple[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_companion != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_tuple != canonical_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    return status_companion, saved_out


def test_source_contains_iq1781_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert f"I32 {HC_FN_IQ1781}(" in src
    assert f"status_companion = {HC_FN_COMPANION}(" in src
    assert f"status_canonical = {HC_FN_CANONICAL}(" in src
    assert "if (status_companion != status_canonical)" in src
    assert "// IQ-1781 zero-write diagnostics companion:" in src


def test_gate_missing_vector() -> None:
    saved_out = (5201, 5202, 5203, 5204)
    status, out_values = iq1781_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80), Scenario(12, 24, 48, 96)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=saved_out,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == saved_out


def test_digest_drift_vector() -> None:
    saved_out = (5301, 5302, 5303, 5304)
    status, out_values = iq1781_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=saved_out,
        force_digest_drift_companion=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == saved_out


def test_status_domain_drift_vector() -> None:
    saved_out = (5401, 5402, 5403, 5404)
    status, out_values = iq1781_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=saved_out,
        force_status_domain_drift_canonical=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == saved_out


def test_no_write_parity_vector() -> None:
    saved_out = (5501, 5502, 5503, 5504)
    status, out_values = iq1781_model(
        scenarios=[
            Scenario(8, 16, 32, 64),
            Scenario(12, 24, 10, 22),
            Scenario(-1, 2, 3, 4),
            Scenario(14, 28, 56, 96),
        ],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=saved_out,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == saved_out


def test_deterministic_secure_on_parity_vector() -> None:
    scenarios = [
        Scenario(7, 14, 20, 40),
        Scenario(9, 18, 27, 45),
        Scenario(11, 22, 33, 55),
    ]
    saved_out = (5601, 5602, 5603, 5604)

    first = iq1781_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=saved_out,
    )
    second = iq1781_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=saved_out,
    )

    assert first[0] == GPU_SEC_PERF_ERR_BAD_PARAM
    assert first == second


if __name__ == "__main__":
    test_source_contains_iq1781_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vector()
    test_deterministic_secure_on_parity_vector()
    print("ok")
