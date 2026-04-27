#!/usr/bin/env python3
"""Harness for IQ-1776 batch-audit zero-write diagnostics companion."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1774 as iq1774
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1775 as iq1775

GPU_SEC_PERF_OK = iq1774.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = iq1774.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = iq1774.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

Scenario = iq1774.Scenario

HC_FN_IQ1776 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64CheckedCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly"
HC_FN_COMPANION = iq1775.HC_FN_IQ1775
HC_FN_CANONICAL = iq1774.HC_FN_IQ1774


def _status_is_valid(status: int) -> bool:
    return status in (GPU_SEC_PERF_OK, GPU_SEC_PERF_ERR_BAD_PARAM)


def iq1776_model(
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

    status_companion, staged_tuple = iq1775.iq1775_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        force_digest_drift_primary=force_digest_drift_companion,
        force_status_domain_drift_parity=force_status_domain_drift_companion,
    )

    status_canonical, canonical_tuple = iq1774.iq1774_model(
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


def test_source_contains_iq1776_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert f"I32 {HC_FN_IQ1776}(" in src
    assert f"status_companion = {HC_FN_COMPANION}(" in src
    assert f"status_canonical = {HC_FN_CANONICAL}(" in src
    assert "if (status_companion != status_canonical)" in src
    assert "// IQ-1776 zero-write diagnostics companion:" in src


def test_gate_missing_vector() -> None:
    saved_out = (3201, 3202, 3203, 3204)
    status, out_values = iq1776_model(
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
    saved_out = (3301, 3302, 3303, 3304)
    status, out_values = iq1776_model(
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
    saved_out = (3401, 3402, 3403, 3404)
    status, out_values = iq1776_model(
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
    saved_out = (3501, 3502, 3503, 3504)
    status, out_values = iq1776_model(
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
    saved_out = (3601, 3602, 3603, 3604)

    first = iq1776_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=saved_out,
    )
    second = iq1776_model(
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
    test_source_contains_iq1776_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vector()
    test_deterministic_secure_on_parity_vector()
    print("ok")
