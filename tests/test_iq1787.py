#!/usr/bin/env python3
"""Harness for IQ-1787 short-name zero-write diagnostics companion."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_iq1785 as iq1785
import test_iq1786 as iq1786
import test_gpu_security_perf_fast_path_switch_batch_audit_q64_iq1782 as iq1782

GPU_SEC_PERF_OK = iq1782.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = iq1782.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = iq1782.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

Scenario = iq1782.Scenario
HC_FN_IQ1787 = "GPUSecPerfIQ1787"


def _status_is_valid(status: int) -> bool:
    return status in (GPU_SEC_PERF_OK, GPU_SEC_PERF_ERR_BAD_PARAM)


def _tuple_ok(values: tuple[int, int, int, int], count: int) -> bool:
    return (
        values[0] >= 0
        and values[1] >= 0
        and values[2] >= 0
        and values[3] > 0
        and (values[0] + values[1] + values[2] == count)
    )


def iq1787_model(
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

    status_companion, staged_tuple = iq1786.iq1786_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        force_digest_drift_primary=force_digest_drift_companion,
        force_status_domain_drift_primary=force_status_domain_drift_companion,
    )

    status_canonical, canonical_tuple = iq1785.iq1785_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        initial_out=staged_tuple,
        force_digest_drift_companion=force_digest_drift_canonical,
        force_status_domain_drift_companion=force_status_domain_drift_canonical,
    )

    if not _status_is_valid(status_companion):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_companion == GPU_SEC_PERF_OK and not _tuple_ok(staged_tuple, len(scenarios)):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if status_canonical == GPU_SEC_PERF_OK and not _tuple_ok(canonical_tuple, len(scenarios)):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_companion != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_tuple != canonical_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    return status_companion, saved_out


def test_source_contains_iq1787_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert f"I32 {HC_FN_IQ1787}(" in src
    assert "#define IQ1787_COMP_FN GPUSecPerfIQ1786" in src
    assert "#define IQ1787_CANON_FN GPUSecPerfIQ1785" in src
    assert "status_companion = IQ1787_COMP_FN(" in src
    assert "status_canonical = IQ1787_CANON_FN(" in src
    assert "// IQ-1787 zero-write diagnostics companion:" in src


def test_gate_missing_vector() -> None:
    saved_out = (7101, 7102, 7103, 7104)
    status, out_values = iq1787_model(
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
    saved_out = (7201, 7202, 7203, 7204)
    status, out_values = iq1787_model(
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
    saved_out = (7301, 7302, 7303, 7304)
    status, out_values = iq1787_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=saved_out,
        force_status_domain_drift_companion=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == saved_out


def test_deterministic_secure_on_nowrite_vector() -> None:
    scenarios = [
        Scenario(8, 16, 32, 64),
        Scenario(12, 24, 10, 22),
        Scenario(14, 28, 56, 96),
    ]
    saved_out = (7401, 7402, 7403, 7404)

    first = iq1787_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=saved_out,
    )
    second = iq1787_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=saved_out,
    )

    assert _status_is_valid(first[0])
    assert first[1] == saved_out
    assert first == second


if __name__ == "__main__":
    test_source_contains_iq1787_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_secure_on_nowrite_vector()
    print("ok")
