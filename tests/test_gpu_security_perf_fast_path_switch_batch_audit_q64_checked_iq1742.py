#!/usr/bin/env python3
"""Harness for IQ-1742 fast-path batch audit checked aggregate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = 1

GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW = 0
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD = 1
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_IOMMU_GUARD = 2
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_BOOK_GUARD = 3
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_POLICY_DIGEST_MISMATCH = 4
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH = 5
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH = 6

HC_FN_IQ1742 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64Checked"
HC_FN_CROSS_GATE = "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateChecked"


def _status_is_valid(status_code: int) -> bool:
    return status_code in {0, 1, 2, 3, 4, 5}


def _reason_is_valid(reason_code: int) -> bool:
    return (
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW
        <= reason_code
        <= GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH
    )


@dataclass(frozen=True)
class Scenario:
    p50_overhead_q16: int
    p95_overhead_q16: int
    max_p50_overhead_q16: int
    max_p95_overhead_q16: int


def _cross_gate_model(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    scenario: Scenario,
) -> tuple[int, int, int]:
    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
    if iommu_active != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_IOMMU_GUARD
    if book_of_truth_gpu_hooks != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_BOOK_GUARD
    if policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_POLICY_DIGEST_MISMATCH
    if dispatch_transcript_parity != 1:
        return (
            GPU_SEC_PERF_ERR_POLICY_GUARD,
            0,
            GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_DISPATCH_TRANSCRIPT_MISMATCH,
        )

    if (
        scenario.p50_overhead_q16 < 0
        or scenario.p95_overhead_q16 < 0
        or scenario.max_p50_overhead_q16 < 0
        or scenario.max_p95_overhead_q16 < 0
        or scenario.p95_overhead_q16 < scenario.p50_overhead_q16
        or scenario.max_p95_overhead_q16 < scenario.max_p50_overhead_q16
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD

    if (
        scenario.p50_overhead_q16 > scenario.max_p50_overhead_q16
        or scenario.p95_overhead_q16 > scenario.max_p95_overhead_q16
    ):
        return (
            GPU_SEC_PERF_ERR_POLICY_GUARD,
            0,
            GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH,
        )

    return GPU_SEC_PERF_OK, 1, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW


def iq1742_model(
    scenarios: list[Scenario],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    *,
    initial_out: tuple[int, int, int, int] = (701, 809, 907, 1009),
    force_digest_drift: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    out_ok, out_blocked, out_drift, out_digest = initial_out
    saved_out = (out_ok, out_blocked, out_drift, out_digest)

    if not scenarios:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    staged_ok = 0
    staged_blocked = 0
    staged_drift = 0
    digest_primary = 1469598103934665603
    digest_parity = 1469598103934665603
    stride = 104729

    for i, scenario in enumerate(scenarios):
        status, enabled, reason = _cross_gate_model(
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            dispatch_transcript_parity=dispatch_transcript_parity,
            scenario=scenario,
        )

        if not _status_is_valid(status):
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
        if enabled not in (0, 1):
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
        if not _reason_is_valid(reason):
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

        if status == GPU_SEC_PERF_OK and enabled == 1 and reason == GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW:
            staged_ok += 1
            row_bucket = 0
        elif status == GPU_SEC_PERF_ERR_POLICY_GUARD and enabled == 0 and reason != GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW:
            staged_blocked += 1
            row_bucket = 1
        else:
            staged_drift += 1
            row_bucket = 2

        tuple_fold = (
            (i + 1)
            + secure_local_mode
            + iommu_active
            + book_of_truth_gpu_hooks
            + policy_digest_parity
            + dispatch_transcript_parity
            + enabled
            + reason
            + row_bucket
        )
        digest_primary += tuple_fold * stride
        digest_parity += tuple_fold * stride
        stride += 104729

    if force_digest_drift:
        digest_parity += 1

    if digest_primary != digest_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_ok + staged_blocked + staged_drift != len(scenarios):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    return GPU_SEC_PERF_OK, (staged_ok, staged_blocked, staged_drift, digest_primary)


def test_source_contains_iq1742_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert f"I32 {HC_FN_IQ1742}(" in src
    assert f"status_row = {HC_FN_CROSS_GATE}(" in src
    assert "if (digest_primary != digest_parity)" in src
    assert "if (i != scenario_count)" in src
    assert "// IQ-1742 batch audit:" in src


def test_gate_missing_vector() -> None:
    status, out_values = iq1742_model(
        scenarios=[
            Scenario(8, 16, 32, 64),
            Scenario(10, 20, 40, 80),
            Scenario(12, 24, 48, 96),
        ],
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


def test_injected_digest_drift_vector() -> None:
    status, out_values = iq1742_model(
        scenarios=[
            Scenario(8, 16, 32, 64),
            Scenario(10, 20, 40, 80),
            Scenario(12, 24, 48, 96),
        ],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        force_digest_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (701, 809, 907, 1009)


def test_deterministic_aggregate_parity_vector() -> None:
    scenarios = [
        Scenario(8, 16, 32, 64),
        Scenario(12, 24, 10, 22),
        Scenario(-1, 2, 3, 4),
        Scenario(14, 28, 56, 96),
    ]

    first = iq1742_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )
    second = iq1742_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )

    assert first[0] == GPU_SEC_PERF_OK
    assert first[1][0] == 2
    assert first[1][1] == 1
    assert first[1][2] == 1
    assert first[1][3] > 0
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1742_symbols()
    test_gate_missing_vector()
    test_injected_digest_drift_vector()
    test_deterministic_aggregate_parity_vector()
    print("ok")
