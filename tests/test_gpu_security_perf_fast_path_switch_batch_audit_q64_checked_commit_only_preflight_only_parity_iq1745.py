#!/usr/bin/env python3
"""Harness for IQ-1745 fast-path batch audit commit-only preflight-only parity gate."""

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

HC_FN_IQ1745 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64CheckedCommitOnlyPreflightOnlyParity"
HC_FN_IQ1744 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64CheckedCommitOnlyPreflightOnly"
HC_FN_IQ1743 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64CheckedCommitOnly"
HC_FN_IQ1742 = "GPUSecurityPerfFastPathSwitchBatchAuditQ64Checked"


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


def _iq1742_model(
    scenarios: list[Scenario],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    *,
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    if not scenarios:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

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
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if enabled not in (0, 1):
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if not _reason_is_valid(reason):
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

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
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if staged_ok + staged_blocked + staged_drift != len(scenarios):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if force_status_domain_drift:
        return 99, (staged_ok, staged_blocked, staged_drift, digest_primary)

    return GPU_SEC_PERF_OK, (staged_ok, staged_blocked, staged_drift, digest_primary)


def _iq1743_model(
    scenarios: list[Scenario],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    *,
    force_digest_drift_second: bool = False,
    force_status_domain_drift_second: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    if not scenarios:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    status_primary, tuple_primary = _iq1742_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
    )
    status_parity, tuple_parity = _iq1742_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        force_digest_drift=force_digest_drift_second,
        force_status_domain_drift=force_status_domain_drift_second,
    )

    if not _status_is_valid(status_primary):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_primary == GPU_SEC_PERF_OK:
        if tuple_primary[0] + tuple_primary[1] + tuple_primary[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if tuple_primary[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_parity == GPU_SEC_PERF_OK:
        if tuple_parity[0] + tuple_parity[1] + tuple_parity[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
        if tuple_parity[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    if status_primary != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)
    if tuple_primary != tuple_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, 0, 0)

    return status_primary, tuple_primary


def _iq1744_model(
    scenarios: list[Scenario],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    *,
    initial_out: tuple[int, int, int, int],
    force_digest_drift_staged: bool = False,
    force_status_domain_drift_canonical: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    saved_out = initial_out

    if not scenarios:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    status_staged, tuple_staged = _iq1743_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        force_digest_drift_second=force_digest_drift_staged,
    )

    status_canonical, tuple_canonical = _iq1742_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        force_status_domain_drift=force_status_domain_drift_canonical,
    )

    if not _status_is_valid(status_staged):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_staged == GPU_SEC_PERF_OK:
        if tuple_staged[0] + tuple_staged[1] + tuple_staged[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
        if tuple_staged[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_canonical == GPU_SEC_PERF_OK:
        if tuple_canonical[0] + tuple_canonical[1] + tuple_canonical[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
        if tuple_canonical[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_staged != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if tuple_staged != tuple_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    return status_staged, saved_out


def iq1745_model(
    scenarios: list[Scenario],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    dispatch_transcript_parity: int,
    *,
    initial_out: tuple[int, int, int, int] = (433, 439, 443, 449),
    force_digest_drift_staged: bool = False,
    force_status_domain_drift_canonical: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    saved_out = initial_out

    if not scenarios:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    status_commit_only, canonical_tuple = _iq1743_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        force_status_domain_drift_second=force_status_domain_drift_canonical,
    )

    status_preflight_only, staged_tuple = _iq1744_model(
        scenarios=scenarios,
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        initial_out=canonical_tuple,
        force_digest_drift_staged=force_digest_drift_staged,
    )

    if not _status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _status_is_valid(status_commit_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_preflight_only == GPU_SEC_PERF_OK:
        if staged_tuple[0] + staged_tuple[1] + staged_tuple[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
        if staged_tuple[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_commit_only == GPU_SEC_PERF_OK:
        if canonical_tuple[0] + canonical_tuple[1] + canonical_tuple[2] != len(scenarios):
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
        if canonical_tuple[3] <= 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_preflight_only != status_commit_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_tuple != canonical_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    return status_preflight_only, saved_out


def test_source_contains_iq1745_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert f"I32 {HC_FN_IQ1745}(" in src
    assert f"status_preflight_only = {HC_FN_IQ1744}(" in src
    assert f"status_commit_only = {HC_FN_IQ1743}(" in src
    assert "if (status_preflight_only != status_commit_only)" in src
    assert "// IQ-1745 strict diagnostics parity gate:" in src


def test_gate_missing_vector() -> None:
    status, out_values = iq1745_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80), Scenario(12, 24, 48, 96)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )
    assert status == GPU_SEC_PERF_OK
    assert out_values == (433, 439, 443, 449)


def test_digest_drift_vector() -> None:
    status, out_values = iq1745_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        force_digest_drift_staged=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (433, 439, 443, 449)


def test_status_domain_drift_vector() -> None:
    status, out_values = iq1745_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(10, 20, 40, 80)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        force_status_domain_drift_canonical=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (433, 439, 443, 449)


def test_no_write_parity_vector() -> None:
    status, out_values = iq1745_model(
        scenarios=[Scenario(8, 16, 32, 64), Scenario(14, 18, 20, 22)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
        initial_out=(461, 463, 467, 479),
    )
    assert status == GPU_SEC_PERF_OK
    assert out_values == (461, 463, 467, 479)


def test_deterministic_secure_on_parity_vector() -> None:
    scenarios = [
        Scenario(8, 16, 32, 64),
        Scenario(12, 24, 10, 22),
        Scenario(-1, 2, 3, 4),
        Scenario(14, 28, 56, 96),
    ]

    first = iq1745_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )
    second = iq1745_model(
        scenarios=scenarios,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        dispatch_transcript_parity=1,
    )

    assert first[0] == GPU_SEC_PERF_OK
    assert first[1] == (433, 439, 443, 449)
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1745_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vector()
    test_deterministic_secure_on_parity_vector()
    print("ok")
