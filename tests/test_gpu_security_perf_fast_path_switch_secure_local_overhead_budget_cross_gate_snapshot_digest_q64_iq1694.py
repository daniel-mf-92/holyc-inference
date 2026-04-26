#!/usr/bin/env python3
"""Harness for IQ-1694 snapshot-digest strict parity gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BASE_IQ1680_PATH = Path(
    "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1680.py"
)
_BASE_IQ1680_SPEC = importlib.util.spec_from_file_location("iq1680_models", _BASE_IQ1680_PATH)
assert _BASE_IQ1680_SPEC is not None and _BASE_IQ1680_SPEC.loader is not None
_BASE_IQ1680_MOD = importlib.util.module_from_spec(_BASE_IQ1680_SPEC)
sys.modules[_BASE_IQ1680_SPEC.name] = _BASE_IQ1680_MOD
_BASE_IQ1680_SPEC.loader.exec_module(_BASE_IQ1680_MOD)

_BASE_IQ1684_PATH = Path(
    "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1684.py"
)
_BASE_IQ1684_SPEC = importlib.util.spec_from_file_location("iq1684_models", _BASE_IQ1684_PATH)
assert _BASE_IQ1684_SPEC is not None and _BASE_IQ1684_SPEC.loader is not None
_BASE_IQ1684_MOD = importlib.util.module_from_spec(_BASE_IQ1684_SPEC)
sys.modules[_BASE_IQ1684_SPEC.name] = _BASE_IQ1684_MOD
_BASE_IQ1684_SPEC.loader.exec_module(_BASE_IQ1684_MOD)

GPU_SEC_PERF_OK = _BASE_IQ1684_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_IQ1684_MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = _BASE_IQ1684_MOD.GPU_SEC_PERF_ERR_POLICY_GUARD

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_IQ1684_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW = _BASE_IQ1684_MOD.GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD = (
    _BASE_IQ1684_MOD.GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD
)
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH = 6


def _status_is_valid(status_code: int) -> bool:
    return status_code in {0, 1, 2, 3, 4, 5}


def _flag_is_binary(value: int) -> bool:
    return value in (0, 1)


def _disable_reason_is_valid(reason_code: int) -> bool:
    return (
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW
        <= reason_code
        <= GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH
    )


def fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_iq1694(
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
    initial_out: tuple[int, int, int] = (409, 12, 2049),
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
    force_no_write_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    out_enabled, out_reason, out_digest = initial_out
    saved_out = (out_enabled, out_reason, out_digest)

    status_preflight, _ = (
        _BASE_IQ1684_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_iq1684(
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            dispatch_transcript_parity=dispatch_transcript_parity,
            p50_overhead_q16=p50_overhead_q16,
            p95_overhead_q16=p95_overhead_q16,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
        )
    )
    _, staged_enabled, staged_reason, staged_digest_q64 = (
        _BASE_IQ1680_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_iq1680(
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            dispatch_transcript_parity=dispatch_transcript_parity,
            p50_overhead_q16=p50_overhead_q16,
            p95_overhead_q16=p95_overhead_q16,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
        )
    )

    status_canonical, canonical_enabled, canonical_reason, canonical_digest_q64 = (
        _BASE_IQ1680_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_iq1680(
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            dispatch_transcript_parity=dispatch_transcript_parity,
            p50_overhead_q16=p50_overhead_q16,
            p95_overhead_q16=p95_overhead_q16,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
        )
    )

    if force_status_domain_drift:
        status_canonical = 99
    if force_digest_drift:
        canonical_digest_q64 += 1
    if force_no_write_drift:
        out_reason += 1

    if not _status_is_valid(status_preflight) or not _status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _flag_is_binary(staged_enabled) or not _flag_is_binary(canonical_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _disable_reason_is_valid(staged_reason) or not _disable_reason_is_valid(canonical_reason):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_preflight == GPU_SEC_PERF_OK and (staged_digest_q64 <= 0 or canonical_digest_q64 <= 0):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_preflight != status_canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_enabled != canonical_enabled or staged_reason != canonical_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_digest_q64 != canonical_digest_q64:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if (out_enabled, out_reason, out_digest) != saved_out:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    return status_preflight, saved_out


def test_source_contains_iq1694_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert (
        "I32 GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnlyParity("
        in src
    )
    assert (
        "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64CheckedCommitOnlyPreflightOnly("
        in src
    )
    assert (
        "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64CheckedCommitOnly("
        in src
    )
    assert "if (staged_snapshot_digest_q64 != canonical_snapshot_digest_q64)" in src
    assert "// IQ-1694 strict diagnostics parity gate:" in src


def test_gate_missing_vector() -> None:
    status, out_values = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_iq1694(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=0,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=100,
            p95_overhead_q16=200,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
        )
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert out_values == (409, 12, 2049)


def test_digest_drift_vector() -> None:
    status, out_values = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_iq1694(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=120,
            p95_overhead_q16=240,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
            force_digest_drift=True,
        )
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (409, 12, 2049)


def test_status_domain_drift_vector() -> None:
    status, out_values = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_iq1694(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=120,
            p95_overhead_q16=240,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
            force_status_domain_drift=True,
        )
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (409, 12, 2049)


def test_no_write_parity_vector() -> None:
    status, out_values = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_iq1694(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=120,
            p95_overhead_q16=240,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
            force_no_write_drift=True,
        )
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (409, 12, 2049)


def test_deterministic_secure_on_parity_vector() -> None:
    first = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_iq1694(
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
    )
    second = (
        fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_preflight_only_parity_iq1694(
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
    )

    assert first[0] == GPU_SEC_PERF_OK
    assert first[1] == (409, 12, 2049)
    assert second == first


def test_underlying_commit_only_reason_vector() -> None:
    status, enabled, reason, digest_q64 = (
        _BASE_IQ1680_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_iq1680(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=150,
            p95_overhead_q16=350,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
        )
    )
    assert status == GPU_SEC_PERF_OK
    assert enabled == 1
    assert reason == GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_ALLOW
    assert digest_q64 > 0


def test_underlying_budget_breach_reason_vector() -> None:
    status, enabled, reason, digest_q64 = (
        _BASE_IQ1680_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_iq1680(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=1,
            dispatch_transcript_parity=1,
            p50_overhead_q16=900,
            p95_overhead_q16=1200,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
        )
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert enabled == 0
    assert reason == GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_OVERHEAD_BUDGET_BREACH
    assert digest_q64 == 0


def test_underlying_policy_block_reason_vector() -> None:
    status, enabled, reason, digest_q64 = (
        _BASE_IQ1680_MOD.fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_commit_only_iq1680(
            secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
            iommu_active=1,
            book_of_truth_gpu_hooks=1,
            policy_digest_parity=0,
            dispatch_transcript_parity=1,
            p50_overhead_q16=120,
            p95_overhead_q16=180,
            max_p50_overhead_q16=500,
            max_p95_overhead_q16=700,
        )
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert enabled == 0
    assert reason == 4
    assert digest_q64 == 0


if __name__ == "__main__":
    test_source_contains_iq1694_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vector()
    test_deterministic_secure_on_parity_vector()
    test_underlying_commit_only_reason_vector()
    test_underlying_budget_breach_reason_vector()
    test_underlying_policy_block_reason_vector()
    print("ok")
