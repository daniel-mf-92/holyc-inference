#!/usr/bin/env python3
"""Harness for IQ-1729 snapshot-digest strict diagnostics parity gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_BASE_IQ1702_MOD = _load_module(
    Path(
        "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1702.py"
    ),
    "iq1702_models",
)
_BASE_IQ1728_MOD = _load_module(
    Path(
        "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1728.py"
    ),
    "iq1728_models",
)
_BASE_IQ1727_MOD = _load_module(
    Path(
        "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1727.py"
    ),
    "iq1727_models",
)

GPU_SEC_PERF_OK = _BASE_IQ1702_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_IQ1702_MOD.GPU_SEC_PERF_ERR_BAD_PARAM

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_IQ1702_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

HC_FN_IQ1729 = (
    "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
)
HC_FN_IQ1728 = (
    "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly"
)
HC_FN_IQ1727 = (
    "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly"
)


def iq1729_model(
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
    initial_out: tuple[int, int, int] = (313, 11, 1901),
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
    force_no_write_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    out_enabled, out_reason, out_digest = initial_out
    saved_out = (out_enabled, out_reason, out_digest)

    status_preflight_only, staged_out = _BASE_IQ1728_MOD.iq1728_model(
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        p50_overhead_q16=p50_overhead_q16,
        p95_overhead_q16=p95_overhead_q16,
        max_p50_overhead_q16=max_p50_overhead_q16,
        max_p95_overhead_q16=max_p95_overhead_q16,
        initial_out=saved_out,
    )
    staged_enabled, staged_reason, staged_digest_q64 = staged_out

    status_commit_only, canonical_enabled, canonical_reason, canonical_digest_q64 = _BASE_IQ1727_MOD.iq1727_model(
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

    if force_status_domain_drift:
        status_commit_only = 99
    if force_digest_drift:
        canonical_digest_q64 += 1
    if force_no_write_drift:
        out_digest += 1

    if not _BASE_IQ1702_MOD._status_is_valid(status_preflight_only) or not _BASE_IQ1702_MOD._status_is_valid(
        status_commit_only
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _BASE_IQ1702_MOD._flag_is_binary(staged_enabled) or not _BASE_IQ1702_MOD._flag_is_binary(
        canonical_enabled
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if not _BASE_IQ1702_MOD._disable_reason_is_valid(staged_reason) or not _BASE_IQ1702_MOD._disable_reason_is_valid(
        canonical_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_preflight_only == GPU_SEC_PERF_OK and (staged_digest_q64 <= 0 or canonical_digest_q64 <= 0):
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if status_preflight_only != status_commit_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_enabled != canonical_enabled or staged_reason != canonical_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out
    if staged_digest_q64 != canonical_digest_q64:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    if (out_enabled, out_reason, out_digest) != saved_out:
        return GPU_SEC_PERF_ERR_BAD_PARAM, saved_out

    return status_preflight_only, saved_out


def test_source_contains_iq1729_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert f"I32 {HC_FN_IQ1729}(" in src
    assert f"{HC_FN_IQ1728}(" in src
    assert f"{HC_FN_IQ1727}(" in src
    assert "if (staged_snapshot_digest_q64 != canonical_snapshot_digest_q64)" in src
    assert "// IQ-1729 strict diagnostics parity gate:" in src


def test_gate_missing_vector() -> None:
    status, out_values = iq1729_model(
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
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (313, 11, 1901)


def test_digest_drift_vector() -> None:
    status, out_values = iq1729_model(
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
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (313, 11, 1901)


def test_status_domain_drift_vector() -> None:
    status, out_values = iq1729_model(
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
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (313, 11, 1901)


def test_no_write_parity_vector() -> None:
    status, out_values = iq1729_model(
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
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (313, 11, 1901)


def test_deterministic_secure_on_vector() -> None:
    first = iq1729_model(
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
    second = iq1729_model(
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

    assert first[0] == GPU_SEC_PERF_ERR_BAD_PARAM
    assert first[1] == (313, 11, 1901)
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1729_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_no_write_parity_vector()
    test_deterministic_secure_on_vector()
    print("ok")
