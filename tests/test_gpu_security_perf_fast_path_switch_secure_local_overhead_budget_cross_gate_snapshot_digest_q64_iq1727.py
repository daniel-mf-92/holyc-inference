#!/usr/bin/env python3
"""Harness for IQ-1727 snapshot-digest commit-only hardening wrapper."""

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


def _find_model_fn(module, suffix: str):
    for name in dir(module):
        if name.startswith(
            "fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_checked_"
        ) and name.endswith(suffix):
            fn = getattr(module, name)
            if callable(fn):
                return fn
    raise AssertionError(f"unable to find function suffix={suffix!r}")


_BASE_IQ1702_MOD = _load_module(
    Path(
        "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1702.py"
    ),
    "iq1702_models",
)
_BASE_IQ1726_MOD = _load_module(
    Path(
        "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1726.py"
    ),
    "iq1726_models",
)
_BASE_IQ1725_MOD = _load_module(
    Path(
        "tests/test_gpu_security_perf_fast_path_switch_secure_local_overhead_budget_cross_gate_snapshot_digest_q64_iq1725.py"
    ),
    "iq1725_models",
)

_IQ1726_FN = _find_model_fn(_BASE_IQ1726_MOD, "_iq1726")
_IQ1725_FN = _find_model_fn(_BASE_IQ1725_MOD, "_iq1725")

GPU_SEC_PERF_OK = _BASE_IQ1702_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_IQ1702_MOD.GPU_SEC_PERF_ERR_BAD_PARAM

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_IQ1702_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL
GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD = 1

HC_FN_IQ1727 = (
    "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly"
)
HC_FN_IQ1726 = (
    "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
)
HC_FN_IQ1725 = (
    "GPUSecurityPerfFastPathSwitchSecureLocalOverheadBudgetCrossGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly"
)


def iq1727_model(
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
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
) -> tuple[int, int, int, int]:
    status_parity, parity_out = _IQ1726_FN(
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        p50_overhead_q16=p50_overhead_q16,
        p95_overhead_q16=p95_overhead_q16,
        max_p50_overhead_q16=max_p50_overhead_q16,
        max_p95_overhead_q16=max_p95_overhead_q16,
        initial_out=(0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0),
    )
    parity_enabled, parity_reason, parity_digest_q64 = parity_out

    status_preflight_only, staged_out = _IQ1725_FN(
        secure_local_mode=secure_local_mode,
        iommu_active=iommu_active,
        book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
        policy_digest_parity=policy_digest_parity,
        dispatch_transcript_parity=dispatch_transcript_parity,
        p50_overhead_q16=p50_overhead_q16,
        p95_overhead_q16=p95_overhead_q16,
        max_p50_overhead_q16=max_p50_overhead_q16,
        max_p95_overhead_q16=max_p95_overhead_q16,
        initial_out=(parity_enabled, parity_reason, parity_digest_q64),
    )
    staged_enabled, staged_reason, staged_digest_q64 = staged_out

    if force_status_domain_drift:
        status_preflight_only = 99
    if force_digest_drift:
        staged_digest_q64 += 1

    if not _BASE_IQ1702_MOD._status_is_valid(status_parity) or not _BASE_IQ1702_MOD._status_is_valid(
        status_preflight_only
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0
    if not _BASE_IQ1702_MOD._flag_is_binary(parity_enabled) or not _BASE_IQ1702_MOD._flag_is_binary(staged_enabled):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0
    if not _BASE_IQ1702_MOD._disable_reason_is_valid(parity_reason) or not _BASE_IQ1702_MOD._disable_reason_is_valid(
        staged_reason
    ):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0

    if status_parity == GPU_SEC_PERF_OK and (parity_digest_q64 <= 0 or staged_digest_q64 <= 0):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0

    if status_parity != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0
    if parity_enabled != staged_enabled or parity_reason != staged_reason:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0
    if parity_digest_q64 != staged_digest_q64:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0

    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, 0, GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD, 0

    return GPU_SEC_PERF_OK, parity_enabled, parity_reason, parity_digest_q64


def test_source_contains_iq1727_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert f"I32 {HC_FN_IQ1727}(" in src
    assert f"{HC_FN_IQ1726}(" in src
    assert f"{HC_FN_IQ1725}(" in src
    assert "if (parity_snapshot_digest_q64 != staged_snapshot_digest_q64)" in src
    assert "// IQ-1727 commit-only hardening wrapper:" in src


def test_gate_missing_vector() -> None:
    status, enabled, reason, digest_q64 = iq1727_model(
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
    assert (status, enabled, reason, digest_q64) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        0,
    )


def test_digest_drift_vector() -> None:
    status, enabled, reason, digest_q64 = iq1727_model(
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
    assert (status, enabled, reason, digest_q64) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        0,
    )


def test_status_domain_drift_vector() -> None:
    status, enabled, reason, digest_q64 = iq1727_model(
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
    assert (status, enabled, reason, digest_q64) == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        0,
    )


def test_deterministic_tuple_parity_vector() -> None:
    first = iq1727_model(
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
    second = iq1727_model(
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
    assert first == (
        GPU_SEC_PERF_ERR_BAD_PARAM,
        0,
        GPU_SEC_PERF_FAST_PATH_DISABLE_REASON_PROFILE_GUARD,
        0,
    )
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1727_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
