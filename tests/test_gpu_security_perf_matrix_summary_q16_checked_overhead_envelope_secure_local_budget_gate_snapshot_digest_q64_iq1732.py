#!/usr/bin/env python3
"""Harness for IQ-1732 snapshot-digest commit-only hardening wrapper."""

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


_BASE_IQ1716_MOD = _load_module(
    Path(
        "tests/test_gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_secure_local_budget_gate_snapshot_digest_q64_iq1716.py"
    ),
    "iq1716_models_for_iq1732",
)
_BASE_IQ1731_MOD = _load_module(
    Path(
        "tests/test_gpu_security_perf_matrix_summary_q16_checked_overhead_envelope_secure_local_budget_gate_snapshot_digest_q64_iq1731.py"
    ),
    "iq1731_models_for_iq1732",
)

GPU_SEC_PERF_OK = _BASE_IQ1716_MOD.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_BAD_PARAM = _BASE_IQ1716_MOD.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_PROFILE_SECURE_LOCAL = _BASE_IQ1716_MOD.GPU_SEC_PERF_PROFILE_SECURE_LOCAL

RowOutput = _BASE_IQ1716_MOD.RowOutput

HC_FN_IQ1732 = (
    "GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly"
)
HC_FN_IQ1731 = (
    "GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
)
HC_FN_IQ1730 = (
    "GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly"
)


def iq1732_model(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    initial_out: tuple[int, int, int, int] = (700, 900, 4, 7777),
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    status_parity, parity_out = (
        _BASE_IQ1731_MOD.matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_iq1731(
            rows=rows,
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
            initial_out=initial_out,
        )
    )
    parity_p50, parity_p95, parity_gate, parity_digest_q64 = parity_out

    status_preflight_only, staged_out = (
        _BASE_IQ1716_MOD.matrix_summary_snapshot_digest_q64_checked_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_iq1716(
            rows=rows,
            secure_local_mode=secure_local_mode,
            iommu_active=iommu_active,
            book_of_truth_gpu_hooks=book_of_truth_gpu_hooks,
            policy_digest_parity=policy_digest_parity,
            max_p50_overhead_q16=max_p50_overhead_q16,
            max_p95_overhead_q16=max_p95_overhead_q16,
            initial_out=initial_out,
        )
    )
    staged_p50, staged_p95, staged_gate, staged_digest_q64 = staged_out

    if force_status_domain_drift:
        status_preflight_only = 99
    if force_digest_drift:
        staged_digest_q64 += 1

    if not _BASE_IQ1716_MOD._BASE_IQ1715_MOD._BASE_IQ1713_MOD._BASE_IQ1711_MOD._BASE_IQ1701_MOD._BASE_IQ1699_MOD._BASE_IQ1698_MOD._BASE_IQ1692_MOD._BASE_IQ1683_MOD._status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if not _BASE_IQ1716_MOD._BASE_IQ1715_MOD._BASE_IQ1713_MOD._BASE_IQ1711_MOD._BASE_IQ1701_MOD._BASE_IQ1699_MOD._BASE_IQ1698_MOD._BASE_IQ1692_MOD._BASE_IQ1683_MOD._status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if not _BASE_IQ1716_MOD._BASE_IQ1715_MOD._BASE_IQ1713_MOD._BASE_IQ1711_MOD._BASE_IQ1701_MOD._BASE_IQ1699_MOD._BASE_IQ1698_MOD._BASE_IQ1692_MOD._BASE_IQ1683_MOD._budget_gate_status_is_valid(parity_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if not _BASE_IQ1716_MOD._BASE_IQ1715_MOD._BASE_IQ1713_MOD._BASE_IQ1711_MOD._BASE_IQ1701_MOD._BASE_IQ1699_MOD._BASE_IQ1698_MOD._BASE_IQ1692_MOD._BASE_IQ1683_MOD._budget_gate_status_is_valid(staged_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out

    if min(parity_p50, parity_p95, staged_p50, staged_p95) < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out

    if status_parity == GPU_SEC_PERF_OK and (parity_digest_q64 <= 0 or staged_digest_q64 <= 0):
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out

    if status_parity != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if parity_p50 != staged_p50 or parity_p95 != staged_p95:
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if parity_gate != staged_gate:
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if parity_digest_q64 != staged_digest_q64:
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out

    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, initial_out

    return GPU_SEC_PERF_OK, (parity_p50, parity_p95, parity_gate, parity_digest_q64)


def test_source_contains_iq1732_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert f"I32 {HC_FN_IQ1732}(" in src
    assert f"status_parity = {HC_FN_IQ1731}(" in src
    assert f"status_preflight_only = {HC_FN_IQ1730}(" in src
    assert "if (parity_snapshot_digest_q64 != staged_snapshot_digest_q64)" in src
    assert "// IQ-1732 commit-only hardening wrapper:" in src


def test_gate_missing_vector() -> None:
    status, out_values = iq1732_model(
        rows=[RowOutput(8), RowOutput(10), RowOutput(12), RowOutput(14), RowOutput(16)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (700, 900, 4, 7777)


def test_digest_drift_vector() -> None:
    status, out_values = iq1732_model(
        rows=[RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15), RowOutput(17)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
        force_digest_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (700, 900, 4, 7777)


def test_status_domain_drift_vector() -> None:
    status, out_values = iq1732_model(
        rows=[RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15), RowOutput(17)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
        force_status_domain_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (700, 900, 4, 7777)


def test_deterministic_tuple_parity_vector() -> None:
    rows = [RowOutput(7), RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15)]

    first = iq1732_model(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    second = iq1732_model(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )

    assert first == (GPU_SEC_PERF_ERR_BAD_PARAM, (700, 900, 4, 7777))
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1732_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
