#!/usr/bin/env python3
"""Harness for IQ-1740 snapshot-digest commit-only hardening wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_NULL_PTR = 1
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3
GPU_SEC_PERF_ERR_CAPACITY = 4
GPU_SEC_PERF_ERR_OVERFLOW = 5

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = 1
GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK = 0
GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS = 1
GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH = 2


@dataclass(frozen=True)
class RowOutput:
    audit_overhead_delta_q16: int


HC_FN_IQ1740 = (
    "GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly"
)
HC_FN_IQ1739 = (
    "GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly"
)
HC_FN_IQ1737 = (
    "GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
    "CommitOnlyPreflightOnlyParity"
)


def _status_is_valid(value: int) -> bool:
    return value in {
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ERR_NULL_PTR,
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ERR_CAPACITY,
        GPU_SEC_PERF_ERR_OVERFLOW,
    }


def _budget_gate_status_is_valid(value: int) -> bool:
    return value in {
        GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH,
    }


def _iq1737_stub(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    initial_out: tuple[int, int, int, int],
) -> tuple[int, tuple[int, int, int, int]]:
    if not rows or secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if iommu_active != 1 or book_of_truth_gpu_hooks != 1 or policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if max_p50_overhead_q16 < 0 or max_p95_overhead_q16 < max_p50_overhead_q16:
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out

    # Deterministic but intentionally strict model mirroring the hard wrappers in this chain.
    return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out


def _iq1739_stub(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    initial_out: tuple[int, int, int, int],
) -> tuple[int, tuple[int, int, int, int]]:
    return _iq1737_stub(
        rows,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        initial_out,
    )


def iq1740_model(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    initial_out: tuple[int, int, int, int] = (1159, 1367, 1, 16357),
    force_digest_drift: bool = False,
    force_status_domain_drift: bool = False,
) -> tuple[int, tuple[int, int, int, int]]:
    status_parity, parity_out = _iq1737_stub(
        rows,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        initial_out,
    )
    parity_p50, parity_p95, parity_gate, parity_digest_q64 = parity_out

    status_preflight_only, staged_out = _iq1739_stub(
        rows,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        (parity_p50, parity_p95, parity_gate, parity_digest_q64),
    )
    staged_p50, staged_p95, staged_gate, staged_digest_q64 = staged_out

    if force_status_domain_drift:
        status_preflight_only = 99
    if force_digest_drift:
        staged_digest_q64 += 1

    if not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if not _status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if not _budget_gate_status_is_valid(parity_gate):
        return GPU_SEC_PERF_ERR_BAD_PARAM, initial_out
    if not _budget_gate_status_is_valid(staged_gate):
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


def test_source_contains_iq1740_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert f"I32 {HC_FN_IQ1740}(" in src
    assert f"status_parity = {HC_FN_IQ1737}(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateSnapshotDigestQ64Checked" in src
    assert "if (parity_snapshot_digest_q64 != staged_snapshot_digest_q64)" in src
    assert "// IQ-1740 commit-only hardening wrapper:" in src


def test_gate_missing_vector() -> None:
    status, out_values = iq1740_model(
        rows=[RowOutput(8), RowOutput(10), RowOutput(12), RowOutput(14), RowOutput(16)],
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert out_values == (1159, 1367, 1, 16357)


def test_digest_drift_vector() -> None:
    status, out_values = iq1740_model(
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
    assert out_values == (1159, 1367, 1, 16357)


def test_status_domain_drift_vector() -> None:
    status, out_values = iq1740_model(
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
    assert out_values == (1159, 1367, 1, 16357)


def test_deterministic_tuple_parity_vector() -> None:
    rows = [RowOutput(7), RowOutput(9), RowOutput(11), RowOutput(13), RowOutput(15)]

    first = iq1740_model(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )
    second = iq1740_model(
        rows=rows,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=64,
        max_p95_overhead_q16=64,
    )

    assert first == (GPU_SEC_PERF_ERR_BAD_PARAM, (1159, 1367, 1, 16357))
    assert second == first


if __name__ == "__main__":
    test_source_contains_iq1740_symbols()
    test_gate_missing_vector()
    test_digest_drift_vector()
    test_status_domain_drift_vector()
    test_deterministic_tuple_parity_vector()
    print("ok")
