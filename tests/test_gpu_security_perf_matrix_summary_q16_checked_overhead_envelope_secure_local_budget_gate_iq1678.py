#!/usr/bin/env python3
"""Harness for IQ-1678 zero-write diagnostics companion over IQ-1677 + IQ-1676."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GPU_SEC_PERF_OK = 0
GPU_SEC_PERF_ERR_BAD_PARAM = 2
GPU_SEC_PERF_ERR_POLICY_GUARD = 3

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = 1

GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK = 0
GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS = 1
GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH = 2


@dataclass(frozen=True)
class RowOutput:
    audit_overhead_delta_q16: int


def _status_is_valid(status_code: int) -> bool:
    return status_code in {0, 1, 2, 3, 4, 5}


def _budget_gate_status_is_valid(status: int) -> bool:
    return status in {
        GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH,
    }


def _rows_digest(rows: list[RowOutput]) -> int:
    digest = 1469598103934665603
    for i, row in enumerate(rows):
        if row.audit_overhead_delta_q16 < 0:
            return -1
        digest += (i + 1) * row.audit_overhead_delta_q16
    return digest


def _nearest_rank(rows: list[RowOutput], percentile_num: int, percentile_den: int) -> int:
    rank = (percentile_num * len(rows) + percentile_den - 1) // percentile_den
    rank = max(1, min(rank, len(rows)))
    ordered = sorted(row.audit_overhead_delta_q16 for row in rows)
    return ordered[rank - 1]


def _canonical_gate(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    force_status_domain_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    out = (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, out
    if max_p50_overhead_q16 < 0 or max_p95_overhead_q16 < 0 or max_p95_overhead_q16 < max_p50_overhead_q16:
        return GPU_SEC_PERF_ERR_BAD_PARAM, out

    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, out
    if iommu_active != 1 or book_of_truth_gpu_hooks != 1 or policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, out

    p50 = _nearest_rank(rows, 50, 100)
    p95 = _nearest_rank(rows, 95, 100)
    gate = GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS
    if p50 > max_p50_overhead_q16 or p95 > max_p95_overhead_q16:
        gate = GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH

    if force_status_domain_drift:
        return 99, (p50, p95, gate)
    if gate != GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (p50, p95, gate)
    return GPU_SEC_PERF_OK, (p50, p95, gate)


def _iq1677_commit_only(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    force_status_domain_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    status_parity, parity_tuple = _canonical_gate(
        rows,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
    )
    status_preflight, preflight_tuple = _canonical_gate(
        rows,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        force_status_domain_drift=force_status_domain_drift,
    )

    if not _status_is_valid(status_parity) or not _status_is_valid(status_preflight):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)
    if not _budget_gate_status_is_valid(parity_tuple[2]) or not _budget_gate_status_is_valid(preflight_tuple[2]):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    if status_parity != status_preflight or parity_tuple != preflight_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    return status_parity, parity_tuple


def _iq1678_preflight_only(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    caller_outputs: tuple[int, int, int],
    *,
    force_commit_status_domain_drift: bool = False,
    force_canonical_status_domain_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    digest_before = _rows_digest(rows)
    if digest_before < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    status_commit, staged = _iq1677_commit_only(
        rows,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        force_status_domain_drift=force_commit_status_domain_drift,
    )
    status_canonical, canonical = _canonical_gate(
        rows,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        force_status_domain_drift=force_canonical_status_domain_drift,
    )

    digest_after = _rows_digest(rows)
    if digest_after != digest_before:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    if not _status_is_valid(status_commit) or not _status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs
    if not _budget_gate_status_is_valid(staged[2]) or not _budget_gate_status_is_valid(canonical[2]):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    if status_commit != status_canonical or staged != canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs
    return status_commit, caller_outputs


def _sample_rows() -> list[RowOutput]:
    return [RowOutput(30), RowOutput(42), RowOutput(50), RowOutput(55), RowOutput(60)]


def test_iq1678_gate_missing_preserves_outputs() -> None:
    status, outputs = _iq1678_preflight_only(
        _sample_rows(),
        GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        0,
        1,
        1,
        100,
        100,
        (7, 8, 9),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs == (7, 8, 9)


def test_iq1678_threshold_breach_preserves_outputs() -> None:
    status, outputs = _iq1678_preflight_only(
        _sample_rows(),
        GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        1,
        1,
        1,
        40,
        40,
        (11, 22, 33),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs == (11, 22, 33)


def test_iq1678_status_domain_drift_fails_closed() -> None:
    status, outputs = _iq1678_preflight_only(
        _sample_rows(),
        GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        1,
        1,
        1,
        100,
        100,
        (1, 2, 3),
        force_canonical_status_domain_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs == (1, 2, 3)


def test_iq1678_no_write_parity_on_success() -> None:
    status, outputs = _iq1678_preflight_only(
        _sample_rows(),
        GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        1,
        1,
        1,
        100,
        100,
        (999, 888, 777),
    )
    assert status == GPU_SEC_PERF_OK
    assert outputs == (999, 888, 777)


def test_iq1678_deterministic_secure_on_vector() -> None:
    status, outputs = _iq1678_preflight_only(
        [RowOutput(12), RowOutput(13), RowOutput(14), RowOutput(15), RowOutput(16)],
        GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        1,
        1,
        1,
        64,
        64,
        (31, 32, 33),
    )
    assert status == GPU_SEC_PERF_OK
    assert outputs == (31, 32, 33)


def test_iq1678_source_has_single_symbol_and_expected_calls() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    fn = (
        "GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGate"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly"
    )
    assert src.count(f"I32 {fn}(") == 1
    assert "// IQ-1678 zero-write diagnostics companion over IQ-1677 commit-only hardening + IQ-1676 canonical parity gate:" in src
    assert (
        "status_commit_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGate"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
    ) in src
    assert (
        "status_canonical = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGate"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    ) in src
