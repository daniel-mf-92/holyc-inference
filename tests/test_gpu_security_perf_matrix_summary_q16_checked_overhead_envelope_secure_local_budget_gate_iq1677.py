#!/usr/bin/env python3
"""Harness for IQ-1677 commit-only hardening wrapper over IQ-1676 strict parity + IQ-1675 preflight-only."""

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


def _iq1677_commit_only_wrapper(
    rows: list[RowOutput],
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    caller_outputs: tuple[int, int, int],
    *,
    force_status_domain_drift: bool = False,
    force_tuple_parity_drift: bool = False,
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
    if force_tuple_parity_drift:
        preflight_tuple = (preflight_tuple[0], preflight_tuple[1], GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    if not _status_is_valid(status_parity) or not _status_is_valid(status_preflight):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs
    if not _budget_gate_status_is_valid(parity_tuple[2]) or not _budget_gate_status_is_valid(preflight_tuple[2]):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    if status_parity != status_preflight or parity_tuple != preflight_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, caller_outputs
    return GPU_SEC_PERF_OK, parity_tuple


def _sample_rows() -> list[RowOutput]:
    return [RowOutput(40), RowOutput(60), RowOutput(50), RowOutput(55), RowOutput(45)]


def test_iq1677_gate_missing_policy_block_preserves_outputs() -> None:
    status, outputs = _iq1677_commit_only_wrapper(
        _sample_rows(),
        GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        0,
        1,
        1,
        100,
        100,
        (777, 888, 9),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs == (777, 888, 9)


def test_iq1677_threshold_breach_preserves_outputs() -> None:
    status, outputs = _iq1677_commit_only_wrapper(
        _sample_rows(),
        GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        1,
        1,
        1,
        45,
        45,
        (11, 22, 33),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs == (11, 22, 33)


def test_iq1677_status_domain_drift_fails_closed() -> None:
    status, outputs = _iq1677_commit_only_wrapper(
        _sample_rows(),
        GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        1,
        1,
        1,
        100,
        100,
        (1, 2, 3),
        force_status_domain_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs == (1, 2, 3)


def test_iq1677_tuple_parity_drift_fails_closed() -> None:
    status, outputs = _iq1677_commit_only_wrapper(
        _sample_rows(),
        GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        1,
        1,
        1,
        100,
        100,
        (5, 6, 7),
        force_tuple_parity_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs == (5, 6, 7)


def test_iq1677_secure_on_success_commits_tuple() -> None:
    status, outputs = _iq1677_commit_only_wrapper(
        _sample_rows(),
        GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        1,
        1,
        1,
        100,
        100,
        (9, 9, 9),
    )
    assert status == GPU_SEC_PERF_OK
    assert outputs == (50, 60, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS)


def test_iq1677_source_has_single_symbol_and_expected_calls() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    fn = (
        "GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGate"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly"
    )
    assert src.count(f"I32 {fn}(") == 1
    assert "// IQ-1677 commit-only hardening wrapper over IQ-1676 strict parity + IQ-1675 preflight-only:" in src
    assert (
        "status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGate"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParity("
    ) in src
    assert (
        "status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGate"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    ) in src
