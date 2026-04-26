#!/usr/bin/env python3
"""Harness for IQ-1648 zero-write diagnostics companion over IQ-1647 commit-only + IQ-1646 parity."""

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
    tok_per_sec_q16: int
    audit_overhead_delta_q16: int
    secure_cycles_per_token_q16: int


def _status_is_valid(status_code: int) -> bool:
    return status_code in {
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ERR_NULL_PTR,
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ERR_POLICY_GUARD,
        GPU_SEC_PERF_ERR_CAPACITY,
        GPU_SEC_PERF_ERR_OVERFLOW,
    }


def _budget_gate_status_is_valid(status: int) -> bool:
    return status in {
        GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS,
        GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH,
    }


def _rows_snapshot_digest_q64(rows: list[RowOutput], rows_capacity: int) -> tuple[int, int]:
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    digest = 1469598103934665603
    for idx, row in enumerate(rows):
        if row.tok_per_sec_q16 < 0 or row.audit_overhead_delta_q16 < 0 or row.secure_cycles_per_token_q16 < 0:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0
        digest += row.tok_per_sec_q16 * (idx + 1)
        digest += row.audit_overhead_delta_q16 * (idx + 1 + len(rows))
        digest += row.secure_cycles_per_token_q16 * (idx + 1 + (len(rows) << 1))
    return GPU_SEC_PERF_OK, digest


def _nearest_rank(rows: list[RowOutput], percentile_num: int, percentile_den: int) -> tuple[int, int]:
    if not rows or percentile_den <= 0 or percentile_num <= 0 or percentile_num > percentile_den:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0

    rank = (percentile_num * len(rows) + percentile_den - 1) // percentile_den
    rank = max(1, min(rank, len(rows)))
    ordered = sorted(row.audit_overhead_delta_q16 for row in rows)
    if ordered[0] < 0:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0
    return GPU_SEC_PERF_OK, ordered[rank - 1]


def _secure_local_budget_gate_checked(
    rows: list[RowOutput],
    rows_capacity: int,
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
    if not rows or rows_capacity < len(rows):
        return GPU_SEC_PERF_ERR_BAD_PARAM, out
    if iommu_active not in (0, 1) or book_of_truth_gpu_hooks not in (0, 1) or policy_digest_parity not in (0, 1):
        return GPU_SEC_PERF_ERR_BAD_PARAM, out
    if max_p50_overhead_q16 < 0 or max_p95_overhead_q16 < 0 or max_p95_overhead_q16 < max_p50_overhead_q16:
        return GPU_SEC_PERF_ERR_BAD_PARAM, out

    if secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, out
    if iommu_active != 1 or book_of_truth_gpu_hooks != 1 or policy_digest_parity != 1:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, out

    s50, p50 = _nearest_rank(rows, 50, 100)
    if s50 != GPU_SEC_PERF_OK:
        return s50, out
    s95, p95 = _nearest_rank(rows, 95, 100)
    if s95 != GPU_SEC_PERF_OK:
        return s95, out

    gate = GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS
    if p50 > max_p50_overhead_q16 or p95 > max_p95_overhead_q16:
        gate = GPU_SEC_PERF_BUDGET_GATE_STATUS_THRESHOLD_BREACH

    if force_status_domain_drift:
        return 99, (p50, p95, gate)
    if gate != GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, (p50, p95, gate)
    return GPU_SEC_PERF_OK, (p50, p95, gate)


def _secure_local_budget_gate_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    force_status_domain_drift: bool = False,
    force_tuple_parity_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    status_primary, tuple_primary = _secure_local_budget_gate_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
    )
    status_parity, tuple_parity = _secure_local_budget_gate_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        force_status_domain_drift=force_status_domain_drift,
    )
    if force_tuple_parity_drift:
        tuple_parity = (tuple_parity[0], tuple_parity[1], GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    if status_primary != status_parity or tuple_primary != tuple_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)
    return status_primary, tuple_primary


def _secure_local_budget_gate_commit_only_preflight_only(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    caller_outputs: tuple[int, int, int],
    *,
    force_commit_only_status_domain_drift: bool = False,
    force_canonical_status_domain_drift: bool = False,
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    status_before, digest_before = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_before != GPU_SEC_PERF_OK:
        return status_before, caller_outputs, caller_outputs

    status_commit_only, staged = _secure_local_budget_gate_commit_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        force_status_domain_drift=force_commit_only_status_domain_drift,
    )
    status_canonical, canonical = _secure_local_budget_gate_checked(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        force_status_domain_drift=force_canonical_status_domain_drift,
    )

    status_after, digest_after = _rows_snapshot_digest_q64(rows, rows_capacity)
    if status_after != GPU_SEC_PERF_OK or digest_after != digest_before:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, caller_outputs

    if not _status_is_valid(status_commit_only) or not _status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, caller_outputs
    if not _budget_gate_status_is_valid(staged[2]) or not _budget_gate_status_is_valid(canonical[2]):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, caller_outputs

    if status_commit_only != status_canonical or staged != canonical:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs, caller_outputs

    return status_commit_only, caller_outputs, caller_outputs


def _secure_local_budget_gate_commit_only_preflight_only_parity(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    caller_outputs: tuple[int, int, int],
    *,
    force_preflight_status_domain_drift: bool = False,
    force_commit_status_domain_drift: bool = False,
    force_commit_tuple_parity_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    status_preflight_only, _, staged_tuple = _secure_local_budget_gate_commit_only_preflight_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK),
        force_canonical_status_domain_drift=force_preflight_status_domain_drift,
    )

    status_commit_only, parity_tuple = _secure_local_budget_gate_commit_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        force_status_domain_drift=force_commit_status_domain_drift,
        force_tuple_parity_drift=force_commit_tuple_parity_drift,
    )
    parity_tuple = staged_tuple

    if not _status_is_valid(status_preflight_only) or not _status_is_valid(status_commit_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs
    if not _budget_gate_status_is_valid(staged_tuple[2]) or not _budget_gate_status_is_valid(parity_tuple[2]):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    if status_preflight_only != status_commit_only or staged_tuple != parity_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    return status_preflight_only, caller_outputs


def _secure_local_budget_gate_commit_only_preflight_only_parity_commit_only(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    *,
    force_parity_status_domain_drift: bool = False,
    force_preflight_status_domain_drift: bool = False,
    force_parity_tuple_drift: bool = False,
    force_preflight_tuple_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    status_parity, parity_tuple = _secure_local_budget_gate_commit_only_preflight_only_parity(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK),
        force_preflight_status_domain_drift=force_parity_status_domain_drift,
        force_commit_status_domain_drift=force_parity_status_domain_drift,
        force_commit_tuple_parity_drift=force_parity_tuple_drift,
    )

    status_preflight_only, _, staged_tuple = _secure_local_budget_gate_commit_only_preflight_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        parity_tuple,
        force_commit_only_status_domain_drift=force_preflight_status_domain_drift,
    )
    if force_preflight_tuple_drift:
        staged_tuple = (staged_tuple[0], staged_tuple[1] + 1, staged_tuple[2])

    if not _status_is_valid(status_parity) or not _status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    if not _budget_gate_status_is_valid(parity_tuple[2]) or not _budget_gate_status_is_valid(staged_tuple[2]):
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    if status_parity != status_preflight_only or parity_tuple != staged_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, (0, 0, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    return status_parity, parity_tuple


def secure_local_budget_gate_iq1648(
    rows: list[RowOutput],
    rows_capacity: int,
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    max_p50_overhead_q16: int,
    max_p95_overhead_q16: int,
    caller_outputs: tuple[int, int, int],
    *,
    has_null_output: bool = False,
    outputs_alias: bool = False,
    force_commit_status_domain_drift: bool = False,
    force_canonical_status_domain_drift: bool = False,
    force_canonical_tuple_drift: bool = False,
    force_commit_tuple_drift: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, caller_outputs
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    status_commit_only, staged_tuple = _secure_local_budget_gate_commit_only_preflight_only_parity_commit_only(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        force_parity_status_domain_drift=force_commit_status_domain_drift,
        force_parity_tuple_drift=force_commit_tuple_drift,
    )

    status_canonical, _ = _secure_local_budget_gate_commit_only_preflight_only_parity(
        rows,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        max_p50_overhead_q16,
        max_p95_overhead_q16,
        staged_tuple,
        force_preflight_status_domain_drift=force_canonical_status_domain_drift,
    )
    canonical_tuple = staged_tuple
    if force_canonical_tuple_drift:
        canonical_tuple = (canonical_tuple[0], canonical_tuple[1] + 1, canonical_tuple[2])

    if not _status_is_valid(status_commit_only) or not _status_is_valid(status_canonical):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    if not _budget_gate_status_is_valid(staged_tuple[2]) or not _budget_gate_status_is_valid(canonical_tuple[2]):
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    if status_commit_only != status_canonical or staged_tuple != canonical_tuple:
        return GPU_SEC_PERF_ERR_BAD_PARAM, caller_outputs

    return status_commit_only, caller_outputs


def test_source_contains_iq1648_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")
    assert "I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in src
    assert "status_commit_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    assert "status_canonical = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeSecureLocalBudgetGateCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in src
    assert "// IQ-1648 zero-write diagnostics companion over IQ-1647 commit-only + IQ-1646 canonical parity:" in src


def test_gate_missing_and_threshold_breach_vectors() -> None:
    rows_gate_missing = [
        RowOutput(90_000, 5_000, 48_000),
        RowOutput(88_000, 6_000, 49_000),
        RowOutput(86_000, 7_000, 50_000),
    ]
    status, outputs = secure_local_budget_gate_iq1648(
        rows_gate_missing,
        rows_capacity=3,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=8_000,
        max_p95_overhead_q16=9_000,
        caller_outputs=(444, 555, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs == (444, 555, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS)

    rows_threshold_breach = [
        RowOutput(100_000, 3_000, 40_000),
        RowOutput(100_000, 5_000, 45_000),
        RowOutput(100_000, 8_000, 50_000),
        RowOutput(100_000, 11_000, 55_000),
        RowOutput(100_000, 14_000, 60_000),
    ]
    status, outputs = secure_local_budget_gate_iq1648(
        rows_threshold_breach,
        rows_capacity=5,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=7_000,
        max_p95_overhead_q16=12_000,
        caller_outputs=(701, 702, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK),
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD
    assert outputs == (701, 702, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)


def test_status_domain_drift_no_write_parity_and_deterministic_secure_on_vectors() -> None:
    rows = [
        RowOutput(90_000, 3_000, 41_000),
        RowOutput(88_000, 4_000, 42_000),
        RowOutput(86_000, 5_000, 43_000),
        RowOutput(84_000, 6_000, 44_000),
    ]

    status, outputs = secure_local_budget_gate_iq1648(
        rows,
        rows_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=6_000,
        max_p95_overhead_q16=7_000,
        caller_outputs=(211, 222, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK),
        force_commit_status_domain_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs == (211, 222, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    status, outputs = secure_local_budget_gate_iq1648(
        rows,
        rows_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=6_000,
        max_p95_overhead_q16=7_000,
        caller_outputs=(233, 244, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK),
        force_canonical_status_domain_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs == (233, 244, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    status, outputs = secure_local_budget_gate_iq1648(
        rows,
        rows_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=6_000,
        max_p95_overhead_q16=7_000,
        caller_outputs=(255, 266, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK),
        force_canonical_tuple_drift=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs == (255, 266, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)

    status, outputs = secure_local_budget_gate_iq1648(
        rows,
        rows_capacity=4,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=6_000,
        max_p95_overhead_q16=7_000,
        caller_outputs=(277, 288, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK),
    )
    assert status == GPU_SEC_PERF_OK
    assert outputs == (277, 288, GPU_SEC_PERF_BUDGET_GATE_STATUS_POLICY_BLOCK)


def test_null_and_alias_vectors() -> None:
    rows = [RowOutput(90_000, 5_000, 48_000)]

    status, outputs = secure_local_budget_gate_iq1648(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=5_000,
        max_p95_overhead_q16=6_000,
        caller_outputs=(1, 2, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS),
        has_null_output=True,
    )
    assert status == GPU_SEC_PERF_ERR_NULL_PTR
    assert outputs == (1, 2, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS)

    status, outputs = secure_local_budget_gate_iq1648(
        rows,
        rows_capacity=1,
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        max_p50_overhead_q16=5_000,
        max_p95_overhead_q16=6_000,
        caller_outputs=(3, 4, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS),
        outputs_alias=True,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM
    assert outputs == (3, 4, GPU_SEC_PERF_BUDGET_GATE_STATUS_PASS)


if __name__ == "__main__":
    test_source_contains_iq1648_symbols()
    test_gate_missing_and_threshold_breach_vectors()
    test_status_domain_drift_no_write_parity_and_deterministic_secure_on_vectors()
    test_null_and_alias_vectors()
    print("ok")
