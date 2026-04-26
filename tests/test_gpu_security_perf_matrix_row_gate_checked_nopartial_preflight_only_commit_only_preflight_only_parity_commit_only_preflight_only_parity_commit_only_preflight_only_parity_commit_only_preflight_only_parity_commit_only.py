#!/usr/bin/env python3
"""Harness for IQ-1574 commit-only hardening wrapper over row-gate parity + preflight diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HELPER_1558_PATH = Path(__file__).with_name(
    "test_gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only.py"
)
_HELPER_1558_SPEC = importlib.util.spec_from_file_location("iq1558_helper", _HELPER_1558_PATH)
assert _HELPER_1558_SPEC is not None and _HELPER_1558_SPEC.loader is not None
iq1558 = importlib.util.module_from_spec(_HELPER_1558_SPEC)
_HELPER_1558_SPEC.loader.exec_module(iq1558)

_HELPER_1557_PATH = Path(__file__).with_name(
    "test_gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only.py"
)
_HELPER_1557_SPEC = importlib.util.spec_from_file_location("iq1557_helper", _HELPER_1557_PATH)
assert _HELPER_1557_SPEC is not None and _HELPER_1557_SPEC.loader is not None
iq1557 = importlib.util.module_from_spec(_HELPER_1557_SPEC)
_HELPER_1557_SPEC.loader.exec_module(iq1557)

_HELPER_1544_PATH = Path(__file__).with_name(
    "test_gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity.py"
)
_HELPER_1544_SPEC = importlib.util.spec_from_file_location("iq1544_helper", _HELPER_1544_PATH)
assert _HELPER_1544_SPEC is not None and _HELPER_1544_SPEC.loader is not None
iq1544 = importlib.util.module_from_spec(_HELPER_1544_SPEC)
_HELPER_1544_SPEC.loader.exec_module(iq1544)

GPU_SEC_PERF_OK = iq1557.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_NULL_PTR = iq1557.GPU_SEC_PERF_ERR_NULL_PTR
GPU_SEC_PERF_ERR_BAD_PARAM = iq1557.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = iq1557.GPU_SEC_PERF_ERR_POLICY_GUARD

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = iq1557.GPU_SEC_PERF_PROFILE_SECURE_LOCAL
GPU_SEC_PERF_PROFILE_DEV_LOCAL = iq1557.GPU_SEC_PERF_PROFILE_DEV_LOCAL

GPU_SEC_PERF_QUANT_Q4_0 = iq1557.GPU_SEC_PERF_QUANT_Q4_0
GPU_SEC_PERF_QUANT_Q8_0 = iq1557.GPU_SEC_PERF_QUANT_Q8_0

GPU_SEC_PERF_ROW_GATE_REASON_ALLOW = iq1557.GPU_SEC_PERF_ROW_GATE_REASON_ALLOW


def _status_is_valid(status: int) -> bool:
    return status in (
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ERR_NULL_PTR,
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ERR_POLICY_GUARD,
    )


def row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
    secure_local_mode: int,
    iommu_active: int,
    book_of_truth_gpu_hooks: int,
    policy_digest_parity: int,
    row_prompt_tokens: int,
    row_batch_size: int,
    row_quant_profile: int,
    current_gate_reason_code: int,
    current_row_allowed: int,
    outputs_alias: bool = False,
    has_null_output: bool = False,
    inject_reason_drift: bool = False,
    inject_status_domain_drift: bool = False,
) -> tuple[int, int, int]:
    if has_null_output:
        return GPU_SEC_PERF_ERR_NULL_PTR, current_gate_reason_code, current_row_allowed
    if outputs_alias:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed

    status_canonical, canonical_reason, canonical_allowed = iq1557.row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        0,
        0,
    )

    status_parity, parity_reason, parity_allowed = iq1544.row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        canonical_reason,
        canonical_allowed,
    )
    status_preflight_only, staged_reason, staged_allowed = iq1558.row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        parity_reason,
        parity_allowed,
    )

    if inject_reason_drift:
        staged_reason = (staged_reason + 1) % 7
    if inject_status_domain_drift:
        status_preflight_only = 99

    if not _status_is_valid(status_parity) or not _status_is_valid(status_preflight_only):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if status_canonical != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if canonical_reason != parity_reason or canonical_allowed != parity_allowed:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if status_parity != status_preflight_only:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if parity_reason != staged_reason or parity_allowed != staged_allowed:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if status_parity != GPU_SEC_PERF_OK:
        return status_parity, current_gate_reason_code, current_row_allowed
    return GPU_SEC_PERF_OK, canonical_reason, canonical_allowed


def test_source_contains_iq1574_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    assert "// IQ-1574 commit-only hardening wrapper:" in src
    assert "status_parity = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in src
    assert "status_preflight_only = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in src
    assert "status_canonical = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_parity))" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_preflight_only))" in src
    assert "if (status_canonical != status_parity)" in src
    assert "if (status_parity != status_preflight_only)" in src
    assert "if (canonical_gate_reason_code != parity_gate_reason_code)" in src
    assert "if (parity_gate_reason_code != staged_gate_reason_code)" in src
    assert "if (status_parity != GPU_SEC_PERF_OK)" in src


def test_null_alias_capacity_vectors() -> None:
    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=24,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=301,
        current_row_allowed=302,
        has_null_output=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_NULL_PTR, 301, 302)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=24,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=303,
        current_row_allowed=304,
        outputs_alias=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 303, 304)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=24,
        row_batch_size=0,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=305,
        current_row_allowed=306,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 305, 306)


def test_gate_missing_status_domain_and_reason_code_parity_vectors() -> None:
    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=48,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=311,
        current_row_allowed=312,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 311, 312)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=48,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=313,
        current_row_allowed=314,
        inject_reason_drift=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 313, 314)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=48,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=315,
        current_row_allowed=316,
        inject_status_domain_drift=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 315, 316)


def test_secure_success_and_profile_guard_publish_behavior() -> None:
    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=96,
        row_batch_size=3,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=321,
        current_row_allowed=322,
    )
    assert status == GPU_SEC_PERF_OK
    assert reason == GPU_SEC_PERF_ROW_GATE_REASON_ALLOW
    assert allowed == 1

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_DEV_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=96,
        row_batch_size=3,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=323,
        current_row_allowed=324,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 323, 324)


if __name__ == "__main__":
    test_source_contains_iq1574_symbols()
    test_null_alias_capacity_vectors()
    test_gate_missing_status_domain_and_reason_code_parity_vectors()
    test_secure_success_and_profile_guard_publish_behavior()
    print("ok")
