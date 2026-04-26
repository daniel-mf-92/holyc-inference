#!/usr/bin/env python3
"""Harness for IQ-1575 zero-write diagnostics companion over IQ-1574 commit-only + IQ-1559 parity wrappers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HELPER_1574_PATH = Path(__file__).with_name(
    "test_gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only.py"
)
_HELPER_1574_SPEC = importlib.util.spec_from_file_location("iq1574_helper", _HELPER_1574_PATH)
assert _HELPER_1574_SPEC is not None and _HELPER_1574_SPEC.loader is not None
iq1574 = importlib.util.module_from_spec(_HELPER_1574_SPEC)
_HELPER_1574_SPEC.loader.exec_module(iq1574)

_HELPER_1559_PATH = Path(__file__).with_name(
    "test_gpu_security_perf_matrix_row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity.py"
)
_HELPER_1559_SPEC = importlib.util.spec_from_file_location("iq1559_helper", _HELPER_1559_PATH)
assert _HELPER_1559_SPEC is not None and _HELPER_1559_SPEC.loader is not None
iq1559 = importlib.util.module_from_spec(_HELPER_1559_SPEC)
_HELPER_1559_SPEC.loader.exec_module(iq1559)

GPU_SEC_PERF_OK = iq1574.GPU_SEC_PERF_OK
GPU_SEC_PERF_ERR_NULL_PTR = iq1574.GPU_SEC_PERF_ERR_NULL_PTR
GPU_SEC_PERF_ERR_BAD_PARAM = iq1574.GPU_SEC_PERF_ERR_BAD_PARAM
GPU_SEC_PERF_ERR_POLICY_GUARD = iq1574.GPU_SEC_PERF_ERR_POLICY_GUARD

GPU_SEC_PERF_PROFILE_SECURE_LOCAL = iq1574.GPU_SEC_PERF_PROFILE_SECURE_LOCAL
GPU_SEC_PERF_PROFILE_DEV_LOCAL = iq1574.GPU_SEC_PERF_PROFILE_DEV_LOCAL

GPU_SEC_PERF_QUANT_Q4_0 = iq1574.GPU_SEC_PERF_QUANT_Q4_0
GPU_SEC_PERF_QUANT_Q8_0 = iq1574.GPU_SEC_PERF_QUANT_Q8_0


def _status_is_valid(status: int) -> bool:
    return status in (
        GPU_SEC_PERF_OK,
        GPU_SEC_PERF_ERR_NULL_PTR,
        GPU_SEC_PERF_ERR_BAD_PARAM,
        GPU_SEC_PERF_ERR_POLICY_GUARD,
    )


def row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    status_commit_only, staged_reason, staged_allowed = iq1574.row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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

    status_parity, canonical_reason, canonical_allowed = iq1559.row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        staged_reason,
        staged_allowed,
    )

    if inject_reason_drift:
        canonical_reason = (canonical_reason + 1) % 7
    if inject_status_domain_drift:
        status_parity = 99

    if not _status_is_valid(status_commit_only) or not _status_is_valid(status_parity):
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if status_commit_only != status_parity:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed
    if staged_reason != canonical_reason or staged_allowed != canonical_allowed:
        return GPU_SEC_PERF_ERR_BAD_PARAM, current_gate_reason_code, current_row_allowed

    return status_commit_only, current_gate_reason_code, current_row_allowed


def test_source_contains_iq1575_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "I32 GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in src
    assert "// IQ-1575 zero-write diagnostics companion:" in src
    assert "status_commit_only = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in src
    assert "status_parity = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_commit_only))" in src
    assert "if (!GPUSecurityPerfStatusIsValid(status_parity))" in src
    assert "if (status_commit_only != status_parity)" in src
    assert "if (staged_gate_reason_code != canonical_gate_reason_code)" in src
    assert "if (staged_row_allowed != canonical_row_allowed)" in src
    assert "*out_gate_reason_code = saved_gate_reason_code;" in src
    assert "*out_row_allowed = saved_row_allowed;" in src


def test_null_alias_capacity_vectors() -> None:
    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=24,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=401,
        current_row_allowed=402,
        has_null_output=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_NULL_PTR, 401, 402)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=24,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=403,
        current_row_allowed=404,
        outputs_alias=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 403, 404)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=24,
        row_batch_size=0,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=405,
        current_row_allowed=406,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 405, 406)


def test_gate_missing_status_domain_and_reason_code_parity_vectors() -> None:
    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=0,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=48,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=411,
        current_row_allowed=412,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 411, 412)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=48,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=413,
        current_row_allowed=414,
        inject_reason_drift=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 413, 414)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=48,
        row_batch_size=2,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q4_0,
        current_gate_reason_code=415,
        current_row_allowed=416,
        inject_status_domain_drift=True,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_BAD_PARAM, 415, 416)


def test_secure_success_and_profile_guard_preserve_outputs() -> None:
    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=96,
        row_batch_size=3,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=421,
        current_row_allowed=422,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_OK, 421, 422)

    status, reason, allowed = row_gate_checked_nopartial_preflight_only_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        secure_local_mode=GPU_SEC_PERF_PROFILE_DEV_LOCAL,
        iommu_active=1,
        book_of_truth_gpu_hooks=1,
        policy_digest_parity=1,
        row_prompt_tokens=96,
        row_batch_size=3,
        row_quant_profile=GPU_SEC_PERF_QUANT_Q8_0,
        current_gate_reason_code=423,
        current_row_allowed=424,
    )
    assert (status, reason, allowed) == (GPU_SEC_PERF_ERR_POLICY_GUARD, 423, 424)


if __name__ == "__main__":
    test_source_contains_iq1575_symbols()
    test_null_alias_capacity_vectors()
    test_gate_missing_status_domain_and_reason_code_parity_vectors()
    test_secure_success_and_profile_guard_preserve_outputs()
    print("ok")
