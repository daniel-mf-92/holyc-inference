#!/usr/bin/env python3
"""Harness for IQ-1442 secure-on GPU security/perf matrix runner."""

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
GPU_SEC_PERF_PROFILE_DEV_LOCAL = 2

GPU_SEC_PERF_QUANT_Q4_0 = 40
GPU_SEC_PERF_QUANT_Q8_0 = 80

GPU_SEC_PERF_Q16_ONE = 65536
GPU_SEC_PERF_CLOCK_HZ = 1_000_000_000
GPU_SEC_PERF_I64_MAX = 0x7FFFFFFFFFFFFFFF


@dataclass(frozen=True)
class RowInput:
    prompt_tokens: int
    batch_size: int
    quant_level: int
    tokens_generated: int
    secure_cycles: int
    relaxed_cycles: int


@dataclass(frozen=True)
class RowOutput:
    tok_per_sec_q16: int
    audit_overhead_delta_q16: int
    secure_cycles_per_token_q16: int


def _add_checked(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs > GPU_SEC_PERF_I64_MAX - rhs:
        return None
    return lhs + rhs


def _mul_checked(lhs: int, rhs: int) -> int | None:
    if lhs < 0 or rhs < 0:
        return None
    if lhs == 0 or rhs == 0:
        return 0
    if lhs > GPU_SEC_PERF_I64_MAX // rhs:
        return None
    return lhs * rhs


def _is_supported_quant(quant_level: int) -> bool:
    return quant_level in (GPU_SEC_PERF_QUANT_Q4_0, GPU_SEC_PERF_QUANT_Q8_0)


def _policy_allow_dispatch_checked(
    profile_id: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> bool:
    if profile_id not in (GPU_SEC_PERF_PROFILE_SECURE_LOCAL, GPU_SEC_PERF_PROFILE_DEV_LOCAL):
        return False
    if iommu_enabled != 1:
        return False
    if bot_dma_log_enabled != 1:
        return False
    if bot_mmio_log_enabled != 1:
        return False
    if bot_dispatch_log_enabled != 1:
        return False
    return True


def gpu_security_perf_matrix_run_checked(
    rows: list[RowInput],
    profile_id: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
    out_capacity: int,
) -> tuple[int, int, int, int, list[RowOutput]]:
    if not rows:
        return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, []
    if out_capacity < len(rows):
        return GPU_SEC_PERF_ERR_CAPACITY, 0, 0, 0, []
    if profile_id != GPU_SEC_PERF_PROFILE_SECURE_LOCAL:
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, []

    if not _policy_allow_dispatch_checked(
        profile_id,
        iommu_enabled,
        bot_dma_log_enabled,
        bot_mmio_log_enabled,
        bot_dispatch_log_enabled,
    ):
        return GPU_SEC_PERF_ERR_POLICY_GUARD, 0, 0, 0, []

    out_rows: list[RowOutput] = []
    tok_total = 0
    overhead_total = 0

    for row in rows:
        if (
            row.prompt_tokens < 0
            or row.batch_size <= 0
            or row.tokens_generated <= 0
            or row.secure_cycles <= 0
            or row.relaxed_cycles <= 0
        ):
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, []

        if not _is_supported_quant(row.quant_level):
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, []

        if row.secure_cycles < row.relaxed_cycles:
            return GPU_SEC_PERF_ERR_BAD_PARAM, 0, 0, 0, []

        tok_num = _mul_checked(row.tokens_generated, GPU_SEC_PERF_CLOCK_HZ)
        if tok_num is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, []

        tok_num_q16 = _mul_checked(tok_num, GPU_SEC_PERF_Q16_ONE)
        if tok_num_q16 is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, []

        tok_per_sec_q16 = tok_num_q16 // row.secure_cycles

        overhead_cycles = row.secure_cycles - row.relaxed_cycles
        overhead_num_q16 = _mul_checked(overhead_cycles, GPU_SEC_PERF_Q16_ONE)
        if overhead_num_q16 is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, []
        overhead_delta_q16 = overhead_num_q16 // row.relaxed_cycles

        cycles_per_token_num_q16 = _mul_checked(row.secure_cycles, GPU_SEC_PERF_Q16_ONE)
        if cycles_per_token_num_q16 is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, []
        cycles_per_token_q16 = cycles_per_token_num_q16 // row.tokens_generated

        tok_total_next = _add_checked(tok_total, tok_per_sec_q16)
        if tok_total_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, []
        overhead_total_next = _add_checked(overhead_total, overhead_delta_q16)
        if overhead_total_next is None:
            return GPU_SEC_PERF_ERR_OVERFLOW, 0, 0, 0, []

        out_rows.append(
            RowOutput(
                tok_per_sec_q16=tok_per_sec_q16,
                audit_overhead_delta_q16=overhead_delta_q16,
                secure_cycles_per_token_q16=cycles_per_token_q16,
            )
        )
        tok_total = tok_total_next
        overhead_total = overhead_total_next

    return GPU_SEC_PERF_OK, len(out_rows), tok_total, overhead_total, out_rows


def test_source_contains_iq1442_symbols() -> None:
    src = Path("src/gpu/security_perf_matrix.HC").read_text(encoding="utf-8")

    assert "class GPUSecurityPerfMatrixRowInput" in src
    assert "class GPUSecurityPerfMatrixRowOutput" in src
    assert "I32 GPUSecurityPerfMatrixRunChecked(" in src
    assert "GPUPolicyAllowDispatchChecked" in src
    assert "GPU_SEC_PERF_PROFILE_SECURE_LOCAL" in src
    assert "tok_per_sec_q16" in src
    assert "audit_overhead_delta_q16" in src


def test_secure_on_matrix_computes_deterministic_q16_metrics() -> None:
    rows = [
        RowInput(64, 1, GPU_SEC_PERF_QUANT_Q4_0, 16, 2_000_000, 1_600_000),
        RowInput(128, 2, GPU_SEC_PERF_QUANT_Q8_0, 32, 4_800_000, 3_840_000),
    ]

    status, written, tok_total, overhead_total, out_rows = gpu_security_perf_matrix_run_checked(
        rows,
        profile_id=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
        out_capacity=2,
    )

    assert status == GPU_SEC_PERF_OK
    assert written == 2
    assert len(out_rows) == 2

    expected_tok0 = (16 * GPU_SEC_PERF_CLOCK_HZ * GPU_SEC_PERF_Q16_ONE) // 2_000_000
    expected_tok1 = (32 * GPU_SEC_PERF_CLOCK_HZ * GPU_SEC_PERF_Q16_ONE) // 4_800_000
    assert out_rows[0].tok_per_sec_q16 == expected_tok0
    assert out_rows[1].tok_per_sec_q16 == expected_tok1
    assert tok_total == expected_tok0 + expected_tok1

    expected_ov0 = ((2_000_000 - 1_600_000) * GPU_SEC_PERF_Q16_ONE) // 1_600_000
    expected_ov1 = ((4_800_000 - 3_840_000) * GPU_SEC_PERF_Q16_ONE) // 3_840_000
    assert out_rows[0].audit_overhead_delta_q16 == expected_ov0
    assert out_rows[1].audit_overhead_delta_q16 == expected_ov1
    assert overhead_total == expected_ov0 + expected_ov1


def test_fail_closed_policy_vectors() -> None:
    rows = [RowInput(32, 1, GPU_SEC_PERF_QUANT_Q4_0, 8, 800_000, 700_000)]

    status, *_ = gpu_security_perf_matrix_run_checked(
        rows,
        profile_id=GPU_SEC_PERF_PROFILE_DEV_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
        out_capacity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, *_ = gpu_security_perf_matrix_run_checked(
        rows,
        profile_id=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=0,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
        out_capacity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD

    status, *_ = gpu_security_perf_matrix_run_checked(
        rows,
        profile_id=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=0,
        bot_dispatch_log_enabled=1,
        out_capacity=1,
    )
    assert status == GPU_SEC_PERF_ERR_POLICY_GUARD


def test_bad_params_and_overhead_monotonic_guard() -> None:
    rows_bad_quant = [RowInput(32, 1, 13, 8, 800_000, 700_000)]
    status, *_ = gpu_security_perf_matrix_run_checked(
        rows_bad_quant,
        profile_id=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
        out_capacity=1,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM

    rows_secure_faster = [RowInput(32, 1, GPU_SEC_PERF_QUANT_Q4_0, 8, 600_000, 700_000)]
    status, *_ = gpu_security_perf_matrix_run_checked(
        rows_secure_faster,
        profile_id=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
        out_capacity=1,
    )
    assert status == GPU_SEC_PERF_ERR_BAD_PARAM


def test_perf_overhead_budget_assertion_secure_on() -> None:
    rows = [
        RowInput(64, 1, GPU_SEC_PERF_QUANT_Q4_0, 16, 2_000_000, 1_920_000),
        RowInput(64, 2, GPU_SEC_PERF_QUANT_Q8_0, 32, 4_000_000, 3_800_000),
    ]
    status, written, _, overhead_total, out_rows = gpu_security_perf_matrix_run_checked(
        rows,
        profile_id=GPU_SEC_PERF_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
        out_capacity=2,
    )

    assert status == GPU_SEC_PERF_OK
    assert written == 2

    # Per-row secure-on audit overhead budget: <= 10%.
    max_overhead_q16 = (GPU_SEC_PERF_Q16_ONE * 10) // 100
    for out_row in out_rows:
        assert out_row.audit_overhead_delta_q16 <= max_overhead_q16

    assert overhead_total <= 2 * max_overhead_q16


if __name__ == "__main__":
    test_source_contains_iq1442_symbols()
    test_secure_on_matrix_computes_deterministic_q16_metrics()
    test_fail_closed_policy_vectors()
    test_bad_params_and_overhead_monotonic_guard()
    test_perf_overhead_budget_assertion_secure_on()
    print("ok")
