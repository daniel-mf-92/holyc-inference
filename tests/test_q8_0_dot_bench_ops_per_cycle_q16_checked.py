#!/usr/bin/env python3
"""Parity harness for IQ-1175 Q8_0DotBenchOpsPerCycleQ16Checked."""

from __future__ import annotations

from pathlib import Path

Q8_0_DOT_BENCH_OK = 0
Q8_0_DOT_BENCH_ERR_NULL_PTR = 1
Q8_0_DOT_BENCH_ERR_BAD_PARAM = 2
Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO = 3
Q8_0_DOT_BENCH_ERR_OVERFLOW = 4
Q8_0_DOT_BENCH_I64_MAX = 0x7FFFFFFFFFFFFFFF


def q8_0_dot_bench_try_add_i64(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > Q8_0_DOT_BENCH_I64_MAX - rhs:
        return Q8_0_DOT_BENCH_ERR_OVERFLOW, 0
    return Q8_0_DOT_BENCH_OK, lhs + rhs


def q8_0_dot_bench_ops_per_cycle_q16_checked(
    total_ops: int,
    total_cycles: int,
    out_ops_per_cycle_q16: list[int] | None,
    out_remainder_ops_q16: list[int] | None,
) -> int:
    if out_ops_per_cycle_q16 is None or out_remainder_ops_q16 is None:
        return Q8_0_DOT_BENCH_ERR_NULL_PTR

    if total_ops < 0 or total_cycles < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    if total_cycles == 0:
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO

    if total_ops > (Q8_0_DOT_BENCH_I64_MAX >> 16):
        return Q8_0_DOT_BENCH_ERR_OVERFLOW
    total_ops_q16 = total_ops << 16

    half_cycles = total_cycles >> 1
    status, rounded_numer = q8_0_dot_bench_try_add_i64(total_ops_q16, half_cycles)
    if status != Q8_0_DOT_BENCH_OK:
        return status

    staged_ops_per_cycle_q16 = rounded_numer // total_cycles
    staged_remainder_ops_q16 = total_ops_q16 % total_cycles

    out_ops_per_cycle_q16[0] = staged_ops_per_cycle_q16
    out_remainder_ops_q16[0] = staged_remainder_ops_q16
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1175_function() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchOpsPerCycleQ16Checked("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "if (total_ops > (Q8_0_DOT_BENCH_I64_MAX >> 16))" in body
    assert "total_ops_q16 = total_ops << 16;" in body
    assert "Q8_0DotBenchTryAddI64(total_ops_q16, half_cycles, &rounded_numer);" in body
    assert "staged_ops_per_cycle_q16 = rounded_numer / total_cycles;" in body
    assert "staged_remainder_ops_q16 = total_ops_q16 % total_cycles;" in body
    assert "*out_ops_per_cycle_q16 = staged_ops_per_cycle_q16;" in body
    assert "*out_remainder_ops_q16 = staged_remainder_ops_q16;" in body


def test_null_ptr_and_domain_guards() -> None:
    out_opc = [111]
    out_rem = [222]

    assert (
        q8_0_dot_bench_ops_per_cycle_q16_checked(100, 10, None, out_rem)
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )
    assert (
        q8_0_dot_bench_ops_per_cycle_q16_checked(100, 10, out_opc, None)
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )
    assert (
        q8_0_dot_bench_ops_per_cycle_q16_checked(-1, 10, out_opc, out_rem)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_ops_per_cycle_q16_checked(100, -7, out_opc, out_rem)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_ops_per_cycle_q16_checked(100, 0, out_opc, out_rem)
        == Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO
    )


def test_deterministic_q16_rounding_and_remainder() -> None:
    out_opc = [-1]
    out_rem = [-1]

    status = q8_0_dot_bench_ops_per_cycle_q16_checked(7, 3, out_opc, out_rem)
    assert status == Q8_0_DOT_BENCH_OK
    # 7/3 cycles^-1 in Q16 domain: round((7*65536)/3) = 152917
    assert out_opc[0] == 152917
    assert out_rem[0] == 1

    status = q8_0_dot_bench_ops_per_cycle_q16_checked(12, 3, out_opc, out_rem)
    assert status == Q8_0_DOT_BENCH_OK
    assert out_opc[0] == (4 << 16)
    assert out_rem[0] == 0


def test_shift_and_bias_overflow_guards_no_partial_publish() -> None:
    out_opc = [777]
    out_rem = [888]

    status = q8_0_dot_bench_ops_per_cycle_q16_checked(
        (Q8_0_DOT_BENCH_I64_MAX >> 16) + 1,
        1,
        out_opc,
        out_rem,
    )
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out_opc[0] == 777
    assert out_rem[0] == 888

    status = q8_0_dot_bench_ops_per_cycle_q16_checked(
        Q8_0_DOT_BENCH_I64_MAX >> 16,
        131073,
        out_opc,
        out_rem,
    )
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out_opc[0] == 777
    assert out_rem[0] == 888


if __name__ == "__main__":
    test_source_contains_iq1175_function()
    test_null_ptr_and_domain_guards()
    test_deterministic_q16_rounding_and_remainder()
    test_shift_and_bias_overflow_guards_no_partial_publish()
    print("q8_0_dot_bench_ops_per_cycle_q16_checked_reference_checks=ok")
