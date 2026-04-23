#!/usr/bin/env python3
"""Parity harness for IQ-1211 Q8_0DotBenchRunDefaultSuiteOpsPerSecondQ16Checked."""

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


def q8_0_dot_bench_try_mul_i64(lhs: int, rhs: int) -> tuple[int, int]:
    if lhs == 0 or rhs == 0:
        return Q8_0_DOT_BENCH_OK, 0
    if lhs < 0 or rhs < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, 0
    if lhs > (Q8_0_DOT_BENCH_I64_MAX // rhs):
        return Q8_0_DOT_BENCH_ERR_OVERFLOW, 0
    return Q8_0_DOT_BENCH_OK, lhs * rhs


def q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(
    total_ops: int,
    total_cycles: int,
    cpu_hz: int,
    out_ops_per_second_q16: list[int] | None,
) -> int:
    if out_ops_per_second_q16 is None:
        return Q8_0_DOT_BENCH_ERR_NULL_PTR

    if total_ops < 0 or total_cycles < 0 or cpu_hz <= 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    if total_cycles == 0:
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO

    status, ops_hz_numer = q8_0_dot_bench_try_mul_i64(total_ops, cpu_hz)
    if status != Q8_0_DOT_BENCH_OK:
        return status

    if ops_hz_numer > (Q8_0_DOT_BENCH_I64_MAX >> 16):
        return Q8_0_DOT_BENCH_ERR_OVERFLOW
    ops_hz_numer_q16 = ops_hz_numer << 16

    half_cycles = total_cycles >> 1
    status, rounded_numer = q8_0_dot_bench_try_add_i64(ops_hz_numer_q16, half_cycles)
    if status != Q8_0_DOT_BENCH_OK:
        return status

    out_ops_per_second_q16[0] = rounded_numer // total_cycles
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1211_function() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuiteOpsPerSecondQ16Checked("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "status = Q8_0DotBenchTryMulI64(total_ops, cpu_hz, &ops_hz_numer);" in body
    assert "if (ops_hz_numer > (Q8_0_DOT_BENCH_I64_MAX >> 16))" in body
    assert "ops_hz_numer_q16 = ops_hz_numer << 16;" in body
    assert "Q8_0DotBenchTryAddI64(ops_hz_numer_q16, half_cycles, &rounded_numer);" in body
    assert "staged_ops_per_second_q16 = rounded_numer / total_cycles;" in body
    assert "*out_ops_per_second_q16 = staged_ops_per_second_q16;" in body


def test_null_ptr_and_domain_guards() -> None:
    out_ops = [777]

    assert (
        q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(10, 5, 1000, None)
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )
    assert (
        q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(-1, 5, 1000, out_ops)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(10, -1, 1000, out_ops)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(10, 5, 0, out_ops)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(10, 0, 1000, out_ops)
        == Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO
    )


def test_q16_ops_per_second_rounding_is_deterministic() -> None:
    out_ops = [-1]

    status = q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(2000, 500, 1000, out_ops)
    assert status == Q8_0_DOT_BENCH_OK
    assert out_ops[0] == (4000 << 16)

    status = q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(7, 3, 5, out_ops)
    assert status == Q8_0_DOT_BENCH_OK
    assert out_ops[0] == 764587


def test_overflow_guards_and_no_partial_publish() -> None:
    out_ops = [123456]

    status = q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(
        Q8_0_DOT_BENCH_I64_MAX,
        1,
        2,
        out_ops,
    )
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out_ops[0] == 123456

    status = q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(
        (Q8_0_DOT_BENCH_I64_MAX >> 16) + 1,
        1,
        1,
        out_ops,
    )
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out_ops[0] == 123456

    status = q8_0_dot_bench_run_default_suite_ops_per_second_q16_checked(
        Q8_0_DOT_BENCH_I64_MAX >> 16,
        131073,
        1,
        out_ops,
    )
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out_ops[0] == 123456


if __name__ == "__main__":
    test_source_contains_iq1211_function()
    test_null_ptr_and_domain_guards()
    test_q16_ops_per_second_rounding_is_deterministic()
    test_overflow_guards_and_no_partial_publish()
    print("q8_0_dot_bench_ops_per_second_q16_checked_reference_checks=ok")
