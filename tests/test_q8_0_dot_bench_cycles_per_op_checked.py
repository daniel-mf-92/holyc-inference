#!/usr/bin/env python3
"""Parity harness for IQ-1174 Q8_0DotBenchCyclesPerOpChecked."""

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


def q8_0_dot_bench_cycles_per_op_checked(
    total_cycles: int,
    total_ops: int,
    out_cycles_per_op: list[int] | None,
    out_remainder_cycles: list[int] | None,
) -> int:
    if out_cycles_per_op is None or out_remainder_cycles is None:
        return Q8_0_DOT_BENCH_ERR_NULL_PTR

    if total_cycles < 0 or total_ops < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    if total_ops == 0:
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO

    half_ops = total_ops >> 1
    status, rounded_numer = q8_0_dot_bench_try_add_i64(total_cycles, half_ops)
    if status != Q8_0_DOT_BENCH_OK:
        return status

    staged_cycles_per_op = rounded_numer // total_ops
    staged_remainder_cycles = total_cycles % total_ops

    out_cycles_per_op[0] = staged_cycles_per_op
    out_remainder_cycles[0] = staged_remainder_cycles
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1174_function() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchCyclesPerOpChecked("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "Q8_0DotBenchTryAddI64(total_cycles, half_ops, &rounded_numer);" in body
    assert "staged_cycles_per_op = rounded_numer / total_ops;" in body
    assert "staged_remainder_cycles = total_cycles % total_ops;" in body
    assert "*out_cycles_per_op = staged_cycles_per_op;" in body
    assert "*out_remainder_cycles = staged_remainder_cycles;" in body


def test_null_ptr_and_domain_guards() -> None:
    out_cpo = [111]
    out_rem = [222]

    assert (
        q8_0_dot_bench_cycles_per_op_checked(100, 10, None, out_rem)
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )
    assert (
        q8_0_dot_bench_cycles_per_op_checked(100, 10, out_cpo, None)
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )
    assert (
        q8_0_dot_bench_cycles_per_op_checked(-1, 10, out_cpo, out_rem)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_cycles_per_op_checked(100, -7, out_cpo, out_rem)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_cycles_per_op_checked(100, 0, out_cpo, out_rem)
        == Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO
    )


def test_integer_rounding_and_remainder_are_deterministic() -> None:
    out_cpo = [-1]
    out_rem = [-1]

    status = q8_0_dot_bench_cycles_per_op_checked(100, 9, out_cpo, out_rem)
    assert status == Q8_0_DOT_BENCH_OK
    assert out_cpo[0] == 11
    assert out_rem[0] == 1

    status = q8_0_dot_bench_cycles_per_op_checked(120, 10, out_cpo, out_rem)
    assert status == Q8_0_DOT_BENCH_OK
    assert out_cpo[0] == 12
    assert out_rem[0] == 0


def test_overflow_guard_and_no_partial_publish() -> None:
    out_cpo = [777]
    out_rem = [888]

    status = q8_0_dot_bench_cycles_per_op_checked(
        Q8_0_DOT_BENCH_I64_MAX,
        3,
        out_cpo,
        out_rem,
    )
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out_cpo[0] == 777
    assert out_rem[0] == 888
