#!/usr/bin/env python3
"""Parity harness for IQ-1213 Q8_0DotBenchEstimateTokPerSecQ16Checked."""

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


def q8_0_dot_bench_estimate_tok_per_sec_q16_checked(
    cpu_hz: int,
    cycles_per_token_q16: int,
    out_tok_per_sec_q16: list[int] | None,
) -> int:
    if out_tok_per_sec_q16 is None:
        return Q8_0_DOT_BENCH_ERR_NULL_PTR

    if cpu_hz <= 0 or cycles_per_token_q16 < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    if cycles_per_token_q16 == 0:
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO

    reciprocal_numer_q48 = 1 << 48
    half_cycles_per_token_q16 = cycles_per_token_q16 >> 1

    status, rounded_reciprocal_numer_q48 = q8_0_dot_bench_try_add_i64(
        reciprocal_numer_q48, half_cycles_per_token_q16
    )
    if status != Q8_0_DOT_BENCH_OK:
        return status

    reciprocal_tok_per_cycle_q32 = (
        rounded_reciprocal_numer_q48 // cycles_per_token_q16
    )

    status, tok_per_sec_q32 = q8_0_dot_bench_try_mul_i64(
        cpu_hz, reciprocal_tok_per_cycle_q32
    )
    if status != Q8_0_DOT_BENCH_OK:
        return status

    status, rounded_tok_per_sec_q32 = q8_0_dot_bench_try_add_i64(tok_per_sec_q32, 32768)
    if status != Q8_0_DOT_BENCH_OK:
        return status

    staged_tok_per_sec_q16 = rounded_tok_per_sec_q32 >> 16
    out_tok_per_sec_q16[0] = staged_tok_per_sec_q16
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1213_function() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchEstimateTokPerSecQ16Checked("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "reciprocal_numer_q48 = 281474976710656;" in body
    assert "status = Q8_0DotBenchTryAddI64(reciprocal_numer_q48," in body
    assert "half_cycles_per_token_q16," in body
    assert "&rounded_reciprocal_numer_q48);" in body
    assert "reciprocal_tok_per_cycle_q32 = rounded_reciprocal_numer_q48 /" in body
    assert "status = Q8_0DotBenchTryMulI64(cpu_hz," in body
    assert "status = Q8_0DotBenchTryAddI64(tok_per_sec_q32," in body
    assert "staged_tok_per_sec_q16 = rounded_tok_per_sec_q32 >> 16;" in body
    assert "*out_tok_per_sec_q16 = staged_tok_per_sec_q16;" in body


def test_null_ptr_and_domain_guards() -> None:
    out_tok = [999]

    assert (
        q8_0_dot_bench_estimate_tok_per_sec_q16_checked(3_000_000_000, 75_000 << 16, None)
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )
    assert (
        q8_0_dot_bench_estimate_tok_per_sec_q16_checked(0, 75_000 << 16, out_tok)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_estimate_tok_per_sec_q16_checked(-1, 75_000 << 16, out_tok)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_estimate_tok_per_sec_q16_checked(3_000_000_000, -1, out_tok)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_estimate_tok_per_sec_q16_checked(3_000_000_000, 0, out_tok)
        == Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO
    )


def test_deterministic_integer_only_estimates() -> None:
    out_tok = [-1]

    # 3 GHz CPU and 75k cycles/token => ~40k tok/s.
    status = q8_0_dot_bench_estimate_tok_per_sec_q16_checked(
        3_000_000_000, 75_000 << 16, out_tok
    )
    assert status == Q8_0_DOT_BENCH_OK
    assert out_tok[0] == 2621429443

    # Another deterministic point; reciprocal remains integer-only.
    status = q8_0_dot_bench_estimate_tok_per_sec_q16_checked(
        2_000_000_000, 100_000 << 16, out_tok
    )
    assert status == Q8_0_DOT_BENCH_OK
    assert out_tok[0] == 1310729980


def test_overflow_and_no_partial_publish() -> None:
    out_tok = [424242]

    # Force multiply overflow in cpu_hz * reciprocal_tok_per_cycle_q32 path.
    status = q8_0_dot_bench_estimate_tok_per_sec_q16_checked(
        32768,
        1,
        out_tok,
    )
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out_tok[0] == 424242


if __name__ == "__main__":
    test_source_contains_iq1213_function()
    test_null_ptr_and_domain_guards()
    test_deterministic_integer_only_estimates()
    test_overflow_and_no_partial_publish()
    print("q8_0_dot_bench_estimate_tok_per_sec_q16_checked_reference_checks=ok")
