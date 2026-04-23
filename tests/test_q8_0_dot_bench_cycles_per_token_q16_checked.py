#!/usr/bin/env python3
"""Parity harness for IQ-1212 Q8_0DotBenchRunDefaultSuiteCyclesPerTokenQ16Checked."""

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


def q8_0_dot_bench_run_default_suite_cycles_per_token_q16_checked(
    total_cycles: int,
    total_tokens: int,
    out_cycles_per_token_q16: list[int] | None,
) -> int:
    if out_cycles_per_token_q16 is None:
        return Q8_0_DOT_BENCH_ERR_NULL_PTR

    if total_cycles < 0 or total_tokens < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    if total_tokens == 0:
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO

    if total_cycles > (Q8_0_DOT_BENCH_I64_MAX >> 16):
        return Q8_0_DOT_BENCH_ERR_OVERFLOW
    total_cycles_q16 = total_cycles << 16

    half_tokens = total_tokens >> 1
    status, rounded_numer = q8_0_dot_bench_try_add_i64(total_cycles_q16, half_tokens)
    if status != Q8_0_DOT_BENCH_OK:
        return status

    staged_cycles_per_token_q16 = rounded_numer // total_tokens

    out_cycles_per_token_q16[0] = staged_cycles_per_token_q16
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1212_function() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuiteCyclesPerTokenQ16Checked("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "if (total_cycles > (Q8_0_DOT_BENCH_I64_MAX >> 16))" in body
    assert "total_cycles_q16 = total_cycles << 16;" in body
    assert "Q8_0DotBenchTryAddI64(total_cycles_q16," in body
    assert "half_tokens," in body
    assert "&rounded_numer);" in body
    assert "staged_cycles_per_token_q16 = rounded_numer / total_tokens;" in body
    assert "*out_cycles_per_token_q16 = staged_cycles_per_token_q16;" in body


def test_null_ptr_and_domain_guards() -> None:
    out_cpt = [777]

    assert (
        q8_0_dot_bench_run_default_suite_cycles_per_token_q16_checked(100, 10, None)
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )
    assert (
        q8_0_dot_bench_run_default_suite_cycles_per_token_q16_checked(-1, 10, out_cpt)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_run_default_suite_cycles_per_token_q16_checked(10, -1, out_cpt)
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )
    assert (
        q8_0_dot_bench_run_default_suite_cycles_per_token_q16_checked(100, 0, out_cpt)
        == Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO
    )


def test_deterministic_q16_rounding() -> None:
    out_cpt = [-1]

    status = q8_0_dot_bench_run_default_suite_cycles_per_token_q16_checked(7, 3, out_cpt)
    assert status == Q8_0_DOT_BENCH_OK
    # round((7 * 65536) / 3) = 152917
    assert out_cpt[0] == 152917

    status = q8_0_dot_bench_run_default_suite_cycles_per_token_q16_checked(12, 3, out_cpt)
    assert status == Q8_0_DOT_BENCH_OK
    assert out_cpt[0] == (4 << 16)


def test_overflow_guards_and_no_partial_publish() -> None:
    out_cpt = [424242]

    status = q8_0_dot_bench_run_default_suite_cycles_per_token_q16_checked(
        (Q8_0_DOT_BENCH_I64_MAX >> 16) + 1,
        1,
        out_cpt,
    )
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out_cpt[0] == 424242

    # Bias add overflow path: rounded_numer = total_cycles_q16 + floor(total_tokens/2)
    status = q8_0_dot_bench_run_default_suite_cycles_per_token_q16_checked(
        Q8_0_DOT_BENCH_I64_MAX >> 16,
        131073,
        out_cpt,
    )
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out_cpt[0] == 424242


if __name__ == "__main__":
    test_source_contains_iq1212_function()
    test_null_ptr_and_domain_guards()
    test_deterministic_q16_rounding()
    test_overflow_guards_and_no_partial_publish()
    print("q8_0_dot_bench_cycles_per_token_q16_checked_reference_checks=ok")
