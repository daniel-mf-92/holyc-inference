#!/usr/bin/env python3
"""Tests for preflight-only wrapper over commit-only tok/sec Q16 estimator."""

Q8_0_DOT_BENCH_OK = 0
Q8_0_DOT_BENCH_ERR_NULL_PTR = 1
Q8_0_DOT_BENCH_ERR_BAD_PARAM = 2
Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO = 3
Q8_0_DOT_BENCH_ERR_OVERFLOW = 4
Q8_0_DOT_BENCH_I64_MAX = 0x7FFFFFFFFFFFFFFF


def try_add_i64(lhs: int, rhs: int):
    if rhs > 0 and lhs > Q8_0_DOT_BENCH_I64_MAX - rhs:
        return Q8_0_DOT_BENCH_ERR_OVERFLOW, 0
    return Q8_0_DOT_BENCH_OK, lhs + rhs


def try_mul_i64(lhs: int, rhs: int):
    if lhs == 0 or rhs == 0:
        return Q8_0_DOT_BENCH_OK, 0
    if lhs < 0 or rhs < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, 0
    if lhs > Q8_0_DOT_BENCH_I64_MAX // rhs:
        return Q8_0_DOT_BENCH_ERR_OVERFLOW, 0
    return Q8_0_DOT_BENCH_OK, lhs * rhs


def canonical_tok_per_sec_q16(cpu_hz: int, cycles_per_token_q16: int):
    if cpu_hz <= 0 or cycles_per_token_q16 < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, None
    if cycles_per_token_q16 == 0:
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO, None

    status, rounded_recip = try_add_i64((1 << 48), cycles_per_token_q16 >> 1)
    if status != Q8_0_DOT_BENCH_OK:
        return status, None

    reciprocal_q32 = rounded_recip // cycles_per_token_q16
    status, tok_q32 = try_mul_i64(cpu_hz, reciprocal_q32)
    if status != Q8_0_DOT_BENCH_OK:
        return status, None

    status, rounded_q32 = try_add_i64(tok_q32, 32768)
    if status != Q8_0_DOT_BENCH_OK:
        return status, None

    return Q8_0_DOT_BENCH_OK, rounded_q32 >> 16


def commit_only(cpu_hz: int, cycles_per_token_q16: int, out_value: int):
    status, staged = canonical_tok_per_sec_q16(cpu_hz, cycles_per_token_q16)
    if status != Q8_0_DOT_BENCH_OK:
        return status, out_value
    return Q8_0_DOT_BENCH_OK, staged


def preflight_only(cpu_hz: int, cycles_per_token_q16: int, out_value: int):
    snapshot_out = out_value

    s1, direct = canonical_tok_per_sec_q16(cpu_hz, cycles_per_token_q16)
    if s1 != Q8_0_DOT_BENCH_OK:
        return s1, out_value

    s2, commit = commit_only(cpu_hz, cycles_per_token_q16, out_value)
    if s2 != Q8_0_DOT_BENCH_OK:
        return s2, out_value

    if direct != commit:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, out_value

    # Zero-write contract.
    if out_value != snapshot_out:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, out_value

    return Q8_0_DOT_BENCH_OK, out_value


def test_null_ptr_contract_modeled():
    assert Q8_0_DOT_BENCH_ERR_NULL_PTR == 1


def test_preflight_zero_write_on_success():
    out_before = 123456
    status, out_after = preflight_only(3_000_000_000, (4 << 16) + 7, out_before)
    assert status == Q8_0_DOT_BENCH_OK
    assert out_after == out_before


def test_bad_params_div_zero_overflow_keep_output_unchanged():
    out_before = 999

    status, out_after = preflight_only(0, 1 << 16, out_before)
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    assert out_after == out_before

    status, out_after = preflight_only(3_000_000_000, 0, out_before)
    assert status == Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO
    assert out_after == out_before

    status, out_after = preflight_only(Q8_0_DOT_BENCH_I64_MAX, 1, out_before)
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out_after == out_before


def test_preflight_matches_canonical_and_commit_paths():
    vectors = [
        (2_000_000_000, 1 << 16),
        (2_500_000_000, (3 << 16) + 2),
        (3_200_000_000, (9 << 16) + 123),
        (4_100_000_000, (17 << 16) + 19),
    ]

    for cpu_hz, cycles_q16 in vectors:
        s0, expect = canonical_tok_per_sec_q16(cpu_hz, cycles_q16)
        s1, got_commit = commit_only(cpu_hz, cycles_q16, -1)
        s2, pre_out = preflight_only(cpu_hz, cycles_q16, 55)

        assert s0 == Q8_0_DOT_BENCH_OK
        assert s1 == Q8_0_DOT_BENCH_OK
        assert s2 == Q8_0_DOT_BENCH_OK
        assert got_commit == expect
        assert pre_out == 55


if __name__ == "__main__":
    test_null_ptr_contract_modeled()
    test_preflight_zero_write_on_success()
    test_bad_params_div_zero_overflow_keep_output_unchanged()
    test_preflight_matches_canonical_and_commit_paths()
    print("ok")
