#!/usr/bin/env python3
"""Parity/spec tests for Q8_0DotBenchEstimateTokPerSecQ16CheckedNoPartialCommitOnly."""

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

    reciprocal_numer_q48 = 1 << 48
    status, rounded_recip = try_add_i64(
        reciprocal_numer_q48, cycles_per_token_q16 >> 1
    )
    if status != Q8_0_DOT_BENCH_OK:
        return status, None

    reciprocal_tok_per_cycle_q32 = rounded_recip // cycles_per_token_q16

    status, tok_per_sec_q32 = try_mul_i64(cpu_hz, reciprocal_tok_per_cycle_q32)
    if status != Q8_0_DOT_BENCH_OK:
        return status, None

    status, rounded_q32 = try_add_i64(tok_per_sec_q32, 32768)
    if status != Q8_0_DOT_BENCH_OK:
        return status, None

    return Q8_0_DOT_BENCH_OK, rounded_q32 >> 16


def commit_only_tok_per_sec_q16(cpu_hz: int, cycles_per_token_q16: int, out_value: int):
    snapshot_cpu_hz = cpu_hz
    snapshot_cycles = cycles_per_token_q16
    snapshot_out = out_value

    status, staged = canonical_tok_per_sec_q16(cpu_hz, cycles_per_token_q16)
    if status != Q8_0_DOT_BENCH_OK:
        return status, snapshot_out

    if snapshot_cpu_hz != cpu_hz or snapshot_cycles != cycles_per_token_q16:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, snapshot_out

    return Q8_0_DOT_BENCH_OK, staged


def test_null_ptr_contract_modeled():
    # HolyC wrapper returns NULL_PTR if output pointer is null.
    assert Q8_0_DOT_BENCH_ERR_NULL_PTR == 1


def test_bad_params_and_div_zero():
    status, out = commit_only_tok_per_sec_q16(0, 1 << 16, 123)
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    assert out == 123

    status, out = commit_only_tok_per_sec_q16(-1, 1 << 16, 123)
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    assert out == 123

    status, out = commit_only_tok_per_sec_q16(3_000_000_000, -1, 123)
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    assert out == 123

    status, out = commit_only_tok_per_sec_q16(3_000_000_000, 0, 123)
    assert status == Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO
    assert out == 123


def test_overflow_guard_in_cpu_hz_multiply():
    # cycles/token very small => huge reciprocal -> multiply overflow.
    status, out = commit_only_tok_per_sec_q16(Q8_0_DOT_BENCH_I64_MAX, 1, 77)
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert out == 77


def test_deterministic_vectors_match_canonical_rounding():
    vectors = [
        (2_000_000_000, 1 << 16),      # 1 cycle/token
        (3_000_000_000, 2 << 16),      # 2 cycles/token
        (3_200_000_000, 3 << 16),      # non power-of-two
        (2_400_000_000, (5 << 16) + 1),
        (4_000_000_000, (17 << 16) + 13),
    ]
    for cpu_hz, cycles_q16 in vectors:
        s1, expected = canonical_tok_per_sec_q16(cpu_hz, cycles_q16)
        s2, got = commit_only_tok_per_sec_q16(cpu_hz, cycles_q16, -999)
        assert s1 == Q8_0_DOT_BENCH_OK
        assert s2 == Q8_0_DOT_BENCH_OK
        assert got == expected


def test_commit_only_keeps_output_on_failure():
    initial = 424242
    status, got = commit_only_tok_per_sec_q16(Q8_0_DOT_BENCH_I64_MAX, 1, initial)
    assert status == Q8_0_DOT_BENCH_ERR_OVERFLOW
    assert got == initial


if __name__ == "__main__":
    test_null_ptr_contract_modeled()
    test_bad_params_and_div_zero()
    test_overflow_guard_in_cpu_hz_multiply()
    test_deterministic_vectors_match_canonical_rounding()
    test_commit_only_keeps_output_on_failure()
    print("ok")
