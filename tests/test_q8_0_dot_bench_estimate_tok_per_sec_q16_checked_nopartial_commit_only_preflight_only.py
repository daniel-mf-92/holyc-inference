#!/usr/bin/env python3
"""Preflight-only no-partial checks for IQ-1314 tok/sec Q16 wrapper."""

from __future__ import annotations

I64_MAX = (1 << 63) - 1

OK = 0
ERR_NULL_PTR = 1
ERR_BAD_PARAM = 2
ERR_DIV_BY_ZERO = 3
ERR_OVERFLOW = 4


def _try_mul_i64(lhs: int, rhs: int):
    if lhs == 0 or rhs == 0:
        return OK, 0
    if lhs < 0 or rhs < 0:
        return ERR_BAD_PARAM, None
    if lhs > I64_MAX // rhs:
        return ERR_OVERFLOW, None
    return OK, lhs * rhs


def _try_add_i64(lhs: int, rhs: int):
    if rhs > 0 and lhs > I64_MAX - rhs:
        return ERR_OVERFLOW, None
    return OK, lhs + rhs


def canonical_tok_per_sec_q16(cpu_hz: int, cycles_per_token_q16: int):
    if cpu_hz <= 0 or cycles_per_token_q16 < 0:
        return ERR_BAD_PARAM, None
    if cycles_per_token_q16 == 0:
        return ERR_DIV_BY_ZERO, None

    status, cpu_hz_q32 = _try_mul_i64(cpu_hz, 1 << 32)
    if status != OK:
        return status, None

    half_denom = cycles_per_token_q16 >> 1
    status, rounded_numer = _try_add_i64(cpu_hz_q32, half_denom)
    if status != OK:
        return status, None

    return OK, rounded_numer // cycles_per_token_q16


def commit_only_tok_per_sec_q16(cpu_hz: int, cycles_per_token_q16: int, out_ptr):
    if out_ptr is None:
        return ERR_NULL_PTR

    out_snapshot = out_ptr[0]
    status, staged = canonical_tok_per_sec_q16(cpu_hz, cycles_per_token_q16)
    if status != OK:
        out_ptr[0] = out_snapshot
        return status

    out_ptr[0] = staged
    return OK


def preflight_only_tok_per_sec_q16(cpu_hz: int, cycles_per_token_q16: int, out_ptr):
    if out_ptr is None:
        return ERR_NULL_PTR

    out_snapshot = out_ptr[0]
    hz_snapshot = cpu_hz
    cycles_snapshot = cycles_per_token_q16

    status, staged_direct = canonical_tok_per_sec_q16(cpu_hz, cycles_per_token_q16)
    if status != OK:
        return status

    staged_commit = [0]
    status = commit_only_tok_per_sec_q16(cpu_hz, cycles_per_token_q16, staged_commit)
    if status != OK:
        return status

    if cpu_hz != hz_snapshot or cycles_per_token_q16 != cycles_snapshot:
        return ERR_BAD_PARAM

    if staged_direct != staged_commit[0]:
        return ERR_BAD_PARAM

    if out_ptr[0] != out_snapshot:
        return ERR_BAD_PARAM

    return OK


def run_case(cpu_hz: int, cycles_q16: int, expected_status: int):
    out = [777777]
    status = preflight_only_tok_per_sec_q16(cpu_hz, cycles_q16, out)
    assert status == expected_status, (
        f"status mismatch for ({cpu_hz},{cycles_q16}): got {status}, expected {expected_status}"
    )
    assert out[0] == 777777, "preflight-only must not mutate output"


def main():
    cases = [
        (2_500_000_000, 131072, OK),
        (1, 1, OK),
        (1_000_000_000, 65536, OK),
        (0, 65536, ERR_BAD_PARAM),
        (-1, 65536, ERR_BAD_PARAM),
        (1234, -1, ERR_BAD_PARAM),
        (1234, 0, ERR_DIV_BY_ZERO),
        ((I64_MAX >> 32) + 1, 1, ERR_OVERFLOW),
    ]

    for cpu_hz, cycles_q16, expected in cases:
        run_case(cpu_hz, cycles_q16, expected)

    assert preflight_only_tok_per_sec_q16(1000, 65536, None) == ERR_NULL_PTR

    print("ok")


if __name__ == "__main__":
    main()
