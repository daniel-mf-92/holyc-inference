#!/usr/bin/env python3
"""Preflight-only no-partial checks for IQ-1312 cycles/token Q16 wrapper."""

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


def canonical_cycles_per_token_q16(total_cycles: int, total_tokens: int):
    if total_cycles < 0 or total_tokens < 0:
        return ERR_BAD_PARAM, None
    if total_tokens == 0:
        return ERR_DIV_BY_ZERO, None

    status, total_cycles_q16 = _try_mul_i64(total_cycles, 1 << 16)
    if status != OK:
        return status, None

    half_tokens = total_tokens >> 1
    status, rounded_numer = _try_add_i64(total_cycles_q16, half_tokens)
    if status != OK:
        return status, None

    return OK, rounded_numer // total_tokens


def commit_only_cycles_per_token_q16(total_cycles: int, total_tokens: int, out_ptr):
    if out_ptr is None:
        return ERR_NULL_PTR

    snapshot = out_ptr[0]
    status, staged = canonical_cycles_per_token_q16(total_cycles, total_tokens)
    if status != OK:
        out_ptr[0] = snapshot
        return status

    out_ptr[0] = staged
    return OK


def preflight_only_cycles_per_token_q16(total_cycles: int, total_tokens: int, out_ptr):
    if out_ptr is None:
        return ERR_NULL_PTR

    out_snapshot = out_ptr[0]
    cycles_snapshot = total_cycles
    tokens_snapshot = total_tokens

    status, staged_direct = canonical_cycles_per_token_q16(total_cycles, total_tokens)
    if status != OK:
        return status

    staged_commit = [0]
    status = commit_only_cycles_per_token_q16(total_cycles, total_tokens, staged_commit)
    if status != OK:
        return status

    if total_cycles != cycles_snapshot or total_tokens != tokens_snapshot:
        return ERR_BAD_PARAM

    if staged_direct != staged_commit[0]:
        return ERR_BAD_PARAM

    if out_ptr[0] != out_snapshot:
        return ERR_BAD_PARAM

    return OK


def run_case(total_cycles: int, total_tokens: int, expected_status: int):
    out = [444444444]
    status = preflight_only_cycles_per_token_q16(total_cycles, total_tokens, out)
    assert status == expected_status, (
        f"status mismatch for ({total_cycles},{total_tokens}): got {status}, expected {expected_status}"
    )

    assert out[0] == 444444444, "preflight-only must not mutate output"


def main():
    cases = [
        (10, 1, OK),
        (1000, 10, OK),
        (123456789, 1000, OK),
        (0, 7, OK),
        (-1, 10, ERR_BAD_PARAM),
        (1, -10, ERR_BAD_PARAM),
        (1, 0, ERR_DIV_BY_ZERO),
        ((I64_MAX >> 16) + 1, 1, ERR_OVERFLOW),
    ]

    for total_cycles, total_tokens, expected_status in cases:
        run_case(total_cycles, total_tokens, expected_status)

    print("ok")


if __name__ == "__main__":
    main()
