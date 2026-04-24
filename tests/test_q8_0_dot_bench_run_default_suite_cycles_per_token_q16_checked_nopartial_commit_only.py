#!/usr/bin/env python3
"""No-partial commit-only checks for IQ-1311 cycles/token Q16 wrapper."""

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


def canonical_cycles_per_token_q16(total_cycles: int, total_ops: int):
    if total_cycles < 0 or total_ops < 0:
        return ERR_BAD_PARAM, None
    if total_ops == 0:
        return ERR_DIV_BY_ZERO, None

    status, total_cycles_q16 = _try_mul_i64(total_cycles, 1 << 16)
    if status != OK:
        return status, None

    half_ops = total_ops >> 1
    status, rounded_numer = _try_add_i64(total_cycles_q16, half_ops)
    if status != OK:
        return status, None

    return OK, rounded_numer // total_ops


def commit_only_cycles_per_token_q16(total_cycles: int, total_ops: int, out_ptr):
    if out_ptr is None:
        return ERR_NULL_PTR

    snapshot = out_ptr[0]
    status, staged = canonical_cycles_per_token_q16(total_cycles, total_ops)
    if status != OK:
        out_ptr[0] = snapshot
        return status

    out_ptr[0] = staged
    return OK


def run_case(total_cycles: int, total_ops: int, expected_status: int):
    out = [987654321]
    status = commit_only_cycles_per_token_q16(total_cycles, total_ops, out)
    assert status == expected_status, (
        f"status mismatch for ({total_cycles},{total_ops}): got {status}, expected {expected_status}"
    )

    if expected_status == OK:
        c_status, c_val = canonical_cycles_per_token_q16(total_cycles, total_ops)
        assert c_status == OK
        assert out[0] == c_val
    else:
        assert out[0] == 987654321, "no-partial violated"


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

    for total_cycles, total_ops, expected_status in cases:
        run_case(total_cycles, total_ops, expected_status)

    print("ok")


if __name__ == "__main__":
    main()
