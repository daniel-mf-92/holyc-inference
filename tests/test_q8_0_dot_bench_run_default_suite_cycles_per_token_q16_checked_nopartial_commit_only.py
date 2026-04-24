#!/usr/bin/env python3
"""Parity checks for IQ-1311 commit-only cycles/token Q16 wrapper."""

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


def main():
    out = [987654321]
    status = commit_only_cycles_per_token_q16(123456, 321, out)
    assert status == OK
    assert out[0] == ((123456 << 16) + (321 >> 1)) // 321

    bad = [111]
    status = commit_only_cycles_per_token_q16(-1, 7, bad)
    assert status == ERR_BAD_PARAM
    assert bad[0] == 111

    bad = [222]
    status = commit_only_cycles_per_token_q16(1, 0, bad)
    assert status == ERR_DIV_BY_ZERO
    assert bad[0] == 222

    bad = [333]
    status = commit_only_cycles_per_token_q16((I64_MAX >> 16) + 1, 1, bad)
    assert status == ERR_OVERFLOW
    assert bad[0] == 333

    assert commit_only_cycles_per_token_q16(1, 1, None) == ERR_NULL_PTR

    print("ok")


if __name__ == "__main__":
    main()
