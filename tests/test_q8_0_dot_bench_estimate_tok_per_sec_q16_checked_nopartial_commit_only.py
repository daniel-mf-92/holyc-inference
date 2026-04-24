#!/usr/bin/env python3
"""Parity checks for IQ-1313 commit-only tok/sec Q16 wrapper."""

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


def main():
    out = [999]
    status = commit_only_tok_per_sec_q16(2_500_000_000, 131072, out)
    assert status == OK
    expected = ((2_500_000_000 << 32) + (131072 >> 1)) // 131072
    assert out[0] == expected

    bad = [111]
    status = commit_only_tok_per_sec_q16(0, 65536, bad)
    assert status == ERR_BAD_PARAM
    assert bad[0] == 111

    bad = [222]
    status = commit_only_tok_per_sec_q16(-1, 65536, bad)
    assert status == ERR_BAD_PARAM
    assert bad[0] == 222

    bad = [333]
    status = commit_only_tok_per_sec_q16(1234, -1, bad)
    assert status == ERR_BAD_PARAM
    assert bad[0] == 333

    bad = [444]
    status = commit_only_tok_per_sec_q16(1234, 0, bad)
    assert status == ERR_DIV_BY_ZERO
    assert bad[0] == 444

    bad = [555]
    status = commit_only_tok_per_sec_q16((I64_MAX >> 32) + 1, 1, bad)
    assert status == ERR_OVERFLOW
    assert bad[0] == 555

    assert commit_only_tok_per_sec_q16(1000, 65536, None) == ERR_NULL_PTR

    print("ok")


if __name__ == "__main__":
    main()
