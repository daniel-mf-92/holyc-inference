#!/usr/bin/env python3
"""Preflight-only zero-write parity checks for IQ-1312 cycles/token Q16 wrapper."""

from __future__ import annotations

I64_MAX = (1 << 63) - 1

OK = 0
ERR_NULL_PTR = 1
ERR_BAD_PARAM = 2
ERR_DIV_BY_ZERO = 3
ERR_OVERFLOW = 4


# Canonical integer helper mirror.
def canonical_cycles_per_token_q16(total_cycles: int, total_ops: int):
    if total_cycles < 0 or total_ops < 0:
        return ERR_BAD_PARAM, None
    if total_ops == 0:
        return ERR_DIV_BY_ZERO, None

    total_cycles_q16 = total_cycles * (1 << 16)
    if total_cycles != 0 and total_cycles_q16 // total_cycles != (1 << 16):
        return ERR_OVERFLOW, None
    if total_cycles_q16 > I64_MAX:
        return ERR_OVERFLOW, None

    half_ops = total_ops >> 1
    rounded_numer = total_cycles_q16 + half_ops
    if rounded_numer > I64_MAX:
        return ERR_OVERFLOW, None

    return OK, rounded_numer // total_ops


# IQ-1311 commit-only wrapper mirror.
def commit_only_cycles_per_token_q16(total_cycles: int, total_ops: int, out_ptr):
    if out_ptr is None:
        return ERR_NULL_PTR

    out_snapshot = out_ptr[0]
    status, staged = canonical_cycles_per_token_q16(total_cycles, total_ops)
    if status != OK:
        out_ptr[0] = out_snapshot
        return status

    out_ptr[0] = staged
    return OK


# IQ-1312 preflight-only wrapper mirror.
def preflight_only_cycles_per_token_q16(total_cycles: int, total_ops: int, out_ptr):
    if out_ptr is None:
        return ERR_NULL_PTR

    snapshot_total_cycles = total_cycles
    snapshot_total_ops = total_ops
    snapshot_out_ptr = out_ptr

    commit_tmp = [0]
    status = commit_only_cycles_per_token_q16(total_cycles, total_ops, commit_tmp)
    if status != OK:
        return status

    status, canonical_tmp = canonical_cycles_per_token_q16(total_cycles, total_ops)
    if status != OK:
        return status

    if snapshot_total_cycles != total_cycles:
        return ERR_BAD_PARAM
    if snapshot_total_ops != total_ops:
        return ERR_BAD_PARAM
    if snapshot_out_ptr is not out_ptr:
        return ERR_BAD_PARAM

    if commit_tmp[0] != canonical_tmp:
        return ERR_BAD_PARAM

    # Preflight-only: never writes to caller destination.
    return OK


def main():
    # Success and edge cases where output must remain untouched.
    success_cases = [
        (0, 7),
        (10, 1),
        (1000, 10),
        (123456789, 1000),
    ]
    for total_cycles, total_ops in success_cases:
        out = [13579]
        status = preflight_only_cycles_per_token_q16(total_cycles, total_ops, out)
        assert status == OK
        assert out[0] == 13579, "preflight-only path mutated caller output"

    error_cases = [
        (-1, 10, ERR_BAD_PARAM),
        (1, -10, ERR_BAD_PARAM),
        (1, 0, ERR_DIV_BY_ZERO),
        ((I64_MAX >> 16) + 1, 1, ERR_OVERFLOW),
    ]
    for total_cycles, total_ops, expected_status in error_cases:
        out = [24680]
        status = preflight_only_cycles_per_token_q16(total_cycles, total_ops, out)
        assert status == expected_status
        assert out[0] == 24680, "error path mutated caller output"

    # Null output pointer guard.
    status = preflight_only_cycles_per_token_q16(10, 2, None)
    assert status == ERR_NULL_PTR

    print("ok")


if __name__ == "__main__":
    main()
