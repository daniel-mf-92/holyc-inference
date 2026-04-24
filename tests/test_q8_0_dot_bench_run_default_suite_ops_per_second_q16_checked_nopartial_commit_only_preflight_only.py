#!/usr/bin/env python3
"""Parity + zero-write diagnostics checks for IQ-1310 preflight-only ops/sec wrapper."""

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


def canonical_ops_per_second_q16(total_ops: int, total_cycles: int, cpu_hz: int):
    if total_ops < 0 or total_cycles < 0 or cpu_hz <= 0:
        return ERR_BAD_PARAM, None
    if total_cycles == 0:
        return ERR_DIV_BY_ZERO, None

    status, ops_hz_numer = _try_mul_i64(total_ops, cpu_hz)
    if status != OK:
        return status, None

    if ops_hz_numer > (I64_MAX >> 16):
        return ERR_OVERFLOW, None
    ops_hz_numer_q16 = ops_hz_numer << 16

    half_cycles = total_cycles >> 1
    status, rounded_numer = _try_add_i64(ops_hz_numer_q16, half_cycles)
    if status != OK:
        return status, None

    return OK, rounded_numer // total_cycles


def commit_only_ops_per_second_q16(total_ops: int, total_cycles: int, cpu_hz: int, out_ptr):
    if out_ptr is None:
        return ERR_NULL_PTR
    status, staged = canonical_ops_per_second_q16(total_ops, total_cycles, cpu_hz)
    if status != OK:
        return status
    out_ptr[0] = staged
    return OK


def preflight_only_ops_per_second_q16(total_ops: int, total_cycles: int, cpu_hz: int, out_ptr):
    if out_ptr is None:
        return ERR_NULL_PTR

    snapshot_total_ops = total_ops
    snapshot_total_cycles = total_cycles
    snapshot_cpu_hz = cpu_hz
    snapshot_value = out_ptr[0]

    staged = [0]
    status = commit_only_ops_per_second_q16(total_ops, total_cycles, cpu_hz, staged)
    if status != OK:
        return status

    c_status, c_value = canonical_ops_per_second_q16(total_ops, total_cycles, cpu_hz)
    if c_status != OK:
        return c_status

    if (
        snapshot_total_ops != total_ops
        or snapshot_total_cycles != total_cycles
        or snapshot_cpu_hz != cpu_hz
    ):
        return ERR_BAD_PARAM

    if staged[0] != c_value:
        return ERR_BAD_PARAM

    # Zero-write diagnostics semantics.
    out_ptr[0] = snapshot_value
    return OK


def _run_case(case):
    total_ops, total_cycles, cpu_hz, expected_status = case

    out = [987654321]
    status = preflight_only_ops_per_second_q16(total_ops, total_cycles, cpu_hz, out)
    assert status == expected_status, f"status mismatch for {case}: got {status}"
    assert out[0] == 987654321, f"zero-write violated for {case}"


def main():
    cases = [
        (10, 1, 1000, OK),
        (30000, 10000, 3300000000, OK),
        (0, 7, 4000000000, OK),
        (1000, 123456789, 3600000000, OK),
        (-1, 100, 1000, ERR_BAD_PARAM),
        (1, -100, 1000, ERR_BAD_PARAM),
        (1, 100, 0, ERR_BAD_PARAM),
        (1, 0, 1000, ERR_DIV_BY_ZERO),
        ((I64_MAX // 2) + 1, 1, 2, ERR_OVERFLOW),
        ((I64_MAX >> 16) + 1, 1, 1, ERR_OVERFLOW),
    ]

    for case in cases:
        _run_case(case)

    out = [1234]
    status = preflight_only_ops_per_second_q16(1000, 3, 2500000000, out)
    assert status == OK
    assert out[0] == 1234

    print("ok")


if __name__ == "__main__":
    main()
