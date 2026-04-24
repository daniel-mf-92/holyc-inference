#!/usr/bin/env python3
"""Parity checks for IQ-1316 commit-only telemetry digest wrapper logic."""

from __future__ import annotations

ERR_OK = 0
ERR_NULL_PTR = 1
ERR_BAD_PARAM = 2
ERR_DIV_BY_ZERO = 3
ERR_OVERFLOW = 4
I64_MAX = (1 << 63) - 1


def try_add_i64(lhs: int, rhs: int) -> tuple[int, int | None]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return ERR_OVERFLOW, None
    return ERR_OK, lhs + rhs


def try_mul_i64(lhs: int, rhs: int) -> tuple[int, int | None]:
    if lhs == 0 or rhs == 0:
        return ERR_OK, 0
    if lhs < 0 or rhs < 0:
        return ERR_BAD_PARAM, None
    if lhs > I64_MAX // rhs:
        return ERR_OVERFLOW, None
    return ERR_OK, lhs * rhs


def telemetry_digest_q64_checked(
    total_ops: int,
    total_cycles: int,
    cycles_per_op: int,
    remainder_cycles: int,
    suite_checksum: int,
    cpu_hz: int,
) -> tuple[int, int | None]:
    digest_mod61 = (1 << 61) - 1
    if (
        total_ops < 0
        or total_cycles < 0
        or cycles_per_op < 0
        or remainder_cycles < 0
        or suite_checksum < 0
        or cpu_hz <= 0
    ):
        return ERR_BAD_PARAM, None

    if total_ops < remainder_cycles:
        return ERR_BAD_PARAM, None

    status, folded_ops = try_mul_i64(total_ops, 3)
    if status != ERR_OK:
        return status, None
    status, folded_cycles = try_mul_i64(total_cycles, 5)
    if status != ERR_OK:
        return status, None
    status, folded_cpo = try_mul_i64(cycles_per_op, 7)
    if status != ERR_OK:
        return status, None
    status, folded_rem = try_mul_i64(remainder_cycles, 11)
    if status != ERR_OK:
        return status, None
    status, folded_checksum = try_mul_i64(suite_checksum, 13)
    if status != ERR_OK:
        return status, None
    status, folded_hz = try_mul_i64(cpu_hz, 17)
    if status != ERR_OK:
        return status, None

    digest = 1469598103934665603
    for folded in (
        folded_ops,
        folded_cycles,
        folded_cpo,
        folded_rem,
        folded_checksum,
        folded_hz,
    ):
        status, digest = try_add_i64(digest, folded)
        if status != ERR_OK:
            return status, None
        digest %= digest_mod61
        status, digest = try_mul_i64(digest, 3)
        if status != ERR_OK:
            return status, None
        digest %= digest_mod61

    return ERR_OK, digest


def telemetry_digest_q64_checked_nopartial_commit_only(
    total_ops: int,
    total_cycles: int,
    cycles_per_op: int,
    remainder_cycles: int,
    suite_checksum: int,
    cpu_hz: int,
) -> tuple[int, int | None]:
    if (
        total_ops < 0
        or total_cycles < 0
        or cycles_per_op < 0
        or remainder_cycles < 0
        or suite_checksum < 0
        or cpu_hz <= 0
    ):
        return ERR_BAD_PARAM, None

    # Commit-only wrapper snapshots are modeled by deterministic passthrough in Python.
    status, staged = telemetry_digest_q64_checked(
        total_ops=total_ops,
        total_cycles=total_cycles,
        cycles_per_op=cycles_per_op,
        remainder_cycles=remainder_cycles,
        suite_checksum=suite_checksum,
        cpu_hz=cpu_hz,
    )
    if status != ERR_OK:
        return status, None
    return ERR_OK, staged


def test_commit_only_matches_canonical_digest() -> None:
    args = dict(
        total_ops=1,
        total_cycles=1,
        cycles_per_op=1,
        remainder_cycles=0,
        suite_checksum=1,
        cpu_hz=1,
    )
    c_status, c_digest = telemetry_digest_q64_checked(**args)
    w_status, w_digest = telemetry_digest_q64_checked_nopartial_commit_only(**args)

    assert c_status == ERR_OK
    assert w_status == ERR_OK
    assert c_digest == w_digest


def test_commit_only_is_deterministic() -> None:
    args = dict(
        total_ops=2,
        total_cycles=3,
        cycles_per_op=1,
        remainder_cycles=1,
        suite_checksum=2,
        cpu_hz=1,
    )
    s1, d1 = telemetry_digest_q64_checked_nopartial_commit_only(**args)
    s2, d2 = telemetry_digest_q64_checked_nopartial_commit_only(**args)
    assert s1 == ERR_OK and s2 == ERR_OK
    assert d1 == d2


def test_bad_params_propagate() -> None:
    for args in (
        (-1, 10, 1, 0, 1, 1),
        (1, -1, 1, 0, 1, 1),
        (1, 10, -1, 0, 1, 1),
        (1, 10, 1, -1, 1, 1),
        (1, 10, 1, 0, -1, 1),
        (1, 10, 1, 0, 1, 0),
        (5, 10, 1, 6, 1, 1),
    ):
        status, _ = telemetry_digest_q64_checked_nopartial_commit_only(*args)
        assert status == ERR_BAD_PARAM


def test_overflow_propagates() -> None:
    status, _ = telemetry_digest_q64_checked_nopartial_commit_only(
        total_ops=(I64_MAX // 3) + 1,
        total_cycles=1,
        cycles_per_op=1,
        remainder_cycles=0,
        suite_checksum=1,
        cpu_hz=1,
    )
    assert status == ERR_OVERFLOW

    status, _ = telemetry_digest_q64_checked_nopartial_commit_only(
        total_ops=I64_MAX // 6,
        total_cycles=I64_MAX // 10,
        cycles_per_op=I64_MAX // 14,
        remainder_cycles=0,
        suite_checksum=I64_MAX // 26,
        cpu_hz=1,
    )
    assert status == ERR_OK
