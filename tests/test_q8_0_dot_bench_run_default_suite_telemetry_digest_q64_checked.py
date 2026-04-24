#!/usr/bin/env python3
"""Parity checks for IQ-1315 telemetry digest helper logic."""

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
        status, digest = try_mul_i64(digest, 1099511628211)
        if status != ERR_OK:
            return status, None

    return ERR_OK, digest


def test_happy_path_digest_and_determinism() -> None:
    status, digest = telemetry_digest_q64_checked(
        total_ops=8000,
        total_cycles=152000,
        cycles_per_op=19,
        remainder_cycles=0,
        suite_checksum=0x1234,
        cpu_hz=2_000_000_000,
    )
    assert status == ERR_OK
    assert digest is not None

    status2, digest2 = telemetry_digest_q64_checked(
        total_ops=8000,
        total_cycles=152000,
        cycles_per_op=19,
        remainder_cycles=0,
        suite_checksum=0x1234,
        cpu_hz=2_000_000_000,
    )
    assert status2 == ERR_OK
    assert digest == digest2


def test_input_change_changes_digest() -> None:
    status_a, digest_a = telemetry_digest_q64_checked(
        total_ops=8000,
        total_cycles=152000,
        cycles_per_op=19,
        remainder_cycles=0,
        suite_checksum=0x1234,
        cpu_hz=2_000_000_000,
    )
    status_b, digest_b = telemetry_digest_q64_checked(
        total_ops=8001,
        total_cycles=152000,
        cycles_per_op=19,
        remainder_cycles=0,
        suite_checksum=0x1234,
        cpu_hz=2_000_000_000,
    )
    assert status_a == ERR_OK and status_b == ERR_OK
    assert digest_a != digest_b


def test_bad_params() -> None:
    for args in (
        (-1, 10, 1, 0, 1, 1),
        (1, -1, 1, 0, 1, 1),
        (1, 10, -1, 0, 1, 1),
        (1, 10, 1, -1, 1, 1),
        (1, 10, 1, 0, -1, 1),
        (1, 10, 1, 0, 1, 0),
        (5, 10, 1, 6, 1, 1),
    ):
        status, _ = telemetry_digest_q64_checked(*args)
        assert status == ERR_BAD_PARAM


def test_overflow_paths() -> None:
    status, _ = telemetry_digest_q64_checked(
        total_ops=(I64_MAX // 3) + 1,
        total_cycles=1,
        cycles_per_op=1,
        remainder_cycles=0,
        suite_checksum=1,
        cpu_hz=1,
    )
    assert status == ERR_OVERFLOW

    # Force multiply overflow in digest mixing phase.
    status, _ = telemetry_digest_q64_checked(
        total_ops=I64_MAX // 6,
        total_cycles=I64_MAX // 10,
        cycles_per_op=I64_MAX // 14,
        remainder_cycles=0,
        suite_checksum=I64_MAX // 26,
        cpu_hz=1,
    )
    assert status == ERR_OVERFLOW
