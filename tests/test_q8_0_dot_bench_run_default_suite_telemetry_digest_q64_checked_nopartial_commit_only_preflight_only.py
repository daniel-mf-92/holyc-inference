#!/usr/bin/env python3
"""Parity + zero-write diagnostics checks for IQ-1317 preflight-only digest wrapper."""

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

    return telemetry_digest_q64_checked(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
    )


def telemetry_digest_q64_checked_nopartial_commit_only_preflight_only(
    total_ops: int,
    total_cycles: int,
    cycles_per_op: int,
    remainder_cycles: int,
    suite_checksum: int,
    cpu_hz: int,
    out_ptr: list[int] | None,
) -> int:
    if out_ptr is None:
        return ERR_NULL_PTR

    if (
        total_ops < 0
        or total_cycles < 0
        or cycles_per_op < 0
        or remainder_cycles < 0
        or suite_checksum < 0
        or cpu_hz <= 0
    ):
        return ERR_BAD_PARAM

    out_snapshot = out_ptr[0]
    snapshot = (total_ops, total_cycles, cycles_per_op, remainder_cycles, suite_checksum, cpu_hz)

    status, commit_digest = telemetry_digest_q64_checked_nopartial_commit_only(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
    )
    if status != ERR_OK:
        return status

    status, canonical_digest = telemetry_digest_q64_checked(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
    )
    if status != ERR_OK:
        return status

    if snapshot != (total_ops, total_cycles, cycles_per_op, remainder_cycles, suite_checksum, cpu_hz):
        return ERR_BAD_PARAM

    if commit_digest != canonical_digest:
        return ERR_BAD_PARAM

    if out_ptr[0] != out_snapshot:
        return ERR_BAD_PARAM

    return ERR_OK


def test_happy_path_parity_and_zero_write() -> None:
    out = [0x1234_5678_9ABC_DEF0]
    status = telemetry_digest_q64_checked_nopartial_commit_only_preflight_only(
        total_ops=8_000,
        total_cycles=152_000,
        cycles_per_op=19,
        remainder_cycles=0,
        suite_checksum=0x1234,
        cpu_hz=2_000_000_000,
        out_ptr=out,
    )
    assert status == ERR_OK
    assert out[0] == 0x1234_5678_9ABC_DEF0


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
        out = [999]
        status = telemetry_digest_q64_checked_nopartial_commit_only_preflight_only(*args, out_ptr=out)
        assert status == ERR_BAD_PARAM
        assert out[0] == 999


def test_overflow_propagation_and_zero_write() -> None:
    out = [777]
    status = telemetry_digest_q64_checked_nopartial_commit_only_preflight_only(
        total_ops=(I64_MAX // 3) + 1,
        total_cycles=1,
        cycles_per_op=1,
        remainder_cycles=0,
        suite_checksum=1,
        cpu_hz=1,
        out_ptr=out,
    )
    assert status == ERR_OVERFLOW
    assert out[0] == 777


def test_null_ptr() -> None:
    status = telemetry_digest_q64_checked_nopartial_commit_only_preflight_only(
        total_ops=1,
        total_cycles=1,
        cycles_per_op=1,
        remainder_cycles=0,
        suite_checksum=1,
        cpu_hz=1,
        out_ptr=None,
    )
    assert status == ERR_NULL_PTR


if __name__ == "__main__":
    test_happy_path_parity_and_zero_write()
    test_bad_params()
    test_overflow_propagation_and_zero_write()
    test_null_ptr()
    print("ok")
