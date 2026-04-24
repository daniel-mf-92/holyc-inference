#!/usr/bin/env python3
"""Commit-only hardening wrapper checks for IQ-1347 telemetry digest wrapper."""

from __future__ import annotations

from pathlib import Path

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


def telemetry_digest_q64_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
    total_ops: int,
    total_cycles: int,
    cycles_per_op: int,
    remainder_cycles: int,
    suite_checksum: int,
    cpu_hz: int,
    out_ptr: list[int] | None,
    *,
    force_write: bool = False,
) -> int:
    if out_ptr is None:
        return ERR_NULL_PTR
    out_snapshot = out_ptr[0]
    status, _ = telemetry_digest_q64_checked(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
    )
    if status != ERR_OK:
        return status
    if force_write:
        out_ptr[0] = 7
    if out_ptr[0] != out_snapshot:
        return ERR_BAD_PARAM
    return ERR_OK


def telemetry_digest_q64_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
    total_ops: int,
    total_cycles: int,
    cycles_per_op: int,
    remainder_cycles: int,
    suite_checksum: int,
    cpu_hz: int,
    out_ptr: list[int] | None,
    *,
    force_digest: int | None = None,
) -> int:
    if out_ptr is None:
        return ERR_NULL_PTR
    status, digest = telemetry_digest_q64_checked(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
    )
    if status != ERR_OK:
        return status
    if force_digest is not None:
        digest = force_digest
    if digest is None:
        return ERR_BAD_PARAM
    out_ptr[0] = digest
    return ERR_OK


def telemetry_digest_q64_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
    total_ops: int,
    total_cycles: int,
    cycles_per_op: int,
    remainder_cycles: int,
    suite_checksum: int,
    cpu_hz: int,
    out_ptr: list[int] | None,
    *,
    force_preflight_write: bool = False,
    force_parity_digest: int | None = None,
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

    snapshot = (
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
    )
    out_snapshot = out_ptr[0]

    preflight_probe = [3689348814741910323]
    status = telemetry_digest_q64_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        preflight_probe,
        force_write=force_preflight_write,
    )
    if status != ERR_OK:
        return status

    parity_digest = [0]
    status = telemetry_digest_q64_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        parity_digest,
        force_digest=force_parity_digest,
    )
    if status != ERR_OK:
        return status

    if snapshot != (
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
    ):
        return ERR_BAD_PARAM
    if out_ptr[0] != out_snapshot:
        return ERR_BAD_PARAM

    if preflight_probe[0] != 3689348814741910323:
        return ERR_BAD_PARAM

    out_ptr[0] = parity_digest[0]
    return ERR_OK


def test_source_contains_iq1347_function_and_guards() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = (
        "I32 Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartial"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
    )
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "if (!out_digest_q64)" in body
    assert "preflight_probe_digest_q64 = 3689348814741910323;" in body
    assert "status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "if (snapshot_out_digest_q64 != out_digest_q64)" in body
    assert "if (preflight_probe_digest_q64 != 3689348814741910323)" in body
    assert "*out_digest_q64 = parity_digest_q64;" in body


def test_happy_path_publishes_digest() -> None:
    out = [0xABCD]
    status = telemetry_digest_q64_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        total_ops=6144,
        total_cycles=159744,
        cycles_per_op=26,
        remainder_cycles=0,
        suite_checksum=0x99,
        cpu_hz=3_000_000_000,
        out_ptr=out,
    )
    assert status == ERR_OK
    expected_status, expected_digest = telemetry_digest_q64_checked(6144, 159744, 26, 0, 0x99, 3_000_000_000)
    assert expected_status == ERR_OK
    assert out[0] == expected_digest


def test_preflight_write_rejected() -> None:
    out = [0x99]
    status = telemetry_digest_q64_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        total_ops=1024,
        total_cycles=28672,
        cycles_per_op=28,
        remainder_cycles=0,
        suite_checksum=0x11,
        cpu_hz=2_400_000_000,
        out_ptr=out,
        force_preflight_write=True,
    )
    assert status == ERR_BAD_PARAM
    assert out[0] == 0x99


def test_bad_param_rejected() -> None:
    out = [0x55]
    status = telemetry_digest_q64_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        total_ops=-1,
        total_cycles=1,
        cycles_per_op=1,
        remainder_cycles=0,
        suite_checksum=1,
        cpu_hz=1,
        out_ptr=out,
    )
    assert status == ERR_BAD_PARAM
    assert out[0] == 0x55


def test_null_ptr() -> None:
    status = telemetry_digest_q64_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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
    test_source_contains_iq1347_function_and_guards()
    test_happy_path_publishes_digest()
    test_preflight_write_rejected()
    test_bad_param_rejected()
    test_null_ptr()
    print("ok")
