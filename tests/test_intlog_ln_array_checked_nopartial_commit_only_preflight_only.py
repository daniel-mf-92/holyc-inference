#!/usr/bin/env python3
"""Parity harness for FPQ16LnArrayCheckedNoPartialCommitOnlyPreflightOnly (IQ-1159)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_intlog_ln_array_checked_nopartial import (
    FP_Q16_ONE,
    I64_MAX_VALUE,
    INTLOG_STATUS_BADCOUNT,
    INTLOG_STATUS_BADPTR,
    INTLOG_STATUS_OK,
    INTLOG_STATUS_OVERFLOW,
    fpq16_ln_array_checked_no_partial_preflight,
)


def fpq16_ln_array_checked_nopartial_commit_only_preflight_only(
    input_q16: list[int] | None,
    count: int,
    out_q16: list[int] | None,
    out_capacity_bytes: int,
    required_bytes_out: list[int] | None,
) -> int:
    if required_bytes_out is None:
        return INTLOG_STATUS_BADPTR

    required_bytes_out[0] = 0

    status, preflight_required = fpq16_ln_array_checked_no_partial_preflight(
        input_q16, count, out_q16
    )
    if status != INTLOG_STATUS_OK:
        return status

    if out_capacity_bytes < preflight_required:
        return INTLOG_STATUS_BADCOUNT

    if count == 0:
        geometry_required = 0
    elif count > (I64_MAX_VALUE // 8):
        return INTLOG_STATUS_OVERFLOW
    else:
        geometry_required = count * 8

    if geometry_required != preflight_required:
        return INTLOG_STATUS_OVERFLOW

    required_bytes_out[0] = preflight_required
    return INTLOG_STATUS_OK


def test_source_contains_iq1159_signature_and_no_write_contract() -> None:
    src = Path("src/math/intlog.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16LnArrayCheckedNoPartialCommitOnlyPreflightOnly(I64 *input_q16,"
    assert sig in src
    body = src.split(sig, 1)[1].split("I32 FPQ16LnArrayCheckedNoPartialDefault(", 1)[0]

    assert "if (!required_bytes_out)" in body
    assert "*required_bytes_out = 0;" in body
    assert "status = FPQ16LnArrayCheckedNoPartialPreflight(input_q16," in body
    assert "if (out_capacity_bytes < preflight_required_bytes)" in body
    assert "if (geometry_required_bytes != preflight_required_bytes)" in body
    assert "*required_bytes_out = preflight_required_bytes;" in body
    assert "out_q16[i] =" not in body


def test_null_required_pointer_fails_fast_and_keeps_output() -> None:
    out = [111, 222, 333]
    st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only(
        [FP_Q16_ONE, FP_Q16_ONE * 2, FP_Q16_ONE * 3],
        3,
        out,
        24,
        None,
    )
    assert st == INTLOG_STATUS_BADPTR
    assert out == [111, 222, 333]


def test_preflight_failures_and_capacity_failure_keep_output_unchanged() -> None:
    out = [888, 777, 666]
    need = [123]

    st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only(None, 3, out, 24, need)
    assert st == INTLOG_STATUS_BADPTR
    assert need[0] == 0
    assert out == [888, 777, 666]

    st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only(
        [FP_Q16_ONE], -1, out, 24, need
    )
    assert st == INTLOG_STATUS_BADCOUNT
    assert need[0] == 0
    assert out == [888, 777, 666]

    st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only(
        [FP_Q16_ONE, FP_Q16_ONE * 2, FP_Q16_ONE * 3],
        3,
        out,
        8,
        need,
    )
    assert st == INTLOG_STATUS_BADCOUNT
    assert need[0] == 0
    assert out == [888, 777, 666]


def test_success_returns_required_bytes_without_mutating_output() -> None:
    inp = [0, -1, FP_Q16_ONE // 2, FP_Q16_ONE, FP_Q16_ONE * 4]
    out = [0xAA55AA55AA55AA55] * len(inp)
    need = [0]

    st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only(
        inp, len(inp), out, len(inp) * 8, need
    )
    assert st == INTLOG_STATUS_OK
    assert need[0] == len(inp) * 8
    assert out == [0xAA55AA55AA55AA55] * len(inp)


def test_randomized_required_bytes_match_preflight_geometry() -> None:
    rng = random.Random(1159)
    for _ in range(300):
        count = rng.randint(0, 128)
        inp = [rng.randint(-(1 << 20), 1 << 30) for _ in range(count)]
        out = [0x5A5A5A5A5A5A5A5A] * count
        need = [999999]

        st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only(
            inp, count, out, count * 8, need
        )
        assert st == INTLOG_STATUS_OK
        assert need[0] == count * 8
        assert out == [0x5A5A5A5A5A5A5A5A] * count


def run() -> None:
    test_source_contains_iq1159_signature_and_no_write_contract()
    test_null_required_pointer_fails_fast_and_keeps_output()
    test_preflight_failures_and_capacity_failure_keep_output_unchanged()
    test_success_returns_required_bytes_without_mutating_output()
    test_randomized_required_bytes_match_preflight_geometry()
    print("intlog_ln_array_checked_nopartial_commit_only_preflight_only_reference_checks=ok")


if __name__ == "__main__":
    run()
