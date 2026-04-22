#!/usr/bin/env python3
"""Parity harness for FPQ16LnArrayCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1160)."""

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
    fpq16_ln,
    fpq16_ln_array_checked_no_partial,
)
from test_intlog_ln_array_checked_nopartial_commit_only_preflight_only import (
    fpq16_ln_array_checked_nopartial_commit_only_preflight_only,
)

INTLOG_STATUS_DOMAIN = -4


def fpq16_ln_array_checked_nopartial_commit_only_preflight_only_parity(
    input_q16: list[int] | None,
    count: int,
    out_q16: list[int] | None,
    out_capacity_bytes: int,
    required_bytes_out: list[int] | None,
) -> int:
    if required_bytes_out is None:
        return INTLOG_STATUS_BADPTR

    required_bytes_out[0] = 0

    snapshot_input = input_q16
    snapshot_count = count
    snapshot_out = out_q16
    snapshot_capacity = out_capacity_bytes

    staged_required = [0]
    status = fpq16_ln_array_checked_nopartial_commit_only_preflight_only(
        input_q16,
        count,
        out_q16,
        out_capacity_bytes,
        staged_required,
    )
    if status != INTLOG_STATUS_OK:
        return status

    if (
        snapshot_input is not input_q16
        or snapshot_count != count
        or snapshot_out is not out_q16
        or snapshot_capacity != out_capacity_bytes
    ):
        return INTLOG_STATUS_BADCOUNT

    if count == 0:
        canonical_required = 0
    elif count > (I64_MAX_VALUE // 8):
        return INTLOG_STATUS_OVERFLOW
    else:
        canonical_required = count * 8

    if staged_required[0] != canonical_required:
        return INTLOG_STATUS_OVERFLOW

    if count == 0:
        required_bytes_out[0] = staged_required[0]
        return INTLOG_STATUS_OK

    scratch = [0] * count
    status = fpq16_ln_array_checked_no_partial(input_q16, count, scratch, canonical_required)
    if status != INTLOG_STATUS_OK:
        return status

    if (
        snapshot_input is not input_q16
        or snapshot_count != count
        or snapshot_out is not out_q16
        or snapshot_capacity != out_capacity_bytes
    ):
        return INTLOG_STATUS_BADCOUNT

    assert input_q16 is not None
    for i in range(count):
        if scratch[i] != fpq16_ln(input_q16[i]):
            return INTLOG_STATUS_DOMAIN

    required_bytes_out[0] = staged_required[0]
    return INTLOG_STATUS_OK


def test_source_contains_iq1160_signature_and_parity_contract() -> None:
    src = Path("src/math/intlog.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16LnArrayCheckedNoPartialCommitOnlyPreflightOnlyParity(I64 *input_q16,"
    assert sig in src
    body = src.split(sig, 1)[1].split("I32 FPQ16LnArrayCheckedNoPartialDefault(", 1)[0]

    assert "if (!required_bytes_out)" in body
    assert "*required_bytes_out = 0;" in body
    assert "status = FPQ16LnArrayCheckedNoPartialCommitOnlyPreflightOnly(input_q16," in body
    assert "status = FPQ16LnArrayCheckedNoPartial(input_q16," in body
    assert "if (staged_required_bytes != canonical_required_bytes)" in body
    assert "if (scratch_out_q16[i] != FPQ16Ln(input_q16[i]))" in body
    assert "*required_bytes_out = staged_required_bytes;" in body
    assert "out_q16[i] =" not in body


def test_null_required_pointer_and_preflight_failures_keep_state() -> None:
    out = [777, 888, 999]

    st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only_parity(
        [FP_Q16_ONE, FP_Q16_ONE * 2, FP_Q16_ONE * 3],
        3,
        out,
        24,
        None,
    )
    assert st == INTLOG_STATUS_BADPTR
    assert out == [777, 888, 999]

    need = [12345]
    st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only_parity(
        None,
        3,
        out,
        24,
        need,
    )
    assert st == INTLOG_STATUS_BADPTR
    assert need[0] == 0
    assert out == [777, 888, 999]

    st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only_parity(
        [FP_Q16_ONE],
        -1,
        out,
        24,
        need,
    )
    assert st == INTLOG_STATUS_BADCOUNT
    assert need[0] == 0
    assert out == [777, 888, 999]


def test_capacity_failure_and_success_no_partial_publish() -> None:
    inp = [0, -1, FP_Q16_ONE // 2, FP_Q16_ONE, FP_Q16_ONE * 4]
    out = [0xDEADBEEFDEADBEEF] * len(inp)
    need = [777]

    st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only_parity(
        inp,
        len(inp),
        out,
        8,
        need,
    )
    assert st == INTLOG_STATUS_BADCOUNT
    assert need[0] == 0
    assert out == [0xDEADBEEFDEADBEEF] * len(inp)

    st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only_parity(
        inp,
        len(inp),
        out,
        len(inp) * 8,
        need,
    )
    assert st == INTLOG_STATUS_OK
    assert need[0] == len(inp) * 8
    assert out == [0xDEADBEEFDEADBEEF] * len(inp)


def test_randomized_required_bytes_and_scalar_lane_parity() -> None:
    rng = random.Random(1160)
    for _ in range(300):
        count = rng.randint(0, 96)
        inp = []
        for _ in range(count):
            selector = rng.randint(0, 6)
            if selector == 0:
                inp.append(0)
            elif selector == 1:
                inp.append(-rng.randint(1, 1 << 20))
            else:
                inp.append(rng.randint(1, 1 << 30))

        out = [0xA5A5A5A5A5A5A5A5] * count
        need = [1]

        st = fpq16_ln_array_checked_nopartial_commit_only_preflight_only_parity(
            inp,
            count,
            out,
            count * 8,
            need,
        )
        assert st == INTLOG_STATUS_OK
        assert need[0] == count * 8
        assert out == [0xA5A5A5A5A5A5A5A5] * count


def run() -> None:
    test_source_contains_iq1160_signature_and_parity_contract()
    test_null_required_pointer_and_preflight_failures_keep_state()
    test_capacity_failure_and_success_no_partial_publish()
    test_randomized_required_bytes_and_scalar_lane_parity()
    print("intlog_ln_array_checked_nopartial_commit_only_preflight_only_parity_reference_checks=ok")


if __name__ == "__main__":
    run()
