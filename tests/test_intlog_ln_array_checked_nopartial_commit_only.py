#!/usr/bin/env python3
"""Parity harness for FPQ16LnArrayCheckedNoPartialCommitOnly (IQ-1158)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_intlog_ln_array_checked_nopartial import (
    FP_Q16_ONE,
    INTLOG_STATUS_BADCOUNT,
    INTLOG_STATUS_BADPTR,
    INTLOG_STATUS_OK,
    fpq16_ln,
    fpq16_ln_array_checked_no_partial,
    fpq16_ln_array_checked_no_partial_preflight,
)


def fpq16_ln_array_checked_nopartial_commit_only(
    input_q16: list[int] | None,
    count: int,
    out_q16: list[int] | None,
    out_capacity_bytes: int,
) -> int:
    status, required_bytes = fpq16_ln_array_checked_no_partial_preflight(
        input_q16, count, out_q16
    )
    if status != INTLOG_STATUS_OK:
        return status
    if out_capacity_bytes < required_bytes:
        return INTLOG_STATUS_BADCOUNT

    if count == 0:
        return INTLOG_STATUS_OK

    assert input_q16 is not None
    assert out_q16 is not None

    staged = [0] * count
    for i in range(count):
        staged[i] = fpq16_ln(input_q16[i])
    for i in range(count):
        out_q16[i] = staged[i]
    return INTLOG_STATUS_OK


def test_source_contains_iq1158_signature_and_staged_publish() -> None:
    src = Path("src/math/intlog.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16LnArrayCheckedNoPartialCommitOnly(I64 *input_q16,"
    assert sig in src
    body = src.split(sig, 1)[1].split("I32 FPQ16LnArrayCheckedNoPartialDefault(", 1)[0]

    assert "status = FPQ16LnArrayCheckedNoPartialPreflight(input_q16," in body
    assert "if (out_capacity_bytes < required_bytes)" in body
    assert "if (!count)" in body
    assert "staged_out_q16 = MAlloc(required_bytes);" in body
    assert "for (i = 0; i < count; i++)" in body
    assert "staged_out_q16[i] = FPQ16Ln(input_q16[i]);" in body
    assert "out_q16[i] = staged_out_q16[i];" in body
    assert "Free(staged_out_q16);" in body


def test_preflight_failures_preserve_output_state() -> None:
    out = [111, 222, 333]

    st = fpq16_ln_array_checked_nopartial_commit_only(None, 3, out, 24)
    assert st == INTLOG_STATUS_BADPTR
    assert out == [111, 222, 333]

    st = fpq16_ln_array_checked_nopartial_commit_only([FP_Q16_ONE], -1, out, 24)
    assert st == INTLOG_STATUS_BADCOUNT
    assert out == [111, 222, 333]


def test_capacity_failure_keeps_no_partial_output() -> None:
    inp = [FP_Q16_ONE // 2, FP_Q16_ONE, FP_Q16_ONE * 2]
    out = [999, 999, 999]

    st = fpq16_ln_array_checked_nopartial_commit_only(inp, len(inp), out, 8)
    assert st == INTLOG_STATUS_BADCOUNT
    assert out == [999, 999, 999]


def test_success_matches_non_commit_wrapper_outputs() -> None:
    inp = [
        -1,
        0,
        FP_Q16_ONE // 8,
        FP_Q16_ONE // 2,
        FP_Q16_ONE,
        FP_Q16_ONE * 3,
        FP_Q16_ONE * 16,
    ]
    out_commit = [0] * len(inp)
    out_direct = [0] * len(inp)

    st0 = fpq16_ln_array_checked_nopartial_commit_only(inp, len(inp), out_commit, len(inp) * 8)
    st1 = fpq16_ln_array_checked_no_partial(inp, len(inp), out_direct, len(inp) * 8)

    assert st0 == INTLOG_STATUS_OK
    assert st1 == INTLOG_STATUS_OK
    assert out_commit == out_direct


def test_randomized_parity_vectors() -> None:
    rng = random.Random(1158)
    for _ in range(200):
        count = rng.randint(0, 32)
        inp = []
        for _ in range(count):
            selector = rng.randint(0, 5)
            if selector == 0:
                inp.append(0)
            elif selector == 1:
                inp.append(-rng.randint(1, 1 << 20))
            else:
                inp.append(rng.randint(1, 1 << 30))

        out_commit = [0x55AA55AA55AA55AA] * count
        out_direct = [0] * count

        st = fpq16_ln_array_checked_nopartial_commit_only(inp, count, out_commit, count * 8)
        assert st == INTLOG_STATUS_OK

        st2 = fpq16_ln_array_checked_no_partial(inp, count, out_direct, count * 8)
        assert st2 == INTLOG_STATUS_OK
        assert out_commit == out_direct


def run() -> None:
    test_source_contains_iq1158_signature_and_staged_publish()
    test_preflight_failures_preserve_output_state()
    test_capacity_failure_keeps_no_partial_output()
    test_success_matches_non_commit_wrapper_outputs()
    test_randomized_parity_vectors()
    print("intlog_ln_array_checked_nopartial_commit_only_reference_checks=ok")


if __name__ == "__main__":
    run()
