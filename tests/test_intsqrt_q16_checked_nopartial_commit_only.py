#!/usr/bin/env python3
"""Parity harness for FPQ16SqrtCheckedNoPartialCommitOnly (IQ-1234)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_checked import (  # noqa: E402
    FP_Q16_ERR_DOMAIN,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    q16_from_float,
    q16_to_float,
)

FP_Q16_SHIFT = 16
I64_MAX_VALUE = (1 << 63) - 1


def fpq16_sqrt(x_q16: int) -> int:
    if x_q16 <= 0:
        return 0
    if x_q16 > (I64_MAX_VALUE >> FP_Q16_SHIFT):
        shifted = I64_MAX_VALUE
    else:
        shifted = x_q16 << FP_Q16_SHIFT
    return int(shifted**0.5)


def fpq16_sqrt_checked_nopartial_commit_only_reference(
    x_q16: int,
    out_q16: list[int] | None,
    *,
    mutate_snapshot: bool = False,
) -> int:
    if out_q16 is None:
        return FP_Q16_ERR_NULL_PTR

    if x_q16 < 0:
        return FP_Q16_ERR_DOMAIN

    if x_q16 > (I64_MAX_VALUE >> FP_Q16_SHIFT):
        return FP_Q16_ERR_OVERFLOW

    snapshot_x_q16 = x_q16
    staged_out_q16 = fpq16_sqrt(x_q16)

    if mutate_snapshot:
        x_q16 += 1

    if snapshot_x_q16 != x_q16:
        return 2  # FP_Q16_ERR_BAD_PARAM

    out_q16[0] = staged_out_q16
    return FP_Q16_OK


def explicit_commit_only_composition(x_q16: int, out_q16: list[int] | None) -> int:
    if out_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if x_q16 < 0:
        return FP_Q16_ERR_DOMAIN
    if x_q16 > (I64_MAX_VALUE >> FP_Q16_SHIFT):
        return FP_Q16_ERR_OVERFLOW

    out_q16[0] = fpq16_sqrt(x_q16)
    return FP_Q16_OK


def test_source_contains_iq1234_function() -> None:
    source = Path("src/math/intsqrt.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16SqrtCheckedNoPartialCommitOnly(I64 x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "if (!out_q16)" in body
    assert "if (x_q16 < 0)" in body
    assert "if (x_q16 > (I64_MAX_VALUE >> FP_Q16_SHIFT))" in body
    assert "snapshot_x_q16 = x_q16;" in body
    assert "staged_out_q16 = FPQ16Sqrt(x_q16);" in body
    assert "if (snapshot_x_q16 != x_q16)" in body
    assert "*out_q16 = staged_out_q16;" in body


def test_null_domain_overflow_no_publish() -> None:
    assert fpq16_sqrt_checked_nopartial_commit_only_reference(123, None) == FP_Q16_ERR_NULL_PTR

    out = [999]
    assert fpq16_sqrt_checked_nopartial_commit_only_reference(-1, out) == FP_Q16_ERR_DOMAIN
    assert out == [999]

    out = [555]
    too_large = (I64_MAX_VALUE >> FP_Q16_SHIFT) + 1
    assert (
        fpq16_sqrt_checked_nopartial_commit_only_reference(too_large, out)
        == FP_Q16_ERR_OVERFLOW
    )
    assert out == [555]


def test_success_and_explicit_composition_parity() -> None:
    vectors = [
        0,
        1,
        q16_from_float(0.25),
        q16_from_float(1.0),
        q16_from_float(2.0),
        q16_from_float(123.5),
        (I64_MAX_VALUE >> FP_Q16_SHIFT),
    ]

    for x_q16 in vectors:
        out_a = [0]
        out_b = [0]
        err_a = fpq16_sqrt_checked_nopartial_commit_only_reference(x_q16, out_a)
        err_b = explicit_commit_only_composition(x_q16, out_b)
        assert err_a == FP_Q16_OK
        assert err_b == FP_Q16_OK
        assert out_a == out_b


def test_snapshot_mutation_guard_returns_bad_param() -> None:
    out = [123]
    err = fpq16_sqrt_checked_nopartial_commit_only_reference(
        q16_from_float(9.0), out, mutate_snapshot=True
    )
    assert err == 2  # FP_Q16_ERR_BAD_PARAM
    assert out == [123]


def test_randomized_parity_and_accuracy() -> None:
    rng = random.Random(20260423_1234)

    for _ in range(5000):
        val = rng.uniform(0.0, 50000.0)
        x_q16 = q16_from_float(val)
        out = [0]
        err = fpq16_sqrt_checked_nopartial_commit_only_reference(x_q16, out)
        assert err == FP_Q16_OK

        got = q16_to_float(out[0])
        want = val**0.5
        assert abs(got - want) <= (3.0 / (1 << 16)) + 1e-7


def run() -> None:
    test_source_contains_iq1234_function()
    test_null_domain_overflow_no_publish()
    test_success_and_explicit_composition_parity()
    test_snapshot_mutation_guard_returns_bad_param()
    test_randomized_parity_and_accuracy()
    print("intsqrt_q16_checked_nopartial_commit_only_reference_checks=ok")


if __name__ == "__main__":
    run()
