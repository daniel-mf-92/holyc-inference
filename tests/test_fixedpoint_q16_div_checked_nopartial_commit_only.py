#!/usr/bin/env python3
"""Parity harness for FPQ16DivCheckedNoPartialCommitOnly (IQ-1226)."""

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
    fpq16_div_checked,
    q16_from_float,
    q16_to_float,
)


def fpq16_div_checked_nopartial_commit_only_reference(
    numerator_q16: int,
    denominator_q16: int,
    out_q16: list[int] | None,
    *,
    mutate_snapshot: bool = False,
) -> int:
    if out_q16 is None:
        return FP_Q16_ERR_NULL_PTR

    snapshot_numerator_q16 = numerator_q16
    snapshot_denominator_q16 = denominator_q16

    status, staged_out_q16 = fpq16_div_checked(numerator_q16, denominator_q16)
    if status != FP_Q16_OK:
        return status

    if mutate_snapshot:
        denominator_q16 += 1

    if (
        snapshot_numerator_q16 != numerator_q16
        or snapshot_denominator_q16 != denominator_q16
    ):
        return 2  # FP_Q16_ERR_BAD_PARAM

    out_q16[0] = staged_out_q16
    return FP_Q16_OK


def explicit_commit_only_composition(
    numerator_q16: int,
    denominator_q16: int,
    out_q16: list[int] | None,
) -> int:
    if out_q16 is None:
        return FP_Q16_ERR_NULL_PTR

    status, value = fpq16_div_checked(numerator_q16, denominator_q16)
    if status != FP_Q16_OK:
        return status

    out_q16[0] = value
    return FP_Q16_OK


def test_source_contains_iq1226_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16DivCheckedNoPartialCommitOnly(I64 numerator_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("// Checked elementwise Q16 multiply helper", 1)[0]
    assert "snapshot_numerator_q16 = numerator_q16;" in body
    assert "snapshot_denominator_q16 = denominator_q16;" in body
    assert "status = FPQ16DivChecked(numerator_q16," in body
    assert "if (snapshot_numerator_q16 != numerator_q16 ||" in body
    assert "*out_q16 = staged_out_q16;" in body


def test_null_and_domain_and_no_publish() -> None:
    err = fpq16_div_checked_nopartial_commit_only_reference(123, 456, None)
    assert err == FP_Q16_ERR_NULL_PTR

    out = [777]
    err = fpq16_div_checked_nopartial_commit_only_reference(123, 0, out)
    assert err == FP_Q16_ERR_DOMAIN
    assert out == [777]


def test_success_and_explicit_composition_parity() -> None:
    numerators = [
        q16_from_float(3.5),
        q16_from_float(-7.0),
        q16_from_float(0.125),
        q16_from_float(19.25),
    ]
    denominators = [
        q16_from_float(0.5),
        q16_from_float(2.0),
        q16_from_float(-4.0),
        q16_from_float(3.25),
    ]

    for num_q16, den_q16 in zip(numerators, denominators):
        out_a = [0]
        out_b = [0]

        err_a = fpq16_div_checked_nopartial_commit_only_reference(num_q16, den_q16, out_a)
        err_b = explicit_commit_only_composition(num_q16, den_q16, out_b)

        assert err_a == FP_Q16_OK
        assert err_b == FP_Q16_OK
        assert out_a == out_b


def test_snapshot_mutation_guard_returns_bad_param() -> None:
    out = [123]
    err = fpq16_div_checked_nopartial_commit_only_reference(
        q16_from_float(4.0),
        q16_from_float(2.0),
        out,
        mutate_snapshot=True,
    )
    assert err == 2  # FP_Q16_ERR_BAD_PARAM
    assert out == [123]


def test_randomized_parity_against_checked_divider() -> None:
    rng = random.Random(20260423_1226)

    for _ in range(8000):
        num_q16 = rng.randint(-(1 << 34), 1 << 34)
        den_q16 = rng.randint(-(1 << 34), 1 << 34)
        if den_q16 == 0:
            den_q16 = 1

        out = [0]
        err_commit = fpq16_div_checked_nopartial_commit_only_reference(num_q16, den_q16, out)
        err_base, base = fpq16_div_checked(num_q16, den_q16)

        assert err_commit == err_base
        if err_commit == FP_Q16_OK:
            assert out[0] == base
        else:
            assert err_commit in (FP_Q16_ERR_DOMAIN, FP_Q16_ERR_OVERFLOW)


def test_randomized_value_accuracy_when_ok() -> None:
    rng = random.Random(20260423_1227)

    for _ in range(4000):
        num = rng.uniform(-1024.0, 1024.0)
        den = rng.uniform(-1024.0, 1024.0)
        if abs(den) < 1e-6:
            den = 0.5

        num_q16 = q16_from_float(num)
        den_q16 = q16_from_float(den)
        if den_q16 == 0:
            den_q16 = 1

        out = [0]
        err = fpq16_div_checked_nopartial_commit_only_reference(num_q16, den_q16, out)
        if err != FP_Q16_OK:
            assert err in (FP_Q16_ERR_OVERFLOW, FP_Q16_ERR_DOMAIN)
            continue

        got = q16_to_float(out[0])
        want = (num_q16 / (1 << 16)) / (den_q16 / (1 << 16))
        assert abs(got - want) <= (2.0 / (1 << 16)) + 1e-9


def run() -> None:
    test_source_contains_iq1226_function()
    test_null_and_domain_and_no_publish()
    test_success_and_explicit_composition_parity()
    test_snapshot_mutation_guard_returns_bad_param()
    test_randomized_parity_against_checked_divider()
    test_randomized_value_accuracy_when_ok()
    print("fixedpoint_q16_div_checked_nopartial_commit_only_reference_checks=ok")


if __name__ == "__main__":
    run()
