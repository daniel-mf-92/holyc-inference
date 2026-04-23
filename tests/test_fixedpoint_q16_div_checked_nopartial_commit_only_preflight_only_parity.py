#!/usr/bin/env python3
"""Parity harness for FPQ16DivCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1230)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_checked import (  # noqa: E402
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_DOMAIN,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    fpq16_div_checked,
    q16_from_float,
)
from test_fixedpoint_q16_div_checked_nopartial_commit_only import (  # noqa: E402
    fpq16_div_checked_nopartial_commit_only_reference,
)
from test_fixedpoint_q16_div_checked_nopartial_commit_only_preflight_only import (  # noqa: E402
    fpq16_div_checked_nopartial_commit_only_preflight_only_reference,
)


def fpq16_div_checked_nopartial_commit_only_preflight_only_parity_reference(
    numerator_q16: int,
    denominator_q16: int,
    out_required_out_cells: list[int] | None,
    *,
    mutate_snapshot: bool = False,
    mutate_tuple: bool = False,
) -> int:
    if out_required_out_cells is None:
        return FP_Q16_ERR_NULL_PTR

    snapshot_numerator_q16 = numerator_q16
    snapshot_denominator_q16 = denominator_q16

    staged_preflight_required = [0]
    status = fpq16_div_checked_nopartial_commit_only_preflight_only_reference(
        numerator_q16,
        denominator_q16,
        staged_preflight_required,
    )
    if status != FP_Q16_OK:
        return status

    staged_commit_out = [0]
    status = fpq16_div_checked_nopartial_commit_only_reference(
        numerator_q16,
        denominator_q16,
        staged_commit_out,
    )
    if status != FP_Q16_OK:
        return status

    if mutate_snapshot:
        denominator_q16 += 1

    recomputed_required_out_cells = 1
    if mutate_tuple:
        recomputed_required_out_cells += 1

    if (
        snapshot_numerator_q16 != numerator_q16
        or snapshot_denominator_q16 != denominator_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    if staged_preflight_required[0] != recomputed_required_out_cells:
        return FP_Q16_ERR_BAD_PARAM

    if recomputed_required_out_cells != 1:
        return FP_Q16_ERR_BAD_PARAM

    out_required_out_cells[0] = staged_preflight_required[0]
    return FP_Q16_OK


def explicit_parity_composition(
    numerator_q16: int,
    denominator_q16: int,
    out_required_out_cells: list[int] | None,
) -> int:
    if out_required_out_cells is None:
        return FP_Q16_ERR_NULL_PTR

    staged_preflight = [0]
    status = fpq16_div_checked_nopartial_commit_only_preflight_only_reference(
        numerator_q16,
        denominator_q16,
        staged_preflight,
    )
    if status != FP_Q16_OK:
        return status

    staged_commit = [0]
    status = fpq16_div_checked_nopartial_commit_only_reference(
        numerator_q16,
        denominator_q16,
        staged_commit,
    )
    if status != FP_Q16_OK:
        return status

    if staged_preflight[0] != 1:
        return FP_Q16_ERR_BAD_PARAM

    out_required_out_cells[0] = staged_preflight[0]
    return FP_Q16_OK


def test_source_contains_iq1230_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16DivCheckedNoPartialCommitOnlyPreflightOnlyParity(I64 numerator_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("// Checked elementwise Q16 multiply helper", 1)[0]
    assert "status = FPQ16DivCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "status = FPQ16DivCheckedNoPartialCommitOnly(numerator_q16," in body
    assert "if (snapshot_numerator_q16 != numerator_q16 ||" in body
    assert "recomputed_required_out_cells = 1;" in body
    assert "if (staged_preflight_required_out_cells != recomputed_required_out_cells)" in body
    assert "*out_required_out_cells = staged_preflight_required_out_cells;" in body


def test_null_domain_overflow_and_no_publish() -> None:
    err = fpq16_div_checked_nopartial_commit_only_preflight_only_parity_reference(1, 1, None)
    assert err == FP_Q16_ERR_NULL_PTR

    out = [777]
    err = fpq16_div_checked_nopartial_commit_only_preflight_only_parity_reference(123, 0, out)
    assert err == FP_Q16_ERR_DOMAIN
    assert out == [777]

    huge_num = (1 << 62)
    tiny_den = 1
    out = [888]
    err = fpq16_div_checked_nopartial_commit_only_preflight_only_parity_reference(
        huge_num,
        tiny_den,
        out,
    )
    assert err in (FP_Q16_ERR_OVERFLOW, FP_Q16_ERR_DOMAIN)
    assert out == [888]


def test_snapshot_and_tuple_parity_guards() -> None:
    out = [123]
    err = fpq16_div_checked_nopartial_commit_only_preflight_only_parity_reference(
        q16_from_float(4.0),
        q16_from_float(2.0),
        out,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [123]

    out = [456]
    err = fpq16_div_checked_nopartial_commit_only_preflight_only_parity_reference(
        q16_from_float(4.0),
        q16_from_float(2.0),
        out,
        mutate_tuple=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [456]


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

        err_a = fpq16_div_checked_nopartial_commit_only_preflight_only_parity_reference(
            num_q16,
            den_q16,
            out_a,
        )
        err_b = explicit_parity_composition(num_q16, den_q16, out_b)

        assert err_a == FP_Q16_OK
        assert err_b == FP_Q16_OK
        assert out_a == out_b == [1]


def test_randomized_parity_against_components_and_base() -> None:
    rng = random.Random(20260423_1230)

    for _ in range(12000):
        num_q16 = rng.randint(-(1 << 34), 1 << 34)
        den_q16 = rng.randint(-(1 << 34), 1 << 34)
        if den_q16 == 0:
            den_q16 = 1

        out_required = [9]
        err_parity = fpq16_div_checked_nopartial_commit_only_preflight_only_parity_reference(
            num_q16,
            den_q16,
            out_required,
        )

        out_pre = [0]
        err_pre = fpq16_div_checked_nopartial_commit_only_preflight_only_reference(
            num_q16,
            den_q16,
            out_pre,
        )

        out_commit = [0]
        err_commit = fpq16_div_checked_nopartial_commit_only_reference(
            num_q16,
            den_q16,
            out_commit,
        )

        err_base, _ = fpq16_div_checked(num_q16, den_q16)

        assert err_parity == err_pre == err_commit == err_base
        if err_parity == FP_Q16_OK:
            assert out_required == [1]
            assert out_pre == [1]
        else:
            assert out_required == [9]
            assert err_parity in (FP_Q16_ERR_DOMAIN, FP_Q16_ERR_OVERFLOW)


def run() -> None:
    test_source_contains_iq1230_function()
    test_null_domain_overflow_and_no_publish()
    test_snapshot_and_tuple_parity_guards()
    test_success_and_explicit_composition_parity()
    test_randomized_parity_against_components_and_base()
    print("fixedpoint_q16_div_checked_nopartial_commit_only_preflight_only_parity_reference_checks=ok")


if __name__ == "__main__":
    run()
