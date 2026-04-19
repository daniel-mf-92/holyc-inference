#!/usr/bin/env python3
"""Parity checks for FPQ16ExpPolyCoreLn2ResidualQ16CheckedNoPartial semantics."""

from __future__ import annotations

import pathlib
import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

EXP_Q16_LN2 = 45426
EXP_Q16_POLY_C0 = FP_Q16_ONE
EXP_Q16_POLY_C2_DIV = 2
EXP_Q16_POLY_C3_DIV = 6
EXP_Q16_POLY_C4_DIV = 24
I64_MAX_VALUE = (1 << 63) - 1


def fpq16_mul_checked(a_q16: int, b_q16: int) -> tuple[int, int]:
    product = a_q16 * b_q16
    q16 = int(product >> FP_Q16_SHIFT)
    if q16 > I64_MAX_VALUE or q16 < -(1 << 63):
        return FP_Q16_ERR_OVERFLOW, 0
    return FP_Q16_OK, q16


def fpq16_exp_poly_core_ln2_residual_q16_checked(
    residual_q16: int,
    out_present: bool = True,
) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if residual_q16 < 0 or residual_q16 >= EXP_Q16_LN2:
        return FP_Q16_ERR_BAD_PARAM, 0

    err, r2 = fpq16_mul_checked(residual_q16, residual_q16)
    if err != FP_Q16_OK:
        return err, 0

    err, r3 = fpq16_mul_checked(r2, residual_q16)
    if err != FP_Q16_OK:
        return err, 0

    err, r4 = fpq16_mul_checked(r3, residual_q16)
    if err != FP_Q16_OK:
        return err, 0

    t1 = residual_q16
    t2 = r2 // EXP_Q16_POLY_C2_DIV
    t3 = r3 // EXP_Q16_POLY_C3_DIV
    t4 = r4 // EXP_Q16_POLY_C4_DIV

    poly = EXP_Q16_POLY_C0
    for term in (t1, t2, t3, t4):
        if poly > I64_MAX_VALUE - term:
            return FP_Q16_ERR_OVERFLOW, 0
        poly += term

    return FP_Q16_OK, poly


def fpq16_exp_poly_core_ln2_residual_q16_checked_no_partial(
    residual_q16: int,
    out_poly: list[int] | None,
) -> int:
    if out_poly is None:
        return FP_Q16_ERR_NULL_PTR

    staged = 0
    err, value = fpq16_exp_poly_core_ln2_residual_q16_checked(residual_q16)
    if err != FP_Q16_OK:
        return err

    staged = value
    out_poly[0] = staged
    return FP_Q16_OK


def test_source_contains_no_partial_wrapper() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpPolyCoreLn2ResidualQ16CheckedNoPartial" in source
    assert "FPQ16ExpPolyCoreLn2ResidualQ16Checked(residual_q16," in source
    assert "staged_poly_q16 = 0;" in source


def test_null_pointer_surface() -> None:
    assert fpq16_exp_poly_core_ln2_residual_q16_checked_no_partial(0, None) == FP_Q16_ERR_NULL_PTR


def test_bad_param_no_partial_commit() -> None:
    for bad_residual in (-1, EXP_Q16_LN2, EXP_Q16_LN2 + 1):
        out = [0x77777777]
        err = fpq16_exp_poly_core_ln2_residual_q16_checked_no_partial(bad_residual, out)
        assert err == FP_Q16_ERR_BAD_PARAM
        assert out == [0x77777777]


def test_success_matches_core_and_is_monotonic() -> None:
    previous = None
    for residual in range(0, EXP_Q16_LN2, 137):
        out = [0x5555]
        err = fpq16_exp_poly_core_ln2_residual_q16_checked_no_partial(residual, out)
        core_err, core_value = fpq16_exp_poly_core_ln2_residual_q16_checked(residual)

        assert err == FP_Q16_OK
        assert core_err == FP_Q16_OK
        assert out[0] == core_value

        if previous is not None:
            assert out[0] >= previous
        previous = out[0]


def test_randomized_parity() -> None:
    rng = random.Random(20260419_458)

    for _ in range(40000):
        residual = rng.randint(-10000, EXP_Q16_LN2 + 10000)
        out = [rng.randint(-2**31, 2**31 - 1)]
        seed = out[0]

        err = fpq16_exp_poly_core_ln2_residual_q16_checked_no_partial(residual, out)
        core_err, core_value = fpq16_exp_poly_core_ln2_residual_q16_checked(residual)

        assert err == core_err
        if err == FP_Q16_OK:
            assert out[0] == core_value
            assert out[0] >= FP_Q16_ONE
        else:
            assert out[0] == seed


def run() -> None:
    test_source_contains_no_partial_wrapper()
    test_null_pointer_surface()
    test_bad_param_no_partial_commit()
    test_success_matches_core_and_is_monotonic()
    test_randomized_parity()
    print("intexp_q16_poly_core_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
