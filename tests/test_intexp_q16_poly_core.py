#!/usr/bin/env python3
"""Reference checks for FPQ16ExpPolyCoreLn2ResidualQ16Checked semantics."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

EXP_Q16_LN2 = 45426
EXP_Q16_POLY_C2_DIV = 2
EXP_Q16_POLY_C3_DIV = 6
EXP_Q16_POLY_C4_DIV = 24

I64_MAX_VALUE = (1 << 63) - 1
U64_MAX_VALUE = (1 << 64) - 1


def _abs_i64_to_u64(value: int) -> int:
    if value >= 0:
        return value
    return (-(value + 1)) + 1


def _try_apply_sign_checked(mag: int, is_negative: bool) -> tuple[int, int]:
    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63
    if mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0
    if is_negative:
        if mag == (1 << 63):
            return FP_Q16_OK, -(1 << 63)
        return FP_Q16_OK, -mag
    return FP_Q16_OK, mag


def fpq16_mul_checked(a_q16: int, b_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if not a_q16 or not b_q16:
        return FP_Q16_OK, 0

    abs_a = _abs_i64_to_u64(a_q16)
    abs_b = _abs_i64_to_u64(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    if abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_prod = abs_a * abs_b
    round_bias = 1 << (FP_Q16_SHIFT - 1)

    if abs_prod > U64_MAX_VALUE - round_bias:
        rounded_mag = U64_MAX_VALUE >> FP_Q16_SHIFT
    else:
        rounded_mag = (abs_prod + round_bias) >> FP_Q16_SHIFT

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63
    if rounded_mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    return _try_apply_sign_checked(rounded_mag, is_negative)


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

    poly = FP_Q16_ONE
    for term in (t1, t2, t3, t4):
        if poly > I64_MAX_VALUE - term:
            return FP_Q16_ERR_OVERFLOW, 0
        poly += term

    return FP_Q16_OK, poly


def _legacy_residual_name_reference(residual_q16: int, out_present: bool = True) -> tuple[int, int]:
    return fpq16_exp_poly_core_ln2_residual_q16_checked(residual_q16, out_present)


def test_null_pointer_surface() -> None:
    err, _ = fpq16_exp_poly_core_ln2_residual_q16_checked(0, out_present=False)
    assert err == FP_Q16_ERR_NULL_PTR


def test_residual_window_bounds() -> None:
    err, _ = fpq16_exp_poly_core_ln2_residual_q16_checked(-1)
    assert err == FP_Q16_ERR_BAD_PARAM

    err, _ = fpq16_exp_poly_core_ln2_residual_q16_checked(EXP_Q16_LN2)
    assert err == FP_Q16_ERR_BAD_PARAM

    err, v0 = fpq16_exp_poly_core_ln2_residual_q16_checked(0)
    assert err == FP_Q16_OK
    assert v0 == FP_Q16_ONE

    err, vmax = fpq16_exp_poly_core_ln2_residual_q16_checked(EXP_Q16_LN2 - 1)
    assert err == FP_Q16_OK
    assert vmax >= FP_Q16_ONE


def test_dense_residual_sweep_monotonic() -> None:
    prev = None
    for r_q16 in range(EXP_Q16_LN2):
        err, poly = fpq16_exp_poly_core_ln2_residual_q16_checked(r_q16)
        assert err == FP_Q16_OK
        if prev is not None:
            assert poly >= prev
        prev = poly


def test_randomized_reference_matches_legacy_name() -> None:
    rng = random.Random(20260419_442)
    for _ in range(20000):
        r_q16 = rng.randint(0, EXP_Q16_LN2 - 1)
        err0, poly0 = fpq16_exp_poly_core_ln2_residual_q16_checked(r_q16)
        err1, poly1 = _legacy_residual_name_reference(r_q16)
        assert err0 == err1 == FP_Q16_OK
        assert poly0 == poly1


def run() -> None:
    test_null_pointer_surface()
    test_residual_window_bounds()
    test_dense_residual_sweep_monotonic()
    test_randomized_reference_matches_legacy_name()
    print("intexp_q16_poly_core_reference_checks=ok")


if __name__ == "__main__":
    run()
