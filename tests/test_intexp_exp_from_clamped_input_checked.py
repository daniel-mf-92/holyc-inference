#!/usr/bin/env python3
"""Reference checks for FPQ16ExpFromClampedInputChecked semantics."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

EXP_Q16_LN2 = 45426
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360

I64_MAX_VALUE = (1 << 63) - 1

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2


def fpq16_mul(a: int, b: int) -> int:
    prod = a * b
    if prod >= 0:
        return (prod + (1 << (FP_Q16_SHIFT - 1))) >> FP_Q16_SHIFT
    return -(((-prod) + (1 << (FP_Q16_SHIFT - 1))) >> FP_Q16_SHIFT)


def fpq16_exp_from_clamped_input_checked(clamped_input_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if clamped_input_q16 < EXP_Q16_MIN_INPUT or clamped_input_q16 > EXP_Q16_MAX_INPUT:
        return FP_Q16_ERR_BAD_PARAM, 0

    if clamped_input_q16 >= EXP_Q16_MAX_INPUT:
        return FP_Q16_OK, I64_MAX_VALUE
    if clamped_input_q16 <= EXP_Q16_MIN_INPUT:
        return FP_Q16_OK, 0

    k = clamped_input_q16 // EXP_Q16_LN2
    if clamped_input_q16 < 0 and (clamped_input_q16 % EXP_Q16_LN2):
        k -= 1

    r = clamped_input_q16 - (k * EXP_Q16_LN2)

    r2 = fpq16_mul(r, r)
    r3 = fpq16_mul(r2, r)
    r4 = fpq16_mul(r3, r)

    poly = FP_Q16_ONE + r + (r2 // 2) + (r3 // 6) + (r4 // 24)

    if k >= 0:
        if k >= 30:
            return FP_Q16_OK, I64_MAX_VALUE

        if poly > (I64_MAX_VALUE >> k):
            return FP_Q16_OK, I64_MAX_VALUE

        return FP_Q16_OK, poly << k

    if k <= -63:
        return FP_Q16_OK, 0

    return FP_Q16_OK, poly >> (-k)


def fpq16_exp_clamp_to_input_domain_checked(input_q16: int) -> tuple[int, int]:
    if input_q16 < EXP_Q16_MIN_INPUT:
        return FP_Q16_OK, EXP_Q16_MIN_INPUT
    if input_q16 > EXP_Q16_MAX_INPUT:
        return FP_Q16_OK, EXP_Q16_MAX_INPUT
    return FP_Q16_OK, input_q16


def fpq16_exp_scalar_composed(x_q16: int) -> int:
    err, clamped = fpq16_exp_clamp_to_input_domain_checked(x_q16)
    assert err == FP_Q16_OK
    err2, out = fpq16_exp_from_clamped_input_checked(clamped)
    assert err2 == FP_Q16_OK
    return out


def test_null_pointer_surface() -> None:
    err, out = fpq16_exp_from_clamped_input_checked(0, out_present=False)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == 0


def test_bad_param_on_out_of_domain_input() -> None:
    err, _ = fpq16_exp_from_clamped_input_checked(EXP_Q16_MIN_INPUT - 1)
    assert err == FP_Q16_ERR_BAD_PARAM

    err, _ = fpq16_exp_from_clamped_input_checked(EXP_Q16_MAX_INPUT + 1)
    assert err == FP_Q16_ERR_BAD_PARAM


def test_boundary_saturation_contract() -> None:
    err_min, out_min = fpq16_exp_from_clamped_input_checked(EXP_Q16_MIN_INPUT)
    assert err_min == FP_Q16_OK
    assert out_min == 0

    err_max, out_max = fpq16_exp_from_clamped_input_checked(EXP_Q16_MAX_INPUT)
    assert err_max == FP_Q16_OK
    assert out_max == I64_MAX_VALUE


def test_monotonicity_within_open_domain() -> None:
    samples = [
        -600_000,
        -450_000,
        -300_000,
        -150_000,
        -1,
        0,
        1,
        150_000,
        300_000,
        450_000,
        600_000,
    ]

    prev = None
    for value in samples:
        err, out = fpq16_exp_from_clamped_input_checked(value)
        assert err == FP_Q16_OK
        if prev is not None:
            assert out >= prev
        prev = out


def test_composition_matches_scalar_clamp_then_eval() -> None:
    vectors = [
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 1,
        -123456,
        -1,
        0,
        1,
        123456,
        EXP_Q16_MAX_INPUT - 1,
        EXP_Q16_MAX_INPUT,
    ]

    for x in vectors:
        err, direct = fpq16_exp_from_clamped_input_checked(x)
        assert err == FP_Q16_OK
        composed = fpq16_exp_scalar_composed(x)
        assert direct == composed


def test_randomized_clamped_inputs_match_composed_path() -> None:
    rng = random.Random(20260417_164)
    for _ in range(3000):
        x = rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT)
        err, out = fpq16_exp_from_clamped_input_checked(x)
        assert err == FP_Q16_OK
        assert out == fpq16_exp_scalar_composed(x)


def run() -> None:
    test_null_pointer_surface()
    test_bad_param_on_out_of_domain_input()
    test_boundary_saturation_contract()
    test_monotonicity_within_open_domain()
    test_composition_matches_scalar_clamp_then_eval()
    test_randomized_clamped_inputs_match_composed_path()
    print("intexp_exp_from_clamped_input_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
