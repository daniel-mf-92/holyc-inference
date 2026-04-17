#!/usr/bin/env python3
"""Reference checks for FPQ16ExpClampToInputDomainChecked semantics."""

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


def fpq16_exp_clamp_to_input_domain_checked(input_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if EXP_Q16_MIN_INPUT > EXP_Q16_MAX_INPUT:
        return FP_Q16_ERR_BAD_PARAM, 0

    if input_q16 < EXP_Q16_MIN_INPUT:
        return FP_Q16_OK, EXP_Q16_MIN_INPUT

    if input_q16 > EXP_Q16_MAX_INPUT:
        return FP_Q16_OK, EXP_Q16_MAX_INPUT

    return FP_Q16_OK, input_q16


def fpq16_exp_with_checked_clamp(x_q16: int) -> int:
    err, clamped_x = fpq16_exp_clamp_to_input_domain_checked(x_q16)
    if err != FP_Q16_OK:
        return 0

    if clamped_x >= EXP_Q16_MAX_INPUT:
        return I64_MAX_VALUE
    if clamped_x <= EXP_Q16_MIN_INPUT:
        return 0

    k = clamped_x // EXP_Q16_LN2
    if clamped_x < 0 and (clamped_x % EXP_Q16_LN2):
        k -= 1

    r = clamped_x - (k * EXP_Q16_LN2)

    r2 = fpq16_mul(r, r)
    r3 = fpq16_mul(r2, r)
    r4 = fpq16_mul(r3, r)

    t1 = r
    t2 = r2 // 2
    t3 = r3 // 6
    t4 = r4 // 24

    poly = FP_Q16_ONE + t1 + t2 + t3 + t4

    if k >= 0:
        if k >= 30:
            return I64_MAX_VALUE

        if poly > (I64_MAX_VALUE >> k):
            return I64_MAX_VALUE

        return poly << k

    if k <= -63:
        return 0

    return poly >> (-k)


def test_null_pointer_surface() -> None:
    err, out = fpq16_exp_clamp_to_input_domain_checked(0, out_present=False)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == 0


def test_boundary_and_saturation_clamps() -> None:
    test_values = [
        EXP_Q16_MIN_INPUT - 1,
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 1,
        -1,
        0,
        1,
        EXP_Q16_MAX_INPUT - 1,
        EXP_Q16_MAX_INPUT,
        EXP_Q16_MAX_INPUT + 1,
    ]

    for x_q16 in test_values:
        err, clamped = fpq16_exp_clamp_to_input_domain_checked(x_q16)
        assert err == FP_Q16_OK
        assert EXP_Q16_MIN_INPUT <= clamped <= EXP_Q16_MAX_INPUT

        if x_q16 < EXP_Q16_MIN_INPUT:
            assert clamped == EXP_Q16_MIN_INPUT
        elif x_q16 > EXP_Q16_MAX_INPUT:
            assert clamped == EXP_Q16_MAX_INPUT
        else:
            assert clamped == x_q16


def test_random_clamp_idempotence_and_range() -> None:
    rng = random.Random(20260417_161)

    for _ in range(10000):
        x_q16 = rng.randint(-(1 << 62), (1 << 62))
        err, clamped = fpq16_exp_clamp_to_input_domain_checked(x_q16)
        assert err == FP_Q16_OK
        assert EXP_Q16_MIN_INPUT <= clamped <= EXP_Q16_MAX_INPUT

        err2, clamped2 = fpq16_exp_clamp_to_input_domain_checked(clamped)
        assert err2 == FP_Q16_OK
        assert clamped2 == clamped


def test_exp_saturation_contract_unchanged() -> None:
    assert fpq16_exp_with_checked_clamp(EXP_Q16_MAX_INPUT) == I64_MAX_VALUE
    assert fpq16_exp_with_checked_clamp(EXP_Q16_MAX_INPUT + 12345) == I64_MAX_VALUE

    assert fpq16_exp_with_checked_clamp(EXP_Q16_MIN_INPUT) == 0
    assert fpq16_exp_with_checked_clamp(EXP_Q16_MIN_INPUT - 12345) == 0


def test_exp_monotonicity_inside_domain() -> None:
    prev = fpq16_exp_with_checked_clamp(EXP_Q16_MIN_INPUT + 1)

    for x_q16 in range(EXP_Q16_MIN_INPUT + 513, EXP_Q16_MAX_INPUT, 4096):
        current = fpq16_exp_with_checked_clamp(x_q16)
        assert current >= prev
        prev = current


def run() -> None:
    test_null_pointer_surface()
    test_boundary_and_saturation_clamps()
    test_random_clamp_idempotence_and_range()
    test_exp_saturation_contract_unchanged()
    test_exp_monotonicity_inside_domain()
    print("intexp_clamp_to_input_domain_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
