#!/usr/bin/env python3
"""Reference checks for FPQ16Exp scalar parity against math.exp samples."""

from __future__ import annotations

import math
import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

EXP_Q16_LN2 = 45426
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360

I64_MAX_VALUE = (1 << 63) - 1


def q16_from_float(value: float) -> int:
    return int(round(value * FP_Q16_ONE))


def q16_to_float(value: int) -> float:
    return value / FP_Q16_ONE


def fpq16_mul(a_q16: int, b_q16: int) -> int:
    product = a_q16 * b_q16
    if product >= 0:
        return (product + (1 << (FP_Q16_SHIFT - 1))) >> FP_Q16_SHIFT
    return -(((-product) + (1 << (FP_Q16_SHIFT - 1))) >> FP_Q16_SHIFT)


def fpq16_exp_from_clamped_input_checked(clamped_input_q16: int) -> int:
    if clamped_input_q16 < EXP_Q16_MIN_INPUT or clamped_input_q16 > EXP_Q16_MAX_INPUT:
        raise ValueError("input must already be clamped")

    if clamped_input_q16 >= EXP_Q16_MAX_INPUT:
        return I64_MAX_VALUE
    if clamped_input_q16 <= EXP_Q16_MIN_INPUT:
        return 0

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
            return I64_MAX_VALUE
        if poly > (I64_MAX_VALUE >> k):
            return I64_MAX_VALUE
        return poly << k

    if k <= -63:
        return 0
    return poly >> (-k)


def fpq16_exp(x_q16: int) -> int:
    clamped = x_q16
    if clamped < EXP_Q16_MIN_INPUT:
        clamped = EXP_Q16_MIN_INPUT
    elif clamped > EXP_Q16_MAX_INPUT:
        clamped = EXP_Q16_MAX_INPUT

    return fpq16_exp_from_clamped_input_checked(clamped)


def test_scalar_boundary_saturation_contract() -> None:
    assert fpq16_exp(EXP_Q16_MIN_INPUT - 1) == 0
    assert fpq16_exp(EXP_Q16_MIN_INPUT) == 0
    assert fpq16_exp(EXP_Q16_MAX_INPUT) == I64_MAX_VALUE
    assert fpq16_exp(EXP_Q16_MAX_INPUT + 1) == I64_MAX_VALUE


def test_scalar_is_monotonic_over_domain_samples() -> None:
    prev = fpq16_exp(EXP_Q16_MIN_INPUT)
    for x_q16 in range(EXP_Q16_MIN_INPUT + 257, EXP_Q16_MAX_INPUT, 257):
        cur = fpq16_exp(x_q16)
        assert cur >= prev
        prev = cur


def test_math_exp_parity_on_core_domain_samples() -> None:
    # Core operating range: exp accuracy is tight here and used heavily in
    # softmax normalization before extreme-tail clipping.
    max_abs_err = 0.0
    max_rel_err = 0.0

    for x_q16 in range(q16_from_float(-8.0), q16_from_float(8.0), 97):
        got = q16_to_float(fpq16_exp(x_q16))
        want = math.exp(q16_to_float(x_q16))

        abs_err = abs(got - want)
        rel_err = abs_err / want

        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        assert rel_err <= 0.05

    assert max_abs_err > 0.0
    assert max_rel_err > 0.0


def test_math_exp_parity_on_wide_domain_random_samples() -> None:
    # Full clamped open domain parity against math.exp. The current polynomial
    # intentionally trades edge precision for integer-only determinism.
    rng = random.Random(20260417_021)

    max_abs_err = 0.0
    max_rel_err = 0.0

    for _ in range(4000):
        x_q16 = rng.randint(EXP_Q16_MIN_INPUT + 1, EXP_Q16_MAX_INPUT - 1)

        got = q16_to_float(fpq16_exp(x_q16))
        want = math.exp(q16_to_float(x_q16))

        abs_err = abs(got - want)
        rel_err = abs_err / want

        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        assert rel_err <= 0.34

    assert max_abs_err > 0.0
    assert max_rel_err > 0.0


def test_sampled_against_float_reference_table() -> None:
    samples = [-9.5, -7.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 8.5]
    for sample in samples:
        x_q16 = q16_from_float(sample)
        got = q16_to_float(fpq16_exp(x_q16))
        want = math.exp(sample)
        rel_err = abs(got - want) / want

        if abs(sample) <= 8.0:
            assert rel_err <= 0.05
        else:
            assert rel_err <= 0.34


def run() -> None:
    test_scalar_boundary_saturation_contract()
    test_scalar_is_monotonic_over_domain_samples()
    test_math_exp_parity_on_core_domain_samples()
    test_math_exp_parity_on_wide_domain_random_samples()
    test_sampled_against_float_reference_table()
    print("intexp_q16_reference_checks=ok")


if __name__ == "__main__":
    run()

