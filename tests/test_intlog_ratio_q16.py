#!/usr/bin/env python3
"""Reference checks for FPQ16Log2 and FPQ16LnRatio HolyC semantics."""

from __future__ import annotations

import math
import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

LOG_Q16_LN2 = 45426
LOG_Q16_INV_LN2 = 94549
LOG_Q16_HALF = 1 << (FP_Q16_SHIFT - 1)
LOG_Q16_SPLIT = (FP_Q16_ONE * 3) // 2

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)


def round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    return (value + (1 << (shift - 1))) >> shift


def round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return round_shift_right_unsigned(value, shift)
    return -round_shift_right_unsigned(-value, shift)


def fpq16_mul(a: int, b: int) -> int:
    return round_shift_right_signed(a * b, FP_Q16_SHIFT)


def fpq16_ln1p_poly(y_q16: int) -> int:
    y2 = fpq16_mul(y_q16, y_q16)
    y3 = fpq16_mul(y2, y_q16)
    y4 = fpq16_mul(y3, y_q16)
    y5 = fpq16_mul(y4, y_q16)
    return y_q16 - (y2 // 2) + (y3 // 3) - (y4 // 4) + (y5 // 5)


def fpq16_ln_reduce(x_q16: int) -> tuple[int, int]:
    if x_q16 <= 0:
        return 0, 0

    m_q16 = x_q16
    k = 0

    while m_q16 >= (FP_Q16_ONE << 1):
        m_q16 >>= 1
        k += 1
    while m_q16 < FP_Q16_ONE:
        m_q16 <<= 1
        k -= 1

    return m_q16, k


def fpq16_ln(x_q16: int) -> int:
    if x_q16 <= 0:
        return I64_MIN_VALUE

    m_q16, k = fpq16_ln_reduce(x_q16)
    if m_q16 >= LOG_Q16_SPLIT:
        m_q16 = (m_q16 + 1) >> 1
        k += 1

    y_q16 = m_q16 - FP_Q16_ONE
    poly_q16 = fpq16_ln1p_poly(y_q16)
    base_q16 = k * LOG_Q16_LN2

    if base_q16 > 0 and poly_q16 > I64_MAX_VALUE - base_q16:
        return I64_MAX_VALUE
    if base_q16 < 0 and poly_q16 < I64_MIN_VALUE - base_q16:
        return I64_MIN_VALUE

    return base_q16 + poly_q16


def fpq16_log2(x_q16: int) -> int:
    ln_q16 = fpq16_ln(x_q16)
    if ln_q16 == I64_MIN_VALUE:
        return I64_MIN_VALUE

    abs_ln = -ln_q16 if ln_q16 < 0 else ln_q16
    max_abs_ln = I64_MAX_VALUE // LOG_Q16_INV_LN2
    if abs_ln > max_abs_ln:
        return I64_MIN_VALUE if ln_q16 < 0 else I64_MAX_VALUE

    prod = ln_q16 * LOG_Q16_INV_LN2
    if prod >= 0:
        return (prod + LOG_Q16_HALF) >> FP_Q16_SHIFT
    return -(((-prod) + LOG_Q16_HALF) >> FP_Q16_SHIFT)


def fpq16_ln_ratio(num_q16: int, den_q16: int) -> int:
    if num_q16 <= 0 or den_q16 <= 0:
        return I64_MIN_VALUE

    num_m_q16, num_k = fpq16_ln_reduce(num_q16)
    den_m_q16, den_k = fpq16_ln_reduce(den_q16)

    if num_m_q16 >= LOG_Q16_SPLIT:
        num_m_q16 = (num_m_q16 + 1) >> 1
        num_k += 1
    if den_m_q16 >= LOG_Q16_SPLIT:
        den_m_q16 = (den_m_q16 + 1) >> 1
        den_k += 1

    num_poly_q16 = fpq16_ln1p_poly(num_m_q16 - FP_Q16_ONE)
    den_poly_q16 = fpq16_ln1p_poly(den_m_q16 - FP_Q16_ONE)

    if den_poly_q16 < 0 and num_poly_q16 > I64_MAX_VALUE + den_poly_q16:
        return I64_MAX_VALUE
    if den_poly_q16 > 0 and num_poly_q16 < I64_MIN_VALUE + den_poly_q16:
        return I64_MIN_VALUE
    poly_delta_q16 = num_poly_q16 - den_poly_q16

    if den_k < 0 and num_k > I64_MAX_VALUE + den_k:
        return I64_MAX_VALUE
    if den_k > 0 and num_k < I64_MIN_VALUE + den_k:
        return I64_MIN_VALUE
    k_delta = num_k - den_k

    abs_k_delta = -k_delta if k_delta < 0 else k_delta
    max_k_delta = I64_MAX_VALUE // LOG_Q16_LN2
    if abs_k_delta > max_k_delta:
        return I64_MIN_VALUE if k_delta < 0 else I64_MAX_VALUE
    base_q16 = k_delta * LOG_Q16_LN2

    if base_q16 > 0 and poly_delta_q16 > I64_MAX_VALUE - base_q16:
        return I64_MAX_VALUE
    if base_q16 < 0 and poly_delta_q16 < I64_MIN_VALUE - base_q16:
        return I64_MIN_VALUE

    return base_q16 + poly_delta_q16


def q16_from_float(value: float) -> int:
    return round(value * FP_Q16_ONE)


def q16_to_float(value: int) -> float:
    return value / FP_Q16_ONE


def test_log2_domain_floor_for_non_positive_inputs() -> None:
    assert fpq16_log2(0) == I64_MIN_VALUE
    assert fpq16_log2(-1) == I64_MIN_VALUE
    assert fpq16_log2(-(19 << FP_Q16_SHIFT)) == I64_MIN_VALUE


def test_log2_powers_of_two_exactness() -> None:
    for k in range(-14, 15):
        if k >= 0:
            x_q16 = FP_Q16_ONE << k
        else:
            x_q16 = FP_Q16_ONE >> (-k)
            if x_q16 <= 0:
                continue
        got = fpq16_log2(x_q16)
        want = k * FP_Q16_ONE
        assert abs(got - want) <= 3


def test_log2_random_reference_error_bounds_vs_math_log2() -> None:
    rng = random.Random(19840409)
    max_abs_err = 0.0

    for _ in range(3000):
        x = rng.uniform(1.0 / 512.0, 512.0)
        x_q16 = q16_from_float(x)

        got = q16_to_float(fpq16_log2(x_q16))
        want = math.log2(x_q16 / FP_Q16_ONE)
        err = abs(got - want)
        max_abs_err = max(max_abs_err, err)
        assert err <= 0.0100

    assert max_abs_err > 0.0


def test_ln_ratio_domain_floor_for_non_positive_inputs() -> None:
    assert fpq16_ln_ratio(0, FP_Q16_ONE) == I64_MIN_VALUE
    assert fpq16_ln_ratio(FP_Q16_ONE, 0) == I64_MIN_VALUE
    assert fpq16_ln_ratio(-1, FP_Q16_ONE) == I64_MIN_VALUE
    assert fpq16_ln_ratio(FP_Q16_ONE, -1) == I64_MIN_VALUE


def test_ln_ratio_identity_reciprocal_and_power_law_cases() -> None:
    for k in range(-12, 13):
        if k >= 0:
            num_q16 = FP_Q16_ONE << k
            den_q16 = FP_Q16_ONE
        else:
            num_q16 = FP_Q16_ONE
            den_q16 = FP_Q16_ONE << (-k)

        got = fpq16_ln_ratio(num_q16, den_q16)
        want = k * LOG_Q16_LN2
        assert abs(got - want) <= 4

    rng = random.Random(919191)
    for _ in range(1200):
        num_q16 = rng.randint(1, 1 << 24)
        den_q16 = rng.randint(1, 1 << 24)

        same = fpq16_ln_ratio(num_q16, num_q16)
        assert abs(same) <= 3

        ab = fpq16_ln_ratio(num_q16, den_q16)
        ba = fpq16_ln_ratio(den_q16, num_q16)
        assert abs(ab + ba) <= 8


def test_ln_ratio_random_reference_error_bounds_vs_math_log() -> None:
    rng = random.Random(20260415081)
    max_abs_err = 0.0

    for _ in range(3500):
        num = rng.uniform(1.0 / 512.0, 512.0)
        den = rng.uniform(1.0 / 512.0, 512.0)

        num_q16 = q16_from_float(num)
        den_q16 = q16_from_float(den)

        got = q16_to_float(fpq16_ln_ratio(num_q16, den_q16))
        want = math.log((num_q16 / FP_Q16_ONE) / (den_q16 / FP_Q16_ONE))
        err = abs(got - want)
        max_abs_err = max(max_abs_err, err)

        assert err <= 0.0130

    assert max_abs_err > 0.0


def test_ln_ratio_matches_ln_difference_and_log2_delta() -> None:
    rng = random.Random(515151)

    for _ in range(2200):
        num_q16 = rng.randint(1, 1 << 25)
        den_q16 = rng.randint(1, 1 << 25)

        ratio_ln = fpq16_ln_ratio(num_q16, den_q16)
        ln_delta = fpq16_ln(num_q16) - fpq16_ln(den_q16)
        assert abs(ratio_ln - ln_delta) <= 8

        log2_delta = fpq16_log2(num_q16) - fpq16_log2(den_q16)
        ratio_as_log2 = round_shift_right_signed(ratio_ln * LOG_Q16_INV_LN2, FP_Q16_SHIFT)
        assert abs(ratio_as_log2 - log2_delta) <= 12


def run() -> None:
    test_log2_domain_floor_for_non_positive_inputs()
    test_log2_powers_of_two_exactness()
    test_log2_random_reference_error_bounds_vs_math_log2()
    test_ln_ratio_domain_floor_for_non_positive_inputs()
    test_ln_ratio_identity_reciprocal_and_power_law_cases()
    test_ln_ratio_random_reference_error_bounds_vs_math_log()
    test_ln_ratio_matches_ln_difference_and_log2_delta()
    print("intlog_ratio_q16_reference_checks=ok")


if __name__ == "__main__":
    run()
