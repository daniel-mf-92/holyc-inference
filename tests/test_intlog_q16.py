#!/usr/bin/env python3
"""Reference checks for Q16.16 integer natural-log approximation semantics."""

from __future__ import annotations

import math
import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

LOG_Q16_LN2 = 45426
LOG_Q16_INV_LN2 = 94549
LOG_Q16_SPLIT = (FP_Q16_ONE * 3) // 2

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)


def round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    round_bias = 1 << (shift - 1)
    return (value + round_bias) >> shift


def round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return round_shift_right_unsigned(value, shift)
    return -round_shift_right_unsigned(-value, shift)


def fpq16_mul(a: int, b: int) -> int:
    prod = a * b
    return round_shift_right_signed(prod, FP_Q16_SHIFT)


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

    prod = ln_q16 * LOG_Q16_INV_LN2
    return round_shift_right_signed(prod, FP_Q16_SHIFT)


def q16_from_float(value: float) -> int:
    return round(value * FP_Q16_ONE)


def q16_to_float(value: int) -> float:
    return value / FP_Q16_ONE


def test_domain_floor_for_non_positive_inputs() -> None:
    assert fpq16_ln(0) == I64_MIN_VALUE
    assert fpq16_ln(-1) == I64_MIN_VALUE
    assert fpq16_ln(-(123 << FP_Q16_SHIFT)) == I64_MIN_VALUE


def test_exact_powers_of_two_are_k_ln2() -> None:
    for k in range(-12, 13):
        if k >= 0:
            x_q16 = FP_Q16_ONE << k
        else:
            x_q16 = FP_Q16_ONE >> (-k)
            if x_q16 <= 0:
                continue

        got = fpq16_ln(x_q16)
        expected = k * LOG_Q16_LN2
        assert abs(got - expected) <= 2


def test_monotonicity_on_positive_domain_samples() -> None:
    samples = list(range(1, (32 << FP_Q16_SHIFT), 9973))
    prev = fpq16_ln(samples[0])
    for x_q16 in samples[1:]:
        current = fpq16_ln(x_q16)
        assert current >= prev
        prev = current


def test_random_reference_error_bounds_vs_math_log() -> None:
    rng = random.Random(310519)

    max_abs_err = 0.0
    for _ in range(2000):
        x = rng.uniform(1.0 / 256.0, 256.0)
        x_q16 = q16_from_float(x)

        got_q16 = fpq16_ln(x_q16)
        got = q16_to_float(got_q16)
        want = math.log(x_q16 / FP_Q16_ONE)

        abs_err = abs(got - want)
        if abs_err > max_abs_err:
            max_abs_err = abs_err

        assert abs_err <= 0.0065

    assert max_abs_err > 0.0


def test_range_recomposition_identity_check() -> None:
    rng = random.Random(424242)
    for _ in range(500):
        x_q16 = rng.randint(1, 1 << 30)
        m_q16, k = fpq16_ln_reduce(x_q16)

        recomposed = m_q16
        if k >= 0:
            recomposed <<= k
        else:
            recomposed >>= -k

        assert m_q16 >= FP_Q16_ONE
        assert m_q16 < (FP_Q16_ONE << 1)

        rel_err = abs(recomposed - x_q16) / max(1, x_q16)
        assert rel_err <= (1.0 / FP_Q16_ONE)


def test_log2_domain_floor_for_non_positive_inputs() -> None:
    assert fpq16_log2(0) == I64_MIN_VALUE
    assert fpq16_log2(-1) == I64_MIN_VALUE
    assert fpq16_log2(-(17 << FP_Q16_SHIFT)) == I64_MIN_VALUE


def test_log2_exact_powers_of_two() -> None:
    for k in range(-12, 13):
        if k >= 0:
            x_q16 = FP_Q16_ONE << k
        else:
            x_q16 = FP_Q16_ONE >> (-k)
            if x_q16 <= 0:
                continue

        got = fpq16_log2(x_q16)
        expected = k * FP_Q16_ONE
        assert abs(got - expected) <= 3


def test_log2_random_reference_error_bounds_vs_math_log2() -> None:
    rng = random.Random(120799)

    max_abs_err = 0.0
    for _ in range(2000):
        x = rng.uniform(1.0 / 256.0, 256.0)
        x_q16 = q16_from_float(x)

        got_q16 = fpq16_log2(x_q16)
        got = q16_to_float(got_q16)
        want = math.log2(x_q16 / FP_Q16_ONE)

        abs_err = abs(got - want)
        if abs_err > max_abs_err:
            max_abs_err = abs_err

        assert abs_err <= 0.0095

    assert max_abs_err > 0.0


def run() -> None:
    test_domain_floor_for_non_positive_inputs()
    test_exact_powers_of_two_are_k_ln2()
    test_monotonicity_on_positive_domain_samples()
    test_random_reference_error_bounds_vs_math_log()
    test_range_recomposition_identity_check()
    test_log2_domain_floor_for_non_positive_inputs()
    test_log2_exact_powers_of_two()
    test_log2_random_reference_error_bounds_vs_math_log2()
    print("intlog_q16_reference_checks=ok")


if __name__ == "__main__":
    run()
