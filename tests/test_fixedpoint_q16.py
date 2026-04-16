#!/usr/bin/env python3
"""Reference checks for fixedpoint.HC Q16 helpers and overflow semantics."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)


def abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    return (-(x + 1)) + 1


def sign_apply_from_u64(mag: int, is_negative: bool) -> int:
    if is_negative:
        if mag >= (1 << 63):
            return I64_MIN_VALUE
        return -mag
    if mag > I64_MAX_VALUE:
        return I64_MAX_VALUE
    return mag


def round_shift_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if shift >= 63:
        return 0

    is_negative = value < 0
    abs_value = abs_to_u64(value)
    round_bias = 1 << (shift - 1)

    if abs_value > ((1 << 64) - 1) - round_bias:
        rounded = ((1 << 64) - 1) >> shift
    else:
        rounded = (abs_value + round_bias) >> shift

    return sign_apply_from_u64(rounded, is_negative)


def fpq16_abs(x: int) -> int:
    if x == I64_MIN_VALUE:
        return I64_MAX_VALUE
    return -x if x < 0 else x


def fpq16_from_int(x: int) -> int:
    if x > (I64_MAX_VALUE >> FP_Q16_SHIFT):
        return I64_MAX_VALUE
    if x < (I64_MIN_VALUE >> FP_Q16_SHIFT):
        return I64_MIN_VALUE
    return x << FP_Q16_SHIFT


def fpq16_mul_sat(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0

    abs_a = abs_to_u64(a)
    abs_b = abs_to_u64(b)
    is_negative = (a < 0) ^ (b < 0)

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if abs_a > (limit // abs_b):
        return I64_MIN_VALUE if is_negative else I64_MAX_VALUE

    prod = a * b
    return round_shift_signed(prod, FP_Q16_SHIFT)


def fpq16_mul(a: int, b: int) -> int:
    return fpq16_mul_sat(a, b)


def fpq16_div(num: int, den: int) -> int:
    if den == 0:
        return 0

    abs_num = abs_to_u64(num)
    abs_den = abs_to_u64(den)
    is_negative = (num < 0) ^ (den < 0)

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    int_part = abs_num // abs_den
    if int_part > (limit >> FP_Q16_SHIFT):
        return I64_MIN_VALUE if is_negative else I64_MAX_VALUE

    result_mag = int_part << FP_Q16_SHIFT
    rem = abs_num % abs_den

    for bit in range(FP_Q16_SHIFT - 1, -1, -1):
        rem <<= 1
        if rem >= abs_den:
            rem -= abs_den
            add = 1 << bit
            if result_mag <= limit - add:
                result_mag |= add
            else:
                result_mag = limit

    if rem >= ((abs_den + 1) >> 1):
        if result_mag < limit:
            result_mag += 1
        else:
            result_mag = limit

    return sign_apply_from_u64(result_mag, is_negative)


def q16_to_float(x: int) -> float:
    return x / FP_Q16_ONE


def test_fpq16_abs_min_value_saturates() -> None:
    assert fpq16_abs(I64_MIN_VALUE) == I64_MAX_VALUE
    assert fpq16_abs(-123) == 123


def test_fpq16_from_int_saturates_and_scales() -> None:
    assert fpq16_from_int(7) == 7 * FP_Q16_ONE
    assert fpq16_from_int(I64_MAX_VALUE) == I64_MAX_VALUE
    assert fpq16_from_int(I64_MIN_VALUE) == I64_MIN_VALUE


def test_mul_sat_extremes() -> None:
    assert fpq16_mul_sat(I64_MAX_VALUE, I64_MAX_VALUE) == I64_MAX_VALUE
    assert fpq16_mul_sat(I64_MIN_VALUE, I64_MAX_VALUE) == I64_MIN_VALUE
    assert fpq16_mul_sat(0, I64_MAX_VALUE) == 0


def test_mul_random_reference_bounds() -> None:
    rng = random.Random(916)
    for _ in range(4000):
        a = rng.randint(-(1 << 40), 1 << 40)
        b = rng.randint(-(1 << 40), 1 << 40)
        got = fpq16_mul(a, b)

        if got in (I64_MAX_VALUE, I64_MIN_VALUE):
            continue

        want = (a / FP_Q16_ONE) * (b / FP_Q16_ONE)
        got_f = q16_to_float(got)
        assert abs(got_f - want) <= (2.0 / FP_Q16_ONE)


def test_div_by_zero_returns_zero() -> None:
    assert fpq16_div(123456, 0) == 0


def test_div_saturates_when_integer_part_overflows() -> None:
    assert fpq16_div(I64_MAX_VALUE, 1) == I64_MAX_VALUE
    assert fpq16_div(I64_MIN_VALUE, 1) == I64_MIN_VALUE


def test_div_random_reference_bounds() -> None:
    rng = random.Random(917)
    for _ in range(5000):
        num = rng.randint(-(1 << 50), 1 << 50)
        den = rng.randint(-(1 << 31), 1 << 31)
        if den == 0:
            den = 1

        got = fpq16_div(num, den)
        if got in (I64_MAX_VALUE, I64_MIN_VALUE):
            continue

        want = (num / den)
        got_f = q16_to_float(got)
        assert abs(got_f - want) <= (1.5 / FP_Q16_ONE)


def test_mul_div_round_trip_near_identity() -> None:
    rng = random.Random(918)
    for _ in range(3000):
        base = rng.randint(-(1 << 29), 1 << 29)
        scale = rng.randint(1, 1 << 20)

        mul = fpq16_mul(base, fpq16_from_int(scale))
        if mul in (I64_MAX_VALUE, I64_MIN_VALUE):
            continue

        back = fpq16_div(mul, fpq16_from_int(scale))

        if back in (I64_MAX_VALUE, I64_MIN_VALUE):
            continue

        assert abs(back - base) <= 2


if __name__ == "__main__":
    test_fpq16_abs_min_value_saturates()
    test_fpq16_from_int_saturates_and_scales()
    test_mul_sat_extremes()
    test_mul_random_reference_bounds()
    test_div_by_zero_returns_zero()
    test_div_saturates_when_integer_part_overflows()
    test_div_random_reference_bounds()
    test_mul_div_round_trip_near_identity()
    print("fixedpoint_q16_reference_checks=ok")
