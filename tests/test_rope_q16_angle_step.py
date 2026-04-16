#!/usr/bin/env python3
"""Reference checks for RoPE Q16 angle-step helper semantics."""

from __future__ import annotations

import math
import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)

ROPE_Q16_OK = 0
ROPE_Q16_ERR_NULL_PTR = 1
ROPE_Q16_ERR_BAD_PARAM = 2
ROPE_Q16_ERR_DOMAIN = 3
ROPE_Q16_ERR_OVERFLOW = 4

LOG_Q16_LN2 = 45426
LOG_Q16_SPLIT = (FP_Q16_ONE * 3) // 2
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360


def q16_from_float(x: float) -> int:
    scaled = int(round(x * FP_Q16_ONE))
    if scaled > I64_MAX:
        return I64_MAX
    if scaled < I64_MIN:
        return I64_MIN
    return scaled


def q16_to_float(x_q16: int) -> float:
    return x_q16 / FP_Q16_ONE


def fpq16_mul(a: int, b: int) -> int:
    prod = a * b
    if prod >= 0:
        return (prod + (1 << (FP_Q16_SHIFT - 1))) >> FP_Q16_SHIFT
    return -(((-prod) + (1 << (FP_Q16_SHIFT - 1))) >> FP_Q16_SHIFT)


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
        return I64_MIN

    m_q16, k = fpq16_ln_reduce(x_q16)
    if m_q16 >= LOG_Q16_SPLIT:
        m_q16 = (m_q16 + 1) >> 1
        k += 1

    y_q16 = m_q16 - FP_Q16_ONE
    poly_q16 = fpq16_ln1p_poly(y_q16)
    base_q16 = k * LOG_Q16_LN2

    if base_q16 > 0 and poly_q16 > I64_MAX - base_q16:
        return I64_MAX
    if base_q16 < 0 and poly_q16 < I64_MIN - base_q16:
        return I64_MIN
    return base_q16 + poly_q16


def fpq16_exp(x_q16: int) -> int:
    if x_q16 >= EXP_Q16_MAX_INPUT:
        return I64_MAX
    if x_q16 <= EXP_Q16_MIN_INPUT:
        return 0

    exp_q16_ln2 = 45426
    k = x_q16 // exp_q16_ln2
    if x_q16 < 0 and (x_q16 % exp_q16_ln2):
        k -= 1
    r = x_q16 - (k * exp_q16_ln2)

    r2 = fpq16_mul(r, r)
    r3 = fpq16_mul(r2, r)
    r4 = fpq16_mul(r3, r)
    poly = FP_Q16_ONE + r + (r2 // 2) + (r3 // 6) + (r4 // 24)

    if k >= 0:
        if k >= 30:
            return I64_MAX
        if poly > (I64_MAX >> k):
            return I64_MAX
        return poly << k

    if k <= -63:
        return 0
    return poly >> (-k)


def rope_q16_from_int_checked(value: int) -> tuple[int, int]:
    if value > (I64_MAX >> FP_Q16_SHIFT) or value < (I64_MIN >> FP_Q16_SHIFT):
        return ROPE_Q16_ERR_OVERFLOW, 0
    return ROPE_Q16_OK, value << FP_Q16_SHIFT


def rope_q16_mul_checked(lhs_q16: int, rhs_q16: int) -> tuple[int, int]:
    prod = lhs_q16 * rhs_q16
    if prod > I64_MAX or prod < I64_MIN:
        return ROPE_Q16_ERR_OVERFLOW, 0
    return ROPE_Q16_OK, fpq16_mul(lhs_q16, rhs_q16)


def rope_q16_div_checked(num_q16: int, den_q16: int) -> tuple[int, int]:
    if den_q16 == 0:
        return ROPE_Q16_ERR_DOMAIN, 0

    if num_q16 == I64_MIN and den_q16 == -1:
        return ROPE_Q16_ERR_OVERFLOW, 0

    num_abs = abs(num_q16)
    den_abs = abs(den_q16)
    sign_negative = (num_q16 < 0) ^ (den_q16 < 0)
    limit = (1 << 63) if sign_negative else I64_MAX

    int_part = num_abs // den_abs
    if int_part > (limit >> FP_Q16_SHIFT):
        return ROPE_Q16_ERR_OVERFLOW, 0

    result_mag = int_part << FP_Q16_SHIFT
    rem = num_abs % den_abs
    for bit in range(FP_Q16_SHIFT - 1, -1, -1):
        rem <<= 1
        if rem >= den_abs:
            rem -= den_abs
            add = 1 << bit
            if result_mag > limit - add:
                return ROPE_Q16_ERR_OVERFLOW, 0
            result_mag |= add

    if rem >= ((den_abs + 1) >> 1):
        if result_mag == limit:
            return ROPE_Q16_ERR_OVERFLOW, 0
        result_mag += 1

    if sign_negative:
        if result_mag == (1 << 63):
            return ROPE_Q16_OK, I64_MIN
        return ROPE_Q16_OK, -result_mag
    return ROPE_Q16_OK, result_mag


def rope_q16_angle_step_checked(freq_base_q16: int, head_dim: int, pair_index: int) -> tuple[int, int]:
    if freq_base_q16 <= 0:
        return ROPE_Q16_ERR_DOMAIN, 0
    if head_dim <= 0 or (head_dim & 1):
        return ROPE_Q16_ERR_BAD_PARAM, 0
    if pair_index < 0:
        return ROPE_Q16_ERR_BAD_PARAM, 0

    pair_count = head_dim >> 1
    if pair_index >= pair_count:
        return ROPE_Q16_ERR_BAD_PARAM, 0

    two_i = pair_index + pair_index
    if two_i > I64_MAX:
        return ROPE_Q16_ERR_OVERFLOW, 0

    err, two_i_q16 = rope_q16_from_int_checked(two_i)
    if err != ROPE_Q16_OK:
        return err, 0

    err, head_dim_q16 = rope_q16_from_int_checked(head_dim)
    if err != ROPE_Q16_OK:
        return err, 0

    err, ratio_q16 = rope_q16_div_checked(two_i_q16, head_dim_q16)
    if err != ROPE_Q16_OK:
        return err, 0

    ln_base_q16 = fpq16_ln(freq_base_q16)
    if ln_base_q16 == I64_MIN:
        return ROPE_Q16_ERR_DOMAIN, 0

    err, exponent_mag_q16 = rope_q16_mul_checked(ratio_q16, ln_base_q16)
    if err != ROPE_Q16_OK:
        return err, 0
    if exponent_mag_q16 == I64_MIN:
        return ROPE_Q16_ERR_OVERFLOW, 0

    exponent_q16 = -exponent_mag_q16
    step_q16 = fpq16_exp(exponent_q16)
    return ROPE_Q16_OK, step_q16


def test_domain_and_bad_param_contracts() -> None:
    assert rope_q16_angle_step_checked(0, 128, 0)[0] == ROPE_Q16_ERR_DOMAIN
    assert rope_q16_angle_step_checked(-1, 128, 0)[0] == ROPE_Q16_ERR_DOMAIN

    base_q16 = q16_from_float(10000.0)
    assert rope_q16_angle_step_checked(base_q16, 0, 0)[0] == ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_angle_step_checked(base_q16, 127, 0)[0] == ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_angle_step_checked(base_q16, 128, -1)[0] == ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_angle_step_checked(base_q16, 128, 64)[0] == ROPE_Q16_ERR_BAD_PARAM


def test_pair_zero_exactly_one_q16() -> None:
    base_q16 = q16_from_float(10000.0)
    err, step_q16 = rope_q16_angle_step_checked(base_q16, 128, 0)
    assert err == ROPE_Q16_OK
    assert step_q16 == FP_Q16_ONE


def test_monotonic_nonincreasing_step_across_pairs() -> None:
    base_q16 = q16_from_float(10000.0)
    last = None

    for pair_index in range(64):
        err, step_q16 = rope_q16_angle_step_checked(base_q16, 128, pair_index)
        assert err == ROPE_Q16_OK
        if last is not None:
            assert step_q16 <= last
        last = step_q16


def test_reference_error_bounds_vs_float_formula() -> None:
    base = 10000.0
    head_dim = 128
    base_q16 = q16_from_float(base)

    max_abs_err = 0.0
    for pair_index in range(head_dim // 2):
        err, step_q16 = rope_q16_angle_step_checked(base_q16, head_dim, pair_index)
        assert err == ROPE_Q16_OK

        got = q16_to_float(step_q16)
        want = base ** (-(2.0 * pair_index) / head_dim)

        abs_err = abs(got - want)
        max_abs_err = max(max_abs_err, abs_err)
        assert abs_err <= 0.02

    assert max_abs_err > 0.0


def test_randomized_common_model_shapes() -> None:
    rng = random.Random(20260416133)
    dims = [32, 64, 80, 128, 160, 256]
    bases = [5000.0, 10000.0, 50000.0]

    for _ in range(1200):
        head_dim = rng.choice(dims)
        pair_index = rng.randint(0, (head_dim // 2) - 1)
        base = rng.choice(bases)

        base_q16 = q16_from_float(base)
        err, step_q16 = rope_q16_angle_step_checked(base_q16, head_dim, pair_index)
        assert err == ROPE_Q16_OK

        got = q16_to_float(step_q16)
        want = base ** (-(2.0 * pair_index) / head_dim)
        assert abs(got - want) <= 0.03


def test_overflow_surface_for_q16_int_conversion() -> None:
    base_q16 = q16_from_float(10000.0)
    huge_head_dim = ((I64_MAX >> FP_Q16_SHIFT) + 2) & ~1
    huge_pair_index = (huge_head_dim // 2) - 1

    err, _ = rope_q16_angle_step_checked(base_q16, huge_head_dim, huge_pair_index)
    assert err == ROPE_Q16_ERR_OVERFLOW


def run() -> None:
    test_domain_and_bad_param_contracts()
    test_pair_zero_exactly_one_q16()
    test_monotonic_nonincreasing_step_across_pairs()
    test_reference_error_bounds_vs_float_formula()
    test_randomized_common_model_shapes()
    test_overflow_surface_for_q16_int_conversion()
    print("rope_q16_angle_step_reference_checks=ok")


if __name__ == "__main__":
    run()
