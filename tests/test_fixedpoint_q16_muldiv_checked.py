#!/usr/bin/env python3
"""Reference checks for FPQ16MulDivRoundedChecked one-rounding semantics."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)
U64_MAX_VALUE = (1 << 64) - 1

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_DOMAIN = 3
FP_Q16_ERR_OVERFLOW = 4


def fp_abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    return (-(x + 1)) + 1


def fp_try_apply_sign_from_u64_checked(mag: int, is_negative: bool) -> tuple[int, int]:
    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if is_negative:
        if mag >= (1 << 63):
            return FP_Q16_OK, I64_MIN_VALUE
        return FP_Q16_OK, -mag

    return FP_Q16_OK, mag


def fpq16_mul_div_rounded_checked(a_q16: int, b_q16: int, d_q16: int) -> tuple[int, int]:
    if d_q16 == 0:
        return FP_Q16_ERR_DOMAIN, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    abs_d = fp_abs_to_u64(d_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0) ^ (d_q16 < 0)

    if abs_a != 0 and abs_b != 0 and abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_num = abs_a * abs_b
    q = abs_num // abs_d
    r = abs_num % abs_d

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if q > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if r >= ((abs_d + 1) >> 1):
        if q == limit:
            return FP_Q16_ERR_OVERFLOW, 0
        q += 1

    return fp_try_apply_sign_from_u64_checked(q, is_negative)


def q16_from_float(x: float) -> int:
    if x >= 0.0:
        return int(x * FP_Q16_ONE + 0.5)
    return -int((-x) * FP_Q16_ONE + 0.5)


def q16_to_float(x: int) -> float:
    return x / FP_Q16_ONE


def test_domain_and_overflow_contracts() -> None:
    assert fpq16_mul_div_rounded_checked(1, 1, 0)[0] == FP_Q16_ERR_DOMAIN

    assert fpq16_mul_div_rounded_checked(I64_MAX_VALUE, I64_MAX_VALUE, FP_Q16_ONE)[0] == FP_Q16_ERR_OVERFLOW

    assert fpq16_mul_div_rounded_checked(I64_MAX_VALUE, FP_Q16_ONE, 1)[0] == FP_Q16_ERR_OVERFLOW


def test_sign_and_half_up_rounding() -> None:
    one = FP_Q16_ONE

    err, out = fpq16_mul_div_rounded_checked(one, one, 2 * one)
    assert err == FP_Q16_OK
    assert out == one // 2

    err, out = fpq16_mul_div_rounded_checked(one, one, q16_from_float(3.0))
    assert err == FP_Q16_OK
    assert abs(q16_to_float(out) - (1.0 / 3.0)) <= (1.0 / FP_Q16_ONE)

    err_p, out_p = fpq16_mul_div_rounded_checked(one, one, one)
    err_n, out_n = fpq16_mul_div_rounded_checked(-one, one, one)
    err_nn, out_nn = fpq16_mul_div_rounded_checked(-one, one, -one)
    assert (err_p, out_p) == (FP_Q16_OK, one)
    assert (err_n, out_n) == (FP_Q16_OK, -one)
    assert (err_nn, out_nn) == (FP_Q16_OK, one)


def test_randomized_real_value_parity() -> None:
    rng = random.Random(20260417_151)

    for _ in range(8000):
        a = rng.randint(-(1 << 30), 1 << 30)
        b = rng.randint(-(1 << 30), 1 << 30)
        d = rng.randint(-(1 << 30), 1 << 30)
        if d == 0:
            d = 1

        err, got = fpq16_mul_div_rounded_checked(a, b, d)
        if err != FP_Q16_OK:
            assert err in (FP_Q16_ERR_OVERFLOW, FP_Q16_ERR_DOMAIN)
            continue

        want = (q16_to_float(a) * q16_to_float(b)) / q16_to_float(d)
        got_f = q16_to_float(got)
        assert abs(got_f - want) <= (1.5 / FP_Q16_ONE)


def run() -> None:
    test_domain_and_overflow_contracts()
    test_sign_and_half_up_rounding()
    test_randomized_real_value_parity()
    print("fixedpoint_q16_muldiv_checked_reference_checks=ok")


if __name__ == "__main__":
    run()

