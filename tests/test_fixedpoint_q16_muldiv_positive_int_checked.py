#!/usr/bin/env python3
"""Reference checks for FPQ16MulDivRoundedByPositiveIntChecked semantics."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

I64_MAX_VALUE = (1 << 63) - 1
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
            return FP_Q16_OK, -(1 << 63)
        return FP_Q16_OK, -mag

    return FP_Q16_OK, mag


def fpq16_mul_div_rounded_by_positive_int_checked(a_q16: int, b_q16: int, d_int: int) -> tuple[int, int]:
    if d_int <= 0:
        return FP_Q16_ERR_DOMAIN, 0

    if d_int > (U64_MAX_VALUE >> FP_Q16_SHIFT):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    if abs_a != 0 and abs_b != 0 and abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_num = abs_a * abs_b
    abs_den_q16 = d_int << FP_Q16_SHIFT

    q = abs_num // abs_den_q16
    r = abs_num % abs_den_q16

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if q > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if r >= ((abs_den_q16 + 1) >> 1):
        if q == limit:
            return FP_Q16_ERR_OVERFLOW, 0
        q += 1

    return fp_try_apply_sign_from_u64_checked(q, is_negative)


def q16_to_float(x: int) -> float:
    return x / FP_Q16_ONE


def test_domain_and_overflow_contracts() -> None:
    assert fpq16_mul_div_rounded_by_positive_int_checked(1, 1, 0)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_mul_div_rounded_by_positive_int_checked(1, 1, -7)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_mul_div_rounded_by_positive_int_checked(I64_MAX_VALUE, I64_MAX_VALUE, 1)[0] == FP_Q16_ERR_OVERFLOW


def test_sign_and_half_up_rounding() -> None:
    one = FP_Q16_ONE

    err, out = fpq16_mul_div_rounded_by_positive_int_checked(one, one, 2)
    assert err == FP_Q16_OK
    assert out == (one // 2)

    err, out = fpq16_mul_div_rounded_by_positive_int_checked(one, one, 3)
    assert err == FP_Q16_OK
    assert abs(q16_to_float(out) - (1.0 / 3.0)) <= (1.0 / FP_Q16_ONE)

    err_p, out_p = fpq16_mul_div_rounded_by_positive_int_checked(one, one, 1)
    err_n, out_n = fpq16_mul_div_rounded_by_positive_int_checked(-one, one, 1)
    assert (err_p, out_p) == (FP_Q16_OK, one)
    assert (err_n, out_n) == (FP_Q16_OK, -one)


def test_randomized_real_value_parity() -> None:
    rng = random.Random(20260417_152)

    for _ in range(10000):
        a = rng.randint(-(1 << 30), 1 << 30)
        b = rng.randint(-(1 << 30), 1 << 30)
        d = rng.randint(1, 1 << 24)

        err, got = fpq16_mul_div_rounded_by_positive_int_checked(a, b, d)
        if err != FP_Q16_OK:
            assert err == FP_Q16_ERR_OVERFLOW
            continue

        want = (q16_to_float(a) * q16_to_float(b)) / float(d)
        got_f = q16_to_float(got)
        assert abs(got_f - want) <= (1.5 / FP_Q16_ONE)


def run() -> None:
    test_domain_and_overflow_contracts()
    test_sign_and_half_up_rounding()
    test_randomized_real_value_parity()
    print("fixedpoint_q16_muldiv_positive_int_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
