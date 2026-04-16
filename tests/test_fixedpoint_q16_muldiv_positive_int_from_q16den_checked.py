#!/usr/bin/env python3
"""Reference checks for FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked."""

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


def fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(
    a_q16: int,
    b_q16: int,
    d_q16: int,
) -> tuple[int, int]:
    if d_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, 0

    if (d_q16 & (FP_Q16_ONE - 1)) != 0:
        return FP_Q16_ERR_BAD_PARAM, 0

    d_int = d_q16 >> FP_Q16_SHIFT
    if d_int <= 0:
        return FP_Q16_ERR_DOMAIN, 0

    return fpq16_mul_div_rounded_by_positive_int_checked(a_q16, b_q16, d_int)


def q16_to_float(x: int) -> float:
    return x / FP_Q16_ONE


def test_domain_and_encoding_contracts() -> None:
    one = FP_Q16_ONE

    assert fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(1, 1, 0)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(1, 1, -one)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(1, 1, one + 1)[0] == FP_Q16_ERR_BAD_PARAM


def test_fast_path_routing_matches_positive_int_helper() -> None:
    rng = random.Random(20260417_153)

    for _ in range(10000):
        a = rng.randint(-(1 << 30), 1 << 30)
        b = rng.randint(-(1 << 30), 1 << 30)
        d_int = rng.randint(1, 1 << 20)
        d_q16 = d_int << FP_Q16_SHIFT

        err_ref, out_ref = fpq16_mul_div_rounded_by_positive_int_checked(a, b, d_int)
        err_new, out_new = fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(a, b, d_q16)
        assert (err_new, out_new) == (err_ref, out_ref)


def test_real_value_parity_with_q16_denominator() -> None:
    rng = random.Random(20260417_3153)

    for _ in range(6000):
        a = rng.randint(-(1 << 30), 1 << 30)
        b = rng.randint(-(1 << 30), 1 << 30)
        d_int = rng.randint(1, 1 << 24)
        d_q16 = d_int << FP_Q16_SHIFT

        err, got = fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(a, b, d_q16)
        if err != FP_Q16_OK:
            assert err == FP_Q16_ERR_OVERFLOW
            continue

        want = (q16_to_float(a) * q16_to_float(b)) / float(d_int)
        got_f = q16_to_float(got)
        assert abs(got_f - want) <= (1.5 / FP_Q16_ONE)


def run() -> None:
    test_domain_and_encoding_contracts()
    test_fast_path_routing_matches_positive_int_helper()
    test_real_value_parity_with_q16_denominator()
    print("fixedpoint_q16_muldiv_positive_int_from_q16den_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
