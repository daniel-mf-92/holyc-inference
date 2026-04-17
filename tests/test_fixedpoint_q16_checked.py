#!/usr/bin/env python3
"""Reference checks for fixedpoint checked Q16 helpers."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_DOMAIN = 3
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)
U64_MAX_VALUE = (1 << 64) - 1


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


def fpq16_mul_checked(a_q16: int, b_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if not a_q16 or not b_q16:
        return FP_Q16_OK, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_prod = abs_a * abs_b
    round_bias = 1 << (FP_Q16_SHIFT - 1)

    if abs_prod > U64_MAX_VALUE - round_bias:
        rounded_mag = U64_MAX_VALUE >> FP_Q16_SHIFT
    else:
        rounded_mag = (abs_prod + round_bias) >> FP_Q16_SHIFT

    if rounded_mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    return fp_try_apply_sign_from_u64_checked(rounded_mag, is_negative)


def fpq16_div_checked(num_q16: int, den_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0
    if not den_q16:
        return FP_Q16_ERR_DOMAIN, 0

    abs_num = fp_abs_to_u64(num_q16)
    abs_den = fp_abs_to_u64(den_q16)
    is_negative = (num_q16 < 0) ^ (den_q16 < 0)

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    int_part = abs_num // abs_den
    if int_part > (limit >> FP_Q16_SHIFT):
        return FP_Q16_ERR_OVERFLOW, 0

    result_mag = int_part << FP_Q16_SHIFT
    rem = abs_num % abs_den

    for bit in range(FP_Q16_SHIFT - 1, -1, -1):
        rem <<= 1
        if rem >= abs_den:
            rem -= abs_den
            add_bit = 1 << bit
            if result_mag > (limit - add_bit):
                return FP_Q16_ERR_OVERFLOW, 0
            result_mag |= add_bit

    if rem >= ((abs_den + 1) >> 1):
        if result_mag == limit:
            return FP_Q16_ERR_OVERFLOW, 0
        result_mag += 1

    return fp_try_apply_sign_from_u64_checked(result_mag, is_negative)


def fpq16_mul_array_checked(lhs_q16: list[int], rhs_q16: list[int], out_q16: list[int]) -> tuple[int, list[int]]:
    if lhs_q16 is None or rhs_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR, out_q16
    if len(lhs_q16) != len(rhs_q16) or len(lhs_q16) != len(out_q16):
        return FP_Q16_ERR_BAD_PARAM, out_q16

    scratch: list[int] = []
    for a, b in zip(lhs_q16, rhs_q16):
        err, lane = fpq16_mul_checked(a, b)
        if err != FP_Q16_OK:
            return err, out_q16
        scratch.append(lane)

    for idx, lane in enumerate(scratch):
        out_q16[idx] = lane
    return FP_Q16_OK, out_q16


def fpq16_div_array_checked(num_q16: list[int], den_q16: list[int], out_q16: list[int]) -> tuple[int, list[int]]:
    if num_q16 is None or den_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR, out_q16
    if len(num_q16) != len(den_q16) or len(num_q16) != len(out_q16):
        return FP_Q16_ERR_BAD_PARAM, out_q16

    scratch: list[int] = []
    for n, d in zip(num_q16, den_q16):
        err, lane = fpq16_div_checked(n, d)
        if err != FP_Q16_OK:
            return err, out_q16
        scratch.append(lane)

    for idx, lane in enumerate(scratch):
        out_q16[idx] = lane
    return FP_Q16_OK, out_q16


def q16_from_float(x: float) -> int:
    return int(round(x * FP_Q16_ONE))


def q16_to_float(x_q16: int) -> float:
    return x_q16 / FP_Q16_ONE


def test_mul_checked_null_ptr_and_zero_fast_path() -> None:
    err, out = fpq16_mul_checked(123, 456, out_present=False)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == 0

    err, out = fpq16_mul_checked(0, 999)
    assert err == FP_Q16_OK
    assert out == 0


def test_mul_checked_matches_float_reference_on_random_domain() -> None:
    rng = random.Random(2026041701)

    for _ in range(5000):
        a = rng.uniform(-64.0, 64.0)
        b = rng.uniform(-64.0, 64.0)
        a_q16 = q16_from_float(a)
        b_q16 = q16_from_float(b)

        err, got_q16 = fpq16_mul_checked(a_q16, b_q16)
        assert err == FP_Q16_OK

        got = q16_to_float(got_q16)
        want = (a_q16 / FP_Q16_ONE) * (b_q16 / FP_Q16_ONE)
        assert abs(got - want) <= (1.0 / FP_Q16_ONE) + 1e-9


def test_div_checked_domain_and_sign_rules() -> None:
    err, out = fpq16_div_checked(123, 0)
    assert err == FP_Q16_ERR_DOMAIN
    assert out == 0

    err, out = fpq16_div_checked(q16_from_float(3.5), q16_from_float(-0.5))
    assert err == FP_Q16_OK
    assert out < 0
    assert abs(q16_to_float(out) + 7.0) <= (1.0 / FP_Q16_ONE)


def test_div_checked_matches_float_reference_on_random_domain() -> None:
    rng = random.Random(2026041702)

    for _ in range(5000):
        num = rng.uniform(-256.0, 256.0)
        den = rng.uniform(-256.0, 256.0)
        if abs(den) < 1e-6:
            den = 0.25

        num_q16 = q16_from_float(num)
        den_q16 = q16_from_float(den)
        if den_q16 == 0:
            den_q16 = 1

        err, got_q16 = fpq16_div_checked(num_q16, den_q16)
        assert err == FP_Q16_OK

        got = q16_to_float(got_q16)
        want = (num_q16 / FP_Q16_ONE) / (den_q16 / FP_Q16_ONE)
        assert abs(got - want) <= (2.0 / FP_Q16_ONE) + 1e-9


def test_mul_array_checked_no_partial_write_on_failure() -> None:
    lhs = [q16_from_float(1.5), q16_from_float(2.0), I64_MIN_VALUE]
    rhs = [q16_from_float(2.0), q16_from_float(-3.0), I64_MIN_VALUE]
    out = [111, 222, 333]

    err, out_after = fpq16_mul_array_checked(lhs, rhs, out)
    assert err == FP_Q16_ERR_OVERFLOW
    assert out_after == [111, 222, 333]


def test_div_array_checked_no_partial_write_on_failure() -> None:
    num = [q16_from_float(1.0), q16_from_float(2.0), q16_from_float(3.0)]
    den = [q16_from_float(2.0), 0, q16_from_float(4.0)]
    out = [7, 8, 9]

    err, out_after = fpq16_div_array_checked(num, den, out)
    assert err == FP_Q16_ERR_DOMAIN
    assert out_after == [7, 8, 9]


def test_array_checked_success_paths() -> None:
    lhs = [q16_from_float(1.25), q16_from_float(-2.5), q16_from_float(0.125)]
    rhs = [q16_from_float(8.0), q16_from_float(3.0), q16_from_float(-4.0)]
    out_mul = [0, 0, 0]

    err_mul, got_mul = fpq16_mul_array_checked(lhs, rhs, out_mul)
    assert err_mul == FP_Q16_OK
    assert got_mul == [q16_from_float(10.0), q16_from_float(-7.5), q16_from_float(-0.5)]

    num = [q16_from_float(9.0), q16_from_float(-7.5), q16_from_float(1.0)]
    den = [q16_from_float(3.0), q16_from_float(2.5), q16_from_float(8.0)]
    out_div = [0, 0, 0]

    err_div, got_div = fpq16_div_array_checked(num, den, out_div)
    assert err_div == FP_Q16_OK
    assert got_div == [q16_from_float(3.0), q16_from_float(-3.0), q16_from_float(0.125)]


def run() -> None:
    test_mul_checked_null_ptr_and_zero_fast_path()
    test_mul_checked_matches_float_reference_on_random_domain()
    test_div_checked_domain_and_sign_rules()
    test_div_checked_matches_float_reference_on_random_domain()
    test_mul_array_checked_no_partial_write_on_failure()
    test_div_array_checked_no_partial_write_on_failure()
    test_array_checked_success_paths()
    print("fixedpoint_q16_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
