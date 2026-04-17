#!/usr/bin/env python3
"""Reference checks for FPQ16MulDivArrayRoundedByPositiveIntFromQ16DenCheckedNoAlias semantics."""

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


def fpq16_mul_div_rounded_by_positive_int_checked(
    a_q16: int, b_q16: int, d_int: int
) -> tuple[int, int]:
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


def fpq16_mul_div_rounded_by_positive_int_from_q16den_checked(
    a_q16: int, b_q16: int, d_q16: int
) -> tuple[int, int]:
    if d_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, 0

    if d_q16 & (FP_Q16_ONE - 1):
        return FP_Q16_ERR_BAD_PARAM, 0

    d_int = d_q16 >> FP_Q16_SHIFT
    if d_int <= 0:
        return FP_Q16_ERR_DOMAIN, 0

    return fpq16_mul_div_rounded_by_positive_int_checked(a_q16, b_q16, d_int)


def fpq16_mul_div_array_rounded_by_positive_int_from_q16den_checked(
    a_q16: list[int], b_q16: list[int], d_q16: list[int]
) -> tuple[int, list[int] | None]:
    if len(a_q16) != len(b_q16) or len(a_q16) != len(d_q16):
        return FP_Q16_ERR_BAD_PARAM, None

    for ai, bi, di in zip(a_q16, b_q16, d_q16):
        err, _ = fpq16_mul_div_rounded_by_positive_int_from_q16den_checked(ai, bi, di)
        if err != FP_Q16_OK:
            return err, None

    out: list[int] = []
    for ai, bi, di in zip(a_q16, b_q16, d_q16):
        err, lane = fpq16_mul_div_rounded_by_positive_int_from_q16den_checked(ai, bi, di)
        if err != FP_Q16_OK:
            return err, None
        out.append(lane)

    return FP_Q16_OK, out


def fpq16_mul_div_array_rounded_by_positive_int_from_q16den_checked_no_alias(
    a_q16: list[int], b_q16: list[int], d_q16: list[int], out_alias: str | None
) -> tuple[int, list[int] | None]:
    if out_alias == "a" or out_alias == "b" or out_alias == "d":
        return FP_Q16_ERR_BAD_PARAM, None
    return fpq16_mul_div_array_rounded_by_positive_int_from_q16den_checked(a_q16, b_q16, d_q16)


def q16_to_float(x: int) -> float:
    return x / FP_Q16_ONE


def test_no_alias_rejects_all_input_alias_forms() -> None:
    one = FP_Q16_ONE
    a = [one, 2 * one]
    b = [3 * one, 4 * one]
    d = [one, one]

    err, out = fpq16_mul_div_array_rounded_by_positive_int_from_q16den_checked_no_alias(a, b, d, "a")
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out is None

    err, out = fpq16_mul_div_array_rounded_by_positive_int_from_q16den_checked_no_alias(a, b, d, "b")
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out is None

    err, out = fpq16_mul_div_array_rounded_by_positive_int_from_q16den_checked_no_alias(a, b, d, "d")
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out is None


def test_no_alias_wrapper_matches_core_when_distinct() -> None:
    one = FP_Q16_ONE
    a = [5 * one, -7 * one, 2 * one]
    b = [3 * one, 11 * one, -13 * one]
    d = [one, 2 * one, 4 * one]

    err_core, out_core = fpq16_mul_div_array_rounded_by_positive_int_from_q16den_checked(a, b, d)
    err_wrap, out_wrap = fpq16_mul_div_array_rounded_by_positive_int_from_q16den_checked_no_alias(a, b, d, None)

    assert err_core == FP_Q16_OK
    assert err_wrap == FP_Q16_OK
    assert out_core == out_wrap


def test_randomized_semantics_and_error_surface() -> None:
    rng = random.Random(20260417_205)

    for _ in range(5000):
        n = rng.randint(1, 24)
        a = [rng.randint(-(1 << 30), 1 << 30) for _ in range(n)]
        b = [rng.randint(-(1 << 30), 1 << 30) for _ in range(n)]

        d_q16: list[int] = []
        for _ in range(n):
            if rng.random() < 0.12:
                d_q16.append((rng.randint(1, 1 << 16) << FP_Q16_SHIFT) + rng.randint(1, 255))
            else:
                d_q16.append(rng.randint(1, 1 << 16) << FP_Q16_SHIFT)

        err_core, out_core = fpq16_mul_div_array_rounded_by_positive_int_from_q16den_checked(a, b, d_q16)
        err_wrap, out_wrap = fpq16_mul_div_array_rounded_by_positive_int_from_q16den_checked_no_alias(a, b, d_q16, None)

        assert err_wrap == err_core
        assert out_wrap == out_core

        if err_wrap != FP_Q16_OK:
            assert err_wrap in (FP_Q16_ERR_BAD_PARAM, FP_Q16_ERR_DOMAIN, FP_Q16_ERR_OVERFLOW)
            assert out_wrap is None
            continue

        assert out_wrap is not None
        for ai, bi, di_q16, got in zip(a, b, d_q16, out_wrap):
            d_int = di_q16 >> FP_Q16_SHIFT
            want = (q16_to_float(ai) * q16_to_float(bi)) / float(d_int)
            got_f = q16_to_float(got)
            assert abs(got_f - want) <= (1.5 / FP_Q16_ONE)


def run() -> None:
    test_no_alias_rejects_all_input_alias_forms()
    test_no_alias_wrapper_matches_core_when_distinct()
    test_randomized_semantics_and_error_surface()
    print("fixedpoint_q16_muldiv_array_positive_int_from_q16den_no_alias_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
