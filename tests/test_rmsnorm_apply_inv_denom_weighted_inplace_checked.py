#!/usr/bin/env python3
"""Reference checks for FPQ16RMSNormApplyInvDenomWeightedInPlaceChecked semantics."""

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


def fpq16_mul_div_rounded_checked(a_q16: int, b_q16: int, d_q16: int) -> tuple[int, int]:
    if d_q16 == 0:
        return FP_Q16_ERR_DOMAIN, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    abs_d = fp_abs_to_u64(d_q16)

    if abs_d == 0:
        return FP_Q16_ERR_DOMAIN, 0

    num_negative = (a_q16 < 0) ^ (b_q16 < 0)
    den_negative = d_q16 < 0
    out_negative = num_negative ^ den_negative

    if abs_a != 0 and abs_b != 0 and abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_num = abs_a * abs_b

    q = abs_num // abs_d
    r = abs_num % abs_d

    limit = I64_MAX_VALUE
    if out_negative:
        limit = 1 << 63

    if q > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if r >= ((abs_d + 1) >> 1):
        if q == limit:
            return FP_Q16_ERR_OVERFLOW, 0
        q += 1

    return fp_try_apply_sign_from_u64_checked(q, out_negative)


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


def fpq16_rmsnorm_apply_inv_denom_weighted_checked(
    input_q16: list[int] | None,
    gamma_q16: list[int] | None,
    count: int,
    inv_denom_q16: int,
    den_q16: int,
) -> tuple[int, list[int]]:
    if input_q16 is None or gamma_q16 is None:
        return FP_Q16_ERR_NULL_PTR, []
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if inv_denom_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, []

    out = [0] * max(count, 0)
    for i in range(count):
        err, scaled = fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(
            input_q16[i],
            inv_denom_q16,
            den_q16,
        )
        if err != FP_Q16_OK:
            return err, []

        err, weighted = fpq16_mul_div_rounded_checked(scaled, gamma_q16[i], FP_Q16_ONE)
        if err != FP_Q16_OK:
            return err, []

        out[i] = weighted

    return FP_Q16_OK, out


def fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(
    input_q16: list[int] | None,
    gamma_q16: list[int] | None,
    count: int,
    inv_denom_q16: int,
    den_q16: int,
) -> tuple[int, list[int]]:
    if input_q16 is None or gamma_q16 is None:
        return FP_Q16_ERR_NULL_PTR, []
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if inv_denom_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, []

    out = list(input_q16)
    for i in range(count):
        err, scaled = fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(
            out[i],
            inv_denom_q16,
            den_q16,
        )
        if err != FP_Q16_OK:
            return err, []

        err, weighted = fpq16_mul_div_rounded_checked(scaled, gamma_q16[i], FP_Q16_ONE)
        if err != FP_Q16_OK:
            return err, []

        out[i] = weighted

    return FP_Q16_OK, out


def test_contract_surfaces() -> None:
    vec = [1, 2, 3]
    gamma = [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE]

    assert fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(None, gamma, 3, FP_Q16_ONE, FP_Q16_ONE)[0] == FP_Q16_ERR_NULL_PTR
    assert fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(vec, None, 3, FP_Q16_ONE, FP_Q16_ONE)[0] == FP_Q16_ERR_NULL_PTR
    assert fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(vec, gamma, -1, FP_Q16_ONE, FP_Q16_ONE)[0] == FP_Q16_ERR_BAD_PARAM
    assert fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(vec, gamma, 3, 0, FP_Q16_ONE)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(vec, gamma, 3, FP_Q16_ONE, 0)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(vec, gamma, 3, FP_Q16_ONE, FP_Q16_ONE + 1)[0] == FP_Q16_ERR_BAD_PARAM


def test_count_zero_is_noop_and_buffer_unchanged() -> None:
    vec = [11, 22]
    gamma = [33, 44]
    err, out = fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(vec, gamma, 0, FP_Q16_ONE, FP_Q16_ONE)
    assert err == FP_Q16_OK
    assert out == vec


def test_inplace_matches_out_of_place_reference() -> None:
    rng = random.Random(20260417_1561)

    for _ in range(5000):
        count = rng.randint(1, 64)
        vec = [rng.randint(-(1 << 27), 1 << 27) for _ in range(count)]
        gamma = [rng.randint(-(1 << 17), 1 << 17) for _ in range(count)]
        inv = rng.randint(1, 1 << 27)
        den_int = rng.randint(1, 1 << 16)
        den_q16 = den_int << FP_Q16_SHIFT

        err_in, out_in = fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked(vec, gamma, count, inv, den_q16)
        err_ref, out_ref = fpq16_rmsnorm_apply_inv_denom_weighted_checked(vec, gamma, count, inv, den_q16)

        assert err_in == err_ref
        if err_in == FP_Q16_OK:
            assert out_in == out_ref


def test_overflow_propagation() -> None:
    huge = I64_MAX_VALUE
    err, out = fpq16_rmsnorm_apply_inv_denom_weighted_inplace_checked([FP_Q16_ONE], [huge], 1, huge, FP_Q16_ONE)
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == []


def run() -> None:
    test_contract_surfaces()
    test_count_zero_is_noop_and_buffer_unchanged()
    test_inplace_matches_out_of_place_reference()
    test_overflow_propagation()
    print("rmsnorm_apply_inv_denom_weighted_inplace_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
