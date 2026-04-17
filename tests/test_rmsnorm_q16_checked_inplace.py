#!/usr/bin/env python3
"""Reference checks for FPQ16RMSNormCheckedInPlace semantics."""

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


def fpq16_from_int(x: int) -> int:
    max_int = I64_MAX_VALUE >> FP_Q16_SHIFT
    min_int = -(1 << (63 - FP_Q16_SHIFT))
    if x > max_int:
        return I64_MAX_VALUE
    if x < min_int:
        return -(1 << 63)
    return x << FP_Q16_SHIFT


def fpq16_mul_checked(a_q16: int, b_q16: int) -> tuple[int, int]:
    if a_q16 == 0 or b_q16 == 0:
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


def fpq16_div(num: int, den: int) -> int:
    if den == 0:
        return 0

    abs_num = fp_abs_to_u64(num)
    abs_den = fp_abs_to_u64(den)
    is_negative = (num < 0) ^ (den < 0)

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    int_part = abs_num // abs_den
    if int_part > (limit >> FP_Q16_SHIFT):
        return -(1 << 63) if is_negative else I64_MAX_VALUE

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

    _, signed = fp_try_apply_sign_from_u64_checked(result_mag, is_negative)
    return signed


def int_sqrt_u64(x: int) -> int:
    res = 0
    bit = 1 << 62

    while bit > x:
        bit >>= 2

    while bit != 0:
        if x >= res + bit:
            x -= res + bit
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2

    return res


def fpq16_sqrt(x_q16: int) -> int:
    if x_q16 <= 0:
        return 0

    if x_q16 > (I64_MAX_VALUE >> FP_Q16_SHIFT):
        shifted = I64_MAX_VALUE
    else:
        shifted = x_q16 << FP_Q16_SHIFT

    return int_sqrt_u64(shifted)


def fpq16_rmsnorm_compute_inv_denom_checked(
    input_q16: list[int] | None,
    count: int,
    eps_q16: int,
) -> tuple[int, int, int]:
    if input_q16 is None:
        return FP_Q16_ERR_NULL_PTR, 0, 0
    if count <= 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    if eps_q16 < 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    if count > (I64_MAX_VALUE >> FP_Q16_SHIFT):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    sum_sq_q16 = 0
    for i in range(count):
        err, sq = fpq16_mul_checked(input_q16[i], input_q16[i])
        if err != FP_Q16_OK:
            return err, 0, 0
        if sum_sq_q16 > I64_MAX_VALUE - sq:
            return FP_Q16_ERR_OVERFLOW, 0, 0
        sum_sq_q16 += sq

    count_q16 = fpq16_from_int(count)
    if count_q16 <= 0:
        return FP_Q16_ERR_OVERFLOW, 0, 0

    mean_sq_q16 = fpq16_div(sum_sq_q16, count_q16)
    if mean_sq_q16 < 0:
        return FP_Q16_ERR_DOMAIN, 0, 0

    if mean_sq_q16 > I64_MAX_VALUE - eps_q16:
        return FP_Q16_ERR_OVERFLOW, 0, 0

    denom_arg_q16 = mean_sq_q16 + eps_q16
    denom_q16 = fpq16_sqrt(denom_arg_q16)
    if denom_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, 0, 0

    inv_denom_q16 = fpq16_div(FP_Q16_ONE, denom_q16)
    if inv_denom_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, 0, 0

    return FP_Q16_OK, inv_denom_q16, denom_q16


def fpq16_rmsnorm_checked(
    input_q16: list[int] | None,
    gamma_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    eps_q16: int,
) -> tuple[int, list[int]]:
    if input_q16 is None or gamma_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR, []
    if count <= 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if eps_q16 < 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if input_q16 is output_q16:
        return FP_Q16_ERR_BAD_PARAM, []

    err, inv_denom_q16, _ = fpq16_rmsnorm_compute_inv_denom_checked(input_q16, count, eps_q16)
    if err != FP_Q16_OK:
        return err, []

    for i in range(count):
        err, norm = fpq16_mul_checked(input_q16[i], inv_denom_q16)
        if err != FP_Q16_OK:
            return err, []
        err, _ = fpq16_mul_checked(norm, gamma_q16[i])
        if err != FP_Q16_OK:
            return err, []

    out = list(output_q16)
    for i in range(count):
        err, norm = fpq16_mul_checked(input_q16[i], inv_denom_q16)
        if err != FP_Q16_OK:
            return err, []
        err, weighted = fpq16_mul_checked(norm, gamma_q16[i])
        if err != FP_Q16_OK:
            return err, []
        out[i] = weighted

    return FP_Q16_OK, out


def fpq16_rmsnorm_checked_inplace(
    input_q16: list[int] | None,
    gamma_q16: list[int] | None,
    count: int,
    eps_q16: int,
) -> tuple[int, list[int]]:
    if input_q16 is None or gamma_q16 is None:
        return FP_Q16_ERR_NULL_PTR, []
    if count <= 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if eps_q16 < 0:
        return FP_Q16_ERR_BAD_PARAM, []

    err, inv_denom_q16, _ = fpq16_rmsnorm_compute_inv_denom_checked(input_q16, count, eps_q16)
    if err != FP_Q16_OK:
        return err, []

    for i in range(count):
        err, norm = fpq16_mul_checked(input_q16[i], inv_denom_q16)
        if err != FP_Q16_OK:
            return err, []
        err, _ = fpq16_mul_checked(norm, gamma_q16[i])
        if err != FP_Q16_OK:
            return err, []

    out = list(input_q16)
    for i in range(count):
        err, norm = fpq16_mul_checked(out[i], inv_denom_q16)
        if err != FP_Q16_OK:
            return err, []
        err, weighted = fpq16_mul_checked(norm, gamma_q16[i])
        if err != FP_Q16_OK:
            return err, []
        out[i] = weighted

    return FP_Q16_OK, out


def test_contract_surfaces() -> None:
    vec = [FP_Q16_ONE, 2 * FP_Q16_ONE]
    gamma = [FP_Q16_ONE, FP_Q16_ONE]

    assert fpq16_rmsnorm_checked_inplace(None, gamma, 2, 0)[0] == FP_Q16_ERR_NULL_PTR
    assert fpq16_rmsnorm_checked_inplace(vec, None, 2, 0)[0] == FP_Q16_ERR_NULL_PTR
    assert fpq16_rmsnorm_checked_inplace(vec, gamma, 0, 0)[0] == FP_Q16_ERR_BAD_PARAM
    assert fpq16_rmsnorm_checked_inplace(vec, gamma, -1, 0)[0] == FP_Q16_ERR_BAD_PARAM
    assert fpq16_rmsnorm_checked_inplace(vec, gamma, 2, -1)[0] == FP_Q16_ERR_BAD_PARAM


def test_no_partial_write_on_failure() -> None:
    vec = [I64_MAX_VALUE, I64_MAX_VALUE]
    gamma = [I64_MAX_VALUE, I64_MAX_VALUE]

    err, out = fpq16_rmsnorm_checked_inplace(vec, gamma, 2, 0)
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == []


def test_inplace_matches_out_of_place_reference() -> None:
    rng = random.Random(20260417_187)

    for _ in range(5000):
        count = rng.randint(1, 64)
        inp = [rng.randint(-(1 << 19), 1 << 19) for _ in range(count)]
        gamma = [rng.randint(int(0.25 * FP_Q16_ONE), int(2.25 * FP_Q16_ONE)) for _ in range(count)]
        eps_q16 = rng.randint(0, 2 * FP_Q16_ONE)

        err_in, out_in = fpq16_rmsnorm_checked_inplace(inp, gamma, count, eps_q16)
        err_ref, out_ref = fpq16_rmsnorm_checked(inp, gamma, [0] * count, count, eps_q16)

        assert err_in == err_ref
        if err_in == FP_Q16_OK:
            assert out_in == out_ref


def run() -> None:
    test_contract_surfaces()
    test_no_partial_write_on_failure()
    test_inplace_matches_out_of_place_reference()
    print("rmsnorm_q16_checked_inplace_reference_checks=ok")


if __name__ == "__main__":
    run()
