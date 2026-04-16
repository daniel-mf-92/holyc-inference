#!/usr/bin/env python3
"""Reference checks for FPQ16RMSNormApplyInvDenomChecked semantics."""

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


def fpq16_rmsnorm_apply_inv_denom_checked(
    input_q16: list[int] | None,
    count: int,
    inv_denom_q16: int,
    den_q16: int,
) -> tuple[int, list[int]]:
    if input_q16 is None:
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
        out[i] = scaled

    return FP_Q16_OK, out


def q16_to_float(x: int) -> float:
    return x / FP_Q16_ONE


def test_contract_surfaces() -> None:
    vec = [1, 2, 3]

    assert fpq16_rmsnorm_apply_inv_denom_checked(None, 3, FP_Q16_ONE, FP_Q16_ONE)[0] == FP_Q16_ERR_NULL_PTR
    assert fpq16_rmsnorm_apply_inv_denom_checked(vec, -1, FP_Q16_ONE, FP_Q16_ONE)[0] == FP_Q16_ERR_BAD_PARAM
    assert fpq16_rmsnorm_apply_inv_denom_checked(vec, 3, 0, FP_Q16_ONE)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_rmsnorm_apply_inv_denom_checked(vec, 3, -FP_Q16_ONE, FP_Q16_ONE)[0] == FP_Q16_ERR_DOMAIN

    assert fpq16_rmsnorm_apply_inv_denom_checked(vec, 3, FP_Q16_ONE, 0)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_rmsnorm_apply_inv_denom_checked(vec, 3, FP_Q16_ONE, -FP_Q16_ONE)[0] == FP_Q16_ERR_DOMAIN
    assert fpq16_rmsnorm_apply_inv_denom_checked(vec, 3, FP_Q16_ONE, FP_Q16_ONE + 1)[0] == FP_Q16_ERR_BAD_PARAM


def test_count_zero_is_noop() -> None:
    err, out = fpq16_rmsnorm_apply_inv_denom_checked([11, 22, 33], 0, FP_Q16_ONE, FP_Q16_ONE)
    assert err == FP_Q16_OK
    assert out == []


def test_elementwise_parity_against_helper() -> None:
    rng = random.Random(20260417_1541)

    for _ in range(4000):
        count = rng.randint(1, 64)
        vec = [rng.randint(-(1 << 30), 1 << 30) for _ in range(count)]
        inv = rng.randint(1, 1 << 30)
        den_int = rng.randint(1, 1 << 16)
        den_q16 = den_int << FP_Q16_SHIFT

        err_vec, out_vec = fpq16_rmsnorm_apply_inv_denom_checked(vec, count, inv, den_q16)

        expected = []
        err_ref = FP_Q16_OK
        for x in vec:
            err_ref, scaled = fpq16_mul_div_rounded_by_positive_int_from_q16_den_checked(x, inv, den_q16)
            if err_ref != FP_Q16_OK:
                break
            expected.append(scaled)

        assert err_vec == err_ref
        if err_vec == FP_Q16_OK:
            assert out_vec == expected


def test_real_value_error_bound() -> None:
    rng = random.Random(20260417_1542)

    for _ in range(2000):
        count = rng.randint(1, 32)
        vec = [rng.randint(-(1 << 28), 1 << 28) for _ in range(count)]
        inv = rng.randint(1, 1 << 28)
        den_int = rng.randint(1, 1 << 15)
        den_q16 = den_int << FP_Q16_SHIFT

        err, out = fpq16_rmsnorm_apply_inv_denom_checked(vec, count, inv, den_q16)
        if err != FP_Q16_OK:
            assert err == FP_Q16_ERR_OVERFLOW
            continue

        for x_q16, got_q16 in zip(vec, out):
            want = (q16_to_float(x_q16) * q16_to_float(inv)) / float(den_int)
            got = q16_to_float(got_q16)
            assert abs(got - want) <= (1.5 / FP_Q16_ONE)


def test_overflow_propagation() -> None:
    huge = I64_MAX_VALUE
    err, out = fpq16_rmsnorm_apply_inv_denom_checked([huge], 1, huge, FP_Q16_ONE)
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == []


def run() -> None:
    test_contract_surfaces()
    test_count_zero_is_noop()
    test_elementwise_parity_against_helper()
    test_real_value_error_bound()
    test_overflow_propagation()
    print("rmsnorm_apply_inv_denom_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
