#!/usr/bin/env python3
"""Reference checks for FPQ16MulDivArrayRoundedChecked semantics."""

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


def fpq16_mul_div_array_rounded_checked(
    a_q16: list[int], b_q16: list[int], d_q16: list[int]
) -> tuple[int, list[int] | None]:
    if len(a_q16) != len(b_q16) or len(a_q16) != len(d_q16):
        return FP_Q16_ERR_BAD_PARAM, None

    out: list[int] = []
    for ai, bi, di in zip(a_q16, b_q16, d_q16):
        err, lane = fpq16_mul_div_rounded_checked(ai, bi, di)
        if err != FP_Q16_OK:
            return err, None
        out.append(lane)

    return FP_Q16_OK, out


def q16_to_float(x: int) -> float:
    return x / FP_Q16_ONE


def test_contract_errors_and_preflight_behavior() -> None:
    err, out = fpq16_mul_div_array_rounded_checked([1], [1], [0])
    assert err == FP_Q16_ERR_DOMAIN
    assert out is None

    err, out = fpq16_mul_div_array_rounded_checked(
        [1, I64_MAX_VALUE],
        [1, I64_MAX_VALUE],
        [1, FP_Q16_ONE],
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out is None


def test_randomized_lane_parity() -> None:
    rng = random.Random(20260417_181)

    for _ in range(4000):
        n = rng.randint(1, 24)
        a = [rng.randint(-(1 << 30), 1 << 30) for _ in range(n)]
        b = [rng.randint(-(1 << 30), 1 << 30) for _ in range(n)]
        d = [rng.randint(-(1 << 30), 1 << 30) or 1 for _ in range(n)]

        err, out = fpq16_mul_div_array_rounded_checked(a, b, d)
        if err != FP_Q16_OK:
            assert err in (FP_Q16_ERR_DOMAIN, FP_Q16_ERR_OVERFLOW)
            assert out is None
            continue

        assert out is not None
        for ai, bi, di, got in zip(a, b, d, out):
            want = (q16_to_float(ai) * q16_to_float(bi)) / q16_to_float(di)
            assert abs(q16_to_float(got) - want) <= (1.5 / FP_Q16_ONE)


def run() -> None:
    test_contract_errors_and_preflight_behavior()
    test_randomized_lane_parity()
    print("fixedpoint_q16_muldiv_array_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
