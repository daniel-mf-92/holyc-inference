#!/usr/bin/env python3
"""Reference checks for FPQ16MulDivArrayRoundedByPositiveIntChecked semantics."""

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


def fpq16_mul_div_array_rounded_by_positive_int_checked(
    a_q16: list[int], b_q16: list[int], d_int: list[int]
) -> tuple[int, list[int] | None]:
    if len(a_q16) != len(b_q16) or len(a_q16) != len(d_int):
        return FP_Q16_ERR_BAD_PARAM, None

    out: list[int] = []
    for ai, bi, di in zip(a_q16, b_q16, d_int):
        err, lane = fpq16_mul_div_rounded_by_positive_int_checked(ai, bi, di)
        if err != FP_Q16_OK:
            return err, None
        out.append(lane)
    return FP_Q16_OK, out


def simulate_holyc_array_call_no_partial_write(
    a_q16: list[int], b_q16: list[int], d_int: list[int], out_seed: list[int]
) -> tuple[int, list[int]]:
    if len(a_q16) != len(b_q16) or len(a_q16) != len(d_int) or len(a_q16) != len(out_seed):
        return FP_Q16_ERR_BAD_PARAM, out_seed.copy()

    snapshot = out_seed.copy()

    for ai, bi, di in zip(a_q16, b_q16, d_int):
        err, _ = fpq16_mul_div_rounded_by_positive_int_checked(ai, bi, di)
        if err != FP_Q16_OK:
            return err, snapshot

    out = out_seed.copy()
    for i, (ai, bi, di) in enumerate(zip(a_q16, b_q16, d_int)):
        err, lane = fpq16_mul_div_rounded_by_positive_int_checked(ai, bi, di)
        assert err == FP_Q16_OK
        out[i] = lane
    return FP_Q16_OK, out


def q16_to_float(x: int) -> float:
    return x / FP_Q16_ONE


def test_contract_errors_and_rounding_edges() -> None:
    err, out = fpq16_mul_div_array_rounded_by_positive_int_checked([1], [1], [0])
    assert err == FP_Q16_ERR_DOMAIN
    assert out is None

    err, out = fpq16_mul_div_array_rounded_by_positive_int_checked([1], [1], [-7])
    assert err == FP_Q16_ERR_DOMAIN
    assert out is None

    one = FP_Q16_ONE
    err, out = fpq16_mul_div_array_rounded_by_positive_int_checked([one], [one], [2])
    assert err == FP_Q16_OK
    assert out == [one // 2]

    err, out = fpq16_mul_div_array_rounded_by_positive_int_checked([one], [one], [3])
    assert err == FP_Q16_OK
    assert out == [((one * one + ((3 << FP_Q16_SHIFT) // 2)) // (3 << FP_Q16_SHIFT))]

    err, out = fpq16_mul_div_array_rounded_by_positive_int_checked(
        [I64_MAX_VALUE, I64_MAX_VALUE],
        [I64_MAX_VALUE, 1],
        [1, 1],
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out is None


def test_no_partial_write_behavior() -> None:
    one = FP_Q16_ONE
    out_seed = [111, 222, 333]

    err, out = simulate_holyc_array_call_no_partial_write(
        [one, one, I64_MAX_VALUE],
        [one, one, I64_MAX_VALUE],
        [1, 2, 1],
        out_seed,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == out_seed

    err, out = simulate_holyc_array_call_no_partial_write(
        [one, -one, 3 * one],
        [one, 2 * one, 5 * one],
        [1, 2, 3],
        out_seed,
    )
    assert err == FP_Q16_OK
    assert out != out_seed


def test_randomized_lane_parity() -> None:
    rng = random.Random(20260417_189)

    for _ in range(4000):
        n = rng.randint(1, 24)
        a = [rng.randint(-(1 << 30), 1 << 30) for _ in range(n)]
        b = [rng.randint(-(1 << 30), 1 << 30) for _ in range(n)]
        d = [rng.randint(1, 1 << 15) for _ in range(n)]

        err, out = fpq16_mul_div_array_rounded_by_positive_int_checked(a, b, d)
        if err != FP_Q16_OK:
            assert err == FP_Q16_ERR_OVERFLOW
            assert out is None
            continue

        assert out is not None
        for ai, bi, di, got in zip(a, b, d, out):
            want = (q16_to_float(ai) * q16_to_float(bi)) / float(di)
            assert abs(q16_to_float(got) - want) <= (1.5 / FP_Q16_ONE)


def run() -> None:
    test_contract_errors_and_rounding_edges()
    test_no_partial_write_behavior()
    test_randomized_lane_parity()
    print("fixedpoint_q16_muldiv_array_positive_int_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
