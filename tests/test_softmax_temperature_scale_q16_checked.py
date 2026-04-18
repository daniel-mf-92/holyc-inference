#!/usr/bin/env python3
"""Reference checks for FPQ16TemperatureScaleLogitsChecked semantics."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
I64_MIN_VALUE = -(1 << 63)
U64_MAX_VALUE = 0xFFFFFFFFFFFFFFFF


def fp_abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    if x == I64_MIN_VALUE:
        return 0x8000000000000000
    return -x


def fp_try_apply_sign_from_u64_checked(value_mag: int, is_negative: bool) -> tuple[int, int]:
    if not is_negative:
        if value_mag > I64_MAX_VALUE:
            return FP_Q16_ERR_OVERFLOW, 0
        return FP_Q16_OK, value_mag

    if value_mag > 0x8000000000000000:
        return FP_Q16_ERR_OVERFLOW, 0
    if value_mag == 0x8000000000000000:
        return FP_Q16_OK, I64_MIN_VALUE
    return FP_Q16_OK, -value_mag


def fpq16_mul_div_rounded_checked(a_q16: int, b_q16: int, d_q16: int) -> tuple[int, int]:
    if d_q16 == 0:
        return FP_Q16_ERR_BAD_PARAM, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    abs_d = fp_abs_to_u64(d_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0) ^ (d_q16 < 0)

    if abs_a and abs_b and abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_num = abs_a * abs_b
    q = abs_num // abs_d
    r = abs_num % abs_d

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 0x8000000000000000

    if q > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if r >= ((abs_d + 1) >> 1):
        if q == limit:
            return FP_Q16_ERR_OVERFLOW, 0
        q += 1

    return fp_try_apply_sign_from_u64_checked(q, is_negative)


def fpq16_temperature_scale_logits_checked_reference(
    logits_q16: list[int],
    lane_count: int,
    temperature_q16: int,
    scaled_seed: list[int],
) -> tuple[int, list[int]]:
    scaled_logits_q16 = scaled_seed[:]

    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, scaled_logits_q16
    if temperature_q16 <= 0:
        return FP_Q16_ERR_BAD_PARAM, scaled_logits_q16
    if lane_count == 0:
        return FP_Q16_OK, scaled_logits_q16

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, scaled_logits_q16

    last_byte_offset = last_index << 3
    if last_byte_offset > U64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, scaled_logits_q16

    for i in range(lane_count):
        err, _ = fpq16_mul_div_rounded_checked(logits_q16[i], FP_Q16_ONE, temperature_q16)
        if err != FP_Q16_OK:
            return err, scaled_logits_q16

    for i in range(lane_count):
        err, lane = fpq16_mul_div_rounded_checked(logits_q16[i], FP_Q16_ONE, temperature_q16)
        assert err == FP_Q16_OK
        scaled_logits_q16[i] = lane

    return FP_Q16_OK, scaled_logits_q16


def q16_from_ratio(num: int, den: int) -> int:
    return (num * FP_Q16_ONE) // den


def test_bad_parameter_contracts() -> None:
    seed = [7, 8, 9]
    assert fpq16_temperature_scale_logits_checked_reference([1, 2, 3], -1, FP_Q16_ONE, seed)[0] == FP_Q16_ERR_BAD_PARAM
    assert fpq16_temperature_scale_logits_checked_reference([1, 2, 3], 3, 0, seed)[0] == FP_Q16_ERR_BAD_PARAM
    assert fpq16_temperature_scale_logits_checked_reference([1, 2, 3], 3, -FP_Q16_ONE, seed)[0] == FP_Q16_ERR_BAD_PARAM


def test_zero_lane_count_is_noop() -> None:
    seed = [101, 202]
    err, out = fpq16_temperature_scale_logits_checked_reference([5, 6], 0, FP_Q16_ONE, seed)
    assert err == FP_Q16_OK
    assert out == seed


def test_exact_identity_and_double_temperature() -> None:
    logits = [5 * FP_Q16_ONE, -2 * FP_Q16_ONE, 3 * FP_Q16_ONE]
    seed = [11, 22, 33]

    err, out = fpq16_temperature_scale_logits_checked_reference(logits, len(logits), FP_Q16_ONE, seed)
    assert err == FP_Q16_OK
    assert out == logits

    err, out = fpq16_temperature_scale_logits_checked_reference(logits, len(logits), 2 * FP_Q16_ONE, seed)
    assert err == FP_Q16_OK
    assert out == [
        int(round(5 * FP_Q16_ONE / 2)),
        int(round(-2 * FP_Q16_ONE / 2)),
        int(round(3 * FP_Q16_ONE / 2)),
    ]


def test_fractional_temperature_deterministic_rounding() -> None:
    # T = 0.5 -> scale by 2x.
    t_half = q16_from_ratio(1, 2)
    logits = [
        int(0.75 * FP_Q16_ONE),
        int(-1.25 * FP_Q16_ONE),
        int(2.00 * FP_Q16_ONE),
    ]
    seed = [900, 901, 902]

    err, out = fpq16_temperature_scale_logits_checked_reference(logits, len(logits), t_half, seed)
    assert err == FP_Q16_OK
    assert out[0] == int(round(1.50 * FP_Q16_ONE))
    assert out[1] == int(round(-2.50 * FP_Q16_ONE))
    assert out[2] == int(round(4.00 * FP_Q16_ONE))


def test_overflow_returns_error_without_partial_write() -> None:
    temp_tiny = 1
    logits = [I64_MAX_VALUE, 1]
    seed = [12345, 67890]

    err, out = fpq16_temperature_scale_logits_checked_reference(logits, len(logits), temp_tiny, seed)
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == seed


def test_random_vectors_match_lane_division_reference() -> None:
    rng = random.Random(20260418)

    for _ in range(4000):
        lane_count = rng.randint(1, 32)
        logits = [rng.randint(-(1 << 50), 1 << 50) for _ in range(lane_count)]
        temperature_q16 = rng.randint(1, 16 * FP_Q16_ONE)
        seed = [rng.randint(-(1 << 20), 1 << 20) for _ in range(lane_count)]

        err, scaled = fpq16_temperature_scale_logits_checked_reference(logits, lane_count, temperature_q16, seed)

        if err == FP_Q16_ERR_OVERFLOW:
            assert scaled == seed
            continue

        assert err == FP_Q16_OK
        for i in range(lane_count):
            lane_err, lane_out = fpq16_mul_div_rounded_checked(logits[i], FP_Q16_ONE, temperature_q16)
            assert lane_err == FP_Q16_OK
            assert scaled[i] == lane_out


def run() -> None:
    test_bad_parameter_contracts()
    test_zero_lane_count_is_noop()
    test_exact_identity_and_double_temperature()
    test_fractional_temperature_deterministic_rounding()
    test_overflow_returns_error_without_partial_write()
    test_random_vectors_match_lane_division_reference()
    print("softmax_temperature_scale_q16_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
