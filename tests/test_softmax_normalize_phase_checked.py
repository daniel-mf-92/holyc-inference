#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxNormalizePhaseChecked semantics."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
U64_MAX_VALUE = 0xFFFFFFFFFFFFFFFF


def fp_abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    if x == -0x8000000000000000:
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
        return FP_Q16_OK, -0x8000000000000000
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


def fpq16_softmax_normalize_phase_checked_reference(
    exp_lanes_q16: list[int], lane_count: int, exp_sum_q16: int
) -> tuple[int, list[int]]:
    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if exp_sum_q16 <= 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if lane_count == 0:
        return FP_Q16_OK, []

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, []

    last_byte_offset = last_index << 3
    if last_byte_offset > U64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, []

    probs_q16: list[int] = [0] * lane_count
    max_exp_q16 = exp_lanes_q16[0]
    max_exp_index = 0

    prob_sum_q16 = 0
    for i in range(lane_count):
        err, lane_prob = fpq16_mul_div_rounded_checked(exp_lanes_q16[i], FP_Q16_ONE, exp_sum_q16)
        if err != FP_Q16_OK:
            return err, []

        probs_q16[i] = lane_prob

        if lane_prob > I64_MAX_VALUE - prob_sum_q16:
            return FP_Q16_ERR_OVERFLOW, []
        prob_sum_q16 += lane_prob

        if exp_lanes_q16[i] > max_exp_q16:
            max_exp_q16 = exp_lanes_q16[i]
            max_exp_index = i

    diff_q16 = FP_Q16_ONE - prob_sum_q16
    if diff_q16 == 0:
        return FP_Q16_OK, probs_q16

    if diff_q16 > 0:
        if probs_q16[max_exp_index] > I64_MAX_VALUE - diff_q16:
            return FP_Q16_ERR_OVERFLOW, []
        probs_q16[max_exp_index] += diff_q16
        return FP_Q16_OK, probs_q16

    remaining_q16 = -diff_q16
    for pass_index in range(lane_count):
        if remaining_q16 <= 0:
            break

        idx = max_exp_index + pass_index
        if idx >= lane_count:
            idx -= lane_count

        take_q16 = probs_q16[idx]
        if take_q16 > remaining_q16:
            take_q16 = remaining_q16

        probs_q16[idx] -= take_q16
        remaining_q16 -= take_q16

    if remaining_q16 != 0:
        return FP_Q16_ERR_OVERFLOW, []

    return FP_Q16_OK, probs_q16


def q16_from_float(value: float) -> int:
    return int(round(value * FP_Q16_ONE))


def test_bad_param_paths() -> None:
    err, out = fpq16_softmax_normalize_phase_checked_reference([], -1, FP_Q16_ONE)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == []

    err, out = fpq16_softmax_normalize_phase_checked_reference([1], 1, 0)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == []


def test_zero_lane_count_ok() -> None:
    err, out = fpq16_softmax_normalize_phase_checked_reference([], 0, 123)
    assert err == FP_Q16_OK
    assert out == []


def test_exact_sum_no_remainder() -> None:
    exp_lanes = [FP_Q16_ONE, FP_Q16_ONE]
    err, probs = fpq16_softmax_normalize_phase_checked_reference(exp_lanes, len(exp_lanes), 2 * FP_Q16_ONE)
    assert err == FP_Q16_OK
    assert probs == [FP_Q16_ONE // 2, FP_Q16_ONE // 2]
    assert sum(probs) == FP_Q16_ONE


def test_positive_remainder_goes_to_argmax_lane() -> None:
    exp_lanes = [3, 1]
    exp_sum = 5

    err, probs = fpq16_softmax_normalize_phase_checked_reference(exp_lanes, len(exp_lanes), exp_sum)
    assert err == FP_Q16_OK

    # Rounded lanes are [39322, 13107], diff=13107, should land on argmax lane 0.
    assert probs[0] == 52429
    assert probs[1] == 13107
    assert sum(probs) == FP_Q16_ONE


def test_negative_remainder_subtracts_deterministically() -> None:
    exp_lanes = [1, 1, 1]
    exp_sum = 2

    err, probs = fpq16_softmax_normalize_phase_checked_reference(exp_lanes, len(exp_lanes), exp_sum)
    assert err == FP_Q16_OK

    # Per-lane rounded probs start at 32768 each (sum 98304), so 32768 must be
    # removed starting at lane 0 (argmax tie-first).
    assert probs == [0, 32768, 32768]
    assert sum(probs) == FP_Q16_ONE


def test_random_positive_lanes_produce_normalized_distribution() -> None:
    rng = random.Random(20260417)

    for _ in range(2000):
        lane_count = rng.randint(1, 32)
        exp_lanes = [rng.randint(1, 1_000_000) for _ in range(lane_count)]
        exp_sum = sum(exp_lanes)

        err, probs = fpq16_softmax_normalize_phase_checked_reference(exp_lanes, lane_count, exp_sum)
        assert err == FP_Q16_OK

        assert len(probs) == lane_count
        assert all(p >= 0 for p in probs)
        assert sum(probs) == FP_Q16_ONE


def test_large_denominator_still_conserves_q16_mass() -> None:
    exp_lanes = [q16_from_float(0.001), q16_from_float(0.005), q16_from_float(0.250), q16_from_float(0.744)]
    exp_sum = 9_999_999_937

    err, probs = fpq16_softmax_normalize_phase_checked_reference(exp_lanes, len(exp_lanes), exp_sum)
    assert err == FP_Q16_OK
    assert all(p >= 0 for p in probs)
    assert sum(probs) == FP_Q16_ONE


def run() -> None:
    test_bad_param_paths()
    test_zero_lane_count_ok()
    test_exact_sum_no_remainder()
    test_positive_remainder_goes_to_argmax_lane()
    test_negative_remainder_subtracts_deterministically()
    test_random_positive_lanes_produce_normalized_distribution()
    test_large_denominator_still_conserves_q16_mass()
    print("softmax_normalize_phase_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
