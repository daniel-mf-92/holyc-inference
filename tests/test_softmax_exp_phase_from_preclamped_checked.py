#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxExpPhaseFromPreclampedChecked semantics."""

from __future__ import annotations

import math
import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

EXP_Q16_LN2 = 45426
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360

I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
U64_MAX_VALUE = 0xFFFFFFFFFFFFFFFF


def round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    half = 1 << (shift - 1)
    return (value + half) >> shift


def round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return round_shift_right_unsigned(value, shift)
    return -round_shift_right_unsigned(-value, shift)


def fpq16_mul(a: int, b: int) -> int:
    return round_shift_right_signed(a * b, FP_Q16_SHIFT)


def fpq16_exp_from_clamped_input_checked(clamped_input_q16: int) -> tuple[int, int]:
    if clamped_input_q16 < EXP_Q16_MIN_INPUT or clamped_input_q16 > EXP_Q16_MAX_INPUT:
        return FP_Q16_ERR_BAD_PARAM, 0

    if clamped_input_q16 >= EXP_Q16_MAX_INPUT:
        return FP_Q16_OK, I64_MAX_VALUE
    if clamped_input_q16 <= EXP_Q16_MIN_INPUT:
        return FP_Q16_OK, 0

    k = clamped_input_q16 // EXP_Q16_LN2
    if clamped_input_q16 < 0 and (clamped_input_q16 % EXP_Q16_LN2):
        k -= 1

    r = clamped_input_q16 - (k * EXP_Q16_LN2)

    r2 = fpq16_mul(r, r)
    r3 = fpq16_mul(r2, r)
    r4 = fpq16_mul(r3, r)

    poly = FP_Q16_ONE + r + (r2 // 2) + (r3 // 6) + (r4 // 24)

    if k >= 0:
        if k >= 30:
            return FP_Q16_OK, I64_MAX_VALUE
        if poly > (I64_MAX_VALUE >> k):
            return FP_Q16_OK, I64_MAX_VALUE
        return FP_Q16_OK, poly << k

    if k <= -63:
        return FP_Q16_OK, 0

    return FP_Q16_OK, poly >> (-k)


def fpq16_exp_array_from_clamped_input_checked_reference(input_q16: list[int], count: int) -> tuple[int, list[int]]:
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if count == 0:
        return FP_Q16_OK, []

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, []

    last_byte_offset = last_index << 3
    if last_byte_offset > U64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, []

    for i in range(count):
        if input_q16[i] < EXP_Q16_MIN_INPUT or input_q16[i] > EXP_Q16_MAX_INPUT:
            return FP_Q16_ERR_BAD_PARAM, []

    output_q16: list[int] = []
    for i in range(count):
        err, value = fpq16_exp_from_clamped_input_checked(input_q16[i])
        if err != FP_Q16_OK:
            return err, []
        output_q16.append(value)

    return FP_Q16_OK, output_q16


def fpq16_softmax_exp_phase_from_preclamped_checked_reference(
    preclamped_logits_q16: list[int],
    lane_count: int,
) -> tuple[int, list[int], int]:
    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, [], 0
    if lane_count == 0:
        return FP_Q16_OK, [], 0

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, [], 0

    last_byte_offset = last_index << 3
    if last_byte_offset > U64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, [], 0

    err, exp_lanes_q16 = fpq16_exp_array_from_clamped_input_checked_reference(
        preclamped_logits_q16,
        lane_count,
    )
    if err != FP_Q16_OK:
        return err, [], 0

    exp_sum_q16 = 0
    for lane in exp_lanes_q16:
        if lane > I64_MAX_VALUE - exp_sum_q16:
            return FP_Q16_ERR_OVERFLOW, exp_lanes_q16, 0
        exp_sum_q16 += lane

    return FP_Q16_OK, exp_lanes_q16, exp_sum_q16


def q16_from_float(value: float) -> int:
    return int(round(value * FP_Q16_ONE))


def q16_to_float(value: int) -> float:
    return value / FP_Q16_ONE


def test_empty_vector_returns_zero_sum() -> None:
    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference([], 0)
    assert err == FP_Q16_OK
    assert exp_lanes == []
    assert exp_sum == 0


def test_negative_count_is_bad_param() -> None:
    err, _, _ = fpq16_softmax_exp_phase_from_preclamped_checked_reference([], -1)
    assert err == FP_Q16_ERR_BAD_PARAM


def test_domain_violation_preserves_no_partial_writes() -> None:
    logits = [q16_from_float(-0.25), EXP_Q16_MAX_INPUT + 1, q16_from_float(0.10)]
    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference(logits, len(logits))
    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_lanes == []
    assert exp_sum == 0


def test_known_vector_matches_float_exp_sum() -> None:
    logits = [q16_from_float(-1.5), q16_from_float(0.0), q16_from_float(0.75)]

    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference(logits, len(logits))
    assert err == FP_Q16_OK

    want_exp_lanes: list[int] = []
    for lane in logits:
        lane_err, lane_exp = fpq16_exp_from_clamped_input_checked(lane)
        assert lane_err == FP_Q16_OK
        want_exp_lanes.append(lane_exp)

    assert exp_lanes == want_exp_lanes
    assert exp_sum == sum(want_exp_lanes)

    got_sum = q16_to_float(exp_sum)
    want_sum = sum(math.exp(q16_to_float(x)) for x in logits)
    assert abs(got_sum - want_sum) <= 0.08

    assert len(exp_lanes) == len(logits)
    for lane_in, lane_out in zip(logits, exp_lanes):
        want_lane = math.exp(q16_to_float(lane_in))
        got_lane = q16_to_float(lane_out)
        assert abs(got_lane - want_lane) <= 0.08


def test_random_preclamped_vectors_track_float_reference() -> None:
    rng = random.Random(20260417)

    for _ in range(1200):
        count = rng.randint(1, 24)
        logits = [rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT) for _ in range(count)]

        err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference(logits, count)
        assert err == FP_Q16_OK

        want_exp_lanes: list[int] = []
        for lane in logits:
            lane_err, lane_exp = fpq16_exp_from_clamped_input_checked(lane)
            assert lane_err == FP_Q16_OK
            want_exp_lanes.append(lane_exp)

        assert exp_lanes == want_exp_lanes
        assert exp_sum == sum(want_exp_lanes)

        got_sum = q16_to_float(exp_sum)
        want_sum = sum(math.exp(q16_to_float(x)) for x in logits)
        assert abs(got_sum - want_sum) <= 2500.0

        for lane_in, lane_out in zip(logits, exp_lanes):
            want_lane = math.exp(q16_to_float(lane_in))
            got_lane = q16_to_float(lane_out)
            assert abs(got_lane - want_lane) <= 100.0


def run() -> None:
    test_empty_vector_returns_zero_sum()
    test_negative_count_is_bad_param()
    test_domain_violation_preserves_no_partial_writes()
    test_known_vector_matches_float_exp_sum()
    test_random_preclamped_vectors_track_float_reference()
    print("softmax_exp_phase_from_preclamped_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
