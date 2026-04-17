#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxExpPhaseFromPreclampedChecked semantics."""

from __future__ import annotations

import random
from typing import Optional

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


def fpq16_exp_array_from_clamped_input_checked_reference(
    input_q16: Optional[list[int]],
    output_q16: Optional[list[int]],
    count: int,
) -> tuple[int, Optional[list[int]]]:
    if input_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR, output_q16[:] if output_q16 is not None else None

    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, output_q16[:]

    if count == 0:
        return FP_Q16_OK, output_q16[:]

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, output_q16[:]

    last_byte_offset = last_index << 3
    if last_byte_offset > U64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, output_q16[:]

    for i in range(count):
        if input_q16[i] < EXP_Q16_MIN_INPUT or input_q16[i] > EXP_Q16_MAX_INPUT:
            return FP_Q16_ERR_BAD_PARAM, output_q16[:]

    out = output_q16[:]
    for i in range(count):
        err, value = fpq16_exp_from_clamped_input_checked(input_q16[i])
        if err != FP_Q16_OK:
            return err, output_q16[:]
        out[i] = value

    return FP_Q16_OK, out


def fpq16_softmax_exp_phase_from_preclamped_checked_reference(
    preclamped_logits_q16: Optional[list[int]],
    exp_lanes_q16: Optional[list[int]],
    lane_count: int,
    out_exp_sum_q16: Optional[int],
) -> tuple[int, Optional[list[int]], Optional[int]]:
    if preclamped_logits_q16 is None or exp_lanes_q16 is None or out_exp_sum_q16 is None:
        return (
            FP_Q16_ERR_NULL_PTR,
            exp_lanes_q16[:] if exp_lanes_q16 is not None else None,
            out_exp_sum_q16,
        )

    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, exp_lanes_q16[:], out_exp_sum_q16

    if lane_count == 0:
        return FP_Q16_OK, exp_lanes_q16[:], 0

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, exp_lanes_q16[:], out_exp_sum_q16

    last_byte_offset = last_index << 3
    if last_byte_offset > U64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, exp_lanes_q16[:], out_exp_sum_q16

    err, exp_lanes_q16 = fpq16_exp_array_from_clamped_input_checked_reference(
        preclamped_logits_q16,
        exp_lanes_q16,
        lane_count,
    )
    assert exp_lanes_q16 is not None
    if err != FP_Q16_OK:
        return err, exp_lanes_q16, out_exp_sum_q16

    exp_sum_q16 = 0
    for lane in exp_lanes_q16[:lane_count]:
        if lane > I64_MAX_VALUE - exp_sum_q16:
            return FP_Q16_ERR_OVERFLOW, exp_lanes_q16, out_exp_sum_q16
        exp_sum_q16 += lane

    return FP_Q16_OK, exp_lanes_q16, exp_sum_q16


def q16_from_float(value: float) -> int:
    return int(round(value * FP_Q16_ONE))


def test_empty_vector_returns_zero_sum() -> None:
    out_seed = [11, 22]
    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference([], out_seed, 0, 999)
    assert err == FP_Q16_OK
    assert exp_lanes == out_seed
    assert exp_sum == 0


def test_negative_count_is_bad_param() -> None:
    out_seed = [7, 8, 9]
    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference([1, 2, 3], out_seed, -1, 123)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_lanes == out_seed
    assert exp_sum == 123


def test_null_pointer_contracts_rejected_without_writes() -> None:
    logits = [q16_from_float(-0.25), q16_from_float(0.0), q16_from_float(0.1)]
    out_seed = [1001, 1002, 1003]

    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference(None, out_seed, len(logits), 777)
    assert err == FP_Q16_ERR_NULL_PTR
    assert exp_lanes == out_seed
    assert exp_sum == 777

    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference(logits, None, len(logits), 777)
    assert err == FP_Q16_ERR_NULL_PTR
    assert exp_lanes is None
    assert exp_sum == 777

    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference(logits, out_seed, len(logits), None)
    assert err == FP_Q16_ERR_NULL_PTR
    assert exp_lanes == out_seed
    assert exp_sum is None


def test_domain_violation_preserves_no_partial_writes() -> None:
    logits = [q16_from_float(-0.25), EXP_Q16_MAX_INPUT + 1, q16_from_float(0.10)]
    out_seed = [501, 502, 503]
    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference(
        logits,
        out_seed,
        len(logits),
        222,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_lanes == out_seed
    assert exp_sum == 222


def test_overflow_preflight_preserves_no_partial_writes() -> None:
    logits = [q16_from_float(0.0)]
    out_seed = [999]
    huge_count = (I64_MAX_VALUE >> 3) + 2

    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference(
        logits,
        out_seed,
        huge_count,
        444,
    )

    assert err == FP_Q16_ERR_OVERFLOW
    assert exp_lanes == out_seed
    assert exp_sum == 444


def test_known_vector_matches_scalar_lane_composition() -> None:
    logits = [q16_from_float(-1.5), q16_from_float(0.0), q16_from_float(0.75)]
    out_seed = [0, 0, 0]

    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference(
        logits,
        out_seed,
        len(logits),
        0,
    )
    assert err == FP_Q16_OK
    assert exp_lanes is not None
    assert exp_sum is not None

    want_exp_lanes: list[int] = []
    for lane in logits:
        lane_err, lane_exp = fpq16_exp_from_clamped_input_checked(lane)
        assert lane_err == FP_Q16_OK
        want_exp_lanes.append(lane_exp)

    assert exp_lanes[: len(logits)] == want_exp_lanes
    assert exp_sum == sum(want_exp_lanes)


def test_random_preclamped_vectors_match_scalar_lane_composition() -> None:
    rng = random.Random(2026041701)

    for _ in range(1200):
        count = rng.randint(1, 24)
        logits = [rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT) for _ in range(count)]
        out_seed = [rng.randint(-1000, 1000) for _ in range(count)]
        sum_seed = rng.randint(-2000, 2000)

        err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_checked_reference(
            logits,
            out_seed,
            count,
            sum_seed,
        )
        assert err == FP_Q16_OK
        assert exp_lanes is not None
        assert exp_sum is not None

        want_exp_lanes: list[int] = []
        for lane in logits:
            lane_err, lane_exp = fpq16_exp_from_clamped_input_checked(lane)
            assert lane_err == FP_Q16_OK
            want_exp_lanes.append(lane_exp)

        assert exp_lanes[:count] == want_exp_lanes
        assert exp_sum == sum(want_exp_lanes)


def run() -> None:
    test_empty_vector_returns_zero_sum()
    test_negative_count_is_bad_param()
    test_null_pointer_contracts_rejected_without_writes()
    test_domain_violation_preserves_no_partial_writes()
    test_overflow_preflight_preserves_no_partial_writes()
    test_known_vector_matches_scalar_lane_composition()
    test_random_preclamped_vectors_match_scalar_lane_composition()
    print("softmax_exp_phase_from_preclamped_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
