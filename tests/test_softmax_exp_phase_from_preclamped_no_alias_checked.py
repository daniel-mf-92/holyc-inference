#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxExpPhaseFromPreclampedNoAliasChecked semantics."""

from __future__ import annotations

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


def fpq16_exp_array_from_clamped_input_no_alias_checked_reference(
    input_q16: list[int],
    output_q16: list[int],
    count: int,
) -> tuple[int, list[int]]:
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, output_q16[:]

    if input_q16 is output_q16:
        return FP_Q16_ERR_BAD_PARAM, output_q16[:]

    if count == 0:
        return FP_Q16_OK, output_q16[:]

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, output_q16[:]

    last_byte_offset = last_index << 3
    if last_byte_offset > U64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, output_q16[:]

    # Fail fast before any writes.
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


def fpq16_softmax_exp_phase_from_preclamped_no_alias_checked_reference(
    preclamped_logits_q16: list[int],
    exp_lanes_q16: list[int],
    lane_count: int,
) -> tuple[int, list[int], int]:
    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, exp_lanes_q16[:], 0

    if preclamped_logits_q16 is exp_lanes_q16:
        return FP_Q16_ERR_BAD_PARAM, exp_lanes_q16[:], 0

    if lane_count == 0:
        return FP_Q16_OK, exp_lanes_q16[:], 0

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, exp_lanes_q16[:], 0

    last_byte_offset = last_index << 3
    if last_byte_offset > U64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, exp_lanes_q16[:], 0

    err, out = fpq16_exp_array_from_clamped_input_no_alias_checked_reference(
        preclamped_logits_q16,
        exp_lanes_q16,
        lane_count,
    )
    if err != FP_Q16_OK:
        return err, exp_lanes_q16[:], 0

    exp_sum_q16 = 0
    for lane in out[:lane_count]:
        if lane > I64_MAX_VALUE - exp_sum_q16:
            return FP_Q16_ERR_OVERFLOW, out, 0
        exp_sum_q16 += lane

    return FP_Q16_OK, out, exp_sum_q16


def q16_from_float(value: float) -> int:
    return int(round(value * FP_Q16_ONE))


def test_alias_rejected() -> None:
    shared = [q16_from_float(-0.2), q16_from_float(0.0), q16_from_float(0.3)]
    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_no_alias_checked_reference(
        shared,
        shared,
        len(shared),
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_lanes == shared
    assert exp_sum == 0


def test_negative_count_rejected() -> None:
    err, _, exp_sum = fpq16_softmax_exp_phase_from_preclamped_no_alias_checked_reference(
        [q16_from_float(0.0)],
        [0],
        -1,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_sum == 0


def test_zero_count_writes_nothing() -> None:
    logits = [q16_from_float(-1.0), q16_from_float(1.0)]
    out = [123, 456]
    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_no_alias_checked_reference(
        logits,
        out,
        0,
    )
    assert err == FP_Q16_OK
    assert exp_lanes == out
    assert exp_sum == 0


def test_domain_failure_has_no_partial_writes() -> None:
    logits = [q16_from_float(-0.5), EXP_Q16_MAX_INPUT + 1, q16_from_float(0.1)]
    out = [777, 888, 999]

    err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_no_alias_checked_reference(
        logits,
        out,
        len(logits),
    )

    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_lanes == out
    assert exp_sum == 0


def test_random_vectors_match_lane_exp_and_sum() -> None:
    rng = random.Random(2026041702)

    for _ in range(1000):
        count = rng.randint(1, 24)
        logits = [rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT) for _ in range(count)]
        out_seed = [rng.randint(0, 1024) for _ in range(count)]

        err, exp_lanes, exp_sum = fpq16_softmax_exp_phase_from_preclamped_no_alias_checked_reference(
            logits,
            out_seed,
            count,
        )

        assert err == FP_Q16_OK

        want = []
        for lane in logits:
            lane_err, lane_exp = fpq16_exp_from_clamped_input_checked(lane)
            assert lane_err == FP_Q16_OK
            want.append(lane_exp)

        assert exp_lanes[:count] == want
        assert exp_sum == sum(want)


def run() -> None:
    test_alias_rejected()
    test_negative_count_rejected()
    test_zero_count_writes_nothing()
    test_domain_failure_has_no_partial_writes()
    test_random_vectors_match_lane_exp_and_sum()
    print("softmax_exp_phase_from_preclamped_no_alias_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
