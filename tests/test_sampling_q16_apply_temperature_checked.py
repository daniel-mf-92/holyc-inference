#!/usr/bin/env python3
"""Reference checks for SamplingQ16ApplyTemperatureChecked semantics (IQ-745)."""

from __future__ import annotations

import random
from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4
SAMPLING_Q16_ERR_DOMAIN = 5

SAMPLING_Q16_SHIFT = 16
SAMPLING_Q16_ONE = 1 << SAMPLING_Q16_SHIFT

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)
U64_MAX = (1 << 64) - 1


def abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    if x == I64_MIN:
        return 1 << 63
    return -x


def apply_sign_checked(magnitude: int, is_negative: bool) -> tuple[int, int]:
    if not is_negative:
        if magnitude > I64_MAX:
            return SAMPLING_Q16_ERR_OVERFLOW, 0
        return SAMPLING_Q16_OK, magnitude

    if magnitude > (1 << 63):
        return SAMPLING_Q16_ERR_OVERFLOW, 0
    if magnitude == (1 << 63):
        return SAMPLING_Q16_OK, I64_MIN
    return SAMPLING_Q16_OK, -magnitude


def fpq16_mul_div_rounded_checked(a_q16: int, b_q16: int, d_q16: int) -> tuple[int, int]:
    if d_q16 == 0:
        return SAMPLING_Q16_ERR_DOMAIN, 0

    abs_a = abs_to_u64(a_q16)
    abs_b = abs_to_u64(b_q16)
    abs_d = abs_to_u64(d_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0) ^ (d_q16 < 0)

    if abs_a and abs_b and abs_a > (U64_MAX // abs_b):
        return SAMPLING_Q16_ERR_OVERFLOW, 0

    abs_num = abs_a * abs_b
    q = abs_num // abs_d
    r = abs_num % abs_d

    limit = I64_MAX
    if is_negative:
        limit = 1 << 63

    if q > limit:
        return SAMPLING_Q16_ERR_OVERFLOW, 0

    if r >= ((abs_d + 1) >> 1):
        if q == limit:
            return SAMPLING_Q16_ERR_OVERFLOW, 0
        q += 1

    return apply_sign_checked(q, is_negative)


def sampling_q16_apply_temperature_checked_reference(
    logits_q16,
    logits_capacity: int,
    lane_count: int,
    temperature_q16: int,
    out_logits_q16,
    out_capacity: int,
) -> int:
    if logits_q16 is None or out_logits_q16 is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if logits_capacity < 0 or out_capacity < 0 or lane_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if temperature_q16 <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if lane_count == 0:
        return SAMPLING_Q16_OK

    if lane_count > logits_capacity or lane_count > out_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    for i in range(lane_count):
        err, _ = fpq16_mul_div_rounded_checked(logits_q16[i], SAMPLING_Q16_ONE, temperature_q16)
        if err != SAMPLING_Q16_OK:
            return err

    for i in range(lane_count):
        err, lane = fpq16_mul_div_rounded_checked(logits_q16[i], SAMPLING_Q16_ONE, temperature_q16)
        if err != SAMPLING_Q16_OK:
            return err
        out_logits_q16[i] = lane

    return SAMPLING_Q16_OK


def test_source_contains_signature_and_reciprocal_scaling() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")

    signature = "I32 SamplingQ16ApplyTemperatureChecked(" 
    assert signature in source
    assert "temperature_q16 <= 0" in source
    assert "FPQ16MulDivRoundedChecked(logits_q16[lane_index]," in source
    assert "SAMPLING_Q16_ONE" in source


def test_null_and_bad_param_contracts() -> None:
    out = [0x7777]
    assert (
        sampling_q16_apply_temperature_checked_reference(None, 1, 1, SAMPLING_Q16_ONE, out, 1)
        == SAMPLING_Q16_ERR_NULL_PTR
    )
    assert (
        sampling_q16_apply_temperature_checked_reference([1], 1, 1, SAMPLING_Q16_ONE, None, 1)
        == SAMPLING_Q16_ERR_NULL_PTR
    )
    assert (
        sampling_q16_apply_temperature_checked_reference([1], -1, 1, SAMPLING_Q16_ONE, [0], 1)
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_q16_apply_temperature_checked_reference([1], 1, -1, SAMPLING_Q16_ONE, [0], 1)
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_q16_apply_temperature_checked_reference([1], 1, 1, 0, [0], 1)
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_q16_apply_temperature_checked_reference([1], 1, 1, -SAMPLING_Q16_ONE, [0], 1)
        == SAMPLING_Q16_ERR_BAD_PARAM
    )


def test_targeted_scaling_and_no_partial_behavior() -> None:
    logits = [2 * SAMPLING_Q16_ONE, -3 * SAMPLING_Q16_ONE, 5 * SAMPLING_Q16_ONE]
    out = [0x1111, 0x2222, 0x3333]

    err = sampling_q16_apply_temperature_checked_reference(
        logits,
        len(logits),
        len(logits),
        2 * SAMPLING_Q16_ONE,
        out,
        len(out),
    )
    assert err == SAMPLING_Q16_OK
    assert out == [SAMPLING_Q16_ONE, -(3 * SAMPLING_Q16_ONE) // 2, (5 * SAMPLING_Q16_ONE) // 2]

    huge_logits = [I64_MAX, I64_MAX]
    sentinel = [0xAAAA, 0xBBBB]
    err = sampling_q16_apply_temperature_checked_reference(
        huge_logits,
        2,
        2,
        1,
        sentinel,
        2,
    )
    assert err == SAMPLING_Q16_ERR_OVERFLOW
    assert sentinel == [0xAAAA, 0xBBBB]


def test_randomized_parity_against_explicit_lane_composition() -> None:
    rng = random.Random(20260420_745)

    for _ in range(5000):
        lane_count = rng.randint(0, 256)
        logits_capacity = lane_count + rng.randint(0, 8)
        out_capacity = lane_count + rng.randint(0, 8)
        temperature_q16 = rng.randint(1, 16 * SAMPLING_Q16_ONE)

        logits = [rng.randint(-(1 << 40), 1 << 40) for _ in range(logits_capacity)]
        out = [0x1234ABCD for _ in range(out_capacity)]

        err_a = sampling_q16_apply_temperature_checked_reference(
            logits,
            logits_capacity,
            lane_count,
            temperature_q16,
            out,
            out_capacity,
        )

        if err_a == SAMPLING_Q16_OK:
            for i in range(lane_count):
                err_lane, lane = fpq16_mul_div_rounded_checked(logits[i], SAMPLING_Q16_ONE, temperature_q16)
                assert err_lane == SAMPLING_Q16_OK
                assert out[i] == lane
        else:
            for i in range(min(lane_count, out_capacity)):
                assert out[i] == 0x1234ABCD


def test_capacity_bounds_rejected_before_write() -> None:
    logits = [10, 20, 30]
    out = [0xDEAD, 0xBEEF]

    err = sampling_q16_apply_temperature_checked_reference(
        logits,
        logits_capacity=3,
        lane_count=3,
        temperature_q16=SAMPLING_Q16_ONE,
        out_logits_q16=out,
        out_capacity=2,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out == [0xDEAD, 0xBEEF]


if __name__ == "__main__":
    test_source_contains_signature_and_reciprocal_scaling()
    test_null_and_bad_param_contracts()
    test_targeted_scaling_and_no_partial_behavior()
    test_randomized_parity_against_explicit_lane_composition()
    test_capacity_bounds_rejected_before_write()
    print("sampling_q16_apply_temperature_checked_reference_checks=ok")
