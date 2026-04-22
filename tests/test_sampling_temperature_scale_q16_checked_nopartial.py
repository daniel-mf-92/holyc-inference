#!/usr/bin/env python3
"""Reference checks for SamplingTemperatureScaleQ16CheckedNoPartial semantics (IQ-1167)."""

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


def sampling_temperature_scale_q16_checked_nopartial_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    lane_count: int,
    temperature_q16: int,
    out_logits_q16: list[int] | None,
    out_capacity: int,
    logits_addr: int = 0,
    out_addr: int = 0,
) -> int:
    if logits_q16 is None or out_logits_q16 is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if logits_capacity < 0 or out_capacity < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if lane_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if temperature_q16 <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if lane_count == 0:
        return SAMPLING_Q16_OK

    if lane_count > logits_capacity or lane_count > out_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if len(logits_q16) < lane_count or len(out_logits_q16) < lane_count:
        return SAMPLING_Q16_ERR_BAD_PARAM

    last_index = lane_count - 1
    if last_index > 0x0FFFFFFFFFFFFFFF:
        return SAMPLING_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if logits_addr > (U64_MAX - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW
    if out_addr > (U64_MAX - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW

    err, reciprocal_q16 = fpq16_mul_div_rounded_checked(
        SAMPLING_Q16_ONE,
        SAMPLING_Q16_ONE,
        temperature_q16,
    )
    if err != SAMPLING_Q16_OK:
        return err

    staged = [0] * lane_count
    for i in range(lane_count):
        err, scaled = fpq16_mul_div_rounded_checked(
            logits_q16[i],
            reciprocal_q16,
            SAMPLING_Q16_ONE,
        )
        if err != SAMPLING_Q16_OK:
            return err
        staged[i] = scaled

    for i in range(lane_count):
        out_logits_q16[i] = staged[i]

    return SAMPLING_Q16_OK


def test_source_contains_iq1167_signature_and_reciprocal_math() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 SamplingTemperatureScaleQ16CheckedNoPartial(" in source
    assert "FPQ16MulDivRoundedChecked(SAMPLING_Q16_ONE," in source
    assert "reciprocal_q16" in source
    assert "staged_out_q16 = MAlloc(staged_bytes);" in source


def test_null_and_bad_param_contracts() -> None:
    logits = [SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE]
    out = [111, 222]

    assert (
        sampling_temperature_scale_q16_checked_nopartial_reference(
            None,
            logits_capacity=2,
            lane_count=2,
            temperature_q16=SAMPLING_Q16_ONE,
            out_logits_q16=out,
            out_capacity=2,
        )
        == SAMPLING_Q16_ERR_NULL_PTR
    )
    assert (
        sampling_temperature_scale_q16_checked_nopartial_reference(
            logits,
            logits_capacity=2,
            lane_count=2,
            temperature_q16=SAMPLING_Q16_ONE,
            out_logits_q16=None,
            out_capacity=2,
        )
        == SAMPLING_Q16_ERR_NULL_PTR
    )
    assert (
        sampling_temperature_scale_q16_checked_nopartial_reference(
            logits,
            logits_capacity=-1,
            lane_count=2,
            temperature_q16=SAMPLING_Q16_ONE,
            out_logits_q16=out,
            out_capacity=2,
        )
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_temperature_scale_q16_checked_nopartial_reference(
            logits,
            logits_capacity=2,
            lane_count=-1,
            temperature_q16=SAMPLING_Q16_ONE,
            out_logits_q16=out,
            out_capacity=2,
        )
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_temperature_scale_q16_checked_nopartial_reference(
            logits,
            logits_capacity=2,
            lane_count=2,
            temperature_q16=0,
            out_logits_q16=out,
            out_capacity=2,
        )
        == SAMPLING_Q16_ERR_BAD_PARAM
    )


def test_expected_scaled_values_with_reciprocal_precompute() -> None:
    logits = [
        4 * SAMPLING_Q16_ONE,
        -(3 * SAMPLING_Q16_ONE),
        SAMPLING_Q16_ONE // 2,
        -(SAMPLING_Q16_ONE // 3),
    ]
    out = [0, 0, 0, 0]

    temperature_q16 = int(1.5 * SAMPLING_Q16_ONE)
    err = sampling_temperature_scale_q16_checked_nopartial_reference(
        logits,
        logits_capacity=4,
        lane_count=4,
        temperature_q16=temperature_q16,
        out_logits_q16=out,
        out_capacity=4,
    )
    assert err == SAMPLING_Q16_OK

    err, reciprocal_q16 = fpq16_mul_div_rounded_checked(
        SAMPLING_Q16_ONE,
        SAMPLING_Q16_ONE,
        temperature_q16,
    )
    assert err == SAMPLING_Q16_OK
    assert reciprocal_q16 > 0

    for i, base in enumerate(logits):
        err, expected = fpq16_mul_div_rounded_checked(base, reciprocal_q16, SAMPLING_Q16_ONE)
        assert err == SAMPLING_Q16_OK
        assert out[i] == expected


def test_no_partial_write_on_lane_overflow() -> None:
    logits = [SAMPLING_Q16_ONE, I64_MAX, -(5 * SAMPLING_Q16_ONE)]
    out = [17, 19, 23]
    before = list(out)

    # Tiny positive temperature => huge reciprocal; lane[1] multiply overflows.
    err = sampling_temperature_scale_q16_checked_nopartial_reference(
        logits,
        logits_capacity=3,
        lane_count=3,
        temperature_q16=1,
        out_logits_q16=out,
        out_capacity=3,
    )
    assert err == SAMPLING_Q16_ERR_OVERFLOW
    assert out == before


def test_pointer_span_overflow_guard() -> None:
    logits = [SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE]
    out = [0, 0]

    # lane_count=2 => last_byte_offset=8, so addr > U64_MAX-8 overflows.
    err = sampling_temperature_scale_q16_checked_nopartial_reference(
        logits,
        logits_capacity=2,
        lane_count=2,
        temperature_q16=SAMPLING_Q16_ONE,
        out_logits_q16=out,
        out_capacity=2,
        logits_addr=U64_MAX - 7,
    )
    assert err == SAMPLING_Q16_ERR_OVERFLOW


def test_randomized_reciprocal_temperature_vectors() -> None:
    rng = random.Random(1167)

    for _ in range(200):
        lane_count = rng.randint(1, 32)
        logits = [rng.randint(-8 * SAMPLING_Q16_ONE, 8 * SAMPLING_Q16_ONE) for _ in range(lane_count)]
        out = [0] * lane_count
        temperature_q16 = rng.randint(SAMPLING_Q16_ONE // 8, 8 * SAMPLING_Q16_ONE)

        err = sampling_temperature_scale_q16_checked_nopartial_reference(
            logits,
            logits_capacity=lane_count,
            lane_count=lane_count,
            temperature_q16=temperature_q16,
            out_logits_q16=out,
            out_capacity=lane_count,
        )
        assert err == SAMPLING_Q16_OK

        err, reciprocal_q16 = fpq16_mul_div_rounded_checked(
            SAMPLING_Q16_ONE,
            SAMPLING_Q16_ONE,
            temperature_q16,
        )
        assert err == SAMPLING_Q16_OK

        for i in range(lane_count):
            err, expected = fpq16_mul_div_rounded_checked(logits[i], reciprocal_q16, SAMPLING_Q16_ONE)
            assert err == SAMPLING_Q16_OK
            assert out[i] == expected


if __name__ == "__main__":
    test_source_contains_iq1167_signature_and_reciprocal_math()
    test_null_and_bad_param_contracts()
    test_expected_scaled_values_with_reciprocal_precompute()
    test_no_partial_write_on_lane_overflow()
    test_pointer_span_overflow_guard()
    test_randomized_reciprocal_temperature_vectors()
    print("ok")
