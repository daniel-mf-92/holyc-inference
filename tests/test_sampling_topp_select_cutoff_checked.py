#!/usr/bin/env python3
"""Reference checks for SamplingTopPSelectCutoffChecked semantics (IQ-747)."""

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

I64_MAX_VALUE = (1 << 63) - 1
U64_MAX_VALUE = (1 << 64) - 1


def sampling_topp_select_cutoff_checked_reference(
    probs_q16: list[int] | None,
    probs_capacity: int,
    lane_count: int,
    top_p_q16: int,
    out_cutoff_index: list[int] | None,
    probs_addr: int = 0,
) -> int:
    if probs_q16 is None or out_cutoff_index is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if probs_capacity < 0 or lane_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if lane_count == 0 or lane_count > probs_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if top_p_q16 <= 0 or top_p_q16 > SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if len(probs_q16) < lane_count or len(out_cutoff_index) < 1:
        return SAMPLING_Q16_ERR_BAD_PARAM

    last_index = lane_count - 1
    if last_index > 0x0FFFFFFFFFFFFFFF:
        return SAMPLING_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if probs_addr > (U64_MAX_VALUE - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW

    cumulative_q16 = 0
    prev_cumulative_q16 = 0
    prev_prob_q16 = SAMPLING_Q16_ONE
    cutoff_index = -1

    for lane_index in range(lane_count):
        lane_prob_q16 = probs_q16[lane_index]
        if lane_prob_q16 < 0 or lane_prob_q16 > SAMPLING_Q16_ONE:
            return SAMPLING_Q16_ERR_BAD_PARAM
        if lane_prob_q16 > prev_prob_q16:
            return SAMPLING_Q16_ERR_BAD_PARAM
        if lane_prob_q16 > (I64_MAX_VALUE - cumulative_q16):
            return SAMPLING_Q16_ERR_OVERFLOW

        cumulative_q16 += lane_prob_q16
        if cumulative_q16 < prev_cumulative_q16:
            return SAMPLING_Q16_ERR_OVERFLOW
        if cumulative_q16 > SAMPLING_Q16_ONE:
            return SAMPLING_Q16_ERR_BAD_PARAM

        if cutoff_index < 0 and cumulative_q16 >= top_p_q16:
            cutoff_index = lane_index

        prev_prob_q16 = lane_prob_q16
        prev_cumulative_q16 = cumulative_q16

    if cumulative_q16 != SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if cutoff_index < 0:
        cutoff_index = lane_count - 1

    out_cutoff_index[0] = cutoff_index
    return SAMPLING_Q16_OK


def test_source_contains_signature_and_checked_scan() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 SamplingTopPSelectCutoffChecked(" in source
    assert "if (lane_prob_q16 > prev_prob_q16)" in source
    assert "if (cutoff_index < 0 && cumulative_q16 >= top_p_q16)" in source
    assert "if (cumulative_q16 != SAMPLING_Q16_ONE)" in source


def test_null_and_bad_param_contracts() -> None:
    out = [777]

    assert (
        sampling_topp_select_cutoff_checked_reference(None, 3, 3, SAMPLING_Q16_ONE, out)
        == SAMPLING_Q16_ERR_NULL_PTR
    )
    assert (
        sampling_topp_select_cutoff_checked_reference([1, 2, 3], 3, 3, SAMPLING_Q16_ONE, None)
        == SAMPLING_Q16_ERR_NULL_PTR
    )
    assert (
        sampling_topp_select_cutoff_checked_reference([1, 2, 3], -1, 3, SAMPLING_Q16_ONE, out)
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_topp_select_cutoff_checked_reference([1, 2, 3], 3, 0, SAMPLING_Q16_ONE, out)
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_topp_select_cutoff_checked_reference([1, 2, 3], 2, 3, SAMPLING_Q16_ONE, out)
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_topp_select_cutoff_checked_reference([1, 2, 3], 3, 3, 0, out)
        == SAMPLING_Q16_ERR_BAD_PARAM
    )


def test_deterministic_first_crossing_behavior() -> None:
    probs = [30000, 20000, 10000, 5536]
    top_p = 50000
    out = [123]

    err = sampling_topp_select_cutoff_checked_reference(
        probs, len(probs), len(probs), top_p, out
    )
    assert err == SAMPLING_Q16_OK
    assert out[0] == 1

    top_p = SAMPLING_Q16_ONE
    out = [123]
    err = sampling_topp_select_cutoff_checked_reference(
        probs, len(probs), len(probs), top_p, out
    )
    assert err == SAMPLING_Q16_OK
    assert out[0] == len(probs) - 1


def test_monotonic_and_exact_mass_guards() -> None:
    out = [555]

    # Not descending.
    err = sampling_topp_select_cutoff_checked_reference(
        [20000, 21000, 24536],
        3,
        3,
        22000,
        out,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out == [555]

    # Mass > 1.0_q16.
    err = sampling_topp_select_cutoff_checked_reference(
        [40000, 25537],
        2,
        2,
        45000,
        out,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out == [555]

    # Mass < 1.0_q16.
    err = sampling_topp_select_cutoff_checked_reference(
        [30000, 30000, 5000],
        3,
        3,
        45000,
        out,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out == [555]


def test_pointer_overflow_and_no_partial_write() -> None:
    probs = [40000, 15536, 10000]
    out = [909]

    err = sampling_topp_select_cutoff_checked_reference(
        probs_q16=probs,
        probs_capacity=3,
        lane_count=3,
        top_p_q16=45000,
        out_cutoff_index=out,
        probs_addr=U64_MAX_VALUE - 15,
    )
    assert err == SAMPLING_Q16_ERR_OVERFLOW
    assert out == [909]


def test_randomized_first_crossing_parity() -> None:
    rng = random.Random(20260420_747)

    for _ in range(4000):
        lane_count = rng.randint(1, 64)
        probs_capacity = lane_count + rng.randint(0, 8)

        cuts = sorted([rng.randint(0, SAMPLING_Q16_ONE) for _ in range(lane_count - 1)])
        parts = []
        prev = 0
        for c in cuts:
            parts.append(c - prev)
            prev = c
        parts.append(SAMPLING_Q16_ONE - prev)
        probs = sorted(parts, reverse=True)

        top_p = rng.randint(1, SAMPLING_Q16_ONE)
        out = [-1]

        err = sampling_topp_select_cutoff_checked_reference(
            probs_q16=probs,
            probs_capacity=probs_capacity,
            lane_count=lane_count,
            top_p_q16=top_p,
            out_cutoff_index=out,
        )
        assert err == SAMPLING_Q16_OK

        cumulative = 0
        expected = lane_count - 1
        for i, p in enumerate(probs):
            cumulative += p
            if cumulative >= top_p:
                expected = i
                break
        assert out[0] == expected


if __name__ == "__main__":
    test_source_contains_signature_and_checked_scan()
    test_null_and_bad_param_contracts()
    test_deterministic_first_crossing_behavior()
    test_monotonic_and_exact_mass_guards()
    test_pointer_overflow_and_no_partial_write()
    test_randomized_first_crossing_parity()
    print("sampling_topp_select_cutoff_checked_reference_checks=ok")
