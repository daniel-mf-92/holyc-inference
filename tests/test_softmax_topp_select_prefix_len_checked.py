#!/usr/bin/env python3
"""Reference checks for FPQ16TopPSelectPrefixLenChecked semantics."""

from __future__ import annotations

import random

FP_Q16_ONE = 1 << 16

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
U64_MAX_VALUE = 0xFFFFFFFFFFFFFFFF


def fpq16_topp_select_prefix_len_checked_reference(
    probs_q16: list[int] | None,
    lane_count: int,
    top_p_q16: int,
    out_prefix_len: list[int] | None,
    probs_addr: int = 0,
) -> int:
    if probs_q16 is None or out_prefix_len is None:
        return FP_Q16_ERR_NULL_PTR
    if lane_count <= 0:
        return FP_Q16_ERR_BAD_PARAM
    if top_p_q16 <= 0 or top_p_q16 > FP_Q16_ONE:
        return FP_Q16_ERR_BAD_PARAM

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if probs_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if len(probs_q16) < lane_count:
        return FP_Q16_ERR_BAD_PARAM

    cumulative_q16 = 0
    prev_cumulative_q16 = 0
    prefix_len = -1
    prev_prob_q16 = FP_Q16_ONE

    for idx in range(lane_count):
        p = probs_q16[idx]
        if p < 0 or p > FP_Q16_ONE:
            return FP_Q16_ERR_BAD_PARAM
        if p > prev_prob_q16:
            return FP_Q16_ERR_BAD_PARAM
        if p > (I64_MAX_VALUE - cumulative_q16):
            return FP_Q16_ERR_OVERFLOW

        cumulative_q16 += p
        if cumulative_q16 < prev_cumulative_q16:
            return FP_Q16_ERR_OVERFLOW
        if cumulative_q16 > FP_Q16_ONE:
            return FP_Q16_ERR_BAD_PARAM

        if prefix_len < 0 and cumulative_q16 >= top_p_q16:
            prefix_len = idx + 1

        prev_prob_q16 = p
        prev_cumulative_q16 = cumulative_q16

    if cumulative_q16 != FP_Q16_ONE:
        return FP_Q16_ERR_BAD_PARAM

    if prefix_len < 0:
        prefix_len = lane_count

    out_prefix_len[0] = prefix_len
    return FP_Q16_OK


def test_null_ptr_paths() -> None:
    out = [99]

    err = fpq16_topp_select_prefix_len_checked_reference(None, 3, FP_Q16_ONE // 2, out)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == [99]

    err = fpq16_topp_select_prefix_len_checked_reference([30000, 20000, 15536], 3, FP_Q16_ONE // 2, None)
    assert err == FP_Q16_ERR_NULL_PTR


def test_bad_param_paths_no_partial_write() -> None:
    probs = [30000, 20000, 15536]
    out = [777]

    err = fpq16_topp_select_prefix_len_checked_reference(probs, 0, FP_Q16_ONE // 2, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [777]

    err = fpq16_topp_select_prefix_len_checked_reference(probs, -1, FP_Q16_ONE // 2, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [777]

    err = fpq16_topp_select_prefix_len_checked_reference(probs, 3, 0, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [777]

    err = fpq16_topp_select_prefix_len_checked_reference(probs, 3, FP_Q16_ONE + 1, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [777]


def test_domain_and_descending_guards() -> None:
    out = [42]

    err = fpq16_topp_select_prefix_len_checked_reference([FP_Q16_ONE + 1, 0], 2, FP_Q16_ONE // 2, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [42]

    err = fpq16_topp_select_prefix_len_checked_reference([40000, -1, 25537], 3, FP_Q16_ONE // 2, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [42]

    err = fpq16_topp_select_prefix_len_checked_reference([30000, 31000, 4536], 3, FP_Q16_ONE // 2, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [42]


def test_total_mass_guard() -> None:
    out = [17]

    err = fpq16_topp_select_prefix_len_checked_reference([32000, 20000, 12000], 3, FP_Q16_ONE // 2, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [17]


def test_pointer_span_overflow_guard_no_partial_write() -> None:
    probs = [32768, 16384, 16384]
    out = [5]

    err = fpq16_topp_select_prefix_len_checked_reference(
        probs,
        lane_count=3,
        top_p_q16=FP_Q16_ONE // 2,
        out_prefix_len=out,
        probs_addr=U64_MAX_VALUE - 15,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [5]


def test_threshold_edges() -> None:
    probs = [32768, 16384, 16384]

    out = [-1]
    err = fpq16_topp_select_prefix_len_checked_reference(probs, 3, 1, out)
    assert err == FP_Q16_OK
    assert out[0] == 1

    out = [-1]
    err = fpq16_topp_select_prefix_len_checked_reference(probs, 3, 32768, out)
    assert err == FP_Q16_OK
    assert out[0] == 1

    out = [-1]
    err = fpq16_topp_select_prefix_len_checked_reference(probs, 3, 32769, out)
    assert err == FP_Q16_OK
    assert out[0] == 2

    out = [-1]
    err = fpq16_topp_select_prefix_len_checked_reference(probs, 3, FP_Q16_ONE, out)
    assert err == FP_Q16_OK
    assert out[0] == 3


def test_randomized_matches_stable_cumulative_reference() -> None:
    rng = random.Random(332)

    for _ in range(2000):
        lane_count = rng.randint(1, 64)

        cuts = sorted(rng.sample(range(0, FP_Q16_ONE + 1), lane_count - 1))
        points = [0] + cuts + [FP_Q16_ONE]
        probs = [points[i + 1] - points[i] for i in range(lane_count)]
        probs.sort(reverse=True)

        top_p_q16 = rng.randint(1, FP_Q16_ONE)

        out = [-1]
        err = fpq16_topp_select_prefix_len_checked_reference(
            probs,
            lane_count=lane_count,
            top_p_q16=top_p_q16,
            out_prefix_len=out,
        )
        assert err == FP_Q16_OK

        cumulative = 0
        expected = lane_count
        for idx, p in enumerate(probs):
            cumulative += p
            if cumulative >= top_p_q16:
                expected = idx + 1
                break

        assert out[0] == expected


if __name__ == "__main__":
    test_null_ptr_paths()
    test_bad_param_paths_no_partial_write()
    test_domain_and_descending_guards()
    test_total_mass_guard()
    test_pointer_span_overflow_guard_no_partial_write()
    test_threshold_edges()
    test_randomized_matches_stable_cumulative_reference()
    print("softmax_topp_select_prefix_len_checked_reference_checks=ok")
