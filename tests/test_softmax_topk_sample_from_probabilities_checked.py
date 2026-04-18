#!/usr/bin/env python3
"""Reference checks for FPQ16TopKSampleFromProbabilitiesChecked semantics."""

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


def fpq16_topk_sample_from_probabilities_checked_reference(
    probs_q16: list[int] | None,
    lane_count: int,
    k: int,
    random_q16: int,
    out_index: list[int] | None,
    probs_addr: int = 0,
) -> int:
    if probs_q16 is None or out_index is None:
        return FP_Q16_ERR_NULL_PTR
    if lane_count < 0 or k < 0:
        return FP_Q16_ERR_BAD_PARAM
    if k == 0 or k > lane_count:
        return FP_Q16_ERR_BAD_PARAM
    if random_q16 < 0 or random_q16 >= FP_Q16_ONE:
        return FP_Q16_ERR_BAD_PARAM

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if probs_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if len(probs_q16) < lane_count:
        return FP_Q16_ERR_BAD_PARAM

    lane_sum_q16 = 0
    for idx in range(lane_count):
        p = probs_q16[idx]
        if p < 0 or p > FP_Q16_ONE:
            return FP_Q16_ERR_BAD_PARAM
        if p > (I64_MAX_VALUE - lane_sum_q16):
            return FP_Q16_ERR_OVERFLOW
        lane_sum_q16 += p

    if lane_sum_q16 != FP_Q16_ONE:
        return FP_Q16_ERR_BAD_PARAM

    order = sorted(range(lane_count), key=lambda idx: (-probs_q16[idx], idx))
    topk = order[:k]

    topk_sum_q16 = 0
    for idx in topk:
        if probs_q16[idx] > (I64_MAX_VALUE - topk_sum_q16):
            return FP_Q16_ERR_OVERFLOW
        topk_sum_q16 += probs_q16[idx]

    if topk_sum_q16 <= 0:
        return FP_Q16_ERR_BAD_PARAM
    if topk_sum_q16 > FP_Q16_ONE:
        return FP_Q16_ERR_OVERFLOW

    threshold_q16 = (random_q16 * topk_sum_q16) >> FP_Q16_SHIFT

    cumulative_q16 = 0
    for idx in topk:
        if probs_q16[idx] > (I64_MAX_VALUE - cumulative_q16):
            return FP_Q16_ERR_OVERFLOW
        cumulative_q16 += probs_q16[idx]
        if threshold_q16 < cumulative_q16:
            out_index[0] = idx
            return FP_Q16_OK

    out_index[0] = topk[-1]
    return FP_Q16_OK


def sample_expected(probs_q16: list[int], k: int, random_q16: int) -> int:
    out = [-1]
    err = fpq16_topk_sample_from_probabilities_checked_reference(
        probs_q16,
        lane_count=len(probs_q16),
        k=k,
        random_q16=random_q16,
        out_index=out,
    )
    assert err == FP_Q16_OK
    return out[0]


def test_null_ptr_paths() -> None:
    out = [777]

    err = fpq16_topk_sample_from_probabilities_checked_reference(None, 3, 2, 0, out)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == [777]

    err = fpq16_topk_sample_from_probabilities_checked_reference([1, 2, 3], 3, 2, 0, None)
    assert err == FP_Q16_ERR_NULL_PTR


def test_bad_param_paths_no_partial_write() -> None:
    probs = [FP_Q16_ONE, 0, 0]
    out = [55]

    err = fpq16_topk_sample_from_probabilities_checked_reference(probs, -1, 1, 0, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [55]

    err = fpq16_topk_sample_from_probabilities_checked_reference(probs, 3, 0, 0, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [55]

    err = fpq16_topk_sample_from_probabilities_checked_reference(probs, 3, 4, 0, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [55]

    err = fpq16_topk_sample_from_probabilities_checked_reference(probs, 3, 2, -1, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [55]

    err = fpq16_topk_sample_from_probabilities_checked_reference(probs, 3, 2, FP_Q16_ONE, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [55]


def test_probability_domain_and_sum_guards() -> None:
    out = [42]

    err = fpq16_topk_sample_from_probabilities_checked_reference(
        [FP_Q16_ONE + 1, 0],
        2,
        1,
        0,
        out,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [42]

    err = fpq16_topk_sample_from_probabilities_checked_reference(
        [-1, FP_Q16_ONE + 1],
        2,
        1,
        0,
        out,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [42]

    err = fpq16_topk_sample_from_probabilities_checked_reference(
        [FP_Q16_ONE - 2, 1],
        2,
        1,
        0,
        out,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [42]


def test_pointer_span_overflow_guard_no_partial_write() -> None:
    probs = [FP_Q16_ONE, 0, 0]
    out = [9]

    err = fpq16_topk_sample_from_probabilities_checked_reference(
        probs,
        lane_count=3,
        k=2,
        random_q16=0,
        out_index=out,
        probs_addr=U64_MAX_VALUE - 15,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [9]


def test_tie_break_order_deterministic() -> None:
    probs = [30000, 30000, FP_Q16_ONE - 60000]

    # k=2 => top-k set is indices [0,1] by lower-index tie-break.
    # random=0 always selects first cumulative bucket.
    assert sample_expected(probs, 2, 0) == 0

    # random near 1 maps to the end of top-k mass and selects the second lane.
    assert sample_expected(probs, 2, FP_Q16_ONE - 1) == 1


def test_randomized_matches_stable_cumulative_reference() -> None:
    rng = random.Random(331)

    for _ in range(2000):
        lane_count = rng.randint(1, 48)

        remaining = FP_Q16_ONE
        probs: list[int] = []
        for idx in range(lane_count - 1):
            if remaining == 0:
                probs.append(0)
                continue
            draw = rng.randint(0, remaining)
            probs.append(draw)
            remaining -= draw
        probs.append(remaining)
        rng.shuffle(probs)

        k = rng.randint(1, lane_count)
        random_q16 = rng.randint(0, FP_Q16_ONE - 1)

        out = [-1]
        err = fpq16_topk_sample_from_probabilities_checked_reference(
            probs,
            lane_count=lane_count,
            k=k,
            random_q16=random_q16,
            out_index=out,
        )
        assert err == FP_Q16_OK

        expected = sample_expected(probs, k, random_q16)
        assert out[0] == expected


def test_boundary_random_values() -> None:
    probs = [FP_Q16_ONE // 2, FP_Q16_ONE - (FP_Q16_ONE // 2)]

    assert sample_expected(probs, 2, 0) in (0, 1)
    assert sample_expected(probs, 2, FP_Q16_ONE - 1) == 1


if __name__ == "__main__":
    test_null_ptr_paths()
    test_bad_param_paths_no_partial_write()
    test_probability_domain_and_sum_guards()
    test_pointer_span_overflow_guard_no_partial_write()
    test_tie_break_order_deterministic()
    test_randomized_matches_stable_cumulative_reference()
    test_boundary_random_values()
    print("softmax_topk_sample_from_probabilities_checked_reference_checks=ok")
