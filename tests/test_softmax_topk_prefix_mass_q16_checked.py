#!/usr/bin/env python3
"""Reference checks for FPQ16TopKPrefixMassQ16Checked semantics."""

from __future__ import annotations

import random

FP_Q16_ONE = 65536

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
U64_MAX_VALUE = 0xFFFFFFFFFFFFFFFF


def fpq16_topk_prefix_mass_q16_checked_reference(
    probs_q16: list[int] | None,
    lane_count: int,
    topk_indices: list[int] | None,
    k: int,
    out_prefix_mass_q16: list[int] | None,
    out_mass_capacity: int,
    probs_addr: int = 0x1000,
    topk_indices_addr: int = 0x3000,
    out_prefix_mass_addr: int = 0x5000,
) -> int:
    if probs_q16 is None or topk_indices is None or out_prefix_mass_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if lane_count < 0 or k < 0 or out_mass_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM
    if k > lane_count:
        return FP_Q16_ERR_BAD_PARAM
    if k > out_mass_capacity:
        return FP_Q16_ERR_BAD_PARAM

    if k == 0:
        return FP_Q16_OK

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if probs_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    last_index = k - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if topk_indices_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    last_index = out_mass_capacity - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if out_prefix_mass_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if len(probs_q16) < lane_count:
        return FP_Q16_ERR_BAD_PARAM
    if len(topk_indices) < k:
        return FP_Q16_ERR_BAD_PARAM
    if len(out_prefix_mass_q16) < out_mass_capacity:
        return FP_Q16_ERR_BAD_PARAM

    cumulative_q16 = 0
    prev_cumulative_q16 = 0
    for rank in range(k):
        selected_index = topk_indices[rank]
        if selected_index < 0 or selected_index >= lane_count:
            return FP_Q16_ERR_BAD_PARAM

        for prior in range(rank):
            if topk_indices[prior] == selected_index:
                return FP_Q16_ERR_BAD_PARAM

        lane_prob_q16 = probs_q16[selected_index]
        if lane_prob_q16 < 0 or lane_prob_q16 > FP_Q16_ONE:
            return FP_Q16_ERR_BAD_PARAM
        if lane_prob_q16 > (I64_MAX_VALUE - cumulative_q16):
            return FP_Q16_ERR_OVERFLOW

        cumulative_q16 += lane_prob_q16
        if cumulative_q16 < prev_cumulative_q16:
            return FP_Q16_ERR_OVERFLOW
        if cumulative_q16 > FP_Q16_ONE:
            return FP_Q16_ERR_BAD_PARAM

        prev_cumulative_q16 = cumulative_q16

    cumulative_q16 = 0
    for rank in range(k):
        cumulative_q16 += probs_q16[topk_indices[rank]]
        out_prefix_mass_q16[rank] = cumulative_q16

    return FP_Q16_OK


def test_null_ptr_paths() -> None:
    probs = [32000, 20000, 13536]
    indices = [0, 1]
    out = [999, 999, 999]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        None, 3, indices, 2, out, 3
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == [999, 999, 999]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs, 3, None, 2, out, 3
    )
    assert err == FP_Q16_ERR_NULL_PTR

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs, 3, indices, 2, None, 3
    )
    assert err == FP_Q16_ERR_NULL_PTR


def test_bad_param_paths_no_partial_write() -> None:
    probs = [32000, 20000, 13536]
    indices = [0, 1]
    out = [77, 88, 99]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs, -1, indices, 1, out, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [77, 88, 99]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs, 3, indices, -1, out, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [77, 88, 99]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs, 3, indices, 4, out, 4
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [77, 88, 99]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs, 3, indices, 2, out, 1
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [77, 88, 99]


def test_index_domain_and_uniqueness_guards() -> None:
    probs = [32000, 20000, 13536]
    out = [11, 22, 33]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs, 3, [0, 3], 2, out, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [11, 22, 33]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs, 3, [-1, 1], 2, out, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [11, 22, 33]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs, 3, [1, 1], 2, out, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [11, 22, 33]


def test_probability_domain_and_mass_guards() -> None:
    out = [1, 2, 3]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        [FP_Q16_ONE + 1, 0, 0], 3, [0], 1, out, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [1, 2, 3]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        [-1, 1, FP_Q16_ONE], 3, [0], 1, out, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [1, 2, 3]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        [50000, 30000, 1000], 3, [0, 1], 2, out, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [1, 2, 3]


def test_pointer_span_overflow_guards_no_partial_write() -> None:
    probs = [32768, 16384, 16384]
    indices = [0, 1]
    out = [44, 55, 66]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs,
        lane_count=3,
        topk_indices=indices,
        k=2,
        out_prefix_mass_q16=out,
        out_mass_capacity=3,
        probs_addr=U64_MAX_VALUE - 15,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [44, 55, 66]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs,
        lane_count=3,
        topk_indices=indices,
        k=2,
        out_prefix_mass_q16=out,
        out_mass_capacity=3,
        topk_indices_addr=U64_MAX_VALUE - 7,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [44, 55, 66]


def test_success_prefix_mass_accumulates_selected_order() -> None:
    probs = [20000, 15000, 10000, 20536]
    indices = [3, 0, 1]
    out = [-1, -1, -1, -1]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs,
        lane_count=4,
        topk_indices=indices,
        k=3,
        out_prefix_mass_q16=out,
        out_mass_capacity=4,
    )
    assert err == FP_Q16_OK
    assert out[:3] == [20536, 40536, 55536]
    assert out[3] == -1


def test_k_zero_is_no_write() -> None:
    probs = [32768, 16384, 16384]
    indices = [0, 1, 2]
    out = [9, 8, 7]

    err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs,
        lane_count=3,
        topk_indices=indices,
        k=0,
        out_prefix_mass_q16=out,
        out_mass_capacity=3,
    )
    assert err == FP_Q16_OK
    assert out == [9, 8, 7]


def test_randomized_matches_stable_reference() -> None:
    rng = random.Random(342)

    for _ in range(2000):
        lane_count = rng.randint(1, 64)

        cuts = sorted(rng.sample(range(0, FP_Q16_ONE + 1), lane_count - 1))
        points = [0] + cuts + [FP_Q16_ONE]
        probs = [points[i + 1] - points[i] for i in range(lane_count)]

        k = rng.randint(0, lane_count)
        indices = list(range(lane_count))
        rng.shuffle(indices)
        indices = indices[:k]

        out = [-1] * max(1, lane_count)
        err = fpq16_topk_prefix_mass_q16_checked_reference(
            probs,
            lane_count=lane_count,
            topk_indices=indices,
            k=k,
            out_prefix_mass_q16=out,
            out_mass_capacity=max(1, lane_count),
        )
        assert err == FP_Q16_OK

        running = 0
        for idx, token_index in enumerate(indices):
            running += probs[token_index]
            assert out[idx] == running


if __name__ == "__main__":
    test_null_ptr_paths()
    test_bad_param_paths_no_partial_write()
    test_index_domain_and_uniqueness_guards()
    test_probability_domain_and_mass_guards()
    test_pointer_span_overflow_guards_no_partial_write()
    test_success_prefix_mass_accumulates_selected_order()
    test_k_zero_is_no_write()
    test_randomized_matches_stable_reference()
    print("softmax_topk_prefix_mass_q16_checked_reference_checks=ok")
