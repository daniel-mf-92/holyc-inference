#!/usr/bin/env python3
"""Reference checks for FPQ16TopPSampleIndexChecked semantics."""

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


def fpq16_topp_sample_index_checked_reference(
    probs_q16: list[int] | None,
    lane_count: int,
    prefix_len: int,
    random_q16: int,
    out_index: list[int] | None,
    probs_addr: int = 0,
) -> int:
    if probs_q16 is None or out_index is None:
        return FP_Q16_ERR_NULL_PTR
    if lane_count <= 0:
        return FP_Q16_ERR_BAD_PARAM
    if prefix_len <= 0 or prefix_len > lane_count:
        return FP_Q16_ERR_BAD_PARAM
    if random_q16 < 0 or random_q16 >= FP_Q16_ONE:
        return FP_Q16_ERR_BAD_PARAM

    last_index = prefix_len - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if probs_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if len(probs_q16) < prefix_len:
        return FP_Q16_ERR_BAD_PARAM

    prefix_mass_q16 = 0
    prev_prob_q16 = FP_Q16_ONE
    for idx in range(prefix_len):
        p = probs_q16[idx]
        if p < 0 or p > FP_Q16_ONE:
            return FP_Q16_ERR_BAD_PARAM
        if p > prev_prob_q16:
            return FP_Q16_ERR_BAD_PARAM
        if p > (I64_MAX_VALUE - prefix_mass_q16):
            return FP_Q16_ERR_OVERFLOW

        prefix_mass_q16 += p
        prev_prob_q16 = p

    if prefix_mass_q16 <= 0:
        return FP_Q16_ERR_BAD_PARAM
    if prefix_mass_q16 > FP_Q16_ONE:
        return FP_Q16_ERR_BAD_PARAM

    threshold_q16 = (random_q16 * prefix_mass_q16) >> FP_Q16_SHIFT

    cumulative_q16 = 0
    for idx in range(prefix_len):
        p = probs_q16[idx]
        if p > (I64_MAX_VALUE - cumulative_q16):
            return FP_Q16_ERR_OVERFLOW

        cumulative_q16 += p
        if threshold_q16 < cumulative_q16:
            out_index[0] = idx
            return FP_Q16_OK

    out_index[0] = prefix_len - 1
    return FP_Q16_OK


def sample_expected(probs_q16: list[int], prefix_len: int, random_q16: int) -> int:
    out = [-1]
    err = fpq16_topp_sample_index_checked_reference(
        probs_q16,
        lane_count=len(probs_q16),
        prefix_len=prefix_len,
        random_q16=random_q16,
        out_index=out,
    )
    assert err == FP_Q16_OK
    return out[0]


def test_null_ptr_paths() -> None:
    out = [123]

    err = fpq16_topp_sample_index_checked_reference(None, 3, 2, 0, out)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == [123]

    err = fpq16_topp_sample_index_checked_reference([32768, 16384, 16384], 3, 2, 0, None)
    assert err == FP_Q16_ERR_NULL_PTR


def test_bad_param_paths_no_partial_write() -> None:
    probs = [32768, 16384, 16384]
    out = [88]

    err = fpq16_topp_sample_index_checked_reference(probs, 0, 1, 0, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [88]

    err = fpq16_topp_sample_index_checked_reference(probs, 3, 0, 0, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [88]

    err = fpq16_topp_sample_index_checked_reference(probs, 3, 4, 0, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [88]

    err = fpq16_topp_sample_index_checked_reference(probs, 3, 2, -1, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [88]

    err = fpq16_topp_sample_index_checked_reference(probs, 3, 2, FP_Q16_ONE, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [88]


def test_pointer_span_overflow_guard_no_partial_write() -> None:
    probs = [32768, 16384, 16384]
    out = [77]

    err = fpq16_topp_sample_index_checked_reference(
        probs,
        lane_count=3,
        prefix_len=3,
        random_q16=0,
        out_index=out,
        probs_addr=U64_MAX_VALUE - 15,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [77]


def test_prefix_domain_guards() -> None:
    out = [66]

    err = fpq16_topp_sample_index_checked_reference([40000, 20000, 5536], 3, 3, 0, out)
    assert err == FP_Q16_OK

    err = fpq16_topp_sample_index_checked_reference([30000, 31000, 4536], 3, 3, 0, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [0]

    out = [66]
    err = fpq16_topp_sample_index_checked_reference([FP_Q16_ONE + 1, 0, 0], 3, 3, 0, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [66]

    out = [66]
    err = fpq16_topp_sample_index_checked_reference([0, 0, FP_Q16_ONE], 3, 2, 0, out)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [66]


def test_threshold_and_tie_behavior() -> None:
    probs = [30000, 30000, 5536]

    assert sample_expected(probs, 2, 0) == 0
    assert sample_expected(probs, 2, FP_Q16_ONE // 2 - 1) == 0
    assert sample_expected(probs, 2, FP_Q16_ONE // 2) == 1
    assert sample_expected(probs, 2, FP_Q16_ONE - 1) == 1


def test_prefix_one_always_selects_zero() -> None:
    probs = [32768, 16384, 16384]
    for r in (0, 1, 12345, FP_Q16_ONE - 1):
        assert sample_expected(probs, 1, r) == 0


def test_randomized_matches_stable_cumulative_reference() -> None:
    rng = random.Random(333)

    for _ in range(2000):
        lane_count = rng.randint(1, 64)

        cuts = sorted(rng.sample(range(0, FP_Q16_ONE + 1), lane_count - 1))
        points = [0] + cuts + [FP_Q16_ONE]
        probs = [points[i + 1] - points[i] for i in range(lane_count)]
        probs.sort(reverse=True)

        prefix_len = rng.randint(1, lane_count)
        random_q16 = rng.randint(0, FP_Q16_ONE - 1)

        out = [-1]
        err = fpq16_topp_sample_index_checked_reference(
            probs,
            lane_count=lane_count,
            prefix_len=prefix_len,
            random_q16=random_q16,
            out_index=out,
        )
        assert err == FP_Q16_OK

        expected = sample_expected(probs, prefix_len, random_q16)
        assert out[0] == expected


if __name__ == "__main__":
    test_null_ptr_paths()
    test_bad_param_paths_no_partial_write()
    test_pointer_span_overflow_guard_no_partial_write()
    test_prefix_domain_guards()
    test_threshold_and_tie_behavior()
    test_prefix_one_always_selects_zero()
    test_randomized_matches_stable_cumulative_reference()
    print("softmax_topp_sample_index_checked_reference_checks=ok")
