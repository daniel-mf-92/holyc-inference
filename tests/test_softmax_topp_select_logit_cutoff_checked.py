#!/usr/bin/env python3
"""Reference checks for FPQ16TopPSelectLogitCutoffChecked semantics."""

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


def fpq16_topp_select_logit_cutoff_checked_reference(
    logits_q16: list[int] | None,
    probs_q16: list[int] | None,
    lane_count: int,
    top_p_q16: int,
    out_cutoff_logit_q16: list[int] | None,
    out_nucleus_len: list[int] | None,
    logits_addr: int = 0,
    probs_addr: int = 0,
) -> int:
    if logits_q16 is None or probs_q16 is None or out_cutoff_logit_q16 is None or out_nucleus_len is None:
        return FP_Q16_ERR_NULL_PTR
    if lane_count <= 0:
        return FP_Q16_ERR_BAD_PARAM
    if top_p_q16 <= 0 or top_p_q16 > FP_Q16_ONE:
        return FP_Q16_ERR_BAD_PARAM

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if logits_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if probs_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if len(logits_q16) < lane_count or len(probs_q16) < lane_count:
        return FP_Q16_ERR_BAD_PARAM

    prev_logit = logits_q16[0]
    for idx in range(1, lane_count):
        cur = logits_q16[idx]
        if cur > prev_logit:
            return FP_Q16_ERR_BAD_PARAM
        prev_logit = cur

    prefix_out = [-1]
    status = fpq16_topp_select_prefix_len_checked_reference(
        probs_q16=probs_q16,
        lane_count=lane_count,
        top_p_q16=top_p_q16,
        out_prefix_len=prefix_out,
        probs_addr=probs_addr,
    )
    if status != FP_Q16_OK:
        return status

    prefix_len = prefix_out[0]
    if prefix_len <= 0 or prefix_len > lane_count:
        return FP_Q16_ERR_OVERFLOW

    cutoff_index = prefix_len - 1
    cutoff_logit = logits_q16[cutoff_index]

    extended_len = prefix_len
    while extended_len < lane_count and logits_q16[extended_len] == cutoff_logit:
        extended_len += 1

    out_cutoff_logit_q16[0] = cutoff_logit
    out_nucleus_len[0] = extended_len
    return FP_Q16_OK


def test_null_ptr_paths() -> None:
    logits = [50000, 45000, 40000]
    probs = [30000, 20000, 15536]
    out_cut = [123]
    out_len = [456]

    err = fpq16_topp_select_logit_cutoff_checked_reference(None, probs, 3, FP_Q16_ONE // 2, out_cut, out_len)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_cut == [123]
    assert out_len == [456]

    err = fpq16_topp_select_logit_cutoff_checked_reference(logits, None, 3, FP_Q16_ONE // 2, out_cut, out_len)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_cut == [123]
    assert out_len == [456]

    err = fpq16_topp_select_logit_cutoff_checked_reference(logits, probs, 3, FP_Q16_ONE // 2, None, out_len)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_len == [456]

    err = fpq16_topp_select_logit_cutoff_checked_reference(logits, probs, 3, FP_Q16_ONE // 2, out_cut, None)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_cut == [123]


def test_param_and_pointer_guards_no_partial_write() -> None:
    logits = [50000, 45000, 40000]
    probs = [30000, 20000, 15536]
    out_cut = [777]
    out_len = [888]

    err = fpq16_topp_select_logit_cutoff_checked_reference(logits, probs, 0, FP_Q16_ONE // 2, out_cut, out_len)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cut == [777]
    assert out_len == [888]

    err = fpq16_topp_select_logit_cutoff_checked_reference(logits, probs, 3, 0, out_cut, out_len)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cut == [777]
    assert out_len == [888]

    err = fpq16_topp_select_logit_cutoff_checked_reference(logits, probs, 3, FP_Q16_ONE + 1, out_cut, out_len)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cut == [777]
    assert out_len == [888]

    err = fpq16_topp_select_logit_cutoff_checked_reference(
        logits,
        probs,
        3,
        FP_Q16_ONE // 2,
        out_cut,
        out_len,
        logits_addr=U64_MAX_VALUE - 15,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out_cut == [777]
    assert out_len == [888]

    err = fpq16_topp_select_logit_cutoff_checked_reference(
        logits,
        probs,
        3,
        FP_Q16_ONE // 2,
        out_cut,
        out_len,
        probs_addr=U64_MAX_VALUE - 15,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out_cut == [777]
    assert out_len == [888]


def test_descending_logit_guard() -> None:
    logits = [50000, 51000, 40000]
    probs = [30000, 20000, 15536]
    out_cut = [111]
    out_len = [222]

    err = fpq16_topp_select_logit_cutoff_checked_reference(logits, probs, 3, FP_Q16_ONE // 2, out_cut, out_len)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cut == [111]
    assert out_len == [222]


def test_tie_extension_behavior() -> None:
    logits = [70000, 65000, 62000, 62000, 62000, 59000]
    probs = [25000, 15000, 11000, 9000, 5536, 0]

    out_cut = [-1]
    out_len = [-1]
    err = fpq16_topp_select_logit_cutoff_checked_reference(logits, probs, 6, 51000, out_cut, out_len)
    assert err == FP_Q16_OK
    assert out_cut[0] == 62000
    assert out_len[0] == 5


def test_no_tie_extension_when_boundary_unique() -> None:
    logits = [80000, 79000, 78000, 77000]
    probs = [32768, 16384, 8192, 8192]

    out_cut = [-1]
    out_len = [-1]
    err = fpq16_topp_select_logit_cutoff_checked_reference(logits, probs, 4, 49152, out_cut, out_len)
    assert err == FP_Q16_OK
    assert out_cut[0] == 79000
    assert out_len[0] == 2


def test_randomized_matches_stable_reference() -> None:
    rng = random.Random(347)

    for _ in range(2000):
        lane_count = rng.randint(1, 64)

        cuts = sorted(rng.sample(range(0, FP_Q16_ONE + 1), lane_count - 1))
        points = [0] + cuts + [FP_Q16_ONE]
        probs = [points[i + 1] - points[i] for i in range(lane_count)]
        probs.sort(reverse=True)

        logits = []
        cur = rng.randint(50000, 120000)
        for _idx in range(lane_count):
            logits.append(cur)
            if rng.randint(0, 4) != 0:
                cur -= rng.randint(1, 2048)

        top_p_q16 = rng.randint(1, FP_Q16_ONE)

        out_cut = [-1]
        out_len = [-1]
        err = fpq16_topp_select_logit_cutoff_checked_reference(
            logits_q16=logits,
            probs_q16=probs,
            lane_count=lane_count,
            top_p_q16=top_p_q16,
            out_cutoff_logit_q16=out_cut,
            out_nucleus_len=out_len,
        )
        assert err == FP_Q16_OK

        cumulative = 0
        prefix_len = lane_count
        for idx, p in enumerate(probs):
            cumulative += p
            if cumulative >= top_p_q16:
                prefix_len = idx + 1
                break

        cutoff = logits[prefix_len - 1]
        expected_len = prefix_len
        while expected_len < lane_count and logits[expected_len] == cutoff:
            expected_len += 1

        assert out_cut[0] == cutoff
        assert out_len[0] == expected_len


if __name__ == "__main__":
    test_null_ptr_paths()
    test_param_and_pointer_guards_no_partial_write()
    test_descending_logit_guard()
    test_tie_extension_behavior()
    test_no_tie_extension_when_boundary_unique()
    test_randomized_matches_stable_reference()
    print("softmax_topp_select_logit_cutoff_checked_reference_checks=ok")
