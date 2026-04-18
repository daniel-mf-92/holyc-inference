#!/usr/bin/env python3
"""Reference checks for FPQ16TopKSelectIndicesChecked semantics."""

from __future__ import annotations

import random

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
U64_MAX_VALUE = 0xFFFFFFFFFFFFFFFF



def fpq16_topk_select_indices_checked_reference(
    logits_q16: list[int] | None,
    lane_count: int,
    k: int,
    out_indices: list[int] | None,
    out_index_capacity: int,
    logits_addr: int = 0,
    out_addr: int = 0,
) -> int:
    if logits_q16 is None or out_indices is None:
        return FP_Q16_ERR_NULL_PTR
    if lane_count < 0 or k < 0 or out_index_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM
    if k > lane_count:
        return FP_Q16_ERR_BAD_PARAM
    if k > out_index_capacity:
        return FP_Q16_ERR_BAD_PARAM

    if k == 0:
        return FP_Q16_OK

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if logits_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    last_index = k - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if out_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if len(logits_q16) < lane_count or len(out_indices) < out_index_capacity:
        return FP_Q16_ERR_BAD_PARAM

    for rank in range(k):
        best_index = -1
        best_logit_q16 = 0

        for i in range(lane_count):
            already_selected = False
            for prior in range(rank):
                if out_indices[prior] == i:
                    already_selected = True
                    break
            if already_selected:
                continue

            candidate = logits_q16[i]
            if best_index < 0:
                best_index = i
                best_logit_q16 = candidate
            elif candidate > best_logit_q16:
                best_index = i
                best_logit_q16 = candidate
            elif candidate == best_logit_q16 and i < best_index:
                best_index = i

        if best_index < 0:
            return FP_Q16_ERR_OVERFLOW

        out_indices[rank] = best_index

    return FP_Q16_OK



def stable_topk_reference(logits_q16: list[int], k: int) -> list[int]:
    order = sorted(range(len(logits_q16)), key=lambda idx: (-logits_q16[idx], idx))
    return order[:k]



def test_null_ptr_paths() -> None:
    out = [99, 98, 97]

    err = fpq16_topk_select_indices_checked_reference(None, 2, 1, out, len(out))
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == [99, 98, 97]

    err = fpq16_topk_select_indices_checked_reference([1, 2], 2, 1, None, 1)
    assert err == FP_Q16_ERR_NULL_PTR



def test_bad_param_paths() -> None:
    logits = [5, 1, 3]
    out = [7, 7, 7]

    err = fpq16_topk_select_indices_checked_reference(logits, -1, 1, out, 3)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [7, 7, 7]

    err = fpq16_topk_select_indices_checked_reference(logits, 3, -1, out, 3)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [7, 7, 7]

    err = fpq16_topk_select_indices_checked_reference(logits, 3, 4, out, 3)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [7, 7, 7]

    err = fpq16_topk_select_indices_checked_reference(logits, 3, 2, out, 1)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [7, 7, 7]



def test_zero_k_ok_no_write() -> None:
    logits = [9, 8, 7]
    out = [11, 12, 13]

    err = fpq16_topk_select_indices_checked_reference(logits, 3, 0, out, 3)
    assert err == FP_Q16_OK
    assert out == [11, 12, 13]



def test_tie_break_prefers_lower_index() -> None:
    logits = [5, 5, 5, 3]
    out = [99, 99, 99, 99]

    err = fpq16_topk_select_indices_checked_reference(logits, 4, 3, out, 4)
    assert err == FP_Q16_OK
    assert out[:3] == [0, 1, 2]



def test_matches_sorted_reference_randomized() -> None:
    rng = random.Random(20260418)

    for _ in range(1600):
        lane_count = rng.randint(1, 64)
        k = rng.randint(0, lane_count)
        logits = [rng.randint(-250_000, 250_000) for _ in range(lane_count)]

        out = [-1] * lane_count
        err = fpq16_topk_select_indices_checked_reference(logits, lane_count, k, out, lane_count)
        assert err == FP_Q16_OK
        assert out[:k] == stable_topk_reference(logits, k)

        seen = set(out[:k])
        assert len(seen) == k



def test_pointer_span_overflow_guards_no_partial_write() -> None:
    logits = [1, 2, 3]
    out = [44, 55, 66]

    # logits pointer + ((lane_count-1)<<3) overflow
    err = fpq16_topk_select_indices_checked_reference(
        logits,
        lane_count=3,
        k=2,
        out_indices=out,
        out_index_capacity=3,
        logits_addr=U64_MAX_VALUE - 15,
        out_addr=0,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [44, 55, 66]

    err = fpq16_topk_select_indices_checked_reference(
        logits,
        lane_count=3,
        k=3,
        out_indices=out,
        out_index_capacity=3,
        logits_addr=0,
        out_addr=U64_MAX_VALUE - 15,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [44, 55, 66]



if __name__ == "__main__":
    test_null_ptr_paths()
    test_bad_param_paths()
    test_zero_k_ok_no_write()
    test_tie_break_prefers_lower_index()
    test_matches_sorted_reference_randomized()
    test_pointer_span_overflow_guards_no_partial_write()
    print("softmax_topk_select_indices_checked_reference_checks=ok")
