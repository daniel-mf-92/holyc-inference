#!/usr/bin/env python3
"""Reference checks for FPQ16TopKSelectLogitsCheckedNoAlias semantics."""

from __future__ import annotations

import random

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
U64_MAX_VALUE = 0xFFFFFFFFFFFFFFFF


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or b_end < a_start)


def fpq16_topk_select_logits_checked_no_alias_reference(
    logits_q16: list[int] | None,
    lane_count: int,
    k: int,
    out_topk_logits_q16: list[int] | None,
    out_topk_indices: list[int] | None,
    out_lane_capacity: int,
    logits_addr: int = 0x1000,
    out_logits_addr: int = 0x3000,
    out_indices_addr: int = 0x5000,
) -> int:
    if logits_q16 is None or out_topk_logits_q16 is None or out_topk_indices is None:
        return FP_Q16_ERR_NULL_PTR
    if lane_count < 0 or k < 0 or out_lane_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM
    if k > lane_count:
        return FP_Q16_ERR_BAD_PARAM
    if k > out_lane_capacity:
        return FP_Q16_ERR_BAD_PARAM

    if k == 0:
        return FP_Q16_OK

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if logits_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    logits_start = logits_addr
    logits_end = logits_start + last_byte_offset

    last_index = out_lane_capacity - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3

    if out_logits_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if out_indices_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    out_logits_start = out_logits_addr
    out_logits_end = out_logits_start + last_byte_offset
    out_indices_start = out_indices_addr
    out_indices_end = out_indices_start + last_byte_offset

    if _spans_overlap(logits_start, logits_end, out_logits_start, out_logits_end):
        return FP_Q16_ERR_BAD_PARAM
    if _spans_overlap(logits_start, logits_end, out_indices_start, out_indices_end):
        return FP_Q16_ERR_BAD_PARAM
    if _spans_overlap(out_logits_start, out_logits_end, out_indices_start, out_indices_end):
        return FP_Q16_ERR_BAD_PARAM

    if len(logits_q16) < lane_count:
        return FP_Q16_ERR_BAD_PARAM
    if len(out_topk_logits_q16) < out_lane_capacity or len(out_topk_indices) < out_lane_capacity:
        return FP_Q16_ERR_BAD_PARAM

    # Preflight rank realizability.
    for rank in range(k):
        selected_index = -1
        for i in range(lane_count):
            lane_order_rank = 0
            for j in range(lane_count):
                if logits_q16[j] > logits_q16[i]:
                    lane_order_rank += 1
                elif logits_q16[j] == logits_q16[i] and j < i:
                    lane_order_rank += 1

            if lane_order_rank == rank:
                selected_index = i
                break

        if selected_index < 0:
            return FP_Q16_ERR_OVERFLOW

    for rank in range(k):
        selected_index = -1
        selected_logit_q16 = 0

        for i in range(lane_count):
            lane_order_rank = 0
            for j in range(lane_count):
                if logits_q16[j] > logits_q16[i]:
                    lane_order_rank += 1
                elif logits_q16[j] == logits_q16[i] and j < i:
                    lane_order_rank += 1

            if lane_order_rank == rank:
                selected_index = i
                selected_logit_q16 = logits_q16[i]
                break

        if selected_index < 0:
            return FP_Q16_ERR_OVERFLOW

        out_topk_indices[rank] = selected_index
        out_topk_logits_q16[rank] = selected_logit_q16

    return FP_Q16_OK


def stable_topk_pairs_reference(logits_q16: list[int], k: int) -> tuple[list[int], list[int]]:
    order = sorted(range(len(logits_q16)), key=lambda idx: (-logits_q16[idx], idx))[:k]
    return order, [logits_q16[idx] for idx in order]


def test_null_ptr_paths() -> None:
    logits = [5, 4, 3]
    out_logits = [99, 99, 99]
    out_indices = [77, 77, 77]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        None, 3, 2, out_logits, out_indices, 3
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_logits == [99, 99, 99]
    assert out_indices == [77, 77, 77]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits, 3, 2, None, out_indices, 3
    )
    assert err == FP_Q16_ERR_NULL_PTR

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits, 3, 2, out_logits, None, 3
    )
    assert err == FP_Q16_ERR_NULL_PTR


def test_bad_param_paths_no_partial_write() -> None:
    logits = [9, 8, 7]
    out_logits = [101, 102, 103]
    out_indices = [201, 202, 203]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits, -1, 1, out_logits, out_indices, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_logits == [101, 102, 103]
    assert out_indices == [201, 202, 203]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits, 3, -1, out_logits, out_indices, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_logits == [101, 102, 103]
    assert out_indices == [201, 202, 203]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits, 3, 4, out_logits, out_indices, 3
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_logits == [101, 102, 103]
    assert out_indices == [201, 202, 203]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits, 3, 2, out_logits, out_indices, 1
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_logits == [101, 102, 103]
    assert out_indices == [201, 202, 203]


def test_zero_k_ok_no_write() -> None:
    logits = [9, 8, 7]
    out_logits = [11, 12, 13]
    out_indices = [21, 22, 23]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits, 3, 0, out_logits, out_indices, 3
    )
    assert err == FP_Q16_OK
    assert out_logits == [11, 12, 13]
    assert out_indices == [21, 22, 23]


def test_tie_break_order_and_pairs() -> None:
    logits = [5, 5, 7, 7, 3]
    out_logits = [-1] * 5
    out_indices = [-1] * 5

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits, 5, 4, out_logits, out_indices, 5
    )
    assert err == FP_Q16_OK
    assert out_indices[:4] == [2, 3, 0, 1]
    assert out_logits[:4] == [7, 7, 5, 5]


def test_no_alias_span_guards_no_partial_write() -> None:
    logits = [1, 2, 3, 4]
    out_logits = [10, 10, 10, 10]
    out_indices = [20, 20, 20, 20]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits,
        lane_count=4,
        k=2,
        out_topk_logits_q16=out_logits,
        out_topk_indices=out_indices,
        out_lane_capacity=4,
        logits_addr=100,
        out_logits_addr=108,
        out_indices_addr=1000,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_logits == [10, 10, 10, 10]
    assert out_indices == [20, 20, 20, 20]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits,
        lane_count=4,
        k=2,
        out_topk_logits_q16=out_logits,
        out_topk_indices=out_indices,
        out_lane_capacity=4,
        logits_addr=100,
        out_logits_addr=1000,
        out_indices_addr=1016,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_logits == [10, 10, 10, 10]
    assert out_indices == [20, 20, 20, 20]


def test_pointer_span_overflow_guards_no_partial_write() -> None:
    logits = [5, 4, 3]
    out_logits = [30, 31, 32]
    out_indices = [40, 41, 42]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits,
        lane_count=3,
        k=2,
        out_topk_logits_q16=out_logits,
        out_topk_indices=out_indices,
        out_lane_capacity=3,
        logits_addr=U64_MAX_VALUE - 15,
        out_logits_addr=0,
        out_indices_addr=100,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out_logits == [30, 31, 32]
    assert out_indices == [40, 41, 42]

    err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits,
        lane_count=3,
        k=2,
        out_topk_logits_q16=out_logits,
        out_topk_indices=out_indices,
        out_lane_capacity=3,
        logits_addr=0,
        out_logits_addr=U64_MAX_VALUE - 15,
        out_indices_addr=100,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out_logits == [30, 31, 32]
    assert out_indices == [40, 41, 42]


def test_matches_stable_sorted_reference_randomized() -> None:
    rng = random.Random(340)

    for _ in range(1600):
        lane_count = rng.randint(1, 64)
        k = rng.randint(0, lane_count)
        logits = [rng.randint(-280_000, 280_000) for _ in range(lane_count)]

        out_logits = [-9_999_999] * lane_count
        out_indices = [-1] * lane_count

        err = fpq16_topk_select_logits_checked_no_alias_reference(
            logits,
            lane_count=lane_count,
            k=k,
            out_topk_logits_q16=out_logits,
            out_topk_indices=out_indices,
            out_lane_capacity=lane_count,
            logits_addr=0x1000,
            out_logits_addr=0x3000,
            out_indices_addr=0x5000,
        )
        assert err == FP_Q16_OK

        expected_indices, expected_logits = stable_topk_pairs_reference(logits, k)
        assert out_indices[:k] == expected_indices
        assert out_logits[:k] == expected_logits


if __name__ == "__main__":
    test_null_ptr_paths()
    test_bad_param_paths_no_partial_write()
    test_zero_k_ok_no_write()
    test_tie_break_order_and_pairs()
    test_no_alias_span_guards_no_partial_write()
    test_pointer_span_overflow_guards_no_partial_write()
    test_matches_stable_sorted_reference_randomized()
    print("softmax_topk_select_logits_checked_no_alias_reference_checks=ok")
