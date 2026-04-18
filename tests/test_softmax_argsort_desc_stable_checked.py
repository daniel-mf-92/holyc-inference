#!/usr/bin/env python3
"""Reference checks for FPQ16ArgsortDescStableChecked semantics."""

from __future__ import annotations

import random

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
U64_MAX_VALUE = 0xFFFFFFFFFFFFFFFF


def fpq16_argsort_desc_stable_checked_reference(
    logits_q16: list[int] | None,
    lane_count: int,
    out_indices: list[int] | None,
    out_index_capacity: int,
    logits_addr: int = 0,
    out_addr: int = 0,
) -> int:
    if logits_q16 is None or out_indices is None:
        return FP_Q16_ERR_NULL_PTR
    if lane_count < 0 or out_index_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM
    if out_index_capacity < lane_count:
        return FP_Q16_ERR_BAD_PARAM

    if lane_count == 0:
        return FP_Q16_OK

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if logits_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if out_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if len(logits_q16) < lane_count or len(out_indices) < out_index_capacity:
        return FP_Q16_ERR_BAD_PARAM

    for i in range(lane_count):
        out_indices[i] = i

    for i in range(1, lane_count):
        key_index = out_indices[i]
        key_logit_q16 = logits_q16[key_index]
        j = i - 1

        while j >= 0:
            scan_index = out_indices[j]
            scan_logit_q16 = logits_q16[scan_index]

            if scan_logit_q16 < key_logit_q16:
                out_indices[j + 1] = scan_index
                j -= 1
                continue

            if scan_logit_q16 == key_logit_q16 and scan_index > key_index:
                out_indices[j + 1] = scan_index
                j -= 1
                continue

            break

        out_indices[j + 1] = key_index

    return FP_Q16_OK


def stable_desc_reference(logits_q16: list[int]) -> list[int]:
    return sorted(range(len(logits_q16)), key=lambda idx: (-logits_q16[idx], idx))


def test_null_ptr_paths() -> None:
    out = [77, 78, 79]

    err = fpq16_argsort_desc_stable_checked_reference(None, 2, out, len(out))
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == [77, 78, 79]

    err = fpq16_argsort_desc_stable_checked_reference([1, 2], 2, None, 2)
    assert err == FP_Q16_ERR_NULL_PTR


def test_bad_param_paths() -> None:
    logits = [5, 1, 3]
    out = [7, 7, 7]

    err = fpq16_argsort_desc_stable_checked_reference(logits, -1, out, 3)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [7, 7, 7]

    err = fpq16_argsort_desc_stable_checked_reference(logits, 3, out, -1)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [7, 7, 7]

    err = fpq16_argsort_desc_stable_checked_reference(logits, 3, out, 2)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [7, 7, 7]


def test_zero_count_ok_no_write() -> None:
    out = [11, 12, 13]

    err = fpq16_argsort_desc_stable_checked_reference([9, 8, 7], 0, out, 3)
    assert err == FP_Q16_OK
    assert out == [11, 12, 13]


def test_tie_break_prefers_lower_index_stable() -> None:
    logits = [7, 7, 4, 7, 4, 1]
    out = [-1] * len(logits)

    err = fpq16_argsort_desc_stable_checked_reference(logits, len(logits), out, len(logits))
    assert err == FP_Q16_OK
    assert out == [0, 1, 3, 2, 4, 5]


def test_matches_python_stable_sort_randomized() -> None:
    rng = random.Random(20260418)

    for _ in range(2000):
        lane_count = rng.randint(1, 72)
        logits = [rng.randint(-300_000, 300_000) for _ in range(lane_count)]
        out = [-1] * lane_count

        err = fpq16_argsort_desc_stable_checked_reference(
            logits,
            lane_count,
            out,
            lane_count,
        )
        assert err == FP_Q16_OK
        assert out == stable_desc_reference(logits)


def test_pointer_span_overflow_guards_no_partial_write() -> None:
    logits = [1, 2, 3]
    out = [44, 55, 66]

    err = fpq16_argsort_desc_stable_checked_reference(
        logits,
        lane_count=3,
        out_indices=out,
        out_index_capacity=3,
        logits_addr=U64_MAX_VALUE - 15,
        out_addr=0,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [44, 55, 66]

    err = fpq16_argsort_desc_stable_checked_reference(
        logits,
        lane_count=3,
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
    test_zero_count_ok_no_write()
    test_tie_break_prefers_lower_index_stable()
    test_matches_python_stable_sort_randomized()
    test_pointer_span_overflow_guards_no_partial_write()
    print("softmax_argsort_desc_stable_checked_reference_checks=ok")
