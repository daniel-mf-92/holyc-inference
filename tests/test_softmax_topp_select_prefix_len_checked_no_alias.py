#!/usr/bin/env python3
"""Parity checks for FPQ16TopPSelectPrefixLenCheckedNoAlias semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_softmax_topp_select_logit_cutoff_checked import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
    U64_MAX_VALUE,
    fpq16_topp_select_prefix_len_checked_reference,
)


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or b_end < a_start)


def fpq16_topp_select_prefix_len_checked_no_alias_reference(
    probs_q16: list[int] | None,
    lane_count: int,
    top_p_q16: int,
    out_prefix_len: list[int] | None,
    probs_addr: int = 0x1000,
    out_addr: int = 0x5000,
) -> int:
    if probs_q16 is None or out_prefix_len is None:
        return FP_Q16_ERR_NULL_PTR
    if lane_count <= 0:
        return FP_Q16_ERR_BAD_PARAM

    probs_last_index = lane_count - 1
    if probs_last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    probs_last_byte_offset = probs_last_index << 3

    if probs_addr > (U64_MAX_VALUE - probs_last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    probs_start = probs_addr
    probs_end = probs_start + probs_last_byte_offset

    if out_addr > (U64_MAX_VALUE - 7):
        return FP_Q16_ERR_OVERFLOW
    out_start = out_addr
    out_end = out_start + 7

    if _spans_overlap(probs_start, probs_end, out_start, out_end):
        return FP_Q16_ERR_BAD_PARAM

    return fpq16_topp_select_prefix_len_checked_reference(
        probs_q16=probs_q16,
        lane_count=lane_count,
        top_p_q16=top_p_q16,
        out_prefix_len=out_prefix_len,
        probs_addr=probs_addr,
    )


def test_source_contains_no_alias_wrapper_and_core_delegation() -> None:
    source = pathlib.Path("src/math/softmax.HC").read_text(encoding="utf-8")
    assert "FPQ16TopPSelectPrefixLenCheckedNoAlias" in source
    assert "FPQ16TopPSelectPrefixLenChecked(probs_q16," in source


def test_null_ptr_paths() -> None:
    probs = [32768, 16384, 16384]
    out = [77]

    err = fpq16_topp_select_prefix_len_checked_no_alias_reference(None, 3, 32768, out)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == [77]

    err = fpq16_topp_select_prefix_len_checked_no_alias_reference(probs, 3, 32768, None)
    assert err == FP_Q16_ERR_NULL_PTR


def test_alias_and_non_alias_paths() -> None:
    probs = [30000, 20000, 15536]
    out = [123]

    err = fpq16_topp_select_prefix_len_checked_no_alias_reference(
        probs_q16=probs,
        lane_count=3,
        top_p_q16=32000,
        out_prefix_len=out,
        probs_addr=100,
        out_addr=108,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [123]

    err = fpq16_topp_select_prefix_len_checked_no_alias_reference(
        probs_q16=probs,
        lane_count=3,
        top_p_q16=32000,
        out_prefix_len=out,
        probs_addr=100,
        out_addr=1000,
    )
    assert err == FP_Q16_OK
    assert out[0] == 2


def test_matches_core_for_non_alias_vectors() -> None:
    probs = [25000, 18000, 12000, 10536]

    out_no_alias = [901]
    out_core = [902]

    err_no_alias = fpq16_topp_select_prefix_len_checked_no_alias_reference(
        probs_q16=probs,
        lane_count=4,
        top_p_q16=43000,
        out_prefix_len=out_no_alias,
        probs_addr=0x4000,
        out_addr=0x9000,
    )
    err_core = fpq16_topp_select_prefix_len_checked_reference(
        probs_q16=probs,
        lane_count=4,
        top_p_q16=43000,
        out_prefix_len=out_core,
        probs_addr=0x4000,
    )

    assert err_no_alias == FP_Q16_OK
    assert err_core == FP_Q16_OK
    assert out_no_alias == out_core == [2]


def test_pointer_overflow_guard_no_partial_write() -> None:
    probs = [30000, 20000, 15536]
    out = [444]

    err = fpq16_topp_select_prefix_len_checked_no_alias_reference(
        probs_q16=probs,
        lane_count=3,
        top_p_q16=32768,
        out_prefix_len=out,
        probs_addr=U64_MAX_VALUE - 15,
        out_addr=0x5000,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [444]

    err = fpq16_topp_select_prefix_len_checked_no_alias_reference(
        probs_q16=probs,
        lane_count=3,
        top_p_q16=32768,
        out_prefix_len=out,
        probs_addr=0x5000,
        out_addr=U64_MAX_VALUE - 6,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [444]


def test_randomized_non_alias_parity_against_core() -> None:
    rng = random.Random(362)

    for _ in range(1200):
        lane_count = rng.randint(1, 64)

        cuts = sorted(rng.sample(range(0, 1 << 16), lane_count - 1))
        points = [0] + cuts + [1 << 16]
        probs = [points[i + 1] - points[i] for i in range(lane_count)]
        probs.sort(reverse=True)

        top_p_q16 = rng.randint(1, 1 << 16)

        out_no_alias = [-1]
        out_core = [-1]

        probs_addr = 0x1000
        out_addr = 0x9000

        err_no_alias = fpq16_topp_select_prefix_len_checked_no_alias_reference(
            probs_q16=probs,
            lane_count=lane_count,
            top_p_q16=top_p_q16,
            out_prefix_len=out_no_alias,
            probs_addr=probs_addr,
            out_addr=out_addr,
        )
        err_core = fpq16_topp_select_prefix_len_checked_reference(
            probs_q16=probs,
            lane_count=lane_count,
            top_p_q16=top_p_q16,
            out_prefix_len=out_core,
            probs_addr=probs_addr,
        )

        assert err_no_alias == err_core
        assert out_no_alias == out_core


if __name__ == "__main__":
    test_source_contains_no_alias_wrapper_and_core_delegation()
    test_null_ptr_paths()
    test_alias_and_non_alias_paths()
    test_matches_core_for_non_alias_vectors()
    test_pointer_overflow_guard_no_partial_write()
    test_randomized_non_alias_parity_against_core()
    print("ok")
