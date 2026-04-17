#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxFromPreclampedCheckedWithSumOut semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_softmax_exp_phase_from_preclamped_checked import (
    EXP_Q16_MAX_INPUT,
    EXP_Q16_MIN_INPUT,
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    FP_Q16_ONE,
    fpq16_softmax_exp_phase_from_preclamped_checked_reference,
)
from test_softmax_normalize_phase_checked import fpq16_softmax_normalize_phase_checked_reference


def fpq16_softmax_from_preclamped_checked_with_sum_out_reference(
    preclamped_logits_q16: list[int],
    exp_lanes_q16: list[int],
    lane_count: int,
    probs_q16: list[int],
    out_exp_sum_q16: int | None,
    *,
    logits_exp_alias: bool = False,
    logits_probs_alias: bool = False,
    exp_probs_alias: bool = False,
) -> tuple[int, list[int], list[int], int | None]:
    if out_exp_sum_q16 is None:
        return FP_Q16_ERR_NULL_PTR, exp_lanes_q16[:], probs_q16[:], None

    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, exp_lanes_q16[:], probs_q16[:], out_exp_sum_q16

    if lane_count == 0:
        if logits_exp_alias:
            return FP_Q16_OK, exp_lanes_q16[:], exp_lanes_q16[:], 0
        if exp_probs_alias:
            return FP_Q16_OK, exp_lanes_q16[:], exp_lanes_q16[:], 0
        if logits_probs_alias:
            return FP_Q16_OK, exp_lanes_q16[:], probs_q16[:], 0
        return FP_Q16_OK, exp_lanes_q16[:], probs_q16[:], 0

    read_logits = preclamped_logits_q16
    if logits_exp_alias:
        read_logits = exp_lanes_q16
    elif logits_probs_alias:
        read_logits = probs_q16

    exp_err, exp_out, exp_sum_q16 = fpq16_softmax_exp_phase_from_preclamped_checked_reference(
        read_logits,
        exp_lanes_q16,
        lane_count,
        0,
    )
    if exp_err != FP_Q16_OK:
        return exp_err, exp_lanes_q16[:], probs_q16[:], out_exp_sum_q16

    if logits_exp_alias or exp_probs_alias:
        exp_buf = exp_out
        exp_written = exp_buf[:]
    else:
        exp_buf = exp_out
        exp_written = exp_out

    preflight_err, _ = fpq16_softmax_normalize_phase_checked_reference(
        exp_buf,
        lane_count,
        exp_sum_q16,
    )
    if preflight_err != FP_Q16_OK:
        if logits_exp_alias or exp_probs_alias:
            return preflight_err, exp_written, exp_written, exp_sum_q16
        return preflight_err, exp_written, probs_q16[:], exp_sum_q16

    norm_err, probs_out = fpq16_softmax_normalize_phase_checked_reference(
        exp_buf,
        lane_count,
        exp_sum_q16,
    )
    if norm_err != FP_Q16_OK:
        if logits_exp_alias or exp_probs_alias:
            return norm_err, exp_written, exp_written, exp_sum_q16
        return norm_err, exp_written, probs_q16[:], exp_sum_q16

    if logits_exp_alias:
        return FP_Q16_OK, probs_out, probs_out, exp_sum_q16
    if exp_probs_alias:
        return FP_Q16_OK, exp_written, probs_out, exp_sum_q16
    return FP_Q16_OK, exp_written, probs_out, exp_sum_q16


def test_null_sum_pointer_rejected() -> None:
    err, exp_out, probs_out, sum_out = fpq16_softmax_from_preclamped_checked_with_sum_out_reference(
        [0, 1],
        [7, 8],
        2,
        [9, 10],
        None,
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert exp_out == [7, 8]
    assert probs_out == [9, 10]
    assert sum_out is None


def test_empty_count_writes_zero_sum_and_preserves_outputs() -> None:
    logits = [1, 2]
    exp_seed = [3, 4]
    probs_seed = [5, 6]

    err, exp_out, probs_out, sum_out = fpq16_softmax_from_preclamped_checked_with_sum_out_reference(
        logits,
        exp_seed,
        0,
        probs_seed,
        777,
    )
    assert err == FP_Q16_OK
    assert exp_out == exp_seed
    assert probs_out == probs_seed
    assert sum_out == 0


def test_domain_failure_has_no_partial_writes_and_sum_unchanged() -> None:
    logits = [0, EXP_Q16_MAX_INPUT + 1, 1]
    exp_seed = [101, 202, 303]
    probs_seed = [404, 505, 606]

    err, exp_out, probs_out, sum_out = fpq16_softmax_from_preclamped_checked_with_sum_out_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
        999,
    )

    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_out == exp_seed
    assert probs_out == probs_seed
    assert sum_out == 999


def test_normalize_overflow_is_fail_fast_for_probs_but_sum_is_written() -> None:
    logits = [EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT]
    exp_seed = [1234, 5678]
    probs_seed = [91011, 121314]

    err, exp_out, probs_out, sum_out = fpq16_softmax_from_preclamped_checked_with_sum_out_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
        42,
    )

    assert err == FP_Q16_ERR_OVERFLOW
    assert exp_out != exp_seed
    assert probs_out == probs_seed
    assert sum_out is not None
    assert sum_out > 0


def test_alias_modes_are_supported_with_sum_out() -> None:
    logits = [0, 1, -1, 3]
    exp_seed = [7, 8, 9, 10]
    probs_seed = [11, 12, 13, 14]

    err, exp_out, probs_out, sum_out = fpq16_softmax_from_preclamped_checked_with_sum_out_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
        0,
    )
    assert err == FP_Q16_OK
    assert sum(probs_out) == FP_Q16_ONE
    assert sum_out is not None and sum_out > 0

    err, exp_out, probs_out, sum_out = fpq16_softmax_from_preclamped_checked_with_sum_out_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
        0,
        logits_exp_alias=True,
    )
    assert err == FP_Q16_OK
    assert exp_out == probs_out
    assert sum(probs_out) == FP_Q16_ONE
    assert sum_out is not None and sum_out > 0

    err, exp_out, probs_out, sum_out = fpq16_softmax_from_preclamped_checked_with_sum_out_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
        0,
        exp_probs_alias=True,
    )
    assert err == FP_Q16_OK
    assert sum(probs_out) == FP_Q16_ONE
    assert sum_out is not None and sum_out > 0

    err, exp_out, probs_out, sum_out = fpq16_softmax_from_preclamped_checked_with_sum_out_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
        0,
        logits_probs_alias=True,
    )
    assert err == FP_Q16_OK
    assert sum(probs_out) == FP_Q16_ONE
    assert sum_out is not None and sum_out > 0


def test_random_vectors_match_split_phase_reference() -> None:
    rng = random.Random(20260417210)

    for _ in range(1500):
        lane_count = rng.randint(1, 24)
        logits = [rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT) for _ in range(lane_count)]
        exp_seed = [rng.randint(0, 4096) for _ in range(lane_count)]
        probs_seed = [rng.randint(0, 4096) for _ in range(lane_count)]

        err, exp_out, probs_out, sum_out = fpq16_softmax_from_preclamped_checked_with_sum_out_reference(
            logits,
            exp_seed,
            lane_count,
            probs_seed,
            1,
        )

        if err == FP_Q16_ERR_OVERFLOW:
            assert probs_out == probs_seed
            assert sum_out is not None and sum_out > 0
            continue

        assert err == FP_Q16_OK
        assert len(exp_out) == lane_count
        assert len(probs_out) == lane_count
        assert all(value >= 0 for value in probs_out)
        assert sum(probs_out) == FP_Q16_ONE
        assert sum_out is not None and sum_out > 0


def run() -> None:
    test_null_sum_pointer_rejected()
    test_empty_count_writes_zero_sum_and_preserves_outputs()
    test_domain_failure_has_no_partial_writes_and_sum_unchanged()
    test_normalize_overflow_is_fail_fast_for_probs_but_sum_is_written()
    test_alias_modes_are_supported_with_sum_out()
    test_random_vectors_match_split_phase_reference()
    print("softmax_from_preclamped_checked_with_sum_out_reference_checks=ok")


if __name__ == "__main__":
    run()
