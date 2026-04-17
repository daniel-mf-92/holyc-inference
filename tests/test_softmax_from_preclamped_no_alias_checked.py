#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxFromPreclampedNoAliasChecked semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_softmax_exp_phase_from_preclamped_no_alias_checked import (
    EXP_Q16_MAX_INPUT,
    EXP_Q16_MIN_INPUT,
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    FP_Q16_ONE,
    fpq16_softmax_exp_phase_from_preclamped_no_alias_checked_reference,
)
from test_softmax_normalize_phase_checked import fpq16_softmax_normalize_phase_checked_reference
from test_softmax_normalize_phase_no_alias_checked import (
    fpq16_softmax_normalize_phase_no_alias_checked_reference,
)
from test_softmax_from_preclamped_no_alias_checked_with_sum_out import (
    fpq16_softmax_from_preclamped_no_alias_checked_with_sum_out_reference,
)


def fpq16_softmax_from_preclamped_no_alias_checked_reference(
    preclamped_logits_q16: list[int],
    exp_lanes_q16: list[int],
    lane_count: int,
    probs_q16: list[int],
    *,
    logits_exp_alias: bool = False,
    logits_probs_alias: bool = False,
    exp_probs_alias: bool = False,
) -> tuple[int, list[int], list[int]]:
    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, exp_lanes_q16[:], probs_q16[:]

    if logits_exp_alias or logits_probs_alias or exp_probs_alias:
        return FP_Q16_ERR_BAD_PARAM, exp_lanes_q16[:], probs_q16[:]

    exp_err, exp_out, exp_sum_q16 = fpq16_softmax_exp_phase_from_preclamped_no_alias_checked_reference(
        preclamped_logits_q16,
        exp_lanes_q16,
        lane_count,
    )
    if exp_err != FP_Q16_OK:
        return exp_err, exp_lanes_q16[:], probs_q16[:]

    if lane_count == 0:
        return FP_Q16_OK, exp_out, probs_q16[:]

    preflight_err, _ = fpq16_softmax_normalize_phase_checked_reference(
        exp_out,
        lane_count,
        exp_sum_q16,
    )
    if preflight_err != FP_Q16_OK:
        return preflight_err, exp_out, probs_q16[:]

    norm_err, probs_out = fpq16_softmax_normalize_phase_no_alias_checked_reference(
        exp_out,
        lane_count,
        exp_sum_q16,
        probs_q16,
        alias=False,
    )
    if norm_err != FP_Q16_OK:
        return norm_err, exp_out, probs_q16[:]

    return FP_Q16_OK, exp_out, probs_out


def test_alias_rejection_all_pairs() -> None:
    logits = [0, 1, 2]
    exp_seed = [11, 22, 33]
    probs_seed = [44, 55, 66]

    err, exp_out, probs_out = fpq16_softmax_from_preclamped_no_alias_checked_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
        logits_exp_alias=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_out == exp_seed
    assert probs_out == probs_seed

    err, exp_out, probs_out = fpq16_softmax_from_preclamped_no_alias_checked_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
        logits_probs_alias=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_out == exp_seed
    assert probs_out == probs_seed

    err, exp_out, probs_out = fpq16_softmax_from_preclamped_no_alias_checked_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
        exp_probs_alias=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_out == exp_seed
    assert probs_out == probs_seed


def test_domain_failure_no_partial_writes() -> None:
    logits = [0, EXP_Q16_MAX_INPUT + 1, 1]
    exp_seed = [101, 202, 303]
    probs_seed = [404, 505, 606]

    err, exp_out, probs_out = fpq16_softmax_from_preclamped_no_alias_checked_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
    )

    assert err == FP_Q16_ERR_BAD_PARAM
    assert exp_out == exp_seed
    assert probs_out == probs_seed


def test_normalize_overflow_is_fail_fast_for_probs() -> None:
    logits = [EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT]
    exp_seed = [1234, 5678]
    probs_seed = [91011, 121314]

    err, exp_out, probs_out = fpq16_softmax_from_preclamped_no_alias_checked_reference(
        logits,
        exp_seed,
        len(logits),
        probs_seed,
    )

    assert err == FP_Q16_ERR_OVERFLOW
    assert exp_out != exp_seed
    assert probs_out == probs_seed


def test_random_vectors_match_split_phase_no_alias_reference() -> None:
    rng = random.Random(2026041716)

    for _ in range(1500):
        lane_count = rng.randint(1, 24)
        logits = [rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT) for _ in range(lane_count)]
        exp_seed = [rng.randint(0, 4096) for _ in range(lane_count)]
        probs_seed = [rng.randint(0, 4096) for _ in range(lane_count)]

        err, exp_out, probs_out = fpq16_softmax_from_preclamped_no_alias_checked_reference(
            logits,
            exp_seed,
            lane_count,
            probs_seed,
        )

        if err == FP_Q16_ERR_OVERFLOW:
            # Saturation-heavy vectors can overflow normalization mul/div.
            assert probs_out == probs_seed
            continue

        assert err == FP_Q16_OK
        assert len(exp_out) == lane_count
        assert len(probs_out) == lane_count
        assert all(value >= 0 for value in probs_out)
        assert sum(probs_out) == FP_Q16_ONE


def test_random_vectors_match_sum_out_variant_outputs() -> None:
    rng = random.Random(20260417175)

    for _ in range(1200):
        lane_count = rng.randint(0, 24)
        logits = [rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT) for _ in range(max(1, lane_count))]
        exp_seed = [rng.randint(0, 4096) for _ in range(max(1, lane_count))]
        probs_seed = [rng.randint(0, 4096) for _ in range(max(1, lane_count))]

        err_a, exp_a, probs_a = fpq16_softmax_from_preclamped_no_alias_checked_reference(
            logits,
            exp_seed,
            lane_count,
            probs_seed,
        )
        err_b, exp_b, probs_b, sum_b = (
            fpq16_softmax_from_preclamped_no_alias_checked_with_sum_out_reference(
                logits,
                exp_seed,
                lane_count,
                probs_seed,
                777,
            )
        )

        assert err_a == err_b
        assert exp_a == exp_b
        assert probs_a == probs_b

        if err_b == FP_Q16_OK and lane_count > 0:
            assert sum_b == sum(exp_b)


def run() -> None:
    test_alias_rejection_all_pairs()
    test_domain_failure_no_partial_writes()
    test_normalize_overflow_is_fail_fast_for_probs()
    test_random_vectors_match_split_phase_no_alias_reference()
    test_random_vectors_match_sum_out_variant_outputs()
    print("softmax_from_preclamped_no_alias_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
