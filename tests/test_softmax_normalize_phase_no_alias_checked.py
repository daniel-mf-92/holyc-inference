#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxNormalizePhaseNoAliasChecked semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_softmax_normalize_phase_checked import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    FP_Q16_ONE,
    fpq16_softmax_normalize_phase_checked_reference,
)


def fpq16_softmax_normalize_phase_no_alias_checked_reference(
    exp_lanes_q16: list[int],
    lane_count: int,
    exp_sum_q16: int,
    probs_q16: list[int],
    *,
    alias: bool = False,
) -> tuple[int, list[int]]:
    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, probs_q16[:]
    if exp_sum_q16 <= 0:
        return FP_Q16_ERR_BAD_PARAM, probs_q16[:]
    if alias:
        return FP_Q16_ERR_BAD_PARAM, probs_q16[:]

    err, normalized = fpq16_softmax_normalize_phase_checked_reference(
        exp_lanes_q16,
        lane_count,
        exp_sum_q16,
    )
    if err != FP_Q16_OK:
        return err, probs_q16[:]

    out = probs_q16[:]
    for index in range(lane_count):
        out[index] = normalized[index]
    return FP_Q16_OK, out


def test_alias_rejected() -> None:
    shared = [7, 11, 13]
    err, out = fpq16_softmax_normalize_phase_no_alias_checked_reference(
        shared,
        len(shared),
        sum(shared),
        shared,
        alias=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == shared


def test_bad_params_preserve_output() -> None:
    seed = [123, 456, 789]

    err, out = fpq16_softmax_normalize_phase_no_alias_checked_reference(
        [2, 3, 5],
        -1,
        10,
        seed,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == seed

    err, out = fpq16_softmax_normalize_phase_no_alias_checked_reference(
        [2, 3, 5],
        3,
        0,
        seed,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == seed


def test_equivalent_to_alias_permitted_path() -> None:
    rng = random.Random(2026041704)

    for _ in range(2000):
        lane_count = rng.randint(1, 32)
        exp_lanes = [rng.randint(1, 1_000_000) for _ in range(lane_count)]
        exp_sum = sum(exp_lanes)
        seed = [rng.randint(0, 2048) for _ in range(lane_count)]

        err_no_alias, out_no_alias = fpq16_softmax_normalize_phase_no_alias_checked_reference(
            exp_lanes,
            lane_count,
            exp_sum,
            seed,
        )
        err_checked, out_checked = fpq16_softmax_normalize_phase_checked_reference(
            exp_lanes,
            lane_count,
            exp_sum,
        )

        assert err_no_alias == FP_Q16_OK
        assert err_checked == FP_Q16_OK
        assert out_no_alias == out_checked
        assert sum(out_no_alias) == FP_Q16_ONE


def test_overflow_passthrough_and_no_partial_write() -> None:
    # Crafted to force overflow in shared checked normalizer path.
    exp_lanes = [1, 1]
    exp_sum = 1
    seed = [999, 1000]

    err, out = fpq16_softmax_normalize_phase_no_alias_checked_reference(
        exp_lanes,
        len(exp_lanes),
        exp_sum,
        seed,
    )

    assert err in (FP_Q16_ERR_OVERFLOW, FP_Q16_OK)
    if err == FP_Q16_ERR_OVERFLOW:
        assert out == seed


def run() -> None:
    test_alias_rejected()
    test_bad_params_preserve_output()
    test_equivalent_to_alias_permitted_path()
    test_overflow_passthrough_and_no_partial_write()
    print("softmax_normalize_phase_no_alias_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
