#!/usr/bin/env python3
"""Parity harness for IQ-1168 SamplingTopPSelectCheckedNoPartial."""

from __future__ import annotations

import random
from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4

SAMPLING_Q16_ONE = 1 << 16
I64_MAX = (1 << 63) - 1


def sampling_topp_select_checked_nopartial_reference(
    sorted_probs_q16: list[int] | None,
    sorted_token_ids: list[int] | None,
    lane_capacity: int,
    lane_count: int,
    top_p_q16: int,
    out_selected_token_ids: list[int] | None,
    out_selected_capacity: int,
    out_selected_count: list[int] | None,
) -> int:
    if sorted_probs_q16 is None or sorted_token_ids is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_selected_token_ids is None or out_selected_count is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if lane_capacity < 0 or lane_count < 0 or out_selected_capacity < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if lane_count == 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if lane_count > lane_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if top_p_q16 <= 0 or top_p_q16 > SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    cumulative = 0
    prev_cumulative = 0
    prev_prob = SAMPLING_Q16_ONE
    cutoff_index = -1

    for idx in range(lane_count):
        prob = sorted_probs_q16[idx]
        tok = sorted_token_ids[idx]

        if prob < 0 or prob > SAMPLING_Q16_ONE:
            return SAMPLING_Q16_ERR_BAD_PARAM
        if tok < 0:
            return SAMPLING_Q16_ERR_BAD_PARAM
        if prob > prev_prob:
            return SAMPLING_Q16_ERR_BAD_PARAM

        if prob > I64_MAX - cumulative:
            return SAMPLING_Q16_ERR_OVERFLOW
        cumulative += prob
        if cumulative < prev_cumulative:
            return SAMPLING_Q16_ERR_OVERFLOW
        if cumulative > SAMPLING_Q16_ONE:
            return SAMPLING_Q16_ERR_BAD_PARAM

        if cutoff_index < 0 and cumulative >= top_p_q16:
            cutoff_index = idx

        prev_prob = prob
        prev_cumulative = cumulative

    if cumulative != SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if cutoff_index < 0:
        cutoff_index = lane_count - 1

    required_count = cutoff_index + 1
    if required_count > out_selected_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    cutoff_prob = sorted_probs_q16[cutoff_index]

    strict_count = 0
    while strict_count < lane_count and sorted_probs_q16[strict_count] > cutoff_prob:
        strict_count += 1

    tie_start = strict_count
    tie_end = tie_start
    while tie_end < lane_count and sorted_probs_q16[tie_end] == cutoff_prob:
        tie_end += 1

    tie_bucket = sorted_token_ids[tie_start:tie_end]
    tie_needed = required_count - strict_count
    if tie_needed < 0 or tie_needed > len(tie_bucket):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if len(set(tie_bucket)) != len(tie_bucket):
        return SAMPLING_Q16_ERR_BAD_PARAM

    staged = [0] * required_count
    for i in range(strict_count):
        staged[i] = sorted_token_ids[i]

    for i, tok in enumerate(sorted(tie_bucket)[:tie_needed]):
        staged[strict_count + i] = tok

    for i in range(required_count):
        out_selected_token_ids[i] = staged[i]
    out_selected_count[0] = required_count
    return SAMPLING_Q16_OK


def _make_sorted_probs_and_ids(rng: random.Random, lane_count: int) -> tuple[list[int], list[int]]:
    cuts = sorted(rng.randint(0, SAMPLING_Q16_ONE) for _ in range(lane_count - 1))
    probs = []
    prev = 0
    for cut in cuts:
        probs.append(cut - prev)
        prev = cut
    probs.append(SAMPLING_Q16_ONE - prev)
    probs.sort(reverse=True)

    token_ids = list(range(lane_count))
    rng.shuffle(token_ids)
    return probs, token_ids


def test_source_contains_iq1168_function_and_tie_policy() -> None:
    src = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    sig = "I32 SamplingTopPSelectCheckedNoPartial(I64 *sorted_probs_q16,"
    assert sig in src
    body = src.split(sig, 1)[1].split("// Checked in-place repetition-penalty pass", 1)[0]

    assert "tie_bucket_count" in body
    assert "tie_needed" in body
    assert "best_token_id" in body
    assert "if (token_id_candidate == best_token_id)" in body
    assert "out_selected_token_ids[lane_index] = staged_selected_ids[lane_index];" in body


def test_null_and_domain_guards_preserve_outputs() -> None:
    probs = [32768, 32768]
    ids = [10, 11]
    out_ids = [999, 888, 777]
    out_count = [123]

    assert (
        sampling_topp_select_checked_nopartial_reference(
            None,
            ids,
            2,
            2,
            32768,
            out_ids,
            len(out_ids),
            out_count,
        )
        == SAMPLING_Q16_ERR_NULL_PTR
    )
    assert out_ids == [999, 888, 777]
    assert out_count == [123]

    assert (
        sampling_topp_select_checked_nopartial_reference(
            probs,
            ids,
            2,
            0,
            32768,
            out_ids,
            len(out_ids),
            out_count,
        )
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert out_ids == [999, 888, 777]
    assert out_count == [123]


def test_tie_bucket_selects_lowest_token_ids() -> None:
    probs = [30000, 15000, 15000, 5536]
    ids = [91, 44, 7, 80]
    out_ids = [-1] * 8
    out_count = [-1]

    err = sampling_topp_select_checked_nopartial_reference(
        probs,
        ids,
        len(probs),
        len(probs),
        45000,
        out_ids,
        len(out_ids),
        out_count,
    )
    assert err == SAMPLING_Q16_OK
    assert out_count[0] == 2
    assert out_ids[:2] == [91, 7]


def test_duplicate_token_ids_inside_tie_bucket_rejected_no_partial() -> None:
    probs = [30000, 15000, 15000, 5536]
    ids = [91, 7, 7, 80]
    out_ids = [111, 222, 333]
    out_count = [9]

    err = sampling_topp_select_checked_nopartial_reference(
        probs,
        ids,
        len(probs),
        len(probs),
        45000,
        out_ids,
        len(out_ids),
        out_count,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_ids == [111, 222, 333]
    assert out_count == [9]


def test_randomized_valid_parity_reference_invariants() -> None:
    rng = random.Random(1168001)

    for _ in range(400):
        lane_count = rng.randint(1, 24)
        probs, token_ids = _make_sorted_probs_and_ids(rng, lane_count)

        top_p_q16 = rng.randint(1, SAMPLING_Q16_ONE)
        out_ids = [-(i + 1) for i in range(40)]
        out_count = [-7]

        err = sampling_topp_select_checked_nopartial_reference(
            probs,
            token_ids,
            lane_count,
            lane_count,
            top_p_q16,
            out_ids,
            40,
            out_count,
        )
        assert err == SAMPLING_Q16_OK

        chosen_n = out_count[0]
        assert 1 <= chosen_n <= lane_count

        chosen = out_ids[:chosen_n]
        assert len(chosen) == len(set(chosen))

        cutoff = chosen_n - 1
        cumulative = sum(probs[: chosen_n])
        assert cumulative >= top_p_q16
        if cutoff > 0:
            assert sum(probs[:cutoff]) < top_p_q16


def test_invalid_monotonic_or_sum_preserves_output() -> None:
    out_ids = [31, 32, 33, 34]
    out_count = [5]

    err = sampling_topp_select_checked_nopartial_reference(
        [1000, 2000, SAMPLING_Q16_ONE - 3000],
        [1, 2, 3],
        3,
        3,
        1000,
        out_ids,
        len(out_ids),
        out_count,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_ids == [31, 32, 33, 34]
    assert out_count == [5]

    err = sampling_topp_select_checked_nopartial_reference(
        [20000, 20000, 20000],
        [1, 2, 3],
        3,
        3,
        1000,
        out_ids,
        len(out_ids),
        out_count,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_ids == [31, 32, 33, 34]
    assert out_count == [5]
