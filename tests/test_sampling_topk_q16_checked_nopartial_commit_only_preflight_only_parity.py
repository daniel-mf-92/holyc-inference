#!/usr/bin/env python3
"""IQ-1048 parity checks for SamplingTopKQ16CheckedNoPartialCommitOnlyPreflightOnlyParity."""

from __future__ import annotations

import random
from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1


def sampling_topk_select_indices_checked_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    k: int,
    out_index_capacity: int,
) -> tuple[int, list[int]]:
    if logits_q16 is None:
        return SAMPLING_Q16_ERR_NULL_PTR, []
    if logits_capacity < 0 or vocab_size < 0 or k < 0 or out_index_capacity < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, []
    if vocab_size > logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, []
    if k > vocab_size or k > out_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, []

    order = sorted(range(vocab_size), key=lambda idx: (-logits_q16[idx], idx))
    return SAMPLING_Q16_OK, order[:k]


def sampling_topk_q16_checked_nopartial_commit_only_preflight_only_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    k: int,
    out_index_capacity: int,
    out_score_capacity: int,
) -> tuple[int, tuple[int, int, int] | None]:
    if logits_q16 is None:
        return SAMPLING_Q16_ERR_NULL_PTR, None
    if logits_capacity < 0 or vocab_size < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if k < 0 or out_index_capacity < 0 or out_score_capacity < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if vocab_size > logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if k > vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    required_indices = k
    required_scores = k
    selected_count = k

    if required_indices > out_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_scores > out_score_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    if vocab_size:
        last_index = vocab_size - 1
        if last_index > 0x0FFFFFFFFFFFFFFF:
            return SAMPLING_Q16_ERR_OVERFLOW, None
        last_byte_offset = last_index << 3
        # emulate pointer-span overflow guard; use 0 as canonical base address
        if 0 > (U64_MAX - last_byte_offset):
            return SAMPLING_Q16_ERR_OVERFLOW, None

    return SAMPLING_Q16_OK, (required_indices, required_scores, selected_count)


def sampling_topk_q16_checked_nopartial_commit_only_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    k: int,
    out_index_capacity: int,
    out_score_capacity: int,
) -> tuple[int, tuple[int, int, int] | None]:
    status, preflight = sampling_topk_q16_checked_nopartial_commit_only_preflight_only_reference(
        logits_q16,
        logits_capacity,
        vocab_size,
        k,
        out_index_capacity,
        out_score_capacity,
    )
    if status != SAMPLING_Q16_OK:
        return status, None

    required_indices, required_scores, selected_count = preflight
    status, staged_indices = sampling_topk_select_indices_checked_reference(
        logits_q16,
        logits_capacity,
        vocab_size,
        k,
        required_indices,
    )
    if status != SAMPLING_Q16_OK:
        return status, None

    if len(staged_indices) != selected_count:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    staged_scores: list[int] = []
    for idx in staged_indices:
        if idx < 0 or idx >= vocab_size:
            return SAMPLING_Q16_ERR_BAD_PARAM, None
        staged_scores.append(logits_q16[idx])

    if len(staged_scores) != required_scores:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    return SAMPLING_Q16_OK, (required_indices, required_scores, selected_count)


def sampling_topk_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    k: int,
    out_index_capacity: int,
    out_score_capacity: int,
) -> tuple[int, tuple[int, int, int] | None]:
    status, preflight_tuple = sampling_topk_q16_checked_nopartial_commit_only_preflight_only_reference(
        logits_q16,
        logits_capacity,
        vocab_size,
        k,
        out_index_capacity,
        out_score_capacity,
    )
    if status != SAMPLING_Q16_OK:
        return status, None

    status, commit_tuple = sampling_topk_q16_checked_nopartial_commit_only_reference(
        logits_q16,
        logits_capacity,
        vocab_size,
        k,
        out_index_capacity,
        out_score_capacity,
    )
    if status != SAMPLING_Q16_OK:
        return status, None

    if preflight_tuple != commit_tuple:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    return SAMPLING_Q16_OK, preflight_tuple


def test_source_contains_iq_1048_signature_and_parity_calls() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")

    assert (
        "I32 SamplingTopKQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(" in source
    )
    assert "SamplingTopKQ16CheckedNoPartialCommitOnlyPreflightOnly(" in source
    assert "SamplingTopKQ16CheckedNoPartialCommitOnly(" in source
    assert "commit_required_indices != pre_required_indices" in source
    assert "commit_required_scores != pre_required_scores" in source
    assert "commit_selected_count != pre_selected_count" in source


def test_null_and_bad_param_contracts() -> None:
    status, tup = sampling_topk_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
        None,
        4,
        4,
        2,
        2,
        2,
    )
    assert status == SAMPLING_Q16_ERR_NULL_PTR
    assert tup is None

    logits = [10, 5, 3, -1]

    cases = [
        (-1, 4, 2, 2, 2),
        (4, -1, 2, 2, 2),
        (4, 4, -1, 2, 2),
        (4, 4, 2, -1, 2),
        (4, 5, 2, 2, 2),
        (4, 4, 5, 5, 5),
        (4, 4, 3, 2, 3),
        (4, 4, 3, 3, 2),
    ]

    for logits_capacity, vocab_size, k, out_index_capacity, out_score_capacity in cases:
        status, tup = sampling_topk_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
            logits,
            logits_capacity,
            vocab_size,
            k,
            out_index_capacity,
            out_score_capacity,
        )
        assert status == SAMPLING_Q16_ERR_BAD_PARAM
        assert tup is None


def test_deterministic_tie_break_and_tuple_outputs() -> None:
    logits = [1000, 1000, 999, 0, -100]

    status, tup = sampling_topk_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
        logits,
        logits_capacity=len(logits),
        vocab_size=len(logits),
        k=3,
        out_index_capacity=3,
        out_score_capacity=3,
    )
    assert status == SAMPLING_Q16_OK
    assert tup == (3, 3, 3)

    status, indices = sampling_topk_select_indices_checked_reference(
        logits,
        len(logits),
        len(logits),
        3,
        3,
    )
    assert status == SAMPLING_Q16_OK
    # Stable tie-break requires lower token id first when logits tie.
    assert indices == [0, 1, 2]


def test_zero_k_fast_path() -> None:
    logits = [7, 3, 1]
    status, tup = sampling_topk_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
        logits,
        logits_capacity=3,
        vocab_size=3,
        k=0,
        out_index_capacity=0,
        out_score_capacity=0,
    )
    assert status == SAMPLING_Q16_OK
    assert tup == (0, 0, 0)


def test_randomized_parity_across_geometry() -> None:
    rng = random.Random(20260422_1048)

    for _ in range(5000):
        vocab_size = rng.randint(0, 64)
        logits_capacity = vocab_size + rng.randint(0, 8)
        k = rng.randint(0, vocab_size) if vocab_size else 0

        required = k
        out_index_capacity = required + rng.randint(0, 4)
        out_score_capacity = required + rng.randint(0, 4)

        logits = [rng.randint(-200000, 200000) for _ in range(logits_capacity)]

        status, tup = sampling_topk_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
            logits,
            logits_capacity,
            vocab_size,
            k,
            out_index_capacity,
            out_score_capacity,
        )

        assert status == SAMPLING_Q16_OK
        assert tup == (k, k, k)


def test_randomized_invalid_capacities_rejected() -> None:
    rng = random.Random(20260422_1049)

    for _ in range(2500):
        vocab_size = rng.randint(1, 64)
        logits_capacity = vocab_size + rng.randint(0, 8)
        k = rng.randint(1, vocab_size)

        # Force one capacity below required.
        if rng.randint(0, 1):
            out_index_capacity = k - 1
            out_score_capacity = k + rng.randint(0, 4)
        else:
            out_index_capacity = k + rng.randint(0, 4)
            out_score_capacity = k - 1

        logits = [rng.randint(-50000, 50000) for _ in range(logits_capacity)]

        status, tup = sampling_topk_q16_checked_nopartial_commit_only_preflight_only_parity_reference(
            logits,
            logits_capacity,
            vocab_size,
            k,
            out_index_capacity,
            out_score_capacity,
        )
        assert status == SAMPLING_Q16_ERR_BAD_PARAM
        assert tup is None


if __name__ == "__main__":
    test_source_contains_iq_1048_signature_and_parity_calls()
    test_null_and_bad_param_contracts()
    test_deterministic_tie_break_and_tuple_outputs()
    test_zero_k_fast_path()
    test_randomized_parity_across_geometry()
    test_randomized_invalid_capacities_rejected()
    print("sampling_topk_q16_checked_nopartial_commit_only_preflight_only_parity=ok")
