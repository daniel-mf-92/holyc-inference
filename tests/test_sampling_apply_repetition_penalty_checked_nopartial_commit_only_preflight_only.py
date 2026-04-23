#!/usr/bin/env python3
"""Reference checks for SamplingApplyRepetitionPenaltyCheckedNoPartialCommitOnlyPreflightOnly (IQ-1216)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_sampling_apply_repetition_penalty_checked_nopartial import (
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_NULL_PTR,
    SAMPLING_Q16_ERR_OVERFLOW,
    SAMPLING_Q16_OK,
    SAMPLING_Q16_ONE,
    sampling_apply_repetition_penalty_checked_nopartial_reference,
)
from test_sampling_apply_repetition_penalty_checked_nopartial_commit_only import (
    sampling_apply_repetition_penalty_checked_nopartial_commit_only_reference,
)


def sampling_apply_repetition_penalty_checked_nopartial_commit_only_preflight_only_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    token_history: list[int] | None,
    token_history_capacity: int,
    token_history_count: int,
    penalty_q16: int,
    out_capacity: int,
) -> tuple[int, tuple[int, int, int] | None]:
    if logits_q16 is None:
        return SAMPLING_Q16_ERR_NULL_PTR, None

    if token_history_count > 0 and token_history is None:
        return SAMPLING_Q16_ERR_NULL_PTR, None

    if logits_capacity < 0 or vocab_size < 0 or out_capacity < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if token_history_capacity < 0 or token_history_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if vocab_size > logits_capacity or vocab_size > out_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if token_history_count > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if penalty_q16 < SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    out_commit = [0 for _ in range(vocab_size)]
    status = sampling_apply_repetition_penalty_checked_nopartial_commit_only_reference(
        logits_q16,
        logits_capacity,
        vocab_size,
        token_history,
        token_history_capacity,
        token_history_count,
        penalty_q16,
        out_commit,
        out_capacity,
    )
    if status != SAMPLING_Q16_OK:
        return status, None

    out_canonical = [0 for _ in range(vocab_size)]
    status = sampling_apply_repetition_penalty_checked_nopartial_reference(
        logits_q16,
        logits_capacity,
        vocab_size,
        token_history,
        token_history_capacity,
        token_history_count,
        penalty_q16,
        out_canonical,
        out_capacity,
    )
    if status != SAMPLING_Q16_OK:
        return status, None

    if out_commit != out_canonical:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    seen: set[int] = set()
    unique_penalized_tokens = 0
    last_penalized_token_id = -1
    if token_history is not None:
        for i in range(token_history_count):
            token_id = token_history[i]
            if token_id < 0 or token_id >= vocab_size:
                return SAMPLING_Q16_ERR_BAD_PARAM, None
            if token_id in seen:
                continue
            seen.add(token_id)
            unique_penalized_tokens += 1
            last_penalized_token_id = token_id

    return SAMPLING_Q16_OK, (vocab_size, unique_penalized_tokens, last_penalized_token_id)


def test_source_contains_iq_1216_signature_and_parity_calls() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 SamplingApplyRepetitionPenaltyCheckedNoPartialCommitOnlyPreflightOnly(" in source
    assert "SamplingApplyRepetitionPenaltyCheckedNoPartialCommitOnly(" in source
    assert "SamplingApplyRepetitionPenaltyCheckedNoPartial(" in source
    assert "staged_commit_logits_q16[lane_index] !=" in source
    assert "staged_canonical_logits_q16[lane_index]" in source


def test_null_and_bad_param_contracts() -> None:
    logits = [100, -200, 300]
    history = [0, 1]

    status, tup = sampling_apply_repetition_penalty_checked_nopartial_commit_only_preflight_only_reference(
        None,
        3,
        3,
        history,
        2,
        2,
        SAMPLING_Q16_ONE,
        3,
    )
    assert status == SAMPLING_Q16_ERR_NULL_PTR
    assert tup is None

    bad_cases = [
        (-1, 3, 2, 2, SAMPLING_Q16_ONE, 3),
        (3, -1, 2, 2, SAMPLING_Q16_ONE, 3),
        (3, 4, 2, 2, SAMPLING_Q16_ONE, 3),
        (3, 3, 2, 3, SAMPLING_Q16_ONE, 3),
        (3, 3, 2, 2, SAMPLING_Q16_ONE - 1, 3),
        (3, 3, 2, 2, SAMPLING_Q16_ONE, 2),
    ]
    for logits_capacity, vocab_size, history_capacity, history_count, penalty_q16, out_capacity in bad_cases:
        status, tup = sampling_apply_repetition_penalty_checked_nopartial_commit_only_preflight_only_reference(
            logits,
            logits_capacity,
            vocab_size,
            history,
            history_capacity,
            history_count,
            penalty_q16,
            out_capacity,
        )
        assert status == SAMPLING_Q16_ERR_BAD_PARAM
        assert tup is None


def test_tuple_outputs_and_dedup_policy() -> None:
    logits = [300000, -150000, 90000, -70000, 25000, 0]
    history = [1, 3, 1, 5, 0, 5]
    status, tup = sampling_apply_repetition_penalty_checked_nopartial_commit_only_preflight_only_reference(
        logits,
        logits_capacity=len(logits),
        vocab_size=len(logits),
        token_history=history,
        token_history_capacity=len(history),
        token_history_count=len(history),
        penalty_q16=(3 * SAMPLING_Q16_ONE) // 2,
        out_capacity=len(logits),
    )
    assert status == SAMPLING_Q16_OK
    assert tup == (6, 4, 0)


def test_zero_vocab_fast_path_tuple() -> None:
    status, tup = sampling_apply_repetition_penalty_checked_nopartial_commit_only_preflight_only_reference(
        logits_q16=[],
        logits_capacity=0,
        vocab_size=0,
        token_history=[],
        token_history_capacity=0,
        token_history_count=0,
        penalty_q16=SAMPLING_Q16_ONE,
        out_capacity=0,
    )
    assert status == SAMPLING_Q16_OK
    assert tup == (0, 0, -1)


def test_randomized_parity_and_tuple_invariants() -> None:
    rng = random.Random(20260423_1216)
    for _ in range(3000):
        vocab_size = rng.randint(0, 96)
        logits_capacity = vocab_size + rng.randint(0, 8)
        out_capacity = vocab_size + rng.randint(0, 8)
        logits = [rng.randint(-1_000_000, 1_000_000) for _ in range(logits_capacity)]

        hist_count = rng.randint(0, 24)
        history = [rng.randint(0, vocab_size - 1) for _ in range(hist_count)] if vocab_size else []
        history_count = len(history)

        status, tup = sampling_apply_repetition_penalty_checked_nopartial_commit_only_preflight_only_reference(
            logits,
            logits_capacity,
            vocab_size,
            history,
            history_count,
            history_count,
            rng.randint(SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE),
            out_capacity,
        )

        if vocab_size > out_capacity:
            assert status == SAMPLING_Q16_ERR_BAD_PARAM
            assert tup is None
            continue

        assert status == SAMPLING_Q16_OK
        assert tup is not None
        required, unique_count, last_token = tup
        assert required == vocab_size
        if vocab_size == 0 or history_count == 0:
            assert unique_count == 0
            assert last_token == -1
        else:
            seen: set[int] = set()
            expected_unique = 0
            expected_last = -1
            for token_id in history:
                if token_id in seen:
                    continue
                seen.add(token_id)
                expected_unique += 1
                expected_last = token_id
            assert unique_count == expected_unique
            assert last_token == expected_last


def test_pointer_span_overflow_propagates() -> None:
    # Delegate overflow path from commit-only wrapper through out_addr overflow.
    logits = [1, 2, 3]
    status = sampling_apply_repetition_penalty_checked_nopartial_commit_only_reference(
        logits,
        logits_capacity=3,
        vocab_size=3,
        token_history=[],
        token_history_capacity=0,
        token_history_count=0,
        penalty_q16=SAMPLING_Q16_ONE,
        out_logits_q16=[0, 0, 0],
        out_capacity=3,
        out_addr=(1 << 64) - 1,
    )
    assert status == SAMPLING_Q16_ERR_OVERFLOW


if __name__ == "__main__":
    test_source_contains_iq_1216_signature_and_parity_calls()
    test_null_and_bad_param_contracts()
    test_tuple_outputs_and_dedup_policy()
    test_zero_vocab_fast_path_tuple()
    test_randomized_parity_and_tuple_invariants()
    test_pointer_span_overflow_propagates()
    print("sampling_apply_repetition_penalty_checked_nopartial_commit_only_preflight_only=ok")
