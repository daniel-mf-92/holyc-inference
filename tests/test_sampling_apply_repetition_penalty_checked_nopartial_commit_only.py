#!/usr/bin/env python3
"""Reference checks for SamplingApplyRepetitionPenaltyCheckedNoPartialCommitOnly (IQ-1215)."""

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
    U64_MAX,
    sampling_apply_repetition_penalty_checked_nopartial_reference,
)


def sampling_apply_repetition_penalty_checked_nopartial_commit_only_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    token_history: list[int] | None,
    token_history_capacity: int,
    token_history_count: int,
    penalty_q16: int,
    out_logits_q16: list[int] | None,
    out_capacity: int,
    out_addr: int = 0,
) -> int:
    if out_logits_q16 is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if vocab_size < 0 or out_capacity < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if vocab_size > out_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if vocab_size == 0:
        return SAMPLING_Q16_OK

    if len(out_logits_q16) < vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM

    last_index = vocab_size - 1
    if last_index > 0x0FFFFFFFFFFFFFFF:
        return SAMPLING_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if out_addr > (U64_MAX - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW

    staged = [0 for _ in range(vocab_size)]

    err = sampling_apply_repetition_penalty_checked_nopartial_reference(
        logits_q16,
        logits_capacity,
        vocab_size,
        token_history,
        token_history_capacity,
        token_history_count,
        penalty_q16,
        staged,
        vocab_size,
    )
    if err != SAMPLING_Q16_OK:
        return err

    for i in range(vocab_size):
        out_logits_q16[i] = staged[i]

    return SAMPLING_Q16_OK


def test_source_contains_commit_only_wrapper_and_parity_guard() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 SamplingApplyRepetitionPenaltyCheckedNoPartialCommitOnly(" in source
    assert "status = SamplingApplyRepetitionPenaltyCheckedNoPartial(" in source
    assert "snapshot_out_logits_q16 != out_logits_q16" in source


def test_success_matches_nopartial_reference() -> None:
    logits = [300000, -150000, 90000, -70000, 25000, 0]
    out = [11, 22, 33, 44, 55, 66]
    history = [1, 3, 1, 5, 0]
    penalty = (3 * SAMPLING_Q16_ONE) // 2

    err = sampling_apply_repetition_penalty_checked_nopartial_commit_only_reference(
        logits,
        logits_capacity=len(logits),
        vocab_size=len(logits),
        token_history=history,
        token_history_capacity=len(history),
        token_history_count=len(history),
        penalty_q16=penalty,
        out_logits_q16=out,
        out_capacity=len(out),
    )
    assert err == SAMPLING_Q16_OK

    expected = [7, 8, 9, 10, 11, 12]
    err2 = sampling_apply_repetition_penalty_checked_nopartial_reference(
        logits,
        logits_capacity=len(logits),
        vocab_size=len(logits),
        token_history=history,
        token_history_capacity=len(history),
        token_history_count=len(history),
        penalty_q16=penalty,
        out_logits_q16=expected,
        out_capacity=len(expected),
    )
    assert err2 == SAMPLING_Q16_OK
    assert out == expected


def test_no_partial_publish_on_delegate_failure() -> None:
    logits = [1000, -2000, 3000, -4000]
    out = [42, 42, 42, 42]
    out_before = list(out)

    err = sampling_apply_repetition_penalty_checked_nopartial_commit_only_reference(
        logits,
        logits_capacity=4,
        vocab_size=4,
        token_history=[0, 7],
        token_history_capacity=2,
        token_history_count=2,
        penalty_q16=SAMPLING_Q16_ONE,
        out_logits_q16=out,
        out_capacity=4,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out == out_before


def test_bad_param_and_pointer_span_paths() -> None:
    logits = [1, 2, 3]
    out = [9, 9, 9]

    err = sampling_apply_repetition_penalty_checked_nopartial_commit_only_reference(
        logits,
        logits_capacity=3,
        vocab_size=3,
        token_history=[],
        token_history_capacity=0,
        token_history_count=0,
        penalty_q16=SAMPLING_Q16_ONE,
        out_logits_q16=out,
        out_capacity=2,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM

    err = sampling_apply_repetition_penalty_checked_nopartial_commit_only_reference(
        logits,
        logits_capacity=3,
        vocab_size=3,
        token_history=[],
        token_history_capacity=0,
        token_history_count=0,
        penalty_q16=SAMPLING_Q16_ONE,
        out_logits_q16=out,
        out_capacity=3,
        out_addr=U64_MAX,
    )
    assert err == SAMPLING_Q16_ERR_OVERFLOW


def test_randomized_parity() -> None:
    rng = random.Random(20260423_1215)
    for _ in range(2000):
        vocab_size = rng.randint(1, 64)
        logits = [rng.randint(-1_000_000, 1_000_000) for _ in range(vocab_size)]
        out = [rng.randint(-7, 7) for _ in range(vocab_size)]
        out2 = list(out)
        hist_count = rng.randint(0, 16)
        history = [rng.randint(0, vocab_size - 1) for _ in range(hist_count)]
        penalty = rng.randint(SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE)

        err = sampling_apply_repetition_penalty_checked_nopartial_commit_only_reference(
            logits,
            logits_capacity=vocab_size,
            vocab_size=vocab_size,
            token_history=history,
            token_history_capacity=hist_count,
            token_history_count=hist_count,
            penalty_q16=penalty,
            out_logits_q16=out,
            out_capacity=vocab_size,
        )
        assert err == SAMPLING_Q16_OK

        err2 = sampling_apply_repetition_penalty_checked_nopartial_reference(
            logits,
            logits_capacity=vocab_size,
            vocab_size=vocab_size,
            token_history=history,
            token_history_capacity=hist_count,
            token_history_count=hist_count,
            penalty_q16=penalty,
            out_logits_q16=out2,
            out_capacity=vocab_size,
        )
        assert err2 == SAMPLING_Q16_OK
        assert out == out2


if __name__ == "__main__":
    test_source_contains_commit_only_wrapper_and_parity_guard()
    test_success_matches_nopartial_reference()
    test_no_partial_publish_on_delegate_failure()
    test_bad_param_and_pointer_span_paths()
    test_randomized_parity()
    print("sampling_apply_repetition_penalty_checked_nopartial_commit_only_reference_checks=ok")
