#!/usr/bin/env python3
"""Reference checks for SamplingApplyRepetitionPenaltyChecked semantics (IQ-748)."""

from __future__ import annotations

import random
from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4
SAMPLING_Q16_ERR_DOMAIN = 5

SAMPLING_Q16_SHIFT = 16
SAMPLING_Q16_ONE = 1 << SAMPLING_Q16_SHIFT

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)
U64_MAX = (1 << 64) - 1


def abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    if x == I64_MIN:
        return 1 << 63
    return -x


def apply_sign_checked(magnitude: int, is_negative: bool) -> tuple[int, int]:
    if not is_negative:
        if magnitude > I64_MAX:
            return SAMPLING_Q16_ERR_OVERFLOW, 0
        return SAMPLING_Q16_OK, magnitude

    if magnitude > (1 << 63):
        return SAMPLING_Q16_ERR_OVERFLOW, 0
    if magnitude == (1 << 63):
        return SAMPLING_Q16_OK, I64_MIN
    return SAMPLING_Q16_OK, -magnitude


def fpq16_mul_div_rounded_checked(a_q16: int, b_q16: int, d_q16: int) -> tuple[int, int]:
    if d_q16 == 0:
        return SAMPLING_Q16_ERR_DOMAIN, 0

    abs_a = abs_to_u64(a_q16)
    abs_b = abs_to_u64(b_q16)
    abs_d = abs_to_u64(d_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0) ^ (d_q16 < 0)

    if abs_a and abs_b and abs_a > (U64_MAX // abs_b):
        return SAMPLING_Q16_ERR_OVERFLOW, 0

    abs_num = abs_a * abs_b
    q = abs_num // abs_d
    r = abs_num % abs_d

    limit = I64_MAX
    if is_negative:
        limit = 1 << 63

    if q > limit:
        return SAMPLING_Q16_ERR_OVERFLOW, 0

    if r >= ((abs_d + 1) >> 1):
        if q == limit:
            return SAMPLING_Q16_ERR_OVERFLOW, 0
        q += 1

    return apply_sign_checked(q, is_negative)


def sampling_apply_repetition_penalty_checked_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    token_history: list[int] | None,
    token_history_capacity: int,
    token_history_count: int,
    penalty_q16: int,
    logits_addr: int = 0,
    history_addr: int = 0,
) -> int:
    if logits_q16 is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if token_history_count > 0 and token_history is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if logits_capacity < 0 or vocab_size < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if token_history_capacity < 0 or token_history_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if vocab_size > logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if token_history_count > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if penalty_q16 < SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if vocab_size == 0 or token_history_count == 0:
        return SAMPLING_Q16_OK

    if len(logits_q16) < vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if token_history is None or len(token_history) < token_history_count:
        return SAMPLING_Q16_ERR_BAD_PARAM

    last_index = vocab_size - 1
    if last_index > 0x0FFFFFFFFFFFFFFF:
        return SAMPLING_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if logits_addr > (U64_MAX - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW

    last_index = token_history_count - 1
    if last_index > 0x0FFFFFFFFFFFFFFF:
        return SAMPLING_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if history_addr > (U64_MAX - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW

    # Preflight: validate ids and arithmetic without writes.
    seen: set[int] = set()
    for i in range(token_history_count):
        token_id = token_history[i]
        if token_id < 0 or token_id >= vocab_size:
            return SAMPLING_Q16_ERR_BAD_PARAM
        if token_id in seen:
            continue
        seen.add(token_id)

        base = logits_q16[token_id]
        if base < 0:
            err, _ = fpq16_mul_div_rounded_checked(base, penalty_q16, SAMPLING_Q16_ONE)
        else:
            err, _ = fpq16_mul_div_rounded_checked(base, SAMPLING_Q16_ONE, penalty_q16)
        if err != SAMPLING_Q16_OK:
            return err

    # Commit.
    seen.clear()
    for i in range(token_history_count):
        token_id = token_history[i]
        if token_id in seen:
            continue
        seen.add(token_id)

        base = logits_q16[token_id]
        if base < 0:
            err, adjusted = fpq16_mul_div_rounded_checked(base, penalty_q16, SAMPLING_Q16_ONE)
        else:
            err, adjusted = fpq16_mul_div_rounded_checked(base, SAMPLING_Q16_ONE, penalty_q16)
        if err != SAMPLING_Q16_OK:
            return err
        logits_q16[token_id] = adjusted

    return SAMPLING_Q16_OK


def test_source_contains_signature_and_penalty_rule() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 SamplingApplyRepetitionPenaltyChecked(" in source
    assert "if (base_logit_q16 < 0)" in source
    assert "FPQ16MulDivRoundedChecked(base_logit_q16," in source
    assert "if (token_history[prior_index] == token_id)" in source


def test_null_and_bad_param_contracts() -> None:
    logits = [10, 20, 30]
    history = [0, 1]

    assert (
        sampling_apply_repetition_penalty_checked_reference(
            None, 3, 3, history, 2, 2, SAMPLING_Q16_ONE
        )
        == SAMPLING_Q16_ERR_NULL_PTR
    )
    assert (
        sampling_apply_repetition_penalty_checked_reference(
            logits, 3, 3, None, 2, 1, SAMPLING_Q16_ONE
        )
        == SAMPLING_Q16_ERR_NULL_PTR
    )
    assert (
        sampling_apply_repetition_penalty_checked_reference(
            logits, -1, 3, history, 2, 2, SAMPLING_Q16_ONE
        )
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_apply_repetition_penalty_checked_reference(
            logits, 3, 4, history, 2, 2, SAMPLING_Q16_ONE
        )
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_apply_repetition_penalty_checked_reference(
            logits, 3, 3, history, 1, 2, SAMPLING_Q16_ONE
        )
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_apply_repetition_penalty_checked_reference(
            logits, 3, 3, history, 2, 2, 0
        )
        == SAMPLING_Q16_ERR_BAD_PARAM
    )
    assert (
        sampling_apply_repetition_penalty_checked_reference(
            logits, 3, 3, history, 2, 2, SAMPLING_Q16_ONE - 1
        )
        == SAMPLING_Q16_ERR_BAD_PARAM
    )


def test_positive_negative_and_duplicate_history_behavior() -> None:
    logits = [
        4 * SAMPLING_Q16_ONE,
        -(2 * SAMPLING_Q16_ONE),
        7 * SAMPLING_Q16_ONE,
        1 * SAMPLING_Q16_ONE,
    ]
    history = [1, 0, 1, 0, 3]
    penalty = int(1.5 * SAMPLING_Q16_ONE)

    err = sampling_apply_repetition_penalty_checked_reference(
        logits,
        logits_capacity=len(logits),
        vocab_size=len(logits),
        token_history=history,
        token_history_capacity=len(history),
        token_history_count=len(history),
        penalty_q16=penalty,
    )
    assert err == SAMPLING_Q16_OK

    err0, expected0 = fpq16_mul_div_rounded_checked(4 * SAMPLING_Q16_ONE, SAMPLING_Q16_ONE, penalty)
    err1, expected1 = fpq16_mul_div_rounded_checked(
        -(2 * SAMPLING_Q16_ONE), penalty, SAMPLING_Q16_ONE
    )
    err3, expected3 = fpq16_mul_div_rounded_checked(1 * SAMPLING_Q16_ONE, SAMPLING_Q16_ONE, penalty)
    assert err0 == SAMPLING_Q16_OK
    assert err1 == SAMPLING_Q16_OK
    assert err3 == SAMPLING_Q16_OK
    assert logits[0] == expected0
    assert logits[1] == expected1
    assert logits[2] == 7 * SAMPLING_Q16_ONE
    assert logits[3] == expected3


def test_no_partial_write_on_preflight_failure() -> None:
    logits = [SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE, 3 * SAMPLING_Q16_ONE]
    original = list(logits)
    history = [0, 99]

    err = sampling_apply_repetition_penalty_checked_reference(
        logits,
        logits_capacity=3,
        vocab_size=3,
        token_history=history,
        token_history_capacity=2,
        token_history_count=2,
        penalty_q16=(3 * SAMPLING_Q16_ONE) // 2,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert logits == original


def test_pointer_overflow_and_randomized_parity() -> None:
    logits = [100, 200, 300]
    history = [0, 1, 2]
    snapshot = list(logits)

    err = sampling_apply_repetition_penalty_checked_reference(
        logits,
        logits_capacity=3,
        vocab_size=3,
        token_history=history,
        token_history_capacity=3,
        token_history_count=3,
        penalty_q16=SAMPLING_Q16_ONE,
        logits_addr=U64_MAX - 15,
    )
    assert err == SAMPLING_Q16_ERR_OVERFLOW
    assert logits == snapshot

    rng = random.Random(20260420_748)
    for _ in range(3500):
        vocab_size = rng.randint(1, 128)
        logits_capacity = vocab_size + rng.randint(0, 8)
        history_count = rng.randint(0, 2 * vocab_size)
        history_capacity = history_count + rng.randint(0, 8)

        logits = [rng.randint(-(1 << 42), 1 << 42) for _ in range(logits_capacity)]
        logits_before = list(logits)
        history = [rng.randint(0, vocab_size - 1) for _ in range(history_capacity)]
        penalty_q16 = rng.randint(SAMPLING_Q16_ONE, 4 * SAMPLING_Q16_ONE)

        err = sampling_apply_repetition_penalty_checked_reference(
            logits,
            logits_capacity,
            vocab_size,
            history,
            history_capacity,
            history_count,
            penalty_q16,
        )
        assert err == SAMPLING_Q16_OK

        seen = set()
        for idx in range(history_count):
            tok = history[idx]
            if tok in seen:
                continue
            seen.add(tok)
            base = logits_before[tok]
            if base < 0:
                err_lane, expected = fpq16_mul_div_rounded_checked(base, penalty_q16, SAMPLING_Q16_ONE)
            else:
                err_lane, expected = fpq16_mul_div_rounded_checked(base, SAMPLING_Q16_ONE, penalty_q16)
            assert err_lane == SAMPLING_Q16_OK
            assert logits[tok] == expected


if __name__ == "__main__":
    test_source_contains_signature_and_penalty_rule()
    test_null_and_bad_param_contracts()
    test_positive_negative_and_duplicate_history_behavior()
    test_no_partial_write_on_preflight_failure()
    test_pointer_overflow_and_randomized_parity()
    print("sampling_apply_repetition_penalty_checked_reference_checks=ok")
