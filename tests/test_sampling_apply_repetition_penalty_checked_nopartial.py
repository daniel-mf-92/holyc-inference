#!/usr/bin/env python3
"""Reference checks for SamplingApplyRepetitionPenaltyCheckedNoPartial (IQ-1214)."""

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
) -> tuple[int, list[int] | None]:
    if logits_q16 is None:
        return SAMPLING_Q16_ERR_NULL_PTR, None
    if token_history_count > 0 and token_history is None:
        return SAMPLING_Q16_ERR_NULL_PTR, None

    if logits_capacity < 0 or vocab_size < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if token_history_capacity < 0 or token_history_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if vocab_size > logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if token_history_count > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if penalty_q16 < SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    if vocab_size == 0 or token_history_count == 0:
        return SAMPLING_Q16_OK, list(logits_q16)

    if len(logits_q16) < vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if token_history is None or len(token_history) < token_history_count:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    last_index = vocab_size - 1
    if last_index > 0x0FFFFFFFFFFFFFFF:
        return SAMPLING_Q16_ERR_OVERFLOW, None
    last_byte_offset = last_index << 3
    if logits_addr > (U64_MAX - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW, None

    last_index = token_history_count - 1
    if last_index > 0x0FFFFFFFFFFFFFFF:
        return SAMPLING_Q16_ERR_OVERFLOW, None
    last_byte_offset = last_index << 3
    if history_addr > (U64_MAX - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW, None

    out = list(logits_q16)

    seen: set[int] = set()
    for i in range(token_history_count):
        token_id = token_history[i]
        if token_id < 0 or token_id >= vocab_size:
            return SAMPLING_Q16_ERR_BAD_PARAM, None
        if token_id in seen:
            continue
        seen.add(token_id)

        base = out[token_id]
        if base < 0:
            err, adjusted = fpq16_mul_div_rounded_checked(base, penalty_q16, SAMPLING_Q16_ONE)
        else:
            err, adjusted = fpq16_mul_div_rounded_checked(base, SAMPLING_Q16_ONE, penalty_q16)
        if err != SAMPLING_Q16_OK:
            return err, None
        out[token_id] = adjusted

    return SAMPLING_Q16_OK, out


def sampling_apply_repetition_penalty_checked_nopartial_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    token_history: list[int] | None,
    token_history_capacity: int,
    token_history_count: int,
    penalty_q16: int,
    out_logits_q16: list[int] | None,
    out_capacity: int,
    logits_addr: int = 0,
    history_addr: int = 0,
    out_addr: int = 0,
) -> int:
    if logits_q16 is None or out_logits_q16 is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if token_history_count > 0 and token_history is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if logits_capacity < 0 or vocab_size < 0 or out_capacity < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if token_history_capacity < 0 or token_history_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if vocab_size > logits_capacity or vocab_size > out_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if token_history_count > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if penalty_q16 < SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if vocab_size == 0:
        return SAMPLING_Q16_OK

    if len(logits_q16) < vocab_size or len(out_logits_q16) < vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM

    last_index = vocab_size - 1
    if last_index > 0x0FFFFFFFFFFFFFFF:
        return SAMPLING_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if logits_addr > (U64_MAX - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW
    if out_addr > (U64_MAX - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW

    if token_history_count:
        last_index = token_history_count - 1
        if last_index > 0x0FFFFFFFFFFFFFFF:
            return SAMPLING_Q16_ERR_OVERFLOW
        last_byte_offset = last_index << 3
        if history_addr > (U64_MAX - last_byte_offset):
            return SAMPLING_Q16_ERR_OVERFLOW

    staged = list(logits_q16[:vocab_size])
    err, staged_adjusted = sampling_apply_repetition_penalty_checked_reference(
        staged,
        vocab_size,
        vocab_size,
        token_history,
        token_history_capacity,
        token_history_count,
        penalty_q16,
    )
    if err != SAMPLING_Q16_OK:
        return err
    assert staged_adjusted is not None

    for i in range(vocab_size):
        out_logits_q16[i] = staged_adjusted[i]

    return SAMPLING_Q16_OK


def test_source_contains_wrapper_and_snapshot_guards() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 SamplingApplyRepetitionPenaltyCheckedNoPartial(" in source
    assert "status = SamplingApplyRepetitionPenaltyChecked(staged_logits_q16," in source
    assert "snapshot_logits_q16 != logits_q16" in source
    assert "snapshot_out_logits_q16 != out_logits_q16" in source


def test_success_matches_canonical_transform() -> None:
    logits = [300000, -150000, 90000, -70000, 25000, 0]
    out = [111, 222, 333, 444, 555, 666]
    history = [1, 3, 1, 5, 0]
    penalty = (3 * SAMPLING_Q16_ONE) // 2

    err = sampling_apply_repetition_penalty_checked_nopartial_reference(
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

    err2, expected = sampling_apply_repetition_penalty_checked_reference(
        logits,
        logits_capacity=len(logits),
        vocab_size=len(logits),
        token_history=history,
        token_history_capacity=len(history),
        token_history_count=len(history),
        penalty_q16=penalty,
    )
    assert err2 == SAMPLING_Q16_OK
    assert out == expected


def test_no_partial_publish_on_failure() -> None:
    logits = [1000, -2000, 3000, -4000]
    out = [42, 42, 42, 42]
    out_before = list(out)

    # invalid history token id forces failure after validation path
    err = sampling_apply_repetition_penalty_checked_nopartial_reference(
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


def test_randomized_invariants() -> None:
    rng = random.Random(20260423_1214)
    for _ in range(2000):
        vocab_size = rng.randint(1, 64)
        logits = [rng.randint(-1_000_000, 1_000_000) for _ in range(vocab_size)]
        out = [rng.randint(-7, 7) for _ in range(vocab_size)]
        hist_count = rng.randint(0, 16)
        history = [rng.randint(0, vocab_size - 1) for _ in range(hist_count)]
        penalty = rng.randint(SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE)

        err = sampling_apply_repetition_penalty_checked_nopartial_reference(
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

        err2, expected = sampling_apply_repetition_penalty_checked_reference(
            logits,
            logits_capacity=vocab_size,
            vocab_size=vocab_size,
            token_history=history,
            token_history_capacity=hist_count,
            token_history_count=hist_count,
            penalty_q16=penalty,
        )
        assert err2 == SAMPLING_Q16_OK
        assert out == expected


def test_bad_param_and_pointer_span_paths() -> None:
    logits = [1, 2, 3]
    out = [9, 9, 9]

    err = sampling_apply_repetition_penalty_checked_nopartial_reference(
        logits,
        logits_capacity=3,
        vocab_size=3,
        token_history=[],
        token_history_capacity=0,
        token_history_count=0,
        penalty_q16=SAMPLING_Q16_ONE - 1,
        out_logits_q16=out,
        out_capacity=3,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM

    err = sampling_apply_repetition_penalty_checked_nopartial_reference(
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


if __name__ == "__main__":
    test_source_contains_wrapper_and_snapshot_guards()
    test_success_matches_canonical_transform()
    test_no_partial_publish_on_failure()
    test_randomized_invariants()
    test_bad_param_and_pointer_span_paths()
    print("sampling_apply_repetition_penalty_checked_nopartial_reference_checks=ok")
