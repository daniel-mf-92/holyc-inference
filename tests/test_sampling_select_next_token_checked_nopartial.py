#!/usr/bin/env python3
"""Reference checks for SamplingSelectNextTokenCheckedNoPartial (IQ-1217)."""

from __future__ import annotations

import math
import random
from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2

SAMPLING_Q16_SHIFT = 16
SAMPLING_Q16_ONE = 1 << SAMPLING_Q16_SHIFT


def fpq16_mul_div_rounded_checked(a_q16: int, b_q16: int, d_q16: int) -> tuple[int, int]:
    if d_q16 == 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, 0
    num = a_q16 * b_q16
    q = abs(num) // abs(d_q16)
    r = abs(num) % abs(d_q16)
    if r >= ((abs(d_q16) + 1) >> 1):
        q += 1
    sign = -1 if (num < 0) ^ (d_q16 < 0) else 1
    return SAMPLING_Q16_OK, sign * q


def apply_repetition_penalty(logits_q16: list[int], history: list[int], vocab_size: int, penalty_q16: int) -> int:
    seen: set[int] = set()
    for token_id in history:
        if token_id < 0 or token_id >= vocab_size:
            return SAMPLING_Q16_ERR_BAD_PARAM
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


def apply_temperature(logits_q16: list[int], temperature_q16: int) -> int:
    for i, v in enumerate(logits_q16):
        err, scaled = fpq16_mul_div_rounded_checked(v, SAMPLING_Q16_ONE, temperature_q16)
        if err != SAMPLING_Q16_OK:
            return err
        logits_q16[i] = scaled
    return SAMPLING_Q16_OK


def stable_topk_indices(logits_q16: list[int], k: int) -> list[int]:
    return sorted(range(len(logits_q16)), key=lambda i: (-logits_q16[i], i))[:k]


def softmax_q16(logits_q16: list[int]) -> list[int]:
    if not logits_q16:
        return []

    mx = max(logits_q16)
    exps = [math.exp((v - mx) / SAMPLING_Q16_ONE) for v in logits_q16]
    total = sum(exps)

    raw = [int((e / total) * SAMPLING_Q16_ONE) for e in exps]
    rem = SAMPLING_Q16_ONE - sum(raw)
    order = sorted(range(len(raw)), key=lambda i: (-exps[i], i))
    for i in range(max(0, rem)):
        raw[order[i % len(order)]] += 1

    for i in range(1, len(raw)):
        if raw[i] > raw[i - 1]:
            raw[i] = raw[i - 1]
    raw[0] += SAMPLING_Q16_ONE - sum(raw)
    return raw


def top_p_cutoff(probs_q16: list[int], top_p_q16: int) -> tuple[int, int]:
    cumulative = 0
    prev = SAMPLING_Q16_ONE
    for i, p in enumerate(probs_q16):
        if p < 0 or p > SAMPLING_Q16_ONE or p > prev:
            return SAMPLING_Q16_ERR_BAD_PARAM, -1
        cumulative += p
        if cumulative >= top_p_q16:
            return SAMPLING_Q16_OK, i
        prev = p

    if cumulative != SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1
    return SAMPLING_Q16_OK, len(probs_q16) - 1


def top_p_sample_index(probs_q16: list[int], prefix_len: int, random_q16: int) -> int:
    prefix_mass = sum(probs_q16[:prefix_len])
    threshold = (random_q16 * prefix_mass) >> SAMPLING_Q16_SHIFT
    cumulative = 0
    for i in range(prefix_len):
        cumulative += probs_q16[i]
        if threshold < cumulative:
            return i
    return prefix_len - 1


def sampling_select_next_token_checked_reference(
    logits_q16: list[int],
    vocab_size: int,
    token_history: list[int],
    temperature_q16: int,
    top_k: int,
    top_p_q16: int,
    repetition_penalty_q16: int,
    random_q16: int,
) -> tuple[int, int, int]:
    if vocab_size <= 0 or top_k <= 0 or top_k > vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1, -1
    if temperature_q16 <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1, -1
    if top_p_q16 <= 0 or top_p_q16 > SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1, -1
    if repetition_penalty_q16 < SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1, -1
    if random_q16 < 0 or random_q16 >= SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1, -1

    stage = logits_q16[:vocab_size]

    err = apply_repetition_penalty(stage, token_history, vocab_size, repetition_penalty_q16)
    if err != SAMPLING_Q16_OK:
        return err, -1, -1
    err = apply_temperature(stage, temperature_q16)
    if err != SAMPLING_Q16_OK:
        return err, -1, -1

    topk_idx = stable_topk_indices(stage, top_k)
    topk_logits = [stage[i] for i in topk_idx]
    topk_probs = softmax_q16(topk_logits)

    err, cutoff = top_p_cutoff(topk_probs, top_p_q16)
    if err != SAMPLING_Q16_OK:
        return err, -1, -1

    sampled_rank = top_p_sample_index(topk_probs, cutoff + 1, random_q16)
    return SAMPLING_Q16_OK, topk_idx[sampled_rank], topk_probs[sampled_rank]


def sampling_select_next_token_checked_nopartial_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    token_history: list[int] | None,
    token_history_capacity: int,
    token_history_count: int,
    temperature_q16: int,
    top_k: int,
    top_p_q16: int,
    repetition_penalty_q16: int,
    random_q16: int,
    out_token_id: list[int],
    out_token_prob_q16: list[int],
) -> int:
    if logits_q16 is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if token_history_count > 0 and token_history is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if not out_token_id or not out_token_prob_q16:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_token_id is out_token_prob_q16:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if logits_capacity < 0 or vocab_size < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if token_history_capacity < 0 or token_history_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if token_history_count > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if vocab_size > logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if token_history is None:
        history = []
    else:
        history = token_history[:token_history_count]

    err, token_id, token_prob_q16 = sampling_select_next_token_checked_reference(
        logits_q16=logits_q16,
        vocab_size=vocab_size,
        token_history=history,
        temperature_q16=temperature_q16,
        top_k=top_k,
        top_p_q16=top_p_q16,
        repetition_penalty_q16=repetition_penalty_q16,
        random_q16=random_q16,
    )
    if err != SAMPLING_Q16_OK:
        return err

    out_token_id[0] = token_id
    out_token_prob_q16[0] = token_prob_q16
    return SAMPLING_Q16_OK


def test_source_contains_no_partial_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 SamplingSelectNextTokenCheckedNoPartial(" in source
    assert "status = SamplingSelectNextTokenChecked(" in source
    assert "*out_token_prob_q16 = staged_token_prob_q16;" in source


def test_success_publishes_token_and_probability() -> None:
    logits = [10 * SAMPLING_Q16_ONE, 9 * SAMPLING_Q16_ONE, 5 * SAMPLING_Q16_ONE, 0]
    out_tok = [777]
    out_prob = [888]

    err = sampling_select_next_token_checked_nopartial_reference(
        logits_q16=logits,
        logits_capacity=len(logits),
        vocab_size=len(logits),
        token_history=[0, 2],
        token_history_capacity=2,
        token_history_count=2,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=3,
        top_p_q16=(3 * SAMPLING_Q16_ONE) // 4,
        repetition_penalty_q16=(5 * SAMPLING_Q16_ONE) // 4,
        random_q16=12345,
        out_token_id=out_tok,
        out_token_prob_q16=out_prob,
    )
    assert err == SAMPLING_Q16_OK

    err2, exp_tok, exp_prob = sampling_select_next_token_checked_reference(
        logits_q16=logits,
        vocab_size=len(logits),
        token_history=[0, 2],
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=3,
        top_p_q16=(3 * SAMPLING_Q16_ONE) // 4,
        repetition_penalty_q16=(5 * SAMPLING_Q16_ONE) // 4,
        random_q16=12345,
    )
    assert err2 == SAMPLING_Q16_OK
    assert out_tok[0] == exp_tok
    assert out_prob[0] == exp_prob


def test_no_partial_publish_on_failure() -> None:
    out_tok = [42]
    out_prob = [1337]
    err = sampling_select_next_token_checked_nopartial_reference(
        logits_q16=[1000, 900, 800],
        logits_capacity=3,
        vocab_size=3,
        token_history=[],
        token_history_capacity=0,
        token_history_count=0,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=2,
        top_p_q16=0,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16=0,
        out_token_id=out_tok,
        out_token_prob_q16=out_prob,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_tok[0] == 42
    assert out_prob[0] == 1337


def test_alias_outputs_rejected() -> None:
    same = [0]
    err = sampling_select_next_token_checked_nopartial_reference(
        logits_q16=[100, 90, 10],
        logits_capacity=3,
        vocab_size=3,
        token_history=[],
        token_history_capacity=0,
        token_history_count=0,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=2,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16=0,
        out_token_id=same,
        out_token_prob_q16=same,
    )
    assert err == SAMPLING_Q16_ERR_BAD_PARAM


def test_randomized_determinism_and_probability_range() -> None:
    rng = random.Random(20260423_1217)
    for _ in range(2000):
        vocab_size = rng.randint(2, 64)
        logits = [rng.randint(-250000, 250000) for _ in range(vocab_size)]
        top_k = rng.randint(1, vocab_size)
        history = [rng.randint(0, vocab_size - 1) for _ in range(rng.randint(0, 12))]

        kwargs = dict(
            logits_q16=logits,
            logits_capacity=vocab_size,
            vocab_size=vocab_size,
            token_history=history,
            token_history_capacity=len(history),
            token_history_count=len(history),
            temperature_q16=rng.randint(1, 4 * SAMPLING_Q16_ONE),
            top_k=top_k,
            top_p_q16=rng.randint(1, SAMPLING_Q16_ONE),
            repetition_penalty_q16=rng.randint(SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE),
            random_q16=rng.randint(0, SAMPLING_Q16_ONE - 1),
        )

        out_tok_a = [-1]
        out_prob_a = [-1]
        out_tok_b = [-1]
        out_prob_b = [-1]

        err_a = sampling_select_next_token_checked_nopartial_reference(
            out_token_id=out_tok_a,
            out_token_prob_q16=out_prob_a,
            **kwargs,
        )
        err_b = sampling_select_next_token_checked_nopartial_reference(
            out_token_id=out_tok_b,
            out_token_prob_q16=out_prob_b,
            **kwargs,
        )

        assert err_a == SAMPLING_Q16_OK
        assert err_b == SAMPLING_Q16_OK
        assert out_tok_a[0] == out_tok_b[0]
        assert out_prob_a[0] == out_prob_b[0]
        assert 0 <= out_tok_a[0] < vocab_size
        assert 0 <= out_prob_a[0] <= SAMPLING_Q16_ONE


if __name__ == "__main__":
    test_source_contains_no_partial_wrapper()
    test_success_publishes_token_and_probability()
    test_no_partial_publish_on_failure()
    test_alias_outputs_rejected()
    test_randomized_determinism_and_probability_range()
    print("sampling_select_next_token_checked_nopartial_reference_checks=ok")
