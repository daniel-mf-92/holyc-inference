#!/usr/bin/env python3
"""Reference checks for GenerationStepChecked stage-coded sampling semantics (IQ-750)."""

from __future__ import annotations

import math
import random
from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4
SAMPLING_Q16_ERR_DOMAIN = 5

SAMPLING_Q16_SHIFT = 16
SAMPLING_Q16_ONE = 1 << SAMPLING_Q16_SHIFT

GENERATION_STEP_STAGE_REPETITION_BASE = 0x0100
GENERATION_STEP_STAGE_TEMPERATURE_BASE = 0x0200
GENERATION_STEP_STAGE_TOPK_BASE = 0x0300
GENERATION_STEP_STAGE_SOFTMAX_BASE = 0x0400
GENERATION_STEP_STAGE_TOPP_CUTOFF_BASE = 0x0500
GENERATION_STEP_STAGE_TOPP_SAMPLE_BASE = 0x0600


def compose_stage_error(stage_base: int, stage_status: int) -> int:
    if stage_status == SAMPLING_Q16_OK:
        return SAMPLING_Q16_OK
    if stage_status < 0 or stage_status > 0xFF:
        return stage_base | SAMPLING_Q16_ERR_BAD_PARAM
    return stage_base | stage_status


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


def stable_topk_indices(logits_q16: list[int], k: int) -> tuple[int, list[int]]:
    if k < 0 or k > len(logits_q16):
        return SAMPLING_Q16_ERR_BAD_PARAM, []
    return SAMPLING_Q16_OK, sorted(range(len(logits_q16)), key=lambda i: (-logits_q16[i], i))[:k]


def softmax_q16(logits_q16: list[int]) -> tuple[int, list[int]]:
    if not logits_q16:
        return SAMPLING_Q16_ERR_BAD_PARAM, []
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
    return SAMPLING_Q16_OK, raw


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


def top_p_sample_index(probs_q16: list[int], prefix_len: int, random_q16: int) -> tuple[int, int]:
    if prefix_len <= 0 or prefix_len > len(probs_q16):
        return SAMPLING_Q16_ERR_BAD_PARAM, -1
    if random_q16 < 0 or random_q16 >= SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1
    prefix_mass = sum(probs_q16[:prefix_len])
    threshold = (random_q16 * prefix_mass) >> SAMPLING_Q16_SHIFT
    cumulative = 0
    for i in range(prefix_len):
        cumulative += probs_q16[i]
        if threshold < cumulative:
            return SAMPLING_Q16_OK, i
    return SAMPLING_Q16_OK, prefix_len - 1


def sampling_select_next_token_reference(
    logits_q16: list[int],
    vocab_size: int,
    token_history: list[int],
    temperature_q16: int,
    top_k: int,
    top_p_q16: int,
    repetition_penalty_q16: int,
    random_q16: int,
) -> tuple[int, int]:
    if vocab_size <= 0 or top_k <= 0 or top_k > vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1

    stage = logits_q16[:vocab_size]
    err = apply_repetition_penalty(stage, token_history, vocab_size, repetition_penalty_q16)
    if err != SAMPLING_Q16_OK:
        return err, -1
    err = apply_temperature(stage, temperature_q16)
    if err != SAMPLING_Q16_OK:
        return err, -1

    err, topk_idx = stable_topk_indices(stage, top_k)
    if err != SAMPLING_Q16_OK:
        return err, -1

    topk_logits = [stage[i] for i in topk_idx]
    err, topk_probs = softmax_q16(topk_logits)
    if err != SAMPLING_Q16_OK:
        return err, -1

    err, cutoff = top_p_cutoff(topk_probs, top_p_q16)
    if err != SAMPLING_Q16_OK:
        return err, -1

    err, sampled_rank = top_p_sample_index(topk_probs, cutoff + 1, random_q16)
    if err != SAMPLING_Q16_OK:
        return err, -1

    return SAMPLING_Q16_OK, topk_idx[sampled_rank]


def generation_step_reference(
    logits_q16: list[int],
    vocab_size: int,
    token_history: list[int],
    temperature_q16: int,
    top_k: int,
    top_p_q16: int,
    repetition_penalty_q16: int,
    random_q16: int,
) -> tuple[int, int]:
    if vocab_size <= 0 or top_k <= 0 or top_k > vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1
    if temperature_q16 <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1
    if top_p_q16 <= 0 or top_p_q16 > SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1
    if repetition_penalty_q16 < SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1
    if random_q16 < 0 or random_q16 >= SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, -1

    stage = logits_q16[:vocab_size]

    err = apply_repetition_penalty(stage, token_history, vocab_size, repetition_penalty_q16)
    if err != SAMPLING_Q16_OK:
        return compose_stage_error(GENERATION_STEP_STAGE_REPETITION_BASE, err), -1

    err = apply_temperature(stage, temperature_q16)
    if err != SAMPLING_Q16_OK:
        return compose_stage_error(GENERATION_STEP_STAGE_TEMPERATURE_BASE, err), -1

    err, topk_idx = stable_topk_indices(stage, top_k)
    if err != SAMPLING_Q16_OK:
        return compose_stage_error(GENERATION_STEP_STAGE_TOPK_BASE, err), -1

    topk_logits = [stage[i] for i in topk_idx]

    err, topk_probs = softmax_q16(topk_logits)
    if err != SAMPLING_Q16_OK:
        return compose_stage_error(GENERATION_STEP_STAGE_SOFTMAX_BASE, err), -1

    err, cutoff = top_p_cutoff(topk_probs, top_p_q16)
    if err != SAMPLING_Q16_OK:
        return compose_stage_error(GENERATION_STEP_STAGE_TOPP_CUTOFF_BASE, err), -1

    err, sampled_rank = top_p_sample_index(topk_probs, cutoff + 1, random_q16)
    if err != SAMPLING_Q16_OK:
        return compose_stage_error(GENERATION_STEP_STAGE_TOPP_SAMPLE_BASE, err), -1

    return SAMPLING_Q16_OK, topk_idx[sampled_rank]


def test_source_contains_generation_step_and_stage_bases() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 GenerationStepChecked(" in source
    assert "GENERATION_STEP_STAGE_REPETITION_BASE" in source
    assert "GENERATION_STEP_STAGE_TEMPERATURE_BASE" in source
    assert "GENERATION_STEP_STAGE_TOPK_BASE" in source
    assert "GENERATION_STEP_STAGE_SOFTMAX_BASE" in source
    assert "GENERATION_STEP_STAGE_TOPP_CUTOFF_BASE" in source
    assert "GENERATION_STEP_STAGE_TOPP_SAMPLE_BASE" in source
    assert "GenerationStepComposeStageError(" in source


def test_stage_coded_repetition_bad_param() -> None:
    logits = [3 * SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE, 1 * SAMPLING_Q16_ONE]
    status, token = generation_step_reference(
        logits_q16=logits,
        vocab_size=3,
        token_history=[4],
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=2,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=(5 * SAMPLING_Q16_ONE) // 4,
        random_q16=0,
    )
    assert token == -1
    assert status == (GENERATION_STEP_STAGE_REPETITION_BASE | SAMPLING_Q16_ERR_BAD_PARAM)


def test_success_path_matches_sampling_reference() -> None:
    rng = random.Random(20260420_750)
    for _ in range(3500):
        vocab_size = rng.randint(2, 80)
        logits = [rng.randint(-240000, 240000) for _ in range(vocab_size)]
        top_k = rng.randint(1, vocab_size)
        history = [rng.randint(0, vocab_size - 1) for _ in range(rng.randint(0, 12))]

        args = dict(
            logits_q16=logits,
            vocab_size=vocab_size,
            token_history=history,
            temperature_q16=rng.randint(1, 4 * SAMPLING_Q16_ONE),
            top_k=top_k,
            top_p_q16=rng.randint(1, SAMPLING_Q16_ONE),
            repetition_penalty_q16=rng.randint(SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE),
            random_q16=rng.randint(0, SAMPLING_Q16_ONE - 1),
        )

        g_status, g_tok = generation_step_reference(**args)
        s_status, s_tok = sampling_select_next_token_reference(**args)

        assert g_status == SAMPLING_Q16_OK
        assert s_status == SAMPLING_Q16_OK
        assert g_tok == s_tok
        assert 0 <= g_tok < vocab_size


if __name__ == "__main__":
    test_source_contains_generation_step_and_stage_bases()
    test_stage_coded_repetition_bad_param()
    test_success_path_matches_sampling_reference()
    print("generation_step_checked_reference_checks=ok")
