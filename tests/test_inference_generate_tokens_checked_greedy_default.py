#!/usr/bin/env python3
"""Reference checks for InferenceGenerateTokensCheckedGreedyDefault (IQ-757)."""

from __future__ import annotations

from pathlib import Path
import random

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4

SAMPLING_Q16_SHIFT = 16
SAMPLING_Q16_ONE = 1 << SAMPLING_Q16_SHIFT
I64_MAX = (1 << 63) - 1


def generation_run_contract_reference(
    *,
    max_new_tokens: int,
    vocab_size: int,
    token_history: list[int],
    token_history_count: int,
    temperature_q16: int,
    top_k: int,
    top_p_q16: int,
    repetition_penalty_q16: int,
    random_q16_values: list[int],
    out_generated_tokens: list[int],
    forced_status: int,
) -> tuple[int, list[int], int, list[int]]:
    staged_history = token_history[:]

    if temperature_q16 <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, staged_history
    if repetition_penalty_q16 < SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, staged_history
    if top_k <= 0 or top_k > vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, staged_history
    if top_p_q16 <= 0 or top_p_q16 > SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, staged_history

    for value in random_q16_values[:max_new_tokens]:
        if value < 0 or value >= SAMPLING_Q16_ONE:
            return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, staged_history

    if forced_status != SAMPLING_Q16_OK:
        return forced_status, out_generated_tokens[:], -1, staged_history

    committed_tokens = out_generated_tokens[:]
    for step_index in range(max_new_tokens):
        if top_k == 1:
            token = (token_history_count + step_index) % vocab_size
        else:
            token = random_q16_values[step_index] % vocab_size
        staged_history[token_history_count + step_index] = token
        committed_tokens[step_index] = token

    return SAMPLING_Q16_OK, committed_tokens, max_new_tokens, staged_history


def inference_generate_tokens_checked_reference(
    *,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history_capacity: int,
    token_history_count: int,
    temperature_q16: int,
    top_k: int,
    top_p_q16: int,
    repetition_penalty_q16: int,
    workspace_stage_logits_capacity: int,
    workspace_topk_logits_capacity: int,
    workspace_topk_index_capacity: int,
    token_history: list[int],
    random_q16_values: list[int],
    out_generated_tokens: list[int],
    forced_run_status: int = SAMPLING_Q16_OK,
) -> tuple[int, list[int], int, list[int]]:
    if (
        step_logits_capacity < 0
        or vocab_size < 0
        or max_new_tokens < 0
        or token_history_capacity < 0
        or token_history_count < 0
        or workspace_stage_logits_capacity < 0
        or workspace_topk_logits_capacity < 0
        or workspace_topk_index_capacity < 0
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]

    if vocab_size <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if token_history_count > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]

    required_history_capacity = token_history_count + max_new_tokens
    if required_history_capacity < token_history_count:
        return SAMPLING_Q16_ERR_OVERFLOW, out_generated_tokens[:], -1, token_history[:]

    required_stage_logits_capacity = vocab_size
    required_topk_capacity = vocab_size
    required_random_capacity = max_new_tokens
    required_generated_capacity = max_new_tokens

    if required_history_capacity > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if required_stage_logits_capacity > workspace_stage_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if required_topk_capacity > workspace_topk_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if required_topk_capacity > workspace_topk_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if required_random_capacity > len(random_q16_values):
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if required_generated_capacity > len(out_generated_tokens):
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]

    if max_new_tokens:
        if vocab_size > I64_MAX // max_new_tokens:
            return SAMPLING_Q16_ERR_OVERFLOW, out_generated_tokens[:], -1, token_history[:]
        required_step_logits_cells = vocab_size * max_new_tokens
    else:
        required_step_logits_cells = 0

    if required_step_logits_cells > step_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]

    return generation_run_contract_reference(
        max_new_tokens=max_new_tokens,
        vocab_size=vocab_size,
        token_history=token_history,
        token_history_count=token_history_count,
        temperature_q16=temperature_q16,
        top_k=top_k,
        top_p_q16=top_p_q16,
        repetition_penalty_q16=repetition_penalty_q16,
        random_q16_values=random_q16_values,
        out_generated_tokens=out_generated_tokens,
        forced_status=forced_run_status,
    )


def inference_generate_tokens_checked_greedy_default_reference(
    *,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history_capacity: int,
    token_history_count: int,
    repetition_penalty_q16: int,
    workspace_stage_logits_capacity: int,
    workspace_topk_logits_capacity: int,
    workspace_topk_index_capacity: int,
    token_history: list[int],
    random_q16_values: list[int],
    out_generated_tokens: list[int],
    forced_run_status: int = SAMPLING_Q16_OK,
) -> tuple[int, list[int], int, list[int]]:
    return inference_generate_tokens_checked_reference(
        step_logits_capacity=step_logits_capacity,
        vocab_size=vocab_size,
        max_new_tokens=max_new_tokens,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=1,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=repetition_penalty_q16,
        workspace_stage_logits_capacity=workspace_stage_logits_capacity,
        workspace_topk_logits_capacity=workspace_topk_logits_capacity,
        workspace_topk_index_capacity=workspace_topk_index_capacity,
        token_history=token_history,
        random_q16_values=random_q16_values,
        out_generated_tokens=out_generated_tokens,
        forced_run_status=forced_run_status,
    )


def test_source_contains_greedy_default_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensCheckedGreedyDefault(" in source
    assert "SAMPLING_Q16_ONE," in source
    assert "                                          1," in source


def test_greedy_default_matches_explicit_randomized() -> None:
    rng = random.Random(20260420_757)
    for _ in range(500):
        vocab_size = rng.randint(1, 160)
        max_new_tokens = rng.randint(0, 48)
        token_history_count = rng.randint(0, 24)
        token_history_capacity = token_history_count + max_new_tokens
        step_logits_capacity = vocab_size * max_new_tokens
        repetition_penalty_q16 = rng.randint(SAMPLING_Q16_ONE, SAMPLING_Q16_ONE * 2)

        history = [rng.randint(0, vocab_size - 1) for _ in range(token_history_capacity)]
        random_values = [rng.randint(0, SAMPLING_Q16_ONE - 1) for _ in range(max_new_tokens)]
        out_greedy = [777] * max_new_tokens
        out_explicit = [777] * max_new_tokens

        greedy_result = inference_generate_tokens_checked_greedy_default_reference(
            step_logits_capacity=step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            repetition_penalty_q16=repetition_penalty_q16,
            workspace_stage_logits_capacity=vocab_size,
            workspace_topk_logits_capacity=vocab_size,
            workspace_topk_index_capacity=vocab_size,
            token_history=history,
            random_q16_values=random_values,
            out_generated_tokens=out_greedy,
        )

        explicit_result = inference_generate_tokens_checked_reference(
            step_logits_capacity=step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=1,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=repetition_penalty_q16,
            workspace_stage_logits_capacity=vocab_size,
            workspace_topk_logits_capacity=vocab_size,
            workspace_topk_index_capacity=vocab_size,
            token_history=history,
            random_q16_values=random_values,
            out_generated_tokens=out_explicit,
        )

        assert greedy_result == explicit_result


def test_greedy_default_random_values_do_not_change_success_output() -> None:
    kwargs = dict(
        step_logits_capacity=48,
        vocab_size=12,
        max_new_tokens=4,
        token_history_capacity=9,
        token_history_count=5,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        workspace_stage_logits_capacity=12,
        workspace_topk_logits_capacity=12,
        workspace_topk_index_capacity=12,
        token_history=[1, 2, 3, 4, 5, 0, 0, 0, 0],
        out_generated_tokens=[91, 92, 93, 94],
    )

    result_a = inference_generate_tokens_checked_greedy_default_reference(
        **kwargs,
        random_q16_values=[1, 2, 3, 4],
    )
    result_b = inference_generate_tokens_checked_greedy_default_reference(
        **kwargs,
        random_q16_values=[5000, 6000, 7000, 8000],
    )

    assert result_a[0] == SAMPLING_Q16_OK
    assert result_b[0] == SAMPLING_Q16_OK
    assert result_a[1:] == result_b[1:]


def test_greedy_default_failure_and_no_partial_contracts() -> None:
    out_tokens = [31, 32, 33, 34]
    history = [3, 5, 7, 11, 13, 17, 19]

    status, out_after, out_count, history_after = (
        inference_generate_tokens_checked_greedy_default_reference(
            step_logits_capacity=16,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            repetition_penalty_q16=SAMPLING_Q16_ONE - 1,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            token_history=history,
            random_q16_values=[1, 2, 3, 4],
            out_generated_tokens=out_tokens,
        )
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_after == out_tokens
    assert out_count == -1
    assert history_after == history

    status, out_after, out_count, history_after = (
        inference_generate_tokens_checked_greedy_default_reference(
            step_logits_capacity=I64_MAX,
            vocab_size=1 << 62,
            max_new_tokens=3,
            token_history_capacity=6,
            token_history_count=0,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            workspace_stage_logits_capacity=1 << 62,
            workspace_topk_logits_capacity=1 << 62,
            workspace_topk_index_capacity=1 << 62,
            token_history=[0] * 6,
            random_q16_values=[1, 2, 3],
            out_generated_tokens=[9, 9, 9],
        )
    )

    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert out_after == [9, 9, 9]
    assert out_count == -1
    assert history_after == [0] * 6


def test_greedy_default_propagates_generation_status() -> None:
    status, out_after, out_count, history_after = (
        inference_generate_tokens_checked_greedy_default_reference(
            step_logits_capacity=24,
            vocab_size=8,
            max_new_tokens=3,
            token_history_capacity=6,
            token_history_count=2,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            workspace_stage_logits_capacity=8,
            workspace_topk_logits_capacity=8,
            workspace_topk_index_capacity=8,
            token_history=[4, 5, 0, 0, 0, 0],
            random_q16_values=[1, 2, 3],
            out_generated_tokens=[80, 81, 82],
            forced_run_status=0x0402,
        )
    )

    assert status == 0x0402
    assert out_after == [80, 81, 82]
    assert out_count == -1
    assert history_after == [4, 5, 0, 0, 0, 0]


if __name__ == "__main__":
    test_source_contains_greedy_default_wrapper()
    test_greedy_default_matches_explicit_randomized()
    test_greedy_default_random_values_do_not_change_success_output()
    test_greedy_default_failure_and_no_partial_contracts()
    test_greedy_default_propagates_generation_status()
    print("inference_generate_tokens_checked_greedy_default_reference_checks=ok")
