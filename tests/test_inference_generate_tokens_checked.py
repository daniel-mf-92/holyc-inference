#!/usr/bin/env python3
"""Reference checks for InferenceGenerateTokensChecked default-capacity wrapper (IQ-752)."""

from __future__ import annotations

from pathlib import Path
import random

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
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
    random_q16_values: list[int],
    out_generated_tokens: list[int],
    forced_status: int,
) -> tuple[int, list[int], int, list[int]]:
    staged_history = token_history[:]
    if forced_status != SAMPLING_Q16_OK:
        return forced_status, out_generated_tokens[:], -1, staged_history

    committed_tokens = out_generated_tokens[:]
    for step_index in range(max_new_tokens):
        token = random_q16_values[step_index] % vocab_size
        staged_history[token_history_count + step_index] = token
        committed_tokens[step_index] = token
    return SAMPLING_Q16_OK, committed_tokens, max_new_tokens, staged_history


def inference_generate_tokens_reference(
    *,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history_capacity: int,
    token_history_count: int,
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

    default_history_capacity = token_history_count + max_new_tokens
    if default_history_capacity < token_history_count:
        return SAMPLING_Q16_ERR_OVERFLOW, out_generated_tokens[:], -1, token_history[:]

    if default_history_capacity > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if vocab_size > workspace_stage_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if vocab_size > workspace_topk_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if vocab_size > workspace_topk_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]

    if max_new_tokens:
        if vocab_size > I64_MAX // max_new_tokens:
            return SAMPLING_Q16_ERR_OVERFLOW, out_generated_tokens[:], -1, token_history[:]
        required_logits_cells = vocab_size * max_new_tokens
    else:
        required_logits_cells = 0

    if required_logits_cells > step_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]

    return generation_run_contract_reference(
        max_new_tokens=max_new_tokens,
        vocab_size=vocab_size,
        token_history=token_history,
        token_history_count=token_history_count,
        random_q16_values=random_q16_values,
        out_generated_tokens=out_generated_tokens,
        forced_status=forced_run_status,
    )


def test_source_contains_inference_generate_tokens_checked_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensChecked(" in source
    assert "default_history_capacity = token_history_count + max_new_tokens;" in source
    assert "if (default_history_capacity < token_history_count)" in source
    assert "if (vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens)" in source
    assert "return GenerationRunChecked(step_logits_q16," in source
    assert "default_stage_logits_capacity" in source
    assert "default_topk_capacity" in source


def test_inference_generate_tokens_reference_success_default_capacities() -> None:
    rng = random.Random(20260420_752)
    for _ in range(400):
        vocab_size = rng.randint(2, 128)
        max_new_tokens = rng.randint(0, 32)
        token_history_count = rng.randint(0, 12)
        token_history_capacity = token_history_count + max_new_tokens
        step_logits_capacity = vocab_size * max_new_tokens

        history = [rng.randint(0, vocab_size - 1) for _ in range(token_history_capacity)]
        random_values = [rng.randint(0, SAMPLING_Q16_ONE - 1) for _ in range(max_new_tokens)]
        out_generated_tokens = [999] * max_new_tokens

        status, committed_tokens, committed_count, staged_history = (
            inference_generate_tokens_reference(
                step_logits_capacity=step_logits_capacity,
                vocab_size=vocab_size,
                max_new_tokens=max_new_tokens,
                token_history_capacity=token_history_capacity,
                token_history_count=token_history_count,
                workspace_stage_logits_capacity=vocab_size,
                workspace_topk_logits_capacity=vocab_size,
                workspace_topk_index_capacity=vocab_size,
                token_history=history,
                random_q16_values=random_values,
                out_generated_tokens=out_generated_tokens,
            )
        )

        assert status == SAMPLING_Q16_OK
        assert committed_count == max_new_tokens
        for i in range(max_new_tokens):
            assert 0 <= committed_tokens[i] < vocab_size
            assert staged_history[token_history_count + i] == committed_tokens[i]


def test_inference_generate_tokens_reference_rejects_bad_default_capacities() -> None:
    status, out_tokens, out_count, history = inference_generate_tokens_reference(
        step_logits_capacity=16,
        vocab_size=8,
        max_new_tokens=2,
        token_history_capacity=4,
        token_history_count=3,
        workspace_stage_logits_capacity=7,
        workspace_topk_logits_capacity=8,
        workspace_topk_index_capacity=8,
        token_history=[0, 1, 2, 3],
        random_q16_values=[0, 1],
        out_generated_tokens=[11, 22],
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_tokens == [11, 22]
    assert out_count == -1
    assert history == [0, 1, 2, 3]


def test_inference_generate_tokens_reference_step_logits_overflow_guard() -> None:
    status, out_tokens, out_count, _history = inference_generate_tokens_reference(
        step_logits_capacity=I64_MAX,
        vocab_size=(1 << 62),
        max_new_tokens=3,
        token_history_capacity=10,
        token_history_count=0,
        workspace_stage_logits_capacity=(1 << 62),
        workspace_topk_logits_capacity=(1 << 62),
        workspace_topk_index_capacity=(1 << 62),
        token_history=[0] * 10,
        random_q16_values=[0, 1, 2],
        out_generated_tokens=[9, 9, 9],
    )
    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert out_tokens == [9, 9, 9]
    assert out_count == -1


def test_inference_generate_tokens_reference_propagates_generation_status() -> None:
    status, out_tokens, out_count, history = inference_generate_tokens_reference(
        step_logits_capacity=30,
        vocab_size=10,
        max_new_tokens=3,
        token_history_capacity=5,
        token_history_count=2,
        workspace_stage_logits_capacity=10,
        workspace_topk_logits_capacity=10,
        workspace_topk_index_capacity=10,
        token_history=[1, 2, 0, 0, 0],
        random_q16_values=[1, 2, 3],
        out_generated_tokens=[44, 55, 66],
        forced_run_status=0x0402,
    )
    assert status == 0x0402
    assert out_tokens == [44, 55, 66]
    assert out_count == -1
    assert history == [1, 2, 0, 0, 0]


if __name__ == "__main__":
    test_source_contains_inference_generate_tokens_checked_wrapper()
    test_inference_generate_tokens_reference_success_default_capacities()
    test_inference_generate_tokens_reference_rejects_bad_default_capacities()
    test_inference_generate_tokens_reference_step_logits_overflow_guard()
    test_inference_generate_tokens_reference_propagates_generation_status()
    print("inference_generate_tokens_checked_reference_checks=ok")
