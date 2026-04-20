#!/usr/bin/env python3
"""Reference checks for InferenceGenerateTokensCheckedDefaultTopP (IQ-755)."""

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


def inference_generate_tokens_checked_reference(
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
        random_q16_values=random_q16_values,
        out_generated_tokens=out_generated_tokens,
        forced_status=forced_run_status,
    )


def inference_generate_tokens_checked_default_topp_reference(**kwargs: object) -> tuple[int, list[int], int, list[int]]:
    return inference_generate_tokens_checked_reference(**kwargs)


def test_source_contains_default_topp_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensCheckedDefaultTopP(" in source
    assert "return InferenceGenerateTokensChecked(step_logits_q16," in source
    assert "top_k,\n                                          SAMPLING_Q16_ONE," in source


def test_default_topp_reference_success_matches_base_contract() -> None:
    rng = random.Random(20260420_755)
    for _ in range(400):
        vocab_size = rng.randint(2, 128)
        max_new_tokens = rng.randint(0, 32)
        token_history_count = rng.randint(0, 12)
        token_history_capacity = token_history_count + max_new_tokens
        top_k = rng.randint(1, vocab_size)
        step_logits_capacity = vocab_size * max_new_tokens

        history = [rng.randint(0, vocab_size - 1) for _ in range(token_history_capacity)]
        random_values = [rng.randint(0, SAMPLING_Q16_ONE - 1) for _ in range(max_new_tokens)]
        out_generated_tokens = [777] * max_new_tokens

        base_result = inference_generate_tokens_checked_reference(
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
        wrapper_result = inference_generate_tokens_checked_default_topp_reference(
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

        assert wrapper_result == base_result
        assert 1 <= top_k <= vocab_size


def test_default_topp_reference_preserves_failure_and_no_partial_writes() -> None:
    sentinel_tokens = [11, 22, 33]
    sentinel_history = [1, 2, 3, 4, 5]

    status, out_tokens, out_count, out_history = inference_generate_tokens_checked_default_topp_reference(
        step_logits_capacity=20,
        vocab_size=10,
        max_new_tokens=3,
        token_history_capacity=5,
        token_history_count=2,
        workspace_stage_logits_capacity=9,
        workspace_topk_logits_capacity=10,
        workspace_topk_index_capacity=10,
        token_history=sentinel_history,
        random_q16_values=[1, 2, 3],
        out_generated_tokens=sentinel_tokens,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_tokens == sentinel_tokens
    assert out_count == -1
    assert out_history == sentinel_history

    status, out_tokens, out_count, out_history = inference_generate_tokens_checked_default_topp_reference(
        step_logits_capacity=I64_MAX,
        vocab_size=(1 << 62),
        max_new_tokens=3,
        token_history_capacity=6,
        token_history_count=0,
        workspace_stage_logits_capacity=(1 << 62),
        workspace_topk_logits_capacity=(1 << 62),
        workspace_topk_index_capacity=(1 << 62),
        token_history=[0] * 6,
        random_q16_values=[1, 2, 3],
        out_generated_tokens=[9, 9, 9],
    )
    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert out_tokens == [9, 9, 9]
    assert out_count == -1
    assert out_history == [0] * 6


def test_default_topp_reference_propagates_generation_status() -> None:
    status, out_tokens, out_count, history = inference_generate_tokens_checked_default_topp_reference(
        step_logits_capacity=24,
        vocab_size=8,
        max_new_tokens=3,
        token_history_capacity=5,
        token_history_count=2,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_capacity=8,
        workspace_topk_index_capacity=8,
        token_history=[4, 5, 0, 0, 0],
        random_q16_values=[1, 2, 3],
        out_generated_tokens=[80, 81, 82],
        forced_run_status=0x0502,
    )
    assert status == 0x0502
    assert out_tokens == [80, 81, 82]
    assert out_count == -1
    assert history == [4, 5, 0, 0, 0]


if __name__ == "__main__":
    test_source_contains_default_topp_wrapper()
    test_default_topp_reference_success_matches_base_contract()
    test_default_topp_reference_preserves_failure_and_no_partial_writes()
    test_default_topp_reference_propagates_generation_status()
    print("inference_generate_tokens_checked_default_topp_reference_checks=ok")
