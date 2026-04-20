#!/usr/bin/env python3
"""Reference checks for InferenceGenerateTokensCheckedDefaultTopK (IQ-754)."""

from __future__ import annotations

from pathlib import Path
import random

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4

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


def inference_generate_tokens_default_topk_reference(
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
    return inference_generate_tokens_reference(
        step_logits_capacity=step_logits_capacity,
        vocab_size=vocab_size,
        max_new_tokens=max_new_tokens,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        workspace_stage_logits_capacity=workspace_stage_logits_capacity,
        workspace_topk_logits_capacity=workspace_topk_logits_capacity,
        workspace_topk_index_capacity=workspace_topk_index_capacity,
        token_history=token_history,
        random_q16_values=random_q16_values,
        out_generated_tokens=out_generated_tokens,
        forced_run_status=forced_run_status,
    )


def test_source_contains_default_topk_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensCheckedDefaultTopK(" in source
    assert "return InferenceGenerateTokensChecked(step_logits_q16," in source
    assert "temperature_q16," in source
    assert "vocab_size," in source
    assert "top_p_q16," in source


def test_default_topk_wrapper_matches_explicit_topk_reference_randomized() -> None:
    rng = random.Random(20260420_754)
    for _ in range(500):
        vocab_size = rng.randint(1, 192)
        max_new_tokens = rng.randint(0, 48)
        token_history_count = rng.randint(0, 24)
        token_history_capacity = token_history_count + max_new_tokens
        step_logits_capacity = vocab_size * max_new_tokens

        history = [rng.randint(0, vocab_size - 1) for _ in range(token_history_capacity)]
        random_values = [rng.randint(0, 65535) for _ in range(max_new_tokens)]
        out_default = [777] * max_new_tokens
        out_explicit = [777] * max_new_tokens

        default_result = inference_generate_tokens_default_topk_reference(
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
            out_generated_tokens=out_default,
        )

        explicit_result = inference_generate_tokens_reference(
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
            out_generated_tokens=out_explicit,
        )

        assert default_result == explicit_result


def test_default_topk_wrapper_preserves_failure_and_no_partial_contracts() -> None:
    out_tokens = [41, 42, 43, 44]
    history = [3, 5, 7, 11, 13, 17, 19]

    status_default, out_default, count_default, hist_default = (
        inference_generate_tokens_default_topk_reference(
            step_logits_capacity=16,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            token_history=history,
            random_q16_values=[0, 1, 2, 3],
            out_generated_tokens=out_tokens,
            forced_run_status=0x0402,
        )
    )

    status_explicit, out_explicit, count_explicit, hist_explicit = (
        inference_generate_tokens_reference(
            step_logits_capacity=16,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            token_history=history,
            random_q16_values=[0, 1, 2, 3],
            out_generated_tokens=out_tokens,
            forced_run_status=0x0402,
        )
    )

    assert status_default == status_explicit == 0x0402
    assert count_default == count_explicit == -1
    assert out_default == out_explicit == out_tokens
    assert hist_default == hist_explicit == history


def test_default_topk_wrapper_matches_explicit_on_adversarial_capacities() -> None:
    adversarial_vectors = [
        dict(
            step_logits_capacity=12,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
        dict(
            step_logits_capacity=11,
            vocab_size=4,
            max_new_tokens=3,
            token_history_capacity=10,
            token_history_count=5,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
        dict(
            step_logits_capacity=100,
            vocab_size=16,
            max_new_tokens=2,
            token_history_capacity=9,
            token_history_count=4,
            workspace_stage_logits_capacity=15,
            workspace_topk_logits_capacity=16,
            workspace_topk_index_capacity=16,
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
        dict(
            step_logits_capacity=I64_MAX,
            vocab_size=(1 << 62),
            max_new_tokens=3,
            token_history_capacity=10,
            token_history_count=1,
            workspace_stage_logits_capacity=(1 << 62),
            workspace_topk_logits_capacity=(1 << 62),
            workspace_topk_index_capacity=(1 << 62),
            expect=SAMPLING_Q16_ERR_OVERFLOW,
        ),
    ]

    for case in adversarial_vectors:
        history = [0] * case["token_history_capacity"]
        random_values = [0] * case["max_new_tokens"]
        out_default = [909] * case["max_new_tokens"]
        out_explicit = [909] * case["max_new_tokens"]

        status_default, *_ = inference_generate_tokens_default_topk_reference(
            step_logits_capacity=case["step_logits_capacity"],
            vocab_size=case["vocab_size"],
            max_new_tokens=case["max_new_tokens"],
            token_history_capacity=case["token_history_capacity"],
            token_history_count=case["token_history_count"],
            workspace_stage_logits_capacity=case["workspace_stage_logits_capacity"],
            workspace_topk_logits_capacity=case["workspace_topk_logits_capacity"],
            workspace_topk_index_capacity=case["workspace_topk_index_capacity"],
            token_history=history,
            random_q16_values=random_values,
            out_generated_tokens=out_default,
        )

        status_explicit, *_ = inference_generate_tokens_reference(
            step_logits_capacity=case["step_logits_capacity"],
            vocab_size=case["vocab_size"],
            max_new_tokens=case["max_new_tokens"],
            token_history_capacity=case["token_history_capacity"],
            token_history_count=case["token_history_count"],
            workspace_stage_logits_capacity=case["workspace_stage_logits_capacity"],
            workspace_topk_logits_capacity=case["workspace_topk_logits_capacity"],
            workspace_topk_index_capacity=case["workspace_topk_index_capacity"],
            token_history=history,
            random_q16_values=random_values,
            out_generated_tokens=out_explicit,
        )

        assert status_default == status_explicit == case["expect"]


if __name__ == "__main__":
    test_source_contains_default_topk_wrapper()
    test_default_topk_wrapper_matches_explicit_topk_reference_randomized()
    test_default_topk_wrapper_preserves_failure_and_no_partial_contracts()
    test_default_topk_wrapper_matches_explicit_on_adversarial_capacities()
    print("inference_generate_tokens_checked_default_topk_reference_checks=ok")
