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


def inference_generate_tokens_checked_default_topp_reference(
    *,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history_capacity: int,
    token_history_count: int,
    temperature_q16: int,
    top_k: int,
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
        temperature_q16=temperature_q16,
        top_k=top_k,
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


def test_source_contains_default_topp_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensCheckedDefaultTopP(" in source
    assert "return InferenceGenerateTokensChecked(step_logits_q16," in source
    assert "top_k,\n                                          SAMPLING_Q16_ONE," in source


def test_default_topp_wrapper_matches_explicit_topp_one_reference_randomized() -> None:
    rng = random.Random(20260420_755)
    for _ in range(500):
        vocab_size = rng.randint(1, 160)
        max_new_tokens = rng.randint(0, 48)
        token_history_count = rng.randint(0, 24)
        token_history_capacity = token_history_count + max_new_tokens
        step_logits_capacity = vocab_size * max_new_tokens

        temperature_q16 = rng.randint(1, SAMPLING_Q16_ONE * 3)
        top_k = rng.randint(1, vocab_size)
        repetition_penalty_q16 = rng.randint(SAMPLING_Q16_ONE, SAMPLING_Q16_ONE * 2)

        history = [rng.randint(0, vocab_size - 1) for _ in range(token_history_capacity)]
        random_values = [rng.randint(0, SAMPLING_Q16_ONE - 1) for _ in range(max_new_tokens)]
        out_default = [777] * max_new_tokens
        out_explicit = [777] * max_new_tokens

        default_result = inference_generate_tokens_checked_default_topp_reference(
            step_logits_capacity=step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            temperature_q16=temperature_q16,
            top_k=top_k,
            repetition_penalty_q16=repetition_penalty_q16,
            workspace_stage_logits_capacity=vocab_size,
            workspace_topk_logits_capacity=vocab_size,
            workspace_topk_index_capacity=vocab_size,
            token_history=history,
            random_q16_values=random_values,
            out_generated_tokens=out_default,
        )

        explicit_result = inference_generate_tokens_checked_reference(
            step_logits_capacity=step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            temperature_q16=temperature_q16,
            top_k=top_k,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=repetition_penalty_q16,
            workspace_stage_logits_capacity=vocab_size,
            workspace_topk_logits_capacity=vocab_size,
            workspace_topk_index_capacity=vocab_size,
            token_history=history,
            random_q16_values=random_values,
            out_generated_tokens=out_explicit,
        )

        assert default_result == explicit_result


def test_default_topp_wrapper_preserves_failure_and_no_partial_contracts() -> None:
    out_tokens = [81, 82, 83, 84]
    history = [2, 3, 5, 7, 11, 13, 17]

    status_default, out_default, count_default, hist_default = (
        inference_generate_tokens_checked_default_topp_reference(
            step_logits_capacity=16,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=4,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            token_history=history,
            random_q16_values=[0, 1, 2, 3],
            out_generated_tokens=out_tokens,
            forced_run_status=0x0602,
        )
    )

    status_explicit, out_explicit, count_explicit, hist_explicit = (
        inference_generate_tokens_checked_reference(
            step_logits_capacity=16,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=4,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            token_history=history,
            random_q16_values=[0, 1, 2, 3],
            out_generated_tokens=out_tokens,
            forced_run_status=0x0602,
        )
    )

    assert status_default == status_explicit == 0x0602
    assert count_default == count_explicit == -1
    assert out_default == out_explicit == out_tokens
    assert hist_default == hist_explicit == history


def test_default_topp_wrapper_matches_explicit_on_adversarial_vectors() -> None:
    adversarial_vectors = [
        dict(
            step_logits_capacity=16,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            temperature_q16=0,
            top_k=4,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            random_q16_values=[0, 1, 2, 3],
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
        dict(
            step_logits_capacity=16,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=0,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            random_q16_values=[0, 1, 2, 3],
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
        dict(
            step_logits_capacity=16,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=5,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            random_q16_values=[0, 1, 2, 3],
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
        dict(
            step_logits_capacity=16,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=4,
            repetition_penalty_q16=SAMPLING_Q16_ONE - 1,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            random_q16_values=[0, 1, 2, 3],
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
        dict(
            step_logits_capacity=15,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=4,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            random_q16_values=[0, 1, 2, 3],
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
        dict(
            step_logits_capacity=I64_MAX,
            vocab_size=1 << 62,
            max_new_tokens=3,
            token_history_capacity=10,
            token_history_count=2,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=1,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            workspace_stage_logits_capacity=1 << 62,
            workspace_topk_logits_capacity=1 << 62,
            workspace_topk_index_capacity=1 << 62,
            random_q16_values=[0, 1, 2],
            expect=SAMPLING_Q16_ERR_OVERFLOW,
        ),
        dict(
            step_logits_capacity=16,
            vocab_size=4,
            max_new_tokens=4,
            token_history_capacity=7,
            token_history_count=3,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=4,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            random_q16_values=[0, 1, SAMPLING_Q16_ONE, 3],
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
    ]

    for vector in adversarial_vectors:
        out_default = [9] * vector["max_new_tokens"]
        out_explicit = [9] * vector["max_new_tokens"]
        history_default = [1] * vector["token_history_capacity"]
        history_explicit = [1] * vector["token_history_capacity"]

        default_result = inference_generate_tokens_checked_default_topp_reference(
            step_logits_capacity=vector["step_logits_capacity"],
            vocab_size=vector["vocab_size"],
            max_new_tokens=vector["max_new_tokens"],
            token_history_capacity=vector["token_history_capacity"],
            token_history_count=vector["token_history_count"],
            temperature_q16=vector["temperature_q16"],
            top_k=vector["top_k"],
            repetition_penalty_q16=vector["repetition_penalty_q16"],
            workspace_stage_logits_capacity=vector["workspace_stage_logits_capacity"],
            workspace_topk_logits_capacity=vector["workspace_topk_logits_capacity"],
            workspace_topk_index_capacity=vector["workspace_topk_index_capacity"],
            token_history=history_default,
            random_q16_values=vector["random_q16_values"],
            out_generated_tokens=out_default,
        )

        explicit_result = inference_generate_tokens_checked_reference(
            step_logits_capacity=vector["step_logits_capacity"],
            vocab_size=vector["vocab_size"],
            max_new_tokens=vector["max_new_tokens"],
            token_history_capacity=vector["token_history_capacity"],
            token_history_count=vector["token_history_count"],
            temperature_q16=vector["temperature_q16"],
            top_k=vector["top_k"],
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=vector["repetition_penalty_q16"],
            workspace_stage_logits_capacity=vector["workspace_stage_logits_capacity"],
            workspace_topk_logits_capacity=vector["workspace_topk_logits_capacity"],
            workspace_topk_index_capacity=vector["workspace_topk_index_capacity"],
            token_history=history_explicit,
            random_q16_values=vector["random_q16_values"],
            out_generated_tokens=out_explicit,
        )

        assert default_result[0] == vector["expect"]
        assert default_result == explicit_result


def test_default_topp_wrapper_matches_explicit_when_forcing_step_stage_error() -> None:
    kwargs = dict(
        step_logits_capacity=24,
        vocab_size=8,
        max_new_tokens=3,
        token_history_capacity=6,
        token_history_count=2,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=8,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_capacity=8,
        workspace_topk_index_capacity=8,
        token_history=[4, 5, 0, 0, 0, 0],
        random_q16_values=[1, 2, 3],
        out_generated_tokens=[80, 81, 82],
        forced_run_status=0x0502,
    )

    default_result = inference_generate_tokens_checked_default_topp_reference(**kwargs)
    explicit_result = inference_generate_tokens_checked_reference(
        **{k: v for k, v in kwargs.items() if k != "forced_run_status"},
        top_p_q16=SAMPLING_Q16_ONE,
        forced_run_status=kwargs["forced_run_status"],
    )

    assert default_result == explicit_result
    assert default_result[0] == 0x0502
    assert default_result[1] == [80, 81, 82]
    assert default_result[2] == -1


if __name__ == "__main__":
    test_source_contains_default_topp_wrapper()
    test_default_topp_wrapper_matches_explicit_topp_one_reference_randomized()
    test_default_topp_wrapper_preserves_failure_and_no_partial_contracts()
    test_default_topp_wrapper_matches_explicit_on_adversarial_vectors()
    test_default_topp_wrapper_matches_explicit_when_forcing_step_stage_error()
    print("inference_generate_tokens_checked_default_topp_reference_checks=ok")
