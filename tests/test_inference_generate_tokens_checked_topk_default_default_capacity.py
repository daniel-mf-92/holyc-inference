#!/usr/bin/env python3
"""Reference checks for InferenceGenerateTokensCheckedTopKDefaultDefaultCapacity (IQ-790)."""

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

    if required_history_capacity > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if vocab_size > workspace_stage_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if vocab_size > workspace_topk_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if vocab_size > workspace_topk_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if max_new_tokens > len(random_q16_values):
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]
    if max_new_tokens > len(out_generated_tokens):
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


def inference_generate_tokens_checked_topk_default_reference(
    *,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history_capacity: int,
    token_history_count: int,
    top_p_q16: int,
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
        top_k=vocab_size,
        top_p_q16=top_p_q16,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        workspace_stage_logits_capacity=workspace_stage_logits_capacity,
        workspace_topk_logits_capacity=workspace_topk_logits_capacity,
        workspace_topk_index_capacity=workspace_topk_index_capacity,
        token_history=token_history,
        random_q16_values=random_q16_values,
        out_generated_tokens=out_generated_tokens,
        forced_run_status=forced_run_status,
    )


def inference_generate_tokens_checked_topk_default_default_capacity_reference(
    *,
    vocab_size: int,
    max_new_tokens: int,
    token_history_count: int,
    top_p_q16: int,
    token_history: list[int],
    random_q16_values: list[int],
    out_generated_tokens: list[int],
    forced_run_status: int = SAMPLING_Q16_OK,
) -> tuple[int, list[int], int, list[int]]:
    if vocab_size < 0 or max_new_tokens < 0 or token_history_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, out_generated_tokens[:], -1, token_history[:]

    token_history_capacity = token_history_count + max_new_tokens
    if token_history_capacity < token_history_count:
        return SAMPLING_Q16_ERR_OVERFLOW, out_generated_tokens[:], -1, token_history[:]

    if max_new_tokens:
        if vocab_size > I64_MAX // max_new_tokens:
            return SAMPLING_Q16_ERR_OVERFLOW, out_generated_tokens[:], -1, token_history[:]
        step_logits_capacity = vocab_size * max_new_tokens
    else:
        step_logits_capacity = 0

    return inference_generate_tokens_checked_topk_default_reference(
        step_logits_capacity=step_logits_capacity,
        vocab_size=vocab_size,
        max_new_tokens=max_new_tokens,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        top_p_q16=top_p_q16,
        workspace_stage_logits_capacity=vocab_size,
        workspace_topk_logits_capacity=vocab_size,
        workspace_topk_index_capacity=vocab_size,
        token_history=token_history,
        random_q16_values=random_q16_values,
        out_generated_tokens=out_generated_tokens,
        forced_run_status=forced_run_status,
    )


def test_source_contains_topk_default_default_capacity_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensCheckedTopKDefaultDefaultCapacity(" in source
    assert "return InferenceGenerateTokensCheckedTopKDefault(" in source


def test_source_wrapper_derives_checked_default_capacities() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceGenerateTokensCheckedTopKDefaultDefaultCapacity("
    start = source.rfind(signature)
    assert start >= 0

    brace_open = source.find("{", start)
    assert brace_open >= 0

    depth = 0
    end = -1
    for index in range(brace_open, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break

    assert end > brace_open
    body = source[brace_open + 1 : end]

    assert "token_history_capacity = token_history_count + max_new_tokens;" in body
    assert "vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens" in body
    assert "step_logits_capacity = vocab_size * max_new_tokens;" in body
    assert "workspace_stage_logits_q16," in body
    assert "workspace_topk_logits_q16," in body
    assert "workspace_topk_indices," in body
    assert "vocab_size," in body


def test_default_capacity_matches_explicit_composition_randomized() -> None:
    rng = random.Random(20260420_790)
    for _ in range(1400):
        vocab_size = rng.randint(0, 512)
        max_new_tokens = rng.randint(0, 128)
        token_history_count = rng.randint(0, 128)
        top_p_q16 = rng.randint(-4, SAMPLING_Q16_ONE + 4)

        if token_history_count + max_new_tokens < token_history_count:
            token_history_capacity = token_history_count
        else:
            token_history_capacity = token_history_count + max_new_tokens

        history = [
            rng.randint(0, max(0, vocab_size - 1)) if vocab_size else 0
            for _ in range(token_history_capacity)
        ]
        random_values = [rng.randint(0, SAMPLING_Q16_ONE - 1) for _ in range(max_new_tokens)]
        out_default = [901] * max_new_tokens
        out_explicit = [901] * max_new_tokens
        forced_run_status = rng.choice([SAMPLING_Q16_OK, SAMPLING_Q16_OK, 0x0521])

        result_default = (
            inference_generate_tokens_checked_topk_default_default_capacity_reference(
                vocab_size=vocab_size,
                max_new_tokens=max_new_tokens,
                token_history_count=token_history_count,
                top_p_q16=top_p_q16,
                token_history=history,
                random_q16_values=random_values,
                out_generated_tokens=out_default,
                forced_run_status=forced_run_status,
            )
        )

        if max_new_tokens and vocab_size > I64_MAX // max_new_tokens:
            expected_step_logits_capacity = I64_MAX
        else:
            expected_step_logits_capacity = vocab_size * max_new_tokens

        expected_token_history_capacity = token_history_count + max_new_tokens

        result_explicit = inference_generate_tokens_checked_topk_default_reference(
            step_logits_capacity=expected_step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history_capacity=expected_token_history_capacity,
            token_history_count=token_history_count,
            top_p_q16=top_p_q16,
            workspace_stage_logits_capacity=vocab_size,
            workspace_topk_logits_capacity=vocab_size,
            workspace_topk_index_capacity=vocab_size,
            token_history=history,
            random_q16_values=random_values,
            out_generated_tokens=out_explicit,
            forced_run_status=forced_run_status,
        )

        if max_new_tokens and vocab_size > I64_MAX // max_new_tokens:
            assert result_default[0] == SAMPLING_Q16_ERR_OVERFLOW
        else:
            assert result_default == result_explicit


def test_default_capacity_preserves_failure_no_partial_contracts() -> None:
    history = [3, 5, 7, 11, 13, 17]
    out_tokens = [21, 22, 23]

    default_result = inference_generate_tokens_checked_topk_default_default_capacity_reference(
        vocab_size=9,
        max_new_tokens=3,
        token_history_count=3,
        top_p_q16=SAMPLING_Q16_ONE,
        token_history=history,
        random_q16_values=[1, 2, 3],
        out_generated_tokens=out_tokens,
        forced_run_status=0x0777,
    )

    explicit_result = inference_generate_tokens_checked_topk_default_reference(
        step_logits_capacity=27,
        vocab_size=9,
        max_new_tokens=3,
        token_history_capacity=6,
        token_history_count=3,
        top_p_q16=SAMPLING_Q16_ONE,
        workspace_stage_logits_capacity=9,
        workspace_topk_logits_capacity=9,
        workspace_topk_index_capacity=9,
        token_history=history,
        random_q16_values=[1, 2, 3],
        out_generated_tokens=out_tokens,
        forced_run_status=0x0777,
    )

    assert default_result[0] == explicit_result[0] == 0x0777
    assert default_result[2] == explicit_result[2] == -1
    assert default_result[1] == explicit_result[1] == out_tokens
    assert default_result[3] == explicit_result[3] == history


def test_adversarial_bad_param_and_overflow_vectors() -> None:
    adversarial = [
        dict(vocab_size=-1, max_new_tokens=2, token_history_count=0, top_p_q16=SAMPLING_Q16_ONE, expect=SAMPLING_Q16_ERR_BAD_PARAM),
        dict(vocab_size=16, max_new_tokens=-1, token_history_count=0, top_p_q16=SAMPLING_Q16_ONE, expect=SAMPLING_Q16_ERR_BAD_PARAM),
        dict(vocab_size=16, max_new_tokens=3, token_history_count=-2, top_p_q16=SAMPLING_Q16_ONE, expect=SAMPLING_Q16_ERR_BAD_PARAM),
        dict(vocab_size=(1 << 62), max_new_tokens=3, token_history_count=1, top_p_q16=SAMPLING_Q16_ONE, expect=SAMPLING_Q16_ERR_OVERFLOW),
        dict(vocab_size=8, max_new_tokens=4, token_history_count=2, top_p_q16=0, expect=SAMPLING_Q16_ERR_BAD_PARAM),
        dict(vocab_size=8, max_new_tokens=4, token_history_count=2, top_p_q16=SAMPLING_Q16_ONE + 1, expect=SAMPLING_Q16_ERR_BAD_PARAM),
    ]

    for case in adversarial:
        token_history_capacity = max(0, case["token_history_count"] + max(0, case["max_new_tokens"]))
        history = [0] * token_history_capacity
        random_values = [0] * max(0, case["max_new_tokens"])
        out_tokens = [505] * max(0, case["max_new_tokens"])

        result = inference_generate_tokens_checked_topk_default_default_capacity_reference(
            vocab_size=case["vocab_size"],
            max_new_tokens=case["max_new_tokens"],
            token_history_count=case["token_history_count"],
            top_p_q16=case["top_p_q16"],
            token_history=history,
            random_q16_values=random_values,
            out_generated_tokens=out_tokens,
        )
        assert result[0] == case["expect"]


if __name__ == "__main__":
    test_source_contains_topk_default_default_capacity_wrapper()
    test_source_wrapper_derives_checked_default_capacities()
    test_default_capacity_matches_explicit_composition_randomized()
    test_default_capacity_preserves_failure_no_partial_contracts()
    test_adversarial_bad_param_and_overflow_vectors()
    print("inference_generate_tokens_checked_topk_default_default_capacity_reference_checks=ok")
