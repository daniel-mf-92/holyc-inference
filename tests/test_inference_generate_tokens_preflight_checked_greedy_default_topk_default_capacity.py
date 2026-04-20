#!/usr/bin/env python3
"""Reference checks for InferenceGenerateTokensPreflightCheckedGreedyDefaultTopKDefaultCapacity (IQ-787)."""

from __future__ import annotations

from pathlib import Path
import random

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1


def inference_generate_tokens_preflight_reference(
    *,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history_capacity: int,
    token_history_count: int,
    workspace_stage_logits_capacity: int,
    workspace_topk_logits_capacity: int,
    workspace_topk_index_capacity: int,
    random_q16_capacity: int,
    generated_capacity: int,
) -> tuple[int, dict[str, int] | None]:
    if (
        step_logits_capacity < 0
        or vocab_size < 0
        or max_new_tokens < 0
        or token_history_capacity < 0
        or token_history_count < 0
        or workspace_stage_logits_capacity < 0
        or workspace_topk_logits_capacity < 0
        or workspace_topk_index_capacity < 0
        or random_q16_capacity < 0
        or generated_capacity < 0
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    if vocab_size <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if token_history_count > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    required_history_capacity = token_history_count + max_new_tokens
    if required_history_capacity < token_history_count:
        return SAMPLING_Q16_ERR_OVERFLOW, None

    required_stage_logits_capacity = vocab_size
    required_topk_capacity = vocab_size
    required_random_capacity = max_new_tokens
    required_generated_capacity = max_new_tokens

    if max_new_tokens:
        if vocab_size > I64_MAX // max_new_tokens:
            return SAMPLING_Q16_ERR_OVERFLOW, None
        required_step_logits_cells = vocab_size * max_new_tokens
    else:
        required_step_logits_cells = 0

    if required_history_capacity > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_stage_logits_capacity > workspace_stage_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_topk_capacity > workspace_topk_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_topk_capacity > workspace_topk_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_random_capacity > random_q16_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_generated_capacity > generated_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_step_logits_cells > step_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    return SAMPLING_Q16_OK, {
        "required_step_logits_cells": required_step_logits_cells,
        "required_history_capacity": required_history_capacity,
        "required_stage_logits_capacity": required_stage_logits_capacity,
        "required_topk_capacity": required_topk_capacity,
        "required_random_capacity": required_random_capacity,
        "required_generated_capacity": required_generated_capacity,
    }


def inference_generate_tokens_preflight_greedy_default_topk_default_capacity_reference(
    *,
    vocab_size: int,
    max_new_tokens: int,
    token_history_count: int,
) -> tuple[int, dict[str, int] | None]:
    if vocab_size < 0 or max_new_tokens < 0 or token_history_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    token_history_capacity = token_history_count + max_new_tokens
    if token_history_capacity < token_history_count:
        return SAMPLING_Q16_ERR_OVERFLOW, None

    if max_new_tokens:
        if vocab_size > I64_MAX // max_new_tokens:
            return SAMPLING_Q16_ERR_OVERFLOW, None
        step_logits_capacity = vocab_size * max_new_tokens
    else:
        step_logits_capacity = 0

    return inference_generate_tokens_preflight_reference(
        step_logits_capacity=step_logits_capacity,
        vocab_size=vocab_size,
        max_new_tokens=max_new_tokens,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        workspace_stage_logits_capacity=vocab_size,
        workspace_topk_logits_capacity=vocab_size,
        workspace_topk_index_capacity=vocab_size,
        random_q16_capacity=max_new_tokens,
        generated_capacity=max_new_tokens,
    )


def test_source_contains_default_capacity_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert (
        "I32 InferenceGenerateTokensPreflightCheckedGreedyDefaultTopKDefaultCapacity("
        in source
    )
    assert "return InferenceGenerateTokensPreflightCheckedGreedyDefaultTopK(" in source


def test_source_default_capacity_wrapper_derives_checked_capacities() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceGenerateTokensPreflightCheckedGreedyDefaultTopKDefaultCapacity("
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

    assert "derived_step_logits_capacity" in body
    assert "derived_token_history_capacity" in body
    assert "derived_workspace_stage_logits_capacity" in body
    assert "vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens" in body
    assert "derived_token_history_capacity = token_history_count + max_new_tokens;" in body
    assert "return InferenceGenerateTokensPreflightCheckedGreedyDefaultTopK(" in body


def test_default_capacity_matches_explicit_composition_randomized() -> None:
    rng = random.Random(20260420_787)
    for _ in range(1200):
        vocab_size = rng.randint(0, 512)
        max_new_tokens = rng.randint(0, 128)
        token_history_count = rng.randint(0, 128)

        status_default, diag_default = (
            inference_generate_tokens_preflight_greedy_default_topk_default_capacity_reference(
                vocab_size=vocab_size,
                max_new_tokens=max_new_tokens,
                token_history_count=token_history_count,
            )
        )

        if vocab_size < 0 or max_new_tokens < 0 or token_history_count < 0:
            status_explicit, diag_explicit = SAMPLING_Q16_ERR_BAD_PARAM, None
        else:
            token_history_capacity = token_history_count + max_new_tokens
            if token_history_capacity < token_history_count:
                status_explicit, diag_explicit = SAMPLING_Q16_ERR_OVERFLOW, None
            elif max_new_tokens and vocab_size > I64_MAX // max_new_tokens:
                status_explicit, diag_explicit = SAMPLING_Q16_ERR_OVERFLOW, None
            else:
                step_logits_capacity = vocab_size * max_new_tokens
                status_explicit, diag_explicit = inference_generate_tokens_preflight_reference(
                    step_logits_capacity=step_logits_capacity,
                    vocab_size=vocab_size,
                    max_new_tokens=max_new_tokens,
                    token_history_capacity=token_history_capacity,
                    token_history_count=token_history_count,
                    workspace_stage_logits_capacity=vocab_size,
                    workspace_topk_logits_capacity=vocab_size,
                    workspace_topk_index_capacity=vocab_size,
                    random_q16_capacity=max_new_tokens,
                    generated_capacity=max_new_tokens,
                )

        assert status_default == status_explicit
        assert diag_default == diag_explicit


def test_default_capacity_preserves_failure_classification() -> None:
    cases = [
        dict(
            vocab_size=0,
            max_new_tokens=4,
            token_history_count=3,
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
        dict(
            vocab_size=(1 << 62),
            max_new_tokens=3,
            token_history_count=0,
            expect=SAMPLING_Q16_ERR_OVERFLOW,
        ),
        dict(
            vocab_size=17,
            max_new_tokens=5,
            token_history_count=6,
            expect=SAMPLING_Q16_OK,
        ),
    ]

    for case in cases:
        expected = case.pop("expect")
        status, diagnostics = (
            inference_generate_tokens_preflight_greedy_default_topk_default_capacity_reference(
                **case
            )
        )
        assert status == expected
        if expected == SAMPLING_Q16_OK:
            assert diagnostics == {
                "required_step_logits_cells": 85,
                "required_history_capacity": 11,
                "required_stage_logits_capacity": 17,
                "required_topk_capacity": 17,
                "required_random_capacity": 5,
                "required_generated_capacity": 5,
            }
        else:
            assert diagnostics is None


if __name__ == "__main__":
    test_source_contains_default_capacity_wrapper()
    test_source_default_capacity_wrapper_derives_checked_capacities()
    test_default_capacity_matches_explicit_composition_randomized()
    test_default_capacity_preserves_failure_classification()
    print(
        "inference_generate_tokens_preflight_checked_greedy_default_topk_default_capacity_reference_checks=ok"
    )
