#!/usr/bin/env python3
"""Reference checks for InferenceGenerateTokensPreflightChecked (IQ-753)."""

from __future__ import annotations

from pathlib import Path
import random

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
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


def test_source_contains_inference_generate_tokens_preflight_checked() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensPreflightChecked(" in source
    assert "required_step_logits_cells" in source
    assert "required_history_capacity = token_history_count + max_new_tokens;" in source
    assert "if (vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens)" in source
    assert "if (required_random_capacity > random_q16_capacity)" in source
    assert "if (required_generated_capacity > generated_capacity)" in source


def test_preflight_reference_success_and_diagnostics() -> None:
    rng = random.Random(20260420_753)
    for _ in range(600):
        vocab_size = rng.randint(1, 256)
        max_new_tokens = rng.randint(0, 64)
        token_history_count = rng.randint(0, 32)

        required_history_capacity = token_history_count + max_new_tokens
        required_step_logits_cells = vocab_size * max_new_tokens

        status, diagnostics = inference_generate_tokens_preflight_reference(
            step_logits_capacity=required_step_logits_cells,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history_capacity=required_history_capacity,
            token_history_count=token_history_count,
            workspace_stage_logits_capacity=vocab_size,
            workspace_topk_logits_capacity=vocab_size,
            workspace_topk_index_capacity=vocab_size,
            random_q16_capacity=max_new_tokens,
            generated_capacity=max_new_tokens,
        )

        assert status == SAMPLING_Q16_OK
        assert diagnostics is not None
        assert diagnostics["required_step_logits_cells"] == required_step_logits_cells
        assert diagnostics["required_history_capacity"] == required_history_capacity
        assert diagnostics["required_stage_logits_capacity"] == vocab_size
        assert diagnostics["required_topk_capacity"] == vocab_size
        assert diagnostics["required_random_capacity"] == max_new_tokens
        assert diagnostics["required_generated_capacity"] == max_new_tokens


def test_preflight_reference_failure_classification() -> None:
    status, diagnostics = inference_generate_tokens_preflight_reference(
        step_logits_capacity=50,
        vocab_size=10,
        max_new_tokens=5,
        token_history_capacity=4,
        token_history_count=2,
        workspace_stage_logits_capacity=10,
        workspace_topk_logits_capacity=10,
        workspace_topk_index_capacity=10,
        random_q16_capacity=5,
        generated_capacity=5,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert diagnostics is None

    status, diagnostics = inference_generate_tokens_preflight_reference(
        step_logits_capacity=I64_MAX,
        vocab_size=(1 << 62),
        max_new_tokens=3,
        token_history_capacity=10,
        token_history_count=1,
        workspace_stage_logits_capacity=(1 << 62),
        workspace_topk_logits_capacity=(1 << 62),
        workspace_topk_index_capacity=(1 << 62),
        random_q16_capacity=3,
        generated_capacity=3,
    )
    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert diagnostics is None


def test_preflight_reference_zero_mutation_on_failure_model() -> None:
    staged = {
        "required_step_logits_cells": -11,
        "required_history_capacity": -22,
        "required_stage_logits_capacity": -33,
        "required_topk_capacity": -44,
        "required_random_capacity": -55,
        "required_generated_capacity": -66,
    }

    status, diagnostics = inference_generate_tokens_preflight_reference(
        step_logits_capacity=80,
        vocab_size=8,
        max_new_tokens=10,
        token_history_capacity=9,
        token_history_count=3,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_capacity=8,
        workspace_topk_index_capacity=8,
        random_q16_capacity=10,
        generated_capacity=10,
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert diagnostics is None
    assert staged == {
        "required_step_logits_cells": -11,
        "required_history_capacity": -22,
        "required_stage_logits_capacity": -33,
        "required_topk_capacity": -44,
        "required_random_capacity": -55,
        "required_generated_capacity": -66,
    }


if __name__ == "__main__":
    test_source_contains_inference_generate_tokens_preflight_checked()
    test_preflight_reference_success_and_diagnostics()
    test_preflight_reference_failure_classification()
    test_preflight_reference_zero_mutation_on_failure_model()
    print("inference_generate_tokens_preflight_checked_reference_checks=ok")
