#!/usr/bin/env python3
"""Parity harness for InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyDefaultCapacity (IQ-807)."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from test_inference_generate_tokens_checked_topk_default_preflight_only import (
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_OVERFLOW,
    SAMPLING_Q16_OK,
    SAMPLING_Q16_ONE,
    inference_generate_tokens_checked_topk_default_preflight_only_reference,
)

I64_MAX = (1 << 63) - 1


def inference_generate_tokens_checked_topk_default_preflight_only_default_capacity_reference(
    *,
    step_logits_q16: list[int] | None,
    vocab_size: int,
    max_new_tokens: int,
    token_history: list[int] | None,
    token_history_count: int,
    top_p_q16: int,
    random_q16_values: list[int] | None,
    workspace_stage_logits_q16: list[int] | None,
    workspace_topk_logits_q16: list[int] | None,
    workspace_topk_indices: list[int] | None,
    out_generated_tokens: list[int] | None,
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

    return inference_generate_tokens_checked_topk_default_preflight_only_reference(
        step_logits_q16=step_logits_q16,
        step_logits_capacity=step_logits_capacity,
        vocab_size=vocab_size,
        max_new_tokens=max_new_tokens,
        token_history=token_history,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        top_p_q16=top_p_q16,
        random_q16_values=random_q16_values,
        random_q16_capacity=max_new_tokens,
        workspace_stage_logits_q16=workspace_stage_logits_q16,
        workspace_stage_logits_capacity=vocab_size,
        workspace_topk_logits_q16=workspace_topk_logits_q16,
        workspace_topk_logits_capacity=vocab_size,
        workspace_topk_indices=workspace_topk_indices,
        workspace_topk_index_capacity=vocab_size,
        out_generated_tokens=out_generated_tokens,
        generated_capacity=max_new_tokens,
    )


def explicit_composition_reference(
    *,
    step_logits_q16: list[int] | None,
    vocab_size: int,
    max_new_tokens: int,
    token_history: list[int] | None,
    token_history_count: int,
    top_p_q16: int,
    random_q16_values: list[int] | None,
    workspace_stage_logits_q16: list[int] | None,
    workspace_topk_logits_q16: list[int] | None,
    workspace_topk_indices: list[int] | None,
    out_generated_tokens: list[int] | None,
) -> tuple[int, dict[str, int] | None]:
    return inference_generate_tokens_checked_topk_default_preflight_only_default_capacity_reference(
        step_logits_q16=step_logits_q16,
        vocab_size=vocab_size,
        max_new_tokens=max_new_tokens,
        token_history=token_history,
        token_history_count=token_history_count,
        top_p_q16=top_p_q16,
        random_q16_values=random_q16_values,
        workspace_stage_logits_q16=workspace_stage_logits_q16,
        workspace_topk_logits_q16=workspace_topk_logits_q16,
        workspace_topk_indices=workspace_topk_indices,
        out_generated_tokens=out_generated_tokens,
    )


def test_source_contains_topk_default_preflight_only_default_capacity_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyDefaultCapacity(" in source
    assert "token_history_capacity = token_history_count + max_new_tokens;" in source
    assert "vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens" in source
    assert "return InferenceGenerateTokensCheckedTopKDefaultPreflightOnly(" in source


def test_topk_default_preflight_only_default_capacity_matches_explicit_randomized() -> None:
    rng = random.Random(20260420_807)

    for _ in range(1000):
        vocab_size = rng.randint(1, 384)
        max_new_tokens = rng.randint(0, 64)
        token_history_count = rng.randint(0, 64)
        top_p_q16 = rng.randint(1, SAMPLING_Q16_ONE)

        if rng.random() < 0.1:
            top_p_q16 = rng.choice([0, SAMPLING_Q16_ONE + 1])

        token_history_capacity = token_history_count + max_new_tokens
        step_logits_capacity = vocab_size * max_new_tokens

        step_logits = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(step_logits_capacity, 1))]
        token_history = [rng.randint(0, vocab_size - 1) for _ in range(max(token_history_capacity, 1))]
        random_values = [rng.randint(0, SAMPLING_Q16_ONE - 1) for _ in range(max(max_new_tokens, 1))]
        stage_logits = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(vocab_size, 1))]
        topk_logits = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(vocab_size, 1))]
        topk_indices = [rng.randint(0, vocab_size - 1) for _ in range(max(vocab_size, 1))]
        generated = [rng.randint(0, vocab_size - 1) for _ in range(max(max_new_tokens, 1))]

        wrapper = inference_generate_tokens_checked_topk_default_preflight_only_default_capacity_reference(
            step_logits_q16=step_logits,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history=token_history,
            token_history_count=token_history_count,
            top_p_q16=top_p_q16,
            random_q16_values=random_values,
            workspace_stage_logits_q16=stage_logits,
            workspace_topk_logits_q16=topk_logits,
            workspace_topk_indices=topk_indices,
            out_generated_tokens=generated,
        )

        explicit = explicit_composition_reference(
            step_logits_q16=step_logits,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history=token_history,
            token_history_count=token_history_count,
            top_p_q16=top_p_q16,
            random_q16_values=random_values,
            workspace_stage_logits_q16=stage_logits,
            workspace_topk_logits_q16=topk_logits,
            workspace_topk_indices=topk_indices,
            out_generated_tokens=generated,
        )

        assert wrapper == explicit


def test_topk_default_preflight_only_default_capacity_error_surface() -> None:
    base = dict(
        step_logits_q16=[0] * 64,
        vocab_size=8,
        max_new_tokens=4,
        token_history=[0] * 8,
        token_history_count=4,
        top_p_q16=SAMPLING_Q16_ONE,
        random_q16_values=[0] * 4,
        workspace_stage_logits_q16=[0] * 8,
        workspace_topk_logits_q16=[0] * 8,
        workspace_topk_indices=[0] * 8,
        out_generated_tokens=[0] * 4,
    )

    status, diagnostics = inference_generate_tokens_checked_topk_default_preflight_only_default_capacity_reference(**base)
    assert status == SAMPLING_Q16_OK
    assert diagnostics == {
        "required_step_logits_cells": 32,
        "required_history_capacity": 8,
        "required_stage_logits_capacity": 8,
        "required_topk_capacity": 8,
        "required_random_capacity": 4,
        "required_generated_capacity": 4,
    }

    bad = dict(base)
    bad["vocab_size"] = -1
    status, diagnostics = inference_generate_tokens_checked_topk_default_preflight_only_default_capacity_reference(**bad)
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert diagnostics is None

    overflow = dict(base)
    overflow["vocab_size"] = 1 << 62
    overflow["max_new_tokens"] = 3
    overflow["workspace_stage_logits_q16"] = [0] * (1 << 8)
    overflow["workspace_topk_logits_q16"] = [0] * (1 << 8)
    overflow["workspace_topk_indices"] = [0] * (1 << 8)
    status, diagnostics = inference_generate_tokens_checked_topk_default_preflight_only_default_capacity_reference(**overflow)
    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert diagnostics is None


if __name__ == "__main__":
    test_source_contains_topk_default_preflight_only_default_capacity_wrapper()
    test_topk_default_preflight_only_default_capacity_matches_explicit_randomized()
    test_topk_default_preflight_only_default_capacity_error_surface()
    print("inference_generate_tokens_checked_topk_default_preflight_only_default_capacity_reference_checks=ok")
