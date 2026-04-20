#!/usr/bin/env python3
"""Reference checks for InferenceGenerateTokensCheckedTopKDefaultPreflightOnly (IQ-805)."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from test_inference_generate_tokens_preflight_checked_topk_default import (
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_OVERFLOW,
    SAMPLING_Q16_OK,
    inference_generate_tokens_preflight_topk_default_reference,
)
from test_inference_generate_tokens_checked_topk_default import SAMPLING_Q16_ONE


def inference_generate_tokens_checked_topk_default_preflight_only_reference(
    *,
    step_logits_q16: list[int] | None,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history: list[int] | None,
    token_history_capacity: int,
    token_history_count: int,
    top_p_q16: int,
    random_q16_values: list[int] | None,
    random_q16_capacity: int,
    workspace_stage_logits_q16: list[int] | None,
    workspace_stage_logits_capacity: int,
    workspace_topk_logits_q16: list[int] | None,
    workspace_topk_logits_capacity: int,
    workspace_topk_indices: list[int] | None,
    workspace_topk_index_capacity: int,
    out_generated_tokens: list[int] | None,
    generated_capacity: int,
) -> tuple[int, dict[str, int] | None]:
    if (
        step_logits_q16 is None
        or token_history is None
        or random_q16_values is None
        or workspace_stage_logits_q16 is None
        or workspace_topk_logits_q16 is None
        or workspace_topk_indices is None
        or out_generated_tokens is None
    ):
        return 1, None

    if top_p_q16 <= 0 or top_p_q16 > SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    status, diagnostics = inference_generate_tokens_preflight_topk_default_reference(
        step_logits_capacity=step_logits_capacity,
        vocab_size=vocab_size,
        max_new_tokens=max_new_tokens,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        workspace_stage_logits_capacity=workspace_stage_logits_capacity,
        workspace_topk_logits_capacity=workspace_topk_logits_capacity,
        workspace_topk_index_capacity=workspace_topk_index_capacity,
        random_q16_capacity=random_q16_capacity,
        generated_capacity=generated_capacity,
    )
    if status != SAMPLING_Q16_OK:
        return status, None

    return SAMPLING_Q16_OK, diagnostics


def explicit_composition_reference(
    *,
    step_logits_q16: list[int] | None,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history: list[int] | None,
    token_history_capacity: int,
    token_history_count: int,
    top_p_q16: int,
    random_q16_values: list[int] | None,
    random_q16_capacity: int,
    workspace_stage_logits_q16: list[int] | None,
    workspace_stage_logits_capacity: int,
    workspace_topk_logits_q16: list[int] | None,
    workspace_topk_logits_capacity: int,
    workspace_topk_indices: list[int] | None,
    workspace_topk_index_capacity: int,
    out_generated_tokens: list[int] | None,
    generated_capacity: int,
) -> tuple[int, dict[str, int] | None]:
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
        random_q16_capacity=random_q16_capacity,
        workspace_stage_logits_q16=workspace_stage_logits_q16,
        workspace_stage_logits_capacity=workspace_stage_logits_capacity,
        workspace_topk_logits_q16=workspace_topk_logits_q16,
        workspace_topk_logits_capacity=workspace_topk_logits_capacity,
        workspace_topk_indices=workspace_topk_indices,
        workspace_topk_index_capacity=workspace_topk_index_capacity,
        out_generated_tokens=out_generated_tokens,
        generated_capacity=generated_capacity,
    )


def test_source_contains_topk_default_preflight_only_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnly(" in source
    assert "InferenceGenerateTokensPreflightCheckedTopKDefault(" in source
    assert "if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)" in source
    assert "snapshot_step_logits_capacity" in source


def test_topk_default_preflight_only_matches_explicit_composition_randomized() -> None:
    rng = random.Random(20260420_805)

    for _ in range(1000):
        vocab_size = rng.randint(1, 384)
        max_new_tokens = rng.randint(0, 64)
        token_history_count = rng.randint(0, 48)

        required_history_capacity = token_history_count + max_new_tokens
        required_step_logits_cells = vocab_size * max_new_tokens

        token_history_capacity = required_history_capacity + rng.randint(0, 4)
        step_logits_capacity = required_step_logits_cells + rng.randint(0, 4)
        random_capacity = max_new_tokens + rng.randint(0, 2)
        generated_capacity = max_new_tokens + rng.randint(0, 2)
        stage_capacity = vocab_size + rng.randint(0, 2)
        topk_logits_capacity = vocab_size + rng.randint(0, 2)
        topk_index_capacity = vocab_size + rng.randint(0, 2)

        if max_new_tokens and rng.random() < 0.2:
            step_logits_capacity = max(0, step_logits_capacity - 1)
        if max_new_tokens and rng.random() < 0.2:
            random_capacity = max(0, random_capacity - 1)
        if max_new_tokens and rng.random() < 0.2:
            generated_capacity = max(0, generated_capacity - 1)

        top_p_q16 = rng.randint(1, SAMPLING_Q16_ONE)
        if rng.random() < 0.1:
            top_p_q16 = rng.choice([0, SAMPLING_Q16_ONE + 1])

        step_logits = [rng.randint(-(1 << 18), (1 << 18)) for _ in range(max(step_logits_capacity, 1))]
        token_history = [rng.randint(0, max(vocab_size - 1, 0)) for _ in range(max(token_history_capacity, 1))]
        random_values = [rng.randint(0, SAMPLING_Q16_ONE - 1) for _ in range(max(random_capacity, 1))]
        stage_logits = [rng.randint(-(1 << 18), (1 << 18)) for _ in range(max(stage_capacity, 1))]
        topk_logits = [rng.randint(-(1 << 18), (1 << 18)) for _ in range(max(topk_logits_capacity, 1))]
        topk_indices = [rng.randint(0, max(vocab_size - 1, 0)) for _ in range(max(topk_index_capacity, 1))]
        generated = [rng.randint(0, max(vocab_size - 1, 0)) for _ in range(max(generated_capacity, 1))]

        wrapper = inference_generate_tokens_checked_topk_default_preflight_only_reference(
            step_logits_q16=step_logits,
            step_logits_capacity=step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history=token_history,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            top_p_q16=top_p_q16,
            random_q16_values=random_values,
            random_q16_capacity=random_capacity,
            workspace_stage_logits_q16=stage_logits,
            workspace_stage_logits_capacity=stage_capacity,
            workspace_topk_logits_q16=topk_logits,
            workspace_topk_logits_capacity=topk_logits_capacity,
            workspace_topk_indices=topk_indices,
            workspace_topk_index_capacity=topk_index_capacity,
            out_generated_tokens=generated,
            generated_capacity=generated_capacity,
        )

        explicit = explicit_composition_reference(
            step_logits_q16=step_logits,
            step_logits_capacity=step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history=token_history,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            top_p_q16=top_p_q16,
            random_q16_values=random_values,
            random_q16_capacity=random_capacity,
            workspace_stage_logits_q16=stage_logits,
            workspace_stage_logits_capacity=stage_capacity,
            workspace_topk_logits_q16=topk_logits,
            workspace_topk_logits_capacity=topk_logits_capacity,
            workspace_topk_indices=topk_indices,
            workspace_topk_index_capacity=topk_index_capacity,
            out_generated_tokens=generated,
            generated_capacity=generated_capacity,
        )

        assert wrapper == explicit


def test_topk_default_preflight_only_error_surface() -> None:
    base = dict(
        step_logits_q16=[0] * 32,
        step_logits_capacity=32,
        vocab_size=8,
        max_new_tokens=4,
        token_history=[1, 2, 3, 4, 5, 6],
        token_history_capacity=6,
        token_history_count=2,
        top_p_q16=SAMPLING_Q16_ONE,
        random_q16_values=[0, 1, 2, 3],
        random_q16_capacity=4,
        workspace_stage_logits_q16=[0] * 8,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_q16=[0] * 8,
        workspace_topk_logits_capacity=8,
        workspace_topk_indices=[0] * 8,
        workspace_topk_index_capacity=8,
        out_generated_tokens=[0] * 4,
        generated_capacity=4,
    )

    status, diagnostics = inference_generate_tokens_checked_topk_default_preflight_only_reference(
        **base
    )
    assert status == SAMPLING_Q16_OK
    assert diagnostics == {
        "required_step_logits_cells": 32,
        "required_history_capacity": 6,
        "required_stage_logits_capacity": 8,
        "required_topk_capacity": 8,
        "required_random_capacity": 4,
        "required_generated_capacity": 4,
    }

    bad_top_p = dict(base)
    bad_top_p["top_p_q16"] = 0
    status, diagnostics = inference_generate_tokens_checked_topk_default_preflight_only_reference(
        **bad_top_p
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert diagnostics is None

    null_case = dict(base)
    null_case["workspace_topk_indices"] = None
    status, diagnostics = inference_generate_tokens_checked_topk_default_preflight_only_reference(
        **null_case
    )
    assert status == 1
    assert diagnostics is None

    overflow_case = dict(base)
    overflow_case["vocab_size"] = 1 << 62
    overflow_case["max_new_tokens"] = 3
    overflow_case["step_logits_capacity"] = (1 << 63) - 1
    overflow_case["workspace_stage_logits_capacity"] = 1 << 62
    overflow_case["workspace_topk_logits_capacity"] = 1 << 62
    overflow_case["workspace_topk_index_capacity"] = 1 << 62
    overflow_case["random_q16_capacity"] = 3
    overflow_case["generated_capacity"] = 3
    status, diagnostics = inference_generate_tokens_checked_topk_default_preflight_only_reference(
        **overflow_case
    )
    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert diagnostics is None


if __name__ == "__main__":
    test_source_contains_topk_default_preflight_only_wrapper()
    test_topk_default_preflight_only_matches_explicit_composition_randomized()
    test_topk_default_preflight_only_error_surface()
    print("inference_generate_tokens_checked_topk_default_preflight_only_reference_checks=ok")
