#!/usr/bin/env python3
"""Reference checks for InferenceGenerateTokensCheckedTopKTopPNoPartial (IQ-1222)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_generation_run_checked import (
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_NULL_PTR,
    SAMPLING_Q16_OK,
    SAMPLING_Q16_ONE,
    generation_step_reference,
)
from test_inference_generate_tokens_preflight_checked import (
    inference_generate_tokens_preflight_reference,
)


def inference_generate_tokens_checked_topk_topp_nopartial_reference(
    *,
    step_logits_q16: list[int] | None,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history: list[int] | None,
    token_history_capacity: int,
    token_history_count: int,
    temperature_q16: int,
    top_k: int,
    top_p_q16: int,
    repetition_penalty_q16: int,
    random_q16_values: list[int] | None,
    random_q16_capacity: int,
    workspace_stage_logits_q16: list[int] | None,
    workspace_stage_logits_capacity: int,
    workspace_topk_logits_q16: list[int] | None,
    workspace_topk_logits_capacity: int,
    workspace_topk_indices: list[int] | None,
    workspace_topk_index_capacity: int,
    workspace_history_stage: list[int] | None,
    workspace_history_stage_capacity: int,
    workspace_generated_stage: list[int] | None,
    workspace_generated_stage_capacity: int,
    out_generated_tokens: list[int] | None,
    generated_capacity: int,
    out_generated_count: list[int] | None,
) -> int:
    if (
        step_logits_q16 is None
        or token_history is None
        or random_q16_values is None
        or workspace_stage_logits_q16 is None
        or workspace_topk_logits_q16 is None
        or workspace_topk_indices is None
        or workspace_history_stage is None
        or workspace_generated_stage is None
        or out_generated_tokens is None
        or out_generated_count is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR

    status, diagnostics = inference_generate_tokens_preflight_reference(
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
        return status
    assert diagnostics is not None

    required_history_capacity = diagnostics["required_history_capacity"]
    required_stage_logits_capacity = diagnostics["required_stage_logits_capacity"]
    required_topk_capacity = diagnostics["required_topk_capacity"]
    required_generated_capacity = diagnostics["required_generated_capacity"]

    if required_history_capacity > workspace_history_stage_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if required_generated_capacity > workspace_generated_stage_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    for i in range(token_history_count):
        workspace_history_stage[i] = token_history[i]

    for step_index in range(max_new_tokens):
        logits_row_offset = step_index * vocab_size
        logits_row_end = (step_index + 1) * vocab_size
        if logits_row_end > step_logits_capacity:
            return SAMPLING_Q16_ERR_BAD_PARAM

        status, sampled_token_id = generation_step_reference(
            logits_q16=step_logits_q16[logits_row_offset:logits_row_end],
            vocab_size=vocab_size,
            token_history=workspace_history_stage[: token_history_count + step_index],
            temperature_q16=temperature_q16,
            top_k=top_k,
            top_p_q16=top_p_q16,
            repetition_penalty_q16=repetition_penalty_q16,
            random_q16=random_q16_values[step_index],
        )
        if status != SAMPLING_Q16_OK:
            return status

        workspace_generated_stage[step_index] = sampled_token_id
        workspace_history_stage[token_history_count + step_index] = sampled_token_id

        token_history[token_history_count + step_index] = workspace_history_stage[
            token_history_count + step_index
        ]
        out_generated_tokens[step_index] = workspace_generated_stage[step_index]
        out_generated_count[0] = step_index + 1

    return SAMPLING_Q16_OK


def test_source_contains_inference_generate_tokens_checked_topk_topp_nopartial() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensCheckedTopKTopPNoPartial(" in source
    assert "workspace_history_stage[token_history_count + step_index] = sampled_token_id;" in source
    assert "out_generated_tokens[step_index] = workspace_generated_stage[step_index];" in source
    assert "*out_generated_count = step_index + 1;" in source


def test_known_vector_success_per_step_commit() -> None:
    token_history = [3, 5, 7, 0, 0, 0, 0]
    out_tokens = [777, 777, 777, 777]
    out_count = [99]
    workspace_history_stage = [0] * len(token_history)
    workspace_generated_stage = [0] * 4

    step_logits_rows = [
        [200000, 10000, 5000, -10000, -5000, 4000, 3000, 2000],
        [8000, 7000, 6000, 5000, 120000, 4000, 3000, 2000],
        [3000, 2000, 1000, 150000, 900, 800, 700, 600],
    ]
    step_logits = [v for row in step_logits_rows for v in row]

    status = inference_generate_tokens_checked_topk_topp_nopartial_reference(
        step_logits_q16=step_logits,
        step_logits_capacity=len(step_logits),
        vocab_size=8,
        max_new_tokens=3,
        token_history=token_history,
        token_history_capacity=len(token_history),
        token_history_count=3,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=4,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16_values=[0, 0, 0],
        random_q16_capacity=3,
        workspace_stage_logits_q16=[0] * 8,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_q16=[0] * 8,
        workspace_topk_logits_capacity=8,
        workspace_topk_indices=[0] * 8,
        workspace_topk_index_capacity=8,
        workspace_history_stage=workspace_history_stage,
        workspace_history_stage_capacity=len(workspace_history_stage),
        workspace_generated_stage=workspace_generated_stage,
        workspace_generated_stage_capacity=len(workspace_generated_stage),
        out_generated_tokens=out_tokens,
        generated_capacity=4,
        out_generated_count=out_count,
    )

    assert status == SAMPLING_Q16_OK
    assert out_count == [3]
    assert out_tokens[:3] == token_history[3:6]
    for token in out_tokens[:3]:
        assert 0 <= token < 8


def test_failure_preserves_current_step_atomicity_and_prior_commits() -> None:
    token_history = [1, 2, 0, 0, 0, 0]
    out_tokens = [111, 112, 113]
    out_count = [41]
    workspace_history_stage = [0] * len(token_history)
    workspace_generated_stage = [0] * 3

    step_logits_rows = [
        [50000, 40000, 30000, 20000],
        [15000, 14000, 13000, 12000],
        [11000, 10000, 9000, 8000],
    ]
    step_logits = [v for row in step_logits_rows for v in row]

    status = inference_generate_tokens_checked_topk_topp_nopartial_reference(
        step_logits_q16=step_logits,
        step_logits_capacity=len(step_logits),
        vocab_size=4,
        max_new_tokens=3,
        token_history=token_history,
        token_history_capacity=len(token_history),
        token_history_count=2,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=4,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16_values=[0, SAMPLING_Q16_ONE, 0],
        random_q16_capacity=3,
        workspace_stage_logits_q16=[0] * 4,
        workspace_stage_logits_capacity=4,
        workspace_topk_logits_q16=[0] * 4,
        workspace_topk_logits_capacity=4,
        workspace_topk_indices=[0] * 4,
        workspace_topk_index_capacity=4,
        workspace_history_stage=workspace_history_stage,
        workspace_history_stage_capacity=len(workspace_history_stage),
        workspace_generated_stage=workspace_generated_stage,
        workspace_generated_stage_capacity=len(workspace_generated_stage),
        out_generated_tokens=out_tokens,
        generated_capacity=3,
        out_generated_count=out_count,
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_count == [1]
    assert out_tokens[1:] == [112, 113]
    assert token_history[2] == out_tokens[0]
    assert token_history[3:] == [0, 0, 0]


def test_bad_workspace_history_capacity_rejected() -> None:
    token_history = [0, 1, 2]
    out_tokens = [9, 9]
    out_count = [7]

    status = inference_generate_tokens_checked_topk_topp_nopartial_reference(
        step_logits_q16=[1000, 2000, 3000, 4000],
        step_logits_capacity=4,
        vocab_size=2,
        max_new_tokens=2,
        token_history=token_history,
        token_history_capacity=3,
        token_history_count=1,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=2,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16_values=[0, 0],
        random_q16_capacity=2,
        workspace_stage_logits_q16=[0] * 2,
        workspace_stage_logits_capacity=2,
        workspace_topk_logits_q16=[0] * 2,
        workspace_topk_logits_capacity=2,
        workspace_topk_indices=[0] * 2,
        workspace_topk_index_capacity=2,
        workspace_history_stage=[0],
        workspace_history_stage_capacity=1,
        workspace_generated_stage=[0, 0],
        workspace_generated_stage_capacity=2,
        out_generated_tokens=out_tokens,
        generated_capacity=2,
        out_generated_count=out_count,
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_tokens == [9, 9]
    assert out_count == [7]


def test_randomized_vectors() -> None:
    rng = random.Random(20260423_1222)
    for _ in range(200):
        vocab_size = rng.randint(2, 24)
        max_new_tokens = rng.randint(1, 6)
        token_history_count = rng.randint(0, 5)
        token_history_capacity = token_history_count + max_new_tokens

        token_history = [
            rng.randint(0, vocab_size - 1) if i < token_history_count else 0
            for i in range(token_history_capacity)
        ]

        step_logits = [
            rng.randint(-150000, 150000)
            for _ in range(vocab_size * max_new_tokens)
        ]

        out_tokens = [4444] * max_new_tokens
        out_count = [123]

        status = inference_generate_tokens_checked_topk_topp_nopartial_reference(
            step_logits_q16=step_logits,
            step_logits_capacity=len(step_logits),
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history=token_history,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            temperature_q16=rng.randint(1, 2 * SAMPLING_Q16_ONE),
            top_k=rng.randint(1, vocab_size),
            top_p_q16=rng.randint(1, SAMPLING_Q16_ONE),
            repetition_penalty_q16=rng.randint(SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE),
            random_q16_values=[rng.randint(0, SAMPLING_Q16_ONE - 1) for _ in range(max_new_tokens)],
            random_q16_capacity=max_new_tokens,
            workspace_stage_logits_q16=[0] * vocab_size,
            workspace_stage_logits_capacity=vocab_size,
            workspace_topk_logits_q16=[0] * vocab_size,
            workspace_topk_logits_capacity=vocab_size,
            workspace_topk_indices=[0] * vocab_size,
            workspace_topk_index_capacity=vocab_size,
            workspace_history_stage=[0] * token_history_capacity,
            workspace_history_stage_capacity=token_history_capacity,
            workspace_generated_stage=[0] * max_new_tokens,
            workspace_generated_stage_capacity=max_new_tokens,
            out_generated_tokens=out_tokens,
            generated_capacity=max_new_tokens,
            out_generated_count=out_count,
        )

        assert status == SAMPLING_Q16_OK
        assert out_count[0] == max_new_tokens
        assert out_tokens == token_history[token_history_count : token_history_count + max_new_tokens]
