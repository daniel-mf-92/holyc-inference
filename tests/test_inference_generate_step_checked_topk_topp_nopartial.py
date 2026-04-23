#!/usr/bin/env python3
"""Reference checks for InferenceGenerateStepCheckedTopKTopPNoPartial (IQ-1225)."""

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

SAMPLING_Q16_ERR_OVERFLOW = 4


def inference_generate_step_checked_topk_topp_nopartial_reference(
    *,
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    token_history: list[int] | None,
    token_history_capacity: int,
    token_history_count: int,
    temperature_q16: int,
    top_k: int,
    top_p_q16: int,
    repetition_penalty_q16: int,
    random_q16: int,
    workspace_stage_logits_q16: list[int] | None,
    workspace_stage_logits_capacity: int,
    workspace_topk_logits_q16: list[int] | None,
    workspace_topk_logits_capacity: int,
    workspace_topk_indices: list[int] | None,
    workspace_topk_index_capacity: int,
    workspace_history_stage: list[int] | None,
    workspace_history_stage_capacity: int,
    out_generated_token: list[int] | None,
    out_next_history_count: list[int] | None,
) -> int:
    if (
        logits_q16 is None
        or token_history is None
        or workspace_stage_logits_q16 is None
        or workspace_topk_logits_q16 is None
        or workspace_topk_indices is None
        or workspace_history_stage is None
        or out_generated_token is None
        or out_next_history_count is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR

    if out_generated_token is out_next_history_count:
        return SAMPLING_Q16_ERR_BAD_PARAM

    status, diagnostics = inference_generate_tokens_preflight_reference(
        step_logits_capacity=logits_capacity,
        vocab_size=vocab_size,
        max_new_tokens=1,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        workspace_stage_logits_capacity=workspace_stage_logits_capacity,
        workspace_topk_logits_capacity=workspace_topk_logits_capacity,
        workspace_topk_index_capacity=workspace_topk_index_capacity,
        random_q16_capacity=1,
        generated_capacity=1,
    )
    if status != SAMPLING_Q16_OK:
        return status
    assert diagnostics is not None

    required_step_logits_cells = diagnostics["required_step_logits_cells"]
    required_history_capacity = diagnostics["required_history_capacity"]
    required_stage_logits_capacity = diagnostics["required_stage_logits_capacity"]
    required_topk_capacity = diagnostics["required_topk_capacity"]

    if required_history_capacity > workspace_history_stage_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    for idx in range(token_history_count):
        workspace_history_stage[idx] = token_history[idx]

    status, sampled_token_id = generation_step_reference(
        logits_q16=logits_q16[:required_step_logits_cells],
        vocab_size=vocab_size,
        token_history=workspace_history_stage[:token_history_count],
        temperature_q16=temperature_q16,
        top_k=top_k,
        top_p_q16=top_p_q16,
        repetition_penalty_q16=repetition_penalty_q16,
        random_q16=random_q16,
    )
    if status != SAMPLING_Q16_OK:
        return status

    _ = required_stage_logits_capacity
    _ = required_topk_capacity

    workspace_history_stage[token_history_count] = sampled_token_id

    if token_history_count == 0x7FFFFFFFFFFFFFFF:
        return SAMPLING_Q16_ERR_OVERFLOW

    token_history[token_history_count] = workspace_history_stage[token_history_count]
    out_generated_token[0] = sampled_token_id
    out_next_history_count[0] = token_history_count + 1
    return SAMPLING_Q16_OK


def test_source_contains_inference_generate_step_checked_topk_topp_nopartial() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateStepCheckedTopKTopPNoPartial(" in source
    assert "status = GenerationStepChecked(" in source
    assert "token_history[token_history_count] = workspace_history_stage[token_history_count];" in source
    assert "*out_next_history_count = staged_next_history_count;" in source


def test_known_vector_success_single_step_commit() -> None:
    history = [3, 5, 0, 0]
    out_token = [777]
    out_count = [999]

    logits = [200000, 10000, 5000, -10000, -5000, 4000, 3000, 2000]

    status = inference_generate_step_checked_topk_topp_nopartial_reference(
        logits_q16=logits,
        logits_capacity=len(logits),
        vocab_size=8,
        token_history=history,
        token_history_capacity=len(history),
        token_history_count=2,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=4,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16=0,
        workspace_stage_logits_q16=[0] * 8,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_q16=[0] * 8,
        workspace_topk_logits_capacity=8,
        workspace_topk_indices=[0] * 8,
        workspace_topk_index_capacity=8,
        workspace_history_stage=[0] * len(history),
        workspace_history_stage_capacity=len(history),
        out_generated_token=out_token,
        out_next_history_count=out_count,
    )

    assert status == SAMPLING_Q16_OK
    assert out_count == [3]
    assert history[2] == out_token[0]
    assert 0 <= out_token[0] < 8


def test_failure_preserves_outputs_and_history() -> None:
    history = [1, 2, 0, 0]
    history_before = history.copy()
    out_token = [1234]
    out_count = [66]

    status = inference_generate_step_checked_topk_topp_nopartial_reference(
        logits_q16=[50000, 40000, 30000, 20000],
        logits_capacity=4,
        vocab_size=4,
        token_history=history,
        token_history_capacity=len(history),
        token_history_count=2,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=4,
        top_p_q16=0,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16=0,
        workspace_stage_logits_q16=[0] * 4,
        workspace_stage_logits_capacity=4,
        workspace_topk_logits_q16=[0] * 4,
        workspace_topk_logits_capacity=4,
        workspace_topk_indices=[0] * 4,
        workspace_topk_index_capacity=4,
        workspace_history_stage=[0] * len(history),
        workspace_history_stage_capacity=len(history),
        out_generated_token=out_token,
        out_next_history_count=out_count,
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert history == history_before
    assert out_token == [1234]
    assert out_count == [66]


def test_rejects_insufficient_history_workspace() -> None:
    history = [0, 1, 2]
    out_token = [9]
    out_count = [7]

    status = inference_generate_step_checked_topk_topp_nopartial_reference(
        logits_q16=[1000, 2000, 3000, 4000],
        logits_capacity=4,
        vocab_size=2,
        token_history=history,
        token_history_capacity=3,
        token_history_count=1,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=2,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16=0,
        workspace_stage_logits_q16=[0] * 2,
        workspace_stage_logits_capacity=2,
        workspace_topk_logits_q16=[0] * 2,
        workspace_topk_logits_capacity=2,
        workspace_topk_indices=[0] * 2,
        workspace_topk_index_capacity=2,
        workspace_history_stage=[0],
        workspace_history_stage_capacity=1,
        out_generated_token=out_token,
        out_next_history_count=out_count,
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_token == [9]
    assert out_count == [7]


def test_randomized_vectors() -> None:
    rng = random.Random(20260423_1225)
    for _ in range(300):
        vocab_size = rng.randint(2, 24)
        token_history_count = rng.randint(0, 7)
        token_history_capacity = token_history_count + 1

        history = [
            rng.randint(0, vocab_size - 1) if i < token_history_count else 0
            for i in range(token_history_capacity)
        ]
        out_token = [4444]
        out_count = [123]

        status = inference_generate_step_checked_topk_topp_nopartial_reference(
            logits_q16=[rng.randint(-150000, 150000) for _ in range(vocab_size)],
            logits_capacity=vocab_size,
            vocab_size=vocab_size,
            token_history=history,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            temperature_q16=rng.randint(1, 2 * SAMPLING_Q16_ONE),
            top_k=rng.randint(1, vocab_size),
            top_p_q16=rng.randint(1, SAMPLING_Q16_ONE),
            repetition_penalty_q16=rng.randint(SAMPLING_Q16_ONE, 2 * SAMPLING_Q16_ONE),
            random_q16=rng.randint(0, SAMPLING_Q16_ONE - 1),
            workspace_stage_logits_q16=[0] * vocab_size,
            workspace_stage_logits_capacity=vocab_size,
            workspace_topk_logits_q16=[0] * vocab_size,
            workspace_topk_logits_capacity=vocab_size,
            workspace_topk_indices=[0] * vocab_size,
            workspace_topk_index_capacity=vocab_size,
            workspace_history_stage=[0] * token_history_capacity,
            workspace_history_stage_capacity=token_history_capacity,
            out_generated_token=out_token,
            out_next_history_count=out_count,
        )

        assert status == SAMPLING_Q16_OK
        assert out_count[0] == token_history_count + 1
        assert history[token_history_count] == out_token[0]
        assert 0 <= out_token[0] < vocab_size


if __name__ == "__main__":
    test_source_contains_inference_generate_step_checked_topk_topp_nopartial()
    test_known_vector_success_single_step_commit()
    test_failure_preserves_outputs_and_history()
    test_rejects_insufficient_history_workspace()
    test_randomized_vectors()
    print("ok")
