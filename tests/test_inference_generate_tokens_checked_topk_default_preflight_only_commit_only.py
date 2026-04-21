#!/usr/bin/env python3
"""Parity harness for InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnly (IQ-831)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_inference_generate_tokens_preflight_checked_topk_default import (
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_OVERFLOW,
    SAMPLING_Q16_OK,
)
from test_inference_generate_tokens_checked_topk_default import SAMPLING_Q16_ONE
from test_inference_generate_tokens_checked_topk_default_preflight_only import (
    inference_generate_tokens_checked_topk_default_preflight_only_reference,
)


def inference_generate_tokens_checked_topk_default_preflight_only_commit_only_reference(
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
    out_required_step_logits_cells: list[int] | None,
    out_required_generated_capacity: list[int] | None,
    out_required_random_capacity: list[int] | None,
) -> int:
    if (
        out_required_step_logits_cells is None
        or out_required_generated_capacity is None
        or out_required_random_capacity is None
    ):
        return 1

    if (
        out_required_step_logits_cells is out_required_generated_capacity
        or out_required_step_logits_cells is out_required_random_capacity
        or out_required_generated_capacity is out_required_random_capacity
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        step_logits_q16 is None
        or token_history is None
        or random_q16_values is None
        or workspace_stage_logits_q16 is None
        or workspace_topk_logits_q16 is None
        or workspace_topk_indices is None
        or out_generated_tokens is None
    ):
        return 1

    if top_p_q16 <= 0 or top_p_q16 > SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    status, diagnostics = inference_generate_tokens_checked_topk_default_preflight_only_reference(
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
    if status != SAMPLING_Q16_OK:
        return status

    out_required_step_logits_cells[0] = diagnostics["required_step_logits_cells"]
    out_required_generated_capacity[0] = diagnostics["required_generated_capacity"]
    out_required_random_capacity[0] = diagnostics["required_random_capacity"]
    return SAMPLING_Q16_OK


def test_source_contains_commit_only_helper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnly(" in source
    assert "InferenceGenerateTokensCheckedTopKDefaultPreflightOnly(" in source
    assert "*out_required_generated_capacity = staged_required_generated_capacity;" in source


def test_known_vector() -> None:
    out_step = [7]
    out_gen = [8]
    out_rand = [9]

    status = inference_generate_tokens_checked_topk_default_preflight_only_commit_only_reference(
        step_logits_q16=[0] * 64,
        step_logits_capacity=64,
        vocab_size=16,
        max_new_tokens=4,
        token_history=[0] * 12,
        token_history_capacity=12,
        token_history_count=8,
        top_p_q16=SAMPLING_Q16_ONE,
        random_q16_values=[0, 1, 2, 3],
        random_q16_capacity=4,
        workspace_stage_logits_q16=[0] * 16,
        workspace_stage_logits_capacity=16,
        workspace_topk_logits_q16=[0] * 16,
        workspace_topk_logits_capacity=16,
        workspace_topk_indices=[0] * 16,
        workspace_topk_index_capacity=16,
        out_generated_tokens=[0] * 4,
        generated_capacity=4,
        out_required_step_logits_cells=out_step,
        out_required_generated_capacity=out_gen,
        out_required_random_capacity=out_rand,
    )

    assert status == SAMPLING_Q16_OK
    assert out_step == [64]
    assert out_gen == [4]
    assert out_rand == [4]


def test_error_no_publish() -> None:
    out_step = [101]
    out_gen = [102]
    out_rand = [103]

    status = inference_generate_tokens_checked_topk_default_preflight_only_commit_only_reference(
        step_logits_q16=[0] * 64,
        step_logits_capacity=64,
        vocab_size=16,
        max_new_tokens=4,
        token_history=[0] * 12,
        token_history_capacity=12,
        token_history_count=8,
        top_p_q16=0,
        random_q16_values=[0, 1, 2, 3],
        random_q16_capacity=4,
        workspace_stage_logits_q16=[0] * 16,
        workspace_stage_logits_capacity=16,
        workspace_topk_logits_q16=[0] * 16,
        workspace_topk_logits_capacity=16,
        workspace_topk_indices=[0] * 16,
        workspace_topk_index_capacity=16,
        out_generated_tokens=[0] * 4,
        generated_capacity=4,
        out_required_step_logits_cells=out_step,
        out_required_generated_capacity=out_gen,
        out_required_random_capacity=out_rand,
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_step == [101]
    assert out_gen == [102]
    assert out_rand == [103]


def test_randomized_parity_with_explicit_preflight() -> None:
    rng = random.Random(20260421_831)

    for _ in range(500):
        vocab_size = rng.randint(1, 320)
        max_new_tokens = rng.randint(0, 48)
        token_history_count = rng.randint(0, 64)

        token_history_capacity = token_history_count + max_new_tokens + rng.randint(0, 3)
        step_logits_capacity = vocab_size * max_new_tokens + rng.randint(0, 3)
        random_capacity = max_new_tokens + rng.randint(0, 2)
        generated_capacity = max_new_tokens + rng.randint(0, 2)
        stage_capacity = vocab_size + rng.randint(0, 2)
        topk_logits_capacity = vocab_size + rng.randint(0, 2)
        topk_index_capacity = vocab_size + rng.randint(0, 2)

        top_p_q16 = rng.randint(1, SAMPLING_Q16_ONE)
        if rng.random() < 0.1:
            top_p_q16 = rng.choice([0, SAMPLING_Q16_ONE + 1])

        out_step = [11]
        out_gen = [22]
        out_rand = [33]

        status = inference_generate_tokens_checked_topk_default_preflight_only_commit_only_reference(
            step_logits_q16=[0] * max(1, step_logits_capacity),
            step_logits_capacity=step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history=[0] * max(1, token_history_capacity),
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            top_p_q16=top_p_q16,
            random_q16_values=[0] * max(1, random_capacity),
            random_q16_capacity=random_capacity,
            workspace_stage_logits_q16=[0] * max(1, stage_capacity),
            workspace_stage_logits_capacity=stage_capacity,
            workspace_topk_logits_q16=[0] * max(1, topk_logits_capacity),
            workspace_topk_logits_capacity=topk_logits_capacity,
            workspace_topk_indices=[0] * max(1, topk_index_capacity),
            workspace_topk_index_capacity=topk_index_capacity,
            out_generated_tokens=[0] * max(1, generated_capacity),
            generated_capacity=generated_capacity,
            out_required_step_logits_cells=out_step,
            out_required_generated_capacity=out_gen,
            out_required_random_capacity=out_rand,
        )

        explicit_status, diagnostics = inference_generate_tokens_checked_topk_default_preflight_only_reference(
            step_logits_q16=[0] * max(1, step_logits_capacity),
            step_logits_capacity=step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history=[0] * max(1, token_history_capacity),
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            top_p_q16=top_p_q16,
            random_q16_values=[0] * max(1, random_capacity),
            random_q16_capacity=random_capacity,
            workspace_stage_logits_q16=[0] * max(1, stage_capacity),
            workspace_stage_logits_capacity=stage_capacity,
            workspace_topk_logits_q16=[0] * max(1, topk_logits_capacity),
            workspace_topk_logits_capacity=topk_logits_capacity,
            workspace_topk_indices=[0] * max(1, topk_index_capacity),
            workspace_topk_index_capacity=topk_index_capacity,
            out_generated_tokens=[0] * max(1, generated_capacity),
            generated_capacity=generated_capacity,
        )

        assert status == explicit_status
        if status == SAMPLING_Q16_OK:
            assert out_step == [diagnostics["required_step_logits_cells"]]
            assert out_gen == [diagnostics["required_generated_capacity"]]
            assert out_rand == [diagnostics["required_random_capacity"]]
        else:
            assert out_step == [11]
            assert out_gen == [22]
            assert out_rand == [33]


if __name__ == "__main__":
    test_source_contains_commit_only_helper()
    test_known_vector()
    test_error_no_publish()
    test_randomized_parity_with_explicit_preflight()
    print("ok")
