#!/usr/bin/env python3
"""Parity harness for InferenceGenerateTokensCheckedTopKTopPCommitOnlyPreflightOnlyParity (IQ-1049)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_inference_generate_tokens_preflight_checked import (
    I64_MAX,
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_OVERFLOW,
    SAMPLING_Q16_ERR_NULL_PTR,
    SAMPLING_Q16_OK,
    inference_generate_tokens_preflight_reference,
)
from test_inference_generate_tokens_checked_topk_default import SAMPLING_Q16_ONE


def inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_reference(
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
    out_generated_tokens: list[int] | None,
    generated_capacity: int,
    out_required_logits: list[int] | None,
    out_required_tokens: list[int] | None,
    out_final_token_count: list[int] | None,
) -> int:
    if out_required_logits is None or out_required_tokens is None or out_final_token_count is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if (
        out_required_logits is out_required_tokens
        or out_required_logits is out_final_token_count
        or out_required_tokens is out_final_token_count
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
        return SAMPLING_Q16_ERR_NULL_PTR

    if temperature_q16 <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if repetition_penalty_q16 < SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if top_k <= 0 or top_k > vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if top_p_q16 <= 0 or top_p_q16 > SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    snapshot = (
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16_capacity,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        generated_capacity,
    )

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

    if max_new_tokens:
        if vocab_size > I64_MAX // max_new_tokens:
            return SAMPLING_Q16_ERR_OVERFLOW
        commit_required_logits = vocab_size * max_new_tokens
    else:
        commit_required_logits = 0

    commit_required_tokens = max_new_tokens
    commit_final_token_count = token_history_count + max_new_tokens
    if commit_final_token_count < token_history_count:
        return SAMPLING_Q16_ERR_OVERFLOW

    if snapshot != (
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16_capacity,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        generated_capacity,
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    assert diagnostics is not None
    if diagnostics["required_step_logits_cells"] != commit_required_logits:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if diagnostics["required_generated_capacity"] != commit_required_tokens:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if diagnostics["required_history_capacity"] != commit_final_token_count:
        return SAMPLING_Q16_ERR_BAD_PARAM

    out_required_logits[0] = commit_required_logits
    out_required_tokens[0] = commit_required_tokens
    out_final_token_count[0] = commit_final_token_count
    return SAMPLING_Q16_OK


def test_source_contains_iq_1049_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceGenerateTokensCheckedTopKTopPCommitOnlyPreflightOnlyParity("
    assert signature in source
    body = source[source.index(signature):]
    assert "status = InferenceGenerateTokensPreflightChecked(" in body
    assert "if (required_step_logits_cells != commit_required_logits)" in body
    assert "if (required_generated_capacity != commit_required_tokens)" in body
    assert "if (required_history_capacity != commit_final_token_count)" in body
    assert "*out_required_logits = commit_required_logits;" in body
    assert "*out_required_tokens = commit_required_tokens;" in body
    assert "*out_final_token_count = commit_final_token_count;" in body


def test_parity_reference_success_vectors() -> None:
    rng = random.Random(20260422_1049)
    for _ in range(200):
        vocab_size = rng.randint(1, 128)
        max_new_tokens = rng.randint(0, 32)
        token_history_count = rng.randint(0, 32)
        token_history_capacity = token_history_count + max_new_tokens + rng.randint(0, 8)

        required_logits = vocab_size * max_new_tokens
        step_logits_capacity = required_logits + rng.randint(0, 16)
        random_q16_capacity = max_new_tokens + rng.randint(0, 8)
        generated_capacity = max_new_tokens + rng.randint(0, 8)

        step_logits_q16 = [0] * max(step_logits_capacity, 1)
        token_history = [0] * max(token_history_capacity, 1)
        random_q16_values = [0] * max(random_q16_capacity, 1)
        workspace_stage_logits_q16 = [0] * max(vocab_size + 4, 1)
        workspace_topk_logits_q16 = [0] * max(vocab_size + 4, 1)
        workspace_topk_indices = [0] * max(vocab_size + 4, 1)
        out_generated_tokens = [0] * max(generated_capacity, 1)

        out_required_logits = [-1]
        out_required_tokens = [-2]
        out_final_token_count = [-3]

        status = inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_reference(
            step_logits_q16=step_logits_q16,
            step_logits_capacity=step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history=token_history,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=vocab_size,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16_values=random_q16_values,
            random_q16_capacity=random_q16_capacity,
            workspace_stage_logits_q16=workspace_stage_logits_q16,
            workspace_stage_logits_capacity=len(workspace_stage_logits_q16),
            workspace_topk_logits_q16=workspace_topk_logits_q16,
            workspace_topk_logits_capacity=len(workspace_topk_logits_q16),
            workspace_topk_indices=workspace_topk_indices,
            workspace_topk_index_capacity=len(workspace_topk_indices),
            out_generated_tokens=out_generated_tokens,
            generated_capacity=generated_capacity,
            out_required_logits=out_required_logits,
            out_required_tokens=out_required_tokens,
            out_final_token_count=out_final_token_count,
        )
        assert status == SAMPLING_Q16_OK
        assert out_required_logits[0] == required_logits
        assert out_required_tokens[0] == max_new_tokens
        assert out_final_token_count[0] == token_history_count + max_new_tokens


def test_parity_reference_rejects_bad_sampling_domains() -> None:
    base = dict(
        step_logits_q16=[0] * 64,
        step_logits_capacity=64,
        vocab_size=8,
        max_new_tokens=4,
        token_history=[0] * 16,
        token_history_capacity=16,
        token_history_count=3,
        top_k=8,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16_values=[0] * 8,
        random_q16_capacity=8,
        workspace_stage_logits_q16=[0] * 8,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_q16=[0] * 8,
        workspace_topk_logits_capacity=8,
        workspace_topk_indices=[0] * 8,
        workspace_topk_index_capacity=8,
        out_generated_tokens=[0] * 8,
        generated_capacity=8,
    )

    out_required_logits = [11]
    out_required_tokens = [22]
    out_final_token_count = [33]

    status = inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_reference(
        **base,
        temperature_q16=0,
        out_required_logits=out_required_logits,
        out_required_tokens=out_required_tokens,
        out_final_token_count=out_final_token_count,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_required_logits == [11]
    assert out_required_tokens == [22]
    assert out_final_token_count == [33]

    base_bad_topk = dict(base)
    base_bad_topk["top_k"] = 0
    status = inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_reference(
        **base_bad_topk,
        temperature_q16=SAMPLING_Q16_ONE,
        out_required_logits=out_required_logits,
        out_required_tokens=out_required_tokens,
        out_final_token_count=out_final_token_count,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM


def test_parity_reference_no_partial_on_preflight_failure() -> None:
    out_required_logits = [101]
    out_required_tokens = [202]
    out_final_token_count = [303]

    status = inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_reference(
        step_logits_q16=[0] * 4,
        step_logits_capacity=4,
        vocab_size=8,
        max_new_tokens=2,
        token_history=[0] * 16,
        token_history_capacity=16,
        token_history_count=6,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=8,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16_values=[0] * 2,
        random_q16_capacity=2,
        workspace_stage_logits_q16=[0] * 8,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_q16=[0] * 8,
        workspace_topk_logits_capacity=8,
        workspace_topk_indices=[0] * 8,
        workspace_topk_index_capacity=8,
        out_generated_tokens=[0] * 2,
        generated_capacity=2,
        out_required_logits=out_required_logits,
        out_required_tokens=out_required_tokens,
        out_final_token_count=out_final_token_count,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_required_logits == [101]
    assert out_required_tokens == [202]
    assert out_final_token_count == [303]


if __name__ == "__main__":
    test_source_contains_iq_1049_wrapper()
    test_parity_reference_success_vectors()
    test_parity_reference_rejects_bad_sampling_domains()
    test_parity_reference_no_partial_on_preflight_failure()
    print("inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity=ok")
