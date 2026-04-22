#!/usr/bin/env python3
"""Harness for ...ParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity (IQ-1114)."""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_inference_generate_tokens_checked_topk_default import SAMPLING_Q16_ONE
from test_inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only import (
    inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference,
)
from test_inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only import (
    inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference,
)
from test_inference_generate_tokens_preflight_checked import (
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_NULL_PTR,
    SAMPLING_Q16_ERR_OVERFLOW,
    SAMPLING_Q16_OK,
)


def inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
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
    parity_fn=inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference,
    commit_fn=inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference,
) -> int:
    if (
        out_required_logits is None
        or out_required_tokens is None
        or out_final_token_count is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR

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

    if (
        out_required_logits is out_required_tokens
        or out_required_logits is out_final_token_count
        or out_required_tokens is out_final_token_count
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        step_logits_capacity < 0
        or vocab_size < 0
        or max_new_tokens < 0
        or token_history_capacity < 0
        or token_history_count < 0
        or random_q16_capacity < 0
        or workspace_stage_logits_capacity < 0
        or workspace_topk_logits_capacity < 0
        or workspace_topk_index_capacity < 0
        or generated_capacity < 0
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if temperature_q16 <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if repetition_penalty_q16 < SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if top_k <= 0 or top_k > vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if top_p_q16 <= 0 or top_p_q16 > SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM

    snapshot_step_logits_capacity = step_logits_capacity
    snapshot_vocab_size = vocab_size
    snapshot_max_new_tokens = max_new_tokens
    snapshot_token_history_capacity = token_history_capacity
    snapshot_token_history_count = token_history_count
    snapshot_temperature_q16 = temperature_q16
    snapshot_top_k = top_k
    snapshot_top_p_q16 = top_p_q16
    snapshot_repetition_penalty_q16 = repetition_penalty_q16
    snapshot_random_q16_capacity = random_q16_capacity
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity
    snapshot_generated_capacity = generated_capacity

    snapshot_step_logits_q16 = step_logits_q16
    snapshot_token_history = token_history
    snapshot_random_q16_values = random_q16_values
    snapshot_workspace_stage_logits_q16 = workspace_stage_logits_q16
    snapshot_workspace_topk_logits_q16 = workspace_topk_logits_q16
    snapshot_workspace_topk_indices = workspace_topk_indices
    snapshot_out_generated_tokens = out_generated_tokens
    snapshot_out_required_logits = out_required_logits
    snapshot_out_required_tokens = out_required_tokens
    snapshot_out_final_token_count = out_final_token_count

    staged_parity_required_logits = [0]
    staged_parity_required_tokens = [0]
    staged_parity_final_token_count = [0]

    staged_commit_required_logits = [0]
    staged_commit_required_tokens = [0]
    staged_commit_final_token_count = [0]

    status = parity_fn(
        step_logits_q16=step_logits_q16,
        step_logits_capacity=step_logits_capacity,
        vocab_size=vocab_size,
        max_new_tokens=max_new_tokens,
        token_history=token_history,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        temperature_q16=temperature_q16,
        top_k=top_k,
        top_p_q16=top_p_q16,
        repetition_penalty_q16=repetition_penalty_q16,
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
        out_required_logits=staged_parity_required_logits,
        out_required_tokens=staged_parity_required_tokens,
        out_final_token_count=staged_parity_final_token_count,
    )
    if status != SAMPLING_Q16_OK:
        return status

    status = commit_fn(
        step_logits_q16=step_logits_q16,
        step_logits_capacity=step_logits_capacity,
        vocab_size=vocab_size,
        max_new_tokens=max_new_tokens,
        token_history=token_history,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        temperature_q16=temperature_q16,
        top_k=top_k,
        top_p_q16=top_p_q16,
        repetition_penalty_q16=repetition_penalty_q16,
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
        out_required_logits=staged_commit_required_logits,
        out_required_tokens=staged_commit_required_tokens,
        out_final_token_count=staged_commit_final_token_count,
    )
    if status != SAMPLING_Q16_OK:
        return status

    if (
        snapshot_step_logits_capacity != step_logits_capacity
        or snapshot_vocab_size != vocab_size
        or snapshot_max_new_tokens != max_new_tokens
        or snapshot_token_history_capacity != token_history_capacity
        or snapshot_token_history_count != token_history_count
        or snapshot_temperature_q16 != temperature_q16
        or snapshot_top_k != top_k
        or snapshot_top_p_q16 != top_p_q16
        or snapshot_repetition_penalty_q16 != repetition_penalty_q16
        or snapshot_random_q16_capacity != random_q16_capacity
        or snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity
        or snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity
        or snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity
        or snapshot_generated_capacity != generated_capacity
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        snapshot_step_logits_q16 is not step_logits_q16
        or snapshot_token_history is not token_history
        or snapshot_random_q16_values is not random_q16_values
        or snapshot_workspace_stage_logits_q16 is not workspace_stage_logits_q16
        or snapshot_workspace_topk_logits_q16 is not workspace_topk_logits_q16
        or snapshot_workspace_topk_indices is not workspace_topk_indices
        or snapshot_out_generated_tokens is not out_generated_tokens
        or snapshot_out_required_logits is not out_required_logits
        or snapshot_out_required_tokens is not out_required_tokens
        or snapshot_out_final_token_count is not out_final_token_count
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        out_required_logits is out_required_tokens
        or out_required_logits is out_final_token_count
        or out_required_tokens is out_final_token_count
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        staged_parity_required_logits[0] < 0
        or staged_parity_required_tokens[0] < 0
        or staged_parity_final_token_count[0] < 0
        or staged_commit_required_logits[0] < 0
        or staged_commit_required_tokens[0] < 0
        or staged_commit_final_token_count[0] < 0
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if vocab_size < 0 or max_new_tokens < 0 or token_history_count < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM

    canonical_required_tokens = max_new_tokens
    if max_new_tokens == 0:
        canonical_required_logits = 0
    else:
        if vocab_size > ((1 << 63) - 1) // max_new_tokens:
            return SAMPLING_Q16_ERR_OVERFLOW
        canonical_required_logits = vocab_size * max_new_tokens

    canonical_final_token_count = token_history_count + max_new_tokens
    if canonical_final_token_count < token_history_count:
        return SAMPLING_Q16_ERR_OVERFLOW

    if (
        staged_parity_required_logits[0] != staged_commit_required_logits[0]
        or staged_parity_required_tokens[0] != staged_commit_required_tokens[0]
        or staged_parity_final_token_count[0] != staged_commit_final_token_count[0]
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        staged_parity_required_logits[0] != canonical_required_logits
        or staged_parity_required_tokens[0] != canonical_required_tokens
        or staged_parity_final_token_count[0] != canonical_final_token_count
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if snapshot_token_history_count > snapshot_token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if snapshot_top_k <= 0 or snapshot_top_k > snapshot_vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged_parity_required_logits[0] > snapshot_step_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if snapshot_top_k > snapshot_workspace_topk_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if snapshot_top_k > snapshot_workspace_topk_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged_parity_required_tokens[0] > snapshot_generated_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_parity_required_tokens[0] > snapshot_random_q16_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged_parity_final_token_count[0] > snapshot_token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    out_required_logits[0] = staged_parity_required_logits[0]
    out_required_tokens[0] = staged_parity_required_tokens[0]
    out_final_token_count[0] = staged_parity_final_token_count[0]
    return SAMPLING_Q16_OK


def test_source_contains_iq_1114_symbol() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    sig = (
        "I32 InferenceGenerateTokensCheckedTopKTopPCommitOnlyPreflightOnly"
        "ParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    )
    assert sig in source
    assert len(re.findall(re.escape(sig), source)) == 1
    body = source.split(sig, 1)[1]
    assert "InferenceGenerateTokensCheckedTopKTopPCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "InferenceGenerateTokensCheckedTopKTopPCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (staged_preflight_required_logits != staged_commit_required_logits ||" in body
    assert "snapshot_out_final_token_count != out_final_token_count" in body
    assert "if (!step_logits_q16 || !token_history || !random_q16_values ||" in source
    assert "if (staged_preflight_required_logits != staged_commit_required_logits ||" in body


def test_known_vector_success() -> None:
    out_required_logits = [111]
    out_required_tokens = [222]
    out_final_token_count = [333]

    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 64,
            step_logits_capacity=64,
            vocab_size=16,
            max_new_tokens=4,
            token_history=[0] * 16,
            token_history_capacity=16,
            token_history_count=8,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=16,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
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
            out_required_logits=out_required_logits,
            out_required_tokens=out_required_tokens,
            out_final_token_count=out_final_token_count,
        )
    )

    assert status == SAMPLING_Q16_OK
    assert out_required_logits == [64]
    assert out_required_tokens == [4]
    assert out_final_token_count == [12]


def test_bad_param_and_no_publish() -> None:
    out_required_logits = [1001]
    out_required_tokens = [1002]
    out_final_token_count = [1003]

    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 16,
            step_logits_capacity=16,
            vocab_size=8,
            max_new_tokens=2,
            token_history=[0] * 4,
            token_history_capacity=4,
            token_history_count=2,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=0,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16_values=[0, 1],
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
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_required_logits == [1001]
    assert out_required_tokens == [1002]
    assert out_final_token_count == [1003]


def test_null_output_pointer_rejected() -> None:
    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 4,
            step_logits_capacity=4,
            vocab_size=2,
            max_new_tokens=1,
            token_history=[0] * 2,
            token_history_capacity=2,
            token_history_count=1,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=2,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16_values=[0],
            random_q16_capacity=1,
            workspace_stage_logits_q16=[0] * 2,
            workspace_stage_logits_capacity=2,
            workspace_topk_logits_q16=[0] * 2,
            workspace_topk_logits_capacity=2,
            workspace_topk_indices=[0] * 2,
            workspace_topk_index_capacity=2,
            out_generated_tokens=[0],
            generated_capacity=1,
            out_required_logits=None,
            out_required_tokens=[0],
            out_final_token_count=[0],
        )
    )
    assert status == SAMPLING_Q16_ERR_NULL_PTR


def test_null_required_input_pointer_rejected() -> None:
    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=None,
            step_logits_capacity=4,
            vocab_size=2,
            max_new_tokens=1,
            token_history=[0] * 2,
            token_history_capacity=2,
            token_history_count=1,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=2,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16_values=[0],
            random_q16_capacity=1,
            workspace_stage_logits_q16=[0] * 2,
            workspace_stage_logits_capacity=2,
            workspace_topk_logits_q16=[0] * 2,
            workspace_topk_logits_capacity=2,
            workspace_topk_indices=[0] * 2,
            workspace_topk_index_capacity=2,
            out_generated_tokens=[0],
            generated_capacity=1,
            out_required_logits=[0],
            out_required_tokens=[0],
            out_final_token_count=[0],
        )
    )
    assert status == SAMPLING_Q16_ERR_NULL_PTR


def test_bad_capacity_contract_rejected_without_publish() -> None:
    out_required_logits = [100]
    out_required_tokens = [200]
    out_final_token_count = [300]

    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 15,
            step_logits_capacity=15,
            vocab_size=8,
            max_new_tokens=2,
            token_history=[0] * 6,
            token_history_capacity=6,
            token_history_count=4,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=8,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16_values=[0, 1],
            random_q16_capacity=2,
            workspace_stage_logits_q16=[0] * 15,
            workspace_stage_logits_capacity=15,
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
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_required_logits == [100]
    assert out_required_tokens == [200]
    assert out_final_token_count == [300]


def test_alias_output_pointer_rejected() -> None:
    aliased = [0]
    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 4,
            step_logits_capacity=4,
            vocab_size=2,
            max_new_tokens=1,
            token_history=[0] * 2,
            token_history_capacity=2,
            token_history_count=1,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=2,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16_values=[0],
            random_q16_capacity=1,
            workspace_stage_logits_q16=[0] * 2,
            workspace_stage_logits_capacity=2,
            workspace_topk_logits_q16=[0] * 2,
            workspace_topk_logits_capacity=2,
            workspace_topk_indices=[0] * 2,
            workspace_topk_index_capacity=2,
            out_generated_tokens=[0],
            generated_capacity=1,
            out_required_logits=aliased,
            out_required_tokens=aliased,
            out_final_token_count=[0],
        )
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM


def test_no_publish_when_parity_tuple_mismatch() -> None:
    out_required_logits = [31]
    out_required_tokens = [32]
    out_final_token_count = [33]

    def _bad_parity(**kwargs: object) -> int:
        assert isinstance(kwargs["out_required_logits"], list)
        assert isinstance(kwargs["out_required_tokens"], list)
        assert isinstance(kwargs["out_final_token_count"], list)
        kwargs["out_required_logits"][0] = 77
        kwargs["out_required_tokens"][0] = 5
        kwargs["out_final_token_count"][0] = 19
        return SAMPLING_Q16_OK

    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 64,
            step_logits_capacity=64,
            vocab_size=16,
            max_new_tokens=4,
            token_history=[0] * 16,
            token_history_capacity=16,
            token_history_count=8,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=16,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
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
            out_required_logits=out_required_logits,
            out_required_tokens=out_required_tokens,
            out_final_token_count=out_final_token_count,
            parity_fn=_bad_parity,
        )
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_required_logits == [31]
    assert out_required_tokens == [32]
    assert out_final_token_count == [33]


def test_no_publish_when_commit_tuple_mismatch() -> None:
    out_required_logits = [41]
    out_required_tokens = [42]
    out_final_token_count = [43]

    def _bad_commit(**kwargs: object) -> int:
        assert isinstance(kwargs["out_required_logits"], list)
        assert isinstance(kwargs["out_required_tokens"], list)
        assert isinstance(kwargs["out_final_token_count"], list)
        kwargs["out_required_logits"][0] = 78
        kwargs["out_required_tokens"][0] = 6
        kwargs["out_final_token_count"][0] = 20
        return SAMPLING_Q16_OK

    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 64,
            step_logits_capacity=64,
            vocab_size=16,
            max_new_tokens=4,
            token_history=[0] * 16,
            token_history_capacity=16,
            token_history_count=8,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=16,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
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
            out_required_logits=out_required_logits,
            out_required_tokens=out_required_tokens,
            out_final_token_count=out_final_token_count,
            commit_fn=_bad_commit,
        )
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_required_logits == [41]
    assert out_required_tokens == [42]
    assert out_final_token_count == [43]


def test_no_publish_when_canonical_tuple_mismatch() -> None:
    out_required_logits = [51]
    out_required_tokens = [52]
    out_final_token_count = [53]

    def _bad_parity(**kwargs: object) -> int:
        assert isinstance(kwargs["out_required_logits"], list)
        assert isinstance(kwargs["out_required_tokens"], list)
        assert isinstance(kwargs["out_final_token_count"], list)
        kwargs["out_required_logits"][0] = 1
        kwargs["out_required_tokens"][0] = 2
        kwargs["out_final_token_count"][0] = 3
        return SAMPLING_Q16_OK

    def _bad_commit(**kwargs: object) -> int:
        assert isinstance(kwargs["out_required_logits"], list)
        assert isinstance(kwargs["out_required_tokens"], list)
        assert isinstance(kwargs["out_final_token_count"], list)
        kwargs["out_required_logits"][0] = 1
        kwargs["out_required_tokens"][0] = 2
        kwargs["out_final_token_count"][0] = 3
        return SAMPLING_Q16_OK

    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 64,
            step_logits_capacity=64,
            vocab_size=16,
            max_new_tokens=4,
            token_history=[0] * 16,
            token_history_capacity=16,
            token_history_count=8,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=16,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
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
            out_required_logits=out_required_logits,
            out_required_tokens=out_required_tokens,
            out_final_token_count=out_final_token_count,
            parity_fn=_bad_parity,
            commit_fn=_bad_commit,
        )
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_required_logits == [51]
    assert out_required_tokens == [52]
    assert out_final_token_count == [53]


def test_overflow_guard_for_required_logits_and_no_publish() -> None:
    out_required_logits = [61]
    out_required_tokens = [62]
    out_final_token_count = [63]

    huge_vocab = (1 << 62)

    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0],
            step_logits_capacity=1,
            vocab_size=huge_vocab,
            max_new_tokens=4,
            token_history=[0] * 10,
            token_history_capacity=10,
            token_history_count=3,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=1,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16_values=[0, 1, 2, 3],
            random_q16_capacity=4,
            workspace_stage_logits_q16=[0],
            workspace_stage_logits_capacity=1,
            workspace_topk_logits_q16=[0],
            workspace_topk_logits_capacity=1,
            workspace_topk_indices=[0],
            workspace_topk_index_capacity=1,
            out_generated_tokens=[0] * 4,
            generated_capacity=4,
            out_required_logits=out_required_logits,
            out_required_tokens=out_required_tokens,
            out_final_token_count=out_final_token_count,
            parity_fn=lambda **kwargs: SAMPLING_Q16_OK,
            commit_fn=lambda **kwargs: SAMPLING_Q16_OK,
        )
    )

    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert out_required_logits == [61]
    assert out_required_tokens == [62]
    assert out_final_token_count == [63]


def test_adversarial_matrix_parity_with_explicit_composition() -> None:
    rng = random.Random(1097)

    for _ in range(256):
        vocab_size = rng.randint(1, 64)
        max_new_tokens = rng.randint(0, 12)
        token_history_count = rng.randint(0, 20)
        step_logits_capacity = rng.randint(vocab_size * max_new_tokens, vocab_size * max_new_tokens + 32)
        token_history_capacity = rng.randint(
            token_history_count + max_new_tokens, token_history_count + max_new_tokens + 32
        )
        top_k = rng.randint(1, vocab_size)

        kwargs = dict(
            step_logits_q16=[0] * max(step_logits_capacity, 1),
            step_logits_capacity=step_logits_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history=[0] * max(token_history_capacity, 1),
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=top_k,
            top_p_q16=rng.randint(1, SAMPLING_Q16_ONE),
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16_values=[0] * max(max_new_tokens, 1),
            random_q16_capacity=max_new_tokens,
            workspace_stage_logits_q16=[0] * vocab_size,
            workspace_stage_logits_capacity=vocab_size,
            workspace_topk_logits_q16=[0] * vocab_size,
            workspace_topk_logits_capacity=vocab_size,
            workspace_topk_indices=[0] * vocab_size,
            workspace_topk_index_capacity=vocab_size,
            out_generated_tokens=[0] * max(max_new_tokens, 1),
            generated_capacity=max_new_tokens,
        )

        expected_logits = [0]
        expected_tokens = [0]
        expected_final = [0]
        expected_status = inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
            **kwargs,
            out_required_logits=expected_logits,
            out_required_tokens=expected_tokens,
            out_final_token_count=expected_final,
        )

        out_logits = [909]
        out_tokens = [808]
        out_final = [707]
        status = inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            **kwargs,
            out_required_logits=out_logits,
            out_required_tokens=out_tokens,
            out_final_token_count=out_final,
        )

        assert status == expected_status
        if status == SAMPLING_Q16_OK:
            assert out_logits == expected_logits
            assert out_tokens == expected_tokens
            assert out_final == expected_final
        else:
            assert out_logits == [909]
            assert out_tokens == [808]
            assert out_final == [707]
