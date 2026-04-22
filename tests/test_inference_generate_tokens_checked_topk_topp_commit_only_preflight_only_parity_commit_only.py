#!/usr/bin/env python3
"""Commit-only harness for InferenceGenerateTokensCheckedTopKTopPCommitOnlyPreflightOnlyParityCommitOnly (IQ-1057)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_inference_generate_tokens_checked_topk_default import SAMPLING_Q16_ONE
from test_inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity import (
    inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_reference,
)
from test_inference_generate_tokens_preflight_checked import (
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_NULL_PTR,
    SAMPLING_Q16_ERR_OVERFLOW,
    SAMPLING_Q16_OK,
)


def inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_reference(
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
    parity_fn=inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_reference,
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

    staged_required_logits = out_required_logits[0]
    staged_required_tokens = out_required_tokens[0]
    staged_final_token_count = out_final_token_count[0]

    parity_out_required_logits = [staged_required_logits]
    parity_out_required_tokens = [staged_required_tokens]
    parity_out_final_token_count = [staged_final_token_count]

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
        out_required_logits=parity_out_required_logits,
        out_required_tokens=parity_out_required_tokens,
        out_final_token_count=parity_out_final_token_count,
    )
    if status != SAMPLING_Q16_OK:
        return status

    parity_required_logits = parity_out_required_logits[0]
    parity_required_tokens = parity_out_required_tokens[0]
    parity_final_token_count = parity_out_final_token_count[0]

    if max_new_tokens < 0 or token_history_count < 0 or vocab_size < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if max_new_tokens:
        if vocab_size > 0x7FFFFFFFFFFFFFFF // max_new_tokens:
            return SAMPLING_Q16_ERR_OVERFLOW
        canonical_required_logits = vocab_size * max_new_tokens
    else:
        canonical_required_logits = 0

    canonical_required_tokens = max_new_tokens
    canonical_final_token_count = token_history_count + max_new_tokens
    if canonical_final_token_count > 0x7FFFFFFFFFFFFFFF:
        return SAMPLING_Q16_ERR_OVERFLOW

    if (
        parity_required_logits != canonical_required_logits
        or parity_required_tokens != canonical_required_tokens
        or parity_final_token_count != canonical_final_token_count
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    staged_required_logits = parity_required_logits
    staged_required_tokens = parity_required_tokens
    staged_final_token_count = parity_final_token_count

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

    out_required_logits[0] = staged_required_logits
    out_required_tokens[0] = staged_required_tokens
    out_final_token_count[0] = staged_final_token_count
    return SAMPLING_Q16_OK


def test_source_contains_topk_topp_commit_only_preflight_only_parity_commit_only() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    sig = "I32 InferenceGenerateTokensCheckedTopKTopPCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source
    assert "status = InferenceGenerateTokensCheckedTopKTopPCommitOnlyPreflightOnlyParity(" in source
    assert "canonical_required_logits = vocab_size * max_new_tokens;" in source
    assert "if (parity_required_logits != canonical_required_logits ||" in source
    assert "snapshot_out_final_token_count = out_final_token_count;" in source
    assert "if (snapshot_step_logits_q16 != step_logits_q16 ||" in source
    assert "snapshot_out_final_token_count != out_final_token_count)" in source


def test_known_vector_success() -> None:
    out_required_logits = [71]
    out_required_tokens = [72]
    out_final_token_count = [73]

    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 60,
            step_logits_capacity=60,
            vocab_size=15,
            max_new_tokens=4,
            token_history=[1] * 20,
            token_history_capacity=20,
            token_history_count=16,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=10,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16_values=[0, 1, 2, 3],
            random_q16_capacity=4,
            workspace_stage_logits_q16=[0] * 15,
            workspace_stage_logits_capacity=15,
            workspace_topk_logits_q16=[0] * 15,
            workspace_topk_logits_capacity=15,
            workspace_topk_indices=[0] * 15,
            workspace_topk_index_capacity=15,
            out_generated_tokens=[0] * 4,
            generated_capacity=4,
            out_required_logits=out_required_logits,
            out_required_tokens=out_required_tokens,
            out_final_token_count=out_final_token_count,
        )
    )

    assert status == SAMPLING_Q16_OK
    assert out_required_logits == [60]
    assert out_required_tokens == [4]
    assert out_final_token_count == [20]


def test_bad_param_no_publish() -> None:
    out_required_logits = [101]
    out_required_tokens = [102]
    out_final_token_count = [103]

    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 16,
            step_logits_capacity=16,
            vocab_size=8,
            max_new_tokens=2,
            token_history=[0] * 4,
            token_history_capacity=4,
            token_history_count=2,
            temperature_q16=0,
            top_k=8,
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
    assert out_required_logits == [101]
    assert out_required_tokens == [102]
    assert out_final_token_count == [103]


def test_parity_failure_no_publish() -> None:
    out_required_logits = [21]
    out_required_tokens = [22]
    out_final_token_count = [23]

    def _parity_fail(**_kwargs: int) -> int:
        return SAMPLING_Q16_ERR_BAD_PARAM

    status = (
        inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_reference(
            step_logits_q16=[0] * 24,
            step_logits_capacity=24,
            vocab_size=6,
            max_new_tokens=4,
            token_history=[0] * 10,
            token_history_capacity=10,
            token_history_count=6,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=6,
            top_p_q16=SAMPLING_Q16_ONE,
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16_values=[0] * 4,
            random_q16_capacity=4,
            workspace_stage_logits_q16=[0] * 6,
            workspace_stage_logits_capacity=6,
            workspace_topk_logits_q16=[0] * 6,
            workspace_topk_logits_capacity=6,
            workspace_topk_indices=[0] * 6,
            workspace_topk_index_capacity=6,
            out_generated_tokens=[0] * 4,
            generated_capacity=4,
            out_required_logits=out_required_logits,
            out_required_tokens=out_required_tokens,
            out_final_token_count=out_final_token_count,
            parity_fn=_parity_fail,
        )
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_required_logits == [21]
    assert out_required_tokens == [22]
    assert out_final_token_count == [23]


def test_randomized_success_vectors() -> None:
    rng = random.Random(1057)
    for _ in range(80):
        vocab_size = rng.randint(1, 64)
        max_new_tokens = rng.randint(0, 20)
        token_history_count = rng.randint(0, 50)
        token_history_capacity = token_history_count + max_new_tokens
        step_logits_capacity = vocab_size * max_new_tokens

        out_required_logits = [rng.randint(-1000, 1000)]
        out_required_tokens = [rng.randint(-1000, 1000)]
        out_final_token_count = [rng.randint(-1000, 1000)]

        status = (
            inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_reference(
                step_logits_q16=[0] * max(1, step_logits_capacity),
                step_logits_capacity=step_logits_capacity,
                vocab_size=vocab_size,
                max_new_tokens=max_new_tokens,
                token_history=[0] * max(1, token_history_capacity),
                token_history_capacity=token_history_capacity,
                token_history_count=token_history_count,
                temperature_q16=SAMPLING_Q16_ONE,
                top_k=rng.randint(1, vocab_size),
                top_p_q16=rng.randint(1, SAMPLING_Q16_ONE),
                repetition_penalty_q16=rng.randint(SAMPLING_Q16_ONE, SAMPLING_Q16_ONE + 4096),
                random_q16_values=[0] * max_new_tokens,
                random_q16_capacity=max_new_tokens,
                workspace_stage_logits_q16=[0] * max(1, vocab_size),
                workspace_stage_logits_capacity=vocab_size,
                workspace_topk_logits_q16=[0] * max(1, vocab_size),
                workspace_topk_logits_capacity=vocab_size,
                workspace_topk_indices=[0] * max(1, vocab_size),
                workspace_topk_index_capacity=vocab_size,
                out_generated_tokens=[0] * max_new_tokens,
                generated_capacity=max_new_tokens,
                out_required_logits=out_required_logits,
                out_required_tokens=out_required_tokens,
                out_final_token_count=out_final_token_count,
            )
        )

        assert status == SAMPLING_Q16_OK
        assert out_required_logits[0] == step_logits_capacity
        assert out_required_tokens[0] == max_new_tokens
        assert out_final_token_count[0] == token_history_count + max_new_tokens


def test_canonical_tuple_mismatch_rejected_no_publish() -> None:
    out_required_logits = [901]
    out_required_tokens = [902]
    out_final_token_count = [903]

    def _parity_wrong_tuple(**kwargs: int) -> int:
        kwargs["out_required_logits"][0] = kwargs["step_logits_capacity"] + 1
        kwargs["out_required_tokens"][0] = kwargs["max_new_tokens"]
        kwargs["out_final_token_count"][0] = kwargs["token_history_count"] + kwargs["max_new_tokens"]
        return SAMPLING_Q16_OK

    status = inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_reference(
        step_logits_q16=[0] * 60,
        step_logits_capacity=60,
        vocab_size=15,
        max_new_tokens=4,
        token_history=[0] * 20,
        token_history_capacity=20,
        token_history_count=16,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=15,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16_values=[0, 1, 2, 3],
        random_q16_capacity=4,
        workspace_stage_logits_q16=[0] * 15,
        workspace_stage_logits_capacity=15,
        workspace_topk_logits_q16=[0] * 15,
        workspace_topk_logits_capacity=15,
        workspace_topk_indices=[0] * 15,
        workspace_topk_index_capacity=15,
        out_generated_tokens=[0] * 4,
        generated_capacity=4,
        out_required_logits=out_required_logits,
        out_required_tokens=out_required_tokens,
        out_final_token_count=out_final_token_count,
        parity_fn=_parity_wrong_tuple,
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_required_logits == [901]
    assert out_required_tokens == [902]
    assert out_final_token_count == [903]


def test_final_token_count_overflow_rejected_no_publish() -> None:
    out_required_logits = [111]
    out_required_tokens = [112]
    out_final_token_count = [113]

    def _parity_ok(**kwargs: int) -> int:
        kwargs["out_required_logits"][0] = kwargs["step_logits_capacity"]
        kwargs["out_required_tokens"][0] = kwargs["max_new_tokens"]
        kwargs["out_final_token_count"][0] = kwargs["token_history_count"] + kwargs["max_new_tokens"]
        return SAMPLING_Q16_OK

    status = inference_generate_tokens_checked_topk_topp_commit_only_preflight_only_parity_commit_only_reference(
        step_logits_q16=[0],
        step_logits_capacity=0,
        vocab_size=0x7FFFFFFFFFFFFFFF,
        max_new_tokens=2,
        token_history=[0],
        token_history_capacity=1,
        token_history_count=1,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=1,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16_values=[0],
        random_q16_capacity=1,
        workspace_stage_logits_q16=[0],
        workspace_stage_logits_capacity=1,
        workspace_topk_logits_q16=[0],
        workspace_topk_logits_capacity=1,
        workspace_topk_indices=[0],
        workspace_topk_index_capacity=1,
        out_generated_tokens=[0],
        generated_capacity=1,
        out_required_logits=out_required_logits,
        out_required_tokens=out_required_tokens,
        out_final_token_count=out_final_token_count,
        parity_fn=_parity_ok,
    )

    assert status == SAMPLING_Q16_ERR_OVERFLOW
    assert out_required_logits == [111]
    assert out_required_tokens == [112]
    assert out_final_token_count == [113]
