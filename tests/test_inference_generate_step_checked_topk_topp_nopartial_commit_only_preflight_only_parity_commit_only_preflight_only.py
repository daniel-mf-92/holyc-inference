#!/usr/bin/env python3
"""Harness for IQ-1233 one-step parity-commit preflight-only wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity import (
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_NULL_PTR,
    SAMPLING_Q16_OK,
    SAMPLING_Q16_ONE,
    inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_reference,
)
from test_inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_commit_only import (
    inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_commit_only_reference,
)


def inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
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
    out_required_step_logits_cells: list[int] | None,
    out_required_history_capacity: list[int] | None,
    out_required_stage_logits_capacity: list[int] | None,
    out_required_topk_capacity: list[int] | None,
    commit_only_fn=inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_commit_only_reference,
    parity_fn=inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_reference,
) -> int:
    if (
        out_required_step_logits_cells is None
        or out_required_history_capacity is None
        or out_required_stage_logits_capacity is None
        or out_required_topk_capacity is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR

    if (
        out_required_step_logits_cells is out_required_history_capacity
        or out_required_step_logits_cells is out_required_stage_logits_capacity
        or out_required_step_logits_cells is out_required_topk_capacity
        or out_required_history_capacity is out_required_stage_logits_capacity
        or out_required_history_capacity is out_required_topk_capacity
        or out_required_stage_logits_capacity is out_required_topk_capacity
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        logits_q16 is None
        or token_history is None
        or workspace_stage_logits_q16 is None
        or workspace_topk_logits_q16 is None
        or workspace_topk_indices is None
        or workspace_history_stage is None
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

    snapshot_logits_capacity = logits_capacity
    snapshot_vocab_size = vocab_size
    snapshot_token_history_capacity = token_history_capacity
    snapshot_token_history_count = token_history_count
    snapshot_temperature_q16 = temperature_q16
    snapshot_top_k = top_k
    snapshot_top_p_q16 = top_p_q16
    snapshot_repetition_penalty_q16 = repetition_penalty_q16
    snapshot_random_q16 = random_q16
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity
    snapshot_workspace_history_stage_capacity = workspace_history_stage_capacity

    snapshot_logits_q16 = logits_q16
    snapshot_token_history = token_history
    snapshot_workspace_stage_logits_q16 = workspace_stage_logits_q16
    snapshot_workspace_topk_logits_q16 = workspace_topk_logits_q16
    snapshot_workspace_topk_indices = workspace_topk_indices
    snapshot_workspace_history_stage = workspace_history_stage
    snapshot_out_required_step_logits_cells = out_required_step_logits_cells
    snapshot_out_required_history_capacity = out_required_history_capacity
    snapshot_out_required_stage_logits_capacity = out_required_stage_logits_capacity
    snapshot_out_required_topk_capacity = out_required_topk_capacity

    staged_commit_step = [out_required_step_logits_cells[0]]
    staged_commit_hist = [out_required_history_capacity[0]]
    staged_commit_stage = [out_required_stage_logits_capacity[0]]
    staged_commit_topk = [out_required_topk_capacity[0]]

    status = commit_only_fn(
        logits_q16=logits_q16,
        logits_capacity=logits_capacity,
        vocab_size=vocab_size,
        token_history=token_history,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        temperature_q16=temperature_q16,
        top_k=top_k,
        top_p_q16=top_p_q16,
        repetition_penalty_q16=repetition_penalty_q16,
        random_q16=random_q16,
        workspace_stage_logits_q16=workspace_stage_logits_q16,
        workspace_stage_logits_capacity=workspace_stage_logits_capacity,
        workspace_topk_logits_q16=workspace_topk_logits_q16,
        workspace_topk_logits_capacity=workspace_topk_logits_capacity,
        workspace_topk_indices=workspace_topk_indices,
        workspace_topk_index_capacity=workspace_topk_index_capacity,
        workspace_history_stage=workspace_history_stage,
        workspace_history_stage_capacity=workspace_history_stage_capacity,
        out_required_step_logits_cells=staged_commit_step,
        out_required_history_capacity=staged_commit_hist,
        out_required_stage_logits_capacity=staged_commit_stage,
        out_required_topk_capacity=staged_commit_topk,
    )
    if status != SAMPLING_Q16_OK:
        return status

    staged_parity_step = [out_required_step_logits_cells[0]]
    staged_parity_hist = [out_required_history_capacity[0]]
    staged_parity_stage = [out_required_stage_logits_capacity[0]]
    staged_parity_topk = [out_required_topk_capacity[0]]

    status = parity_fn(
        logits_q16=logits_q16,
        logits_capacity=logits_capacity,
        vocab_size=vocab_size,
        token_history=token_history,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        temperature_q16=temperature_q16,
        top_k=top_k,
        top_p_q16=top_p_q16,
        repetition_penalty_q16=repetition_penalty_q16,
        random_q16=random_q16,
        workspace_stage_logits_q16=workspace_stage_logits_q16,
        workspace_stage_logits_capacity=workspace_stage_logits_capacity,
        workspace_topk_logits_q16=workspace_topk_logits_q16,
        workspace_topk_logits_capacity=workspace_topk_logits_capacity,
        workspace_topk_indices=workspace_topk_indices,
        workspace_topk_index_capacity=workspace_topk_index_capacity,
        workspace_history_stage=workspace_history_stage,
        workspace_history_stage_capacity=workspace_history_stage_capacity,
        out_required_step_logits_cells=staged_parity_step,
        out_required_history_capacity=staged_parity_hist,
        out_required_stage_logits_capacity=staged_parity_stage,
        out_required_topk_capacity=staged_parity_topk,
    )
    if status != SAMPLING_Q16_OK:
        return status

    if (
        snapshot_logits_capacity != logits_capacity
        or snapshot_vocab_size != vocab_size
        or snapshot_token_history_capacity != token_history_capacity
        or snapshot_token_history_count != token_history_count
        or snapshot_temperature_q16 != temperature_q16
        or snapshot_top_k != top_k
        or snapshot_top_p_q16 != top_p_q16
        or snapshot_repetition_penalty_q16 != repetition_penalty_q16
        or snapshot_random_q16 != random_q16
        or snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity
        or snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity
        or snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity
        or snapshot_workspace_history_stage_capacity != workspace_history_stage_capacity
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if (
        snapshot_logits_q16 is not logits_q16
        or snapshot_token_history is not token_history
        or snapshot_workspace_stage_logits_q16 is not workspace_stage_logits_q16
        or snapshot_workspace_topk_logits_q16 is not workspace_topk_logits_q16
        or snapshot_workspace_topk_indices is not workspace_topk_indices
        or snapshot_workspace_history_stage is not workspace_history_stage
        or snapshot_out_required_step_logits_cells is not out_required_step_logits_cells
        or snapshot_out_required_history_capacity is not out_required_history_capacity
        or snapshot_out_required_stage_logits_capacity is not out_required_stage_logits_capacity
        or snapshot_out_required_topk_capacity is not out_required_topk_capacity
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged_commit_step[0] != staged_parity_step[0]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_commit_hist[0] != staged_parity_hist[0]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_commit_stage[0] != staged_parity_stage[0]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_commit_topk[0] != staged_parity_topk[0]:
        return SAMPLING_Q16_ERR_BAD_PARAM

    out_required_step_logits_cells[0] = staged_commit_step[0]
    out_required_history_capacity[0] = staged_commit_hist[0]
    out_required_stage_logits_capacity[0] = staged_commit_stage[0]
    out_required_topk_capacity[0] = staged_commit_topk[0]
    return SAMPLING_Q16_OK


def test_source_contains_iq_1233_signature_and_contract() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    assert source.count(signature) == 1
    body = source.split(signature, 1)[1].split("I32 InferenceGenerateTokensCheckedTopKTopPNoPartial(", 1)[0]

    assert "IQ-1233 diagnostics zero-write companion over one-step commit/parity wrappers." in source
    assert "InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "snapshot_temperature_q16 = temperature_q16;" in body
    assert "snapshot_out_required_step_logits_cells = out_required_step_logits_cells;" in body
    assert "if (staged_commit_required_topk_capacity != staged_parity_required_topk_capacity)" in body
    assert "*out_required_topk_capacity = staged_commit_required_topk_capacity;" in body


def test_success_and_no_publish_on_error() -> None:
    out_step = [101]
    out_hist = [102]
    out_stage = [103]
    out_topk = [104]

    status = inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        logits_q16=[0] * 64,
        logits_capacity=64,
        vocab_size=64,
        token_history=[1, 2, 3, 4, 5, 6],
        token_history_capacity=6,
        token_history_count=5,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=16,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16=111,
        workspace_stage_logits_q16=[0] * 64,
        workspace_stage_logits_capacity=64,
        workspace_topk_logits_q16=[0] * 64,
        workspace_topk_logits_capacity=64,
        workspace_topk_indices=[0] * 64,
        workspace_topk_index_capacity=64,
        workspace_history_stage=[0] * 8,
        workspace_history_stage_capacity=8,
        out_required_step_logits_cells=out_step,
        out_required_history_capacity=out_hist,
        out_required_stage_logits_capacity=out_stage,
        out_required_topk_capacity=out_topk,
    )

    assert status == SAMPLING_Q16_OK
    assert out_step == [64]
    assert out_hist == [6]
    assert out_stage == [64]
    assert out_topk == [64]

    out_step = [201]
    out_hist = [202]
    out_stage = [203]
    out_topk = [204]

    status = inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        logits_q16=[0] * 8,
        logits_capacity=8,
        vocab_size=8,
        token_history=[1, 2, 3],
        token_history_capacity=3,
        token_history_count=2,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=4,
        top_p_q16=0,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16=9,
        workspace_stage_logits_q16=[0] * 8,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_q16=[0] * 8,
        workspace_topk_logits_capacity=8,
        workspace_topk_indices=[0] * 8,
        workspace_topk_index_capacity=8,
        workspace_history_stage=[0] * 3,
        workspace_history_stage_capacity=3,
        out_required_step_logits_cells=out_step,
        out_required_history_capacity=out_hist,
        out_required_stage_logits_capacity=out_stage,
        out_required_topk_capacity=out_topk,
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert out_step == [201]
    assert out_hist == [202]
    assert out_stage == [203]
    assert out_topk == [204]


def test_alias_and_null_rejected() -> None:
    shared = [0]
    status = inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        logits_q16=[0] * 8,
        logits_capacity=8,
        vocab_size=8,
        token_history=[1, 2, 3],
        token_history_capacity=3,
        token_history_count=2,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=4,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16=3,
        workspace_stage_logits_q16=[0] * 8,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_q16=[0] * 8,
        workspace_topk_logits_capacity=8,
        workspace_topk_indices=[0] * 8,
        workspace_topk_index_capacity=8,
        workspace_history_stage=[0] * 3,
        workspace_history_stage_capacity=3,
        out_required_step_logits_cells=shared,
        out_required_history_capacity=shared,
        out_required_stage_logits_capacity=[0],
        out_required_topk_capacity=[0],
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM

    status = inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        logits_q16=None,
        logits_capacity=8,
        vocab_size=8,
        token_history=[1, 2, 3],
        token_history_capacity=3,
        token_history_count=2,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=4,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16=3,
        workspace_stage_logits_q16=[0] * 8,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_q16=[0] * 8,
        workspace_topk_logits_capacity=8,
        workspace_topk_indices=[0] * 8,
        workspace_topk_index_capacity=8,
        workspace_history_stage=[0] * 3,
        workspace_history_stage_capacity=3,
        out_required_step_logits_cells=[0],
        out_required_history_capacity=[0],
        out_required_stage_logits_capacity=[0],
        out_required_topk_capacity=[0],
    )
    assert status == SAMPLING_Q16_ERR_NULL_PTR


def test_fuzz_vectors() -> None:
    rng = random.Random(1233)
    for _ in range(300):
        vocab_size = rng.randint(1, 128)
        token_history_capacity = rng.randint(1, 128)
        token_history_count = rng.randint(0, token_history_capacity - 1)
        logits_capacity = rng.randint(vocab_size, vocab_size + 64)

        out_step = [-1]
        out_hist = [-2]
        out_stage = [-3]
        out_topk = [-4]

        status = inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
            logits_q16=[0] * logits_capacity,
            logits_capacity=logits_capacity,
            vocab_size=vocab_size,
            token_history=[0] * token_history_capacity,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            temperature_q16=SAMPLING_Q16_ONE,
            top_k=rng.randint(1, vocab_size),
            top_p_q16=rng.randint(1, SAMPLING_Q16_ONE),
            repetition_penalty_q16=SAMPLING_Q16_ONE,
            random_q16=rng.randint(0, SAMPLING_Q16_ONE - 1),
            workspace_stage_logits_q16=[0] * (vocab_size + 8),
            workspace_stage_logits_capacity=vocab_size + 8,
            workspace_topk_logits_q16=[0] * (vocab_size + 8),
            workspace_topk_logits_capacity=vocab_size + 8,
            workspace_topk_indices=[0] * (vocab_size + 8),
            workspace_topk_index_capacity=vocab_size + 8,
            workspace_history_stage=[0] * (token_history_capacity + 2),
            workspace_history_stage_capacity=token_history_capacity + 2,
            out_required_step_logits_cells=out_step,
            out_required_history_capacity=out_hist,
            out_required_stage_logits_capacity=out_stage,
            out_required_topk_capacity=out_topk,
        )

        assert status == SAMPLING_Q16_OK
        assert out_step[0] == vocab_size
        assert out_hist[0] == token_history_count + 1
        assert out_stage[0] == vocab_size
        assert out_topk[0] == vocab_size


if __name__ == "__main__":
    test_source_contains_iq_1233_signature_and_contract()
    test_success_and_no_publish_on_error()
    test_alias_and_null_rejected()
    test_fuzz_vectors()
    print("ok")
