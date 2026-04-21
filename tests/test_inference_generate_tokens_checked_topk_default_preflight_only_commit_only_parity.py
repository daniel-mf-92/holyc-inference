#!/usr/bin/env python3
"""Parity harness for InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParity (IQ-843)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_inference_generate_tokens_preflight_checked_topk_default import (
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_OK,
)
from test_inference_generate_tokens_checked_topk_default import SAMPLING_Q16_ONE
from test_inference_generate_tokens_checked_topk_default_preflight_only import (
    inference_generate_tokens_checked_topk_default_preflight_only_reference,
)
from test_inference_generate_tokens_checked_topk_default_preflight_only_commit_only import (
    inference_generate_tokens_checked_topk_default_preflight_only_commit_only_reference,
)


def inference_generate_tokens_checked_topk_default_preflight_only_commit_only_parity_reference(
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

    snapshot_step_logits_capacity = step_logits_capacity
    snapshot_vocab_size = vocab_size
    snapshot_max_new_tokens = max_new_tokens
    snapshot_token_history_capacity = token_history_capacity
    snapshot_token_history_count = token_history_count
    snapshot_top_p_q16 = top_p_q16
    snapshot_random_q16_capacity = random_q16_capacity
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity
    snapshot_generated_capacity = generated_capacity

    commit_required_step_logits_cells = [0]
    commit_required_generated_capacity = [0]
    commit_required_random_capacity = [0]

    status = inference_generate_tokens_checked_topk_default_preflight_only_commit_only_reference(
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
        out_required_step_logits_cells=commit_required_step_logits_cells,
        out_required_generated_capacity=commit_required_generated_capacity,
        out_required_random_capacity=commit_required_random_capacity,
    )
    if status != SAMPLING_Q16_OK:
        return status

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

    if (
        snapshot_step_logits_capacity != step_logits_capacity
        or snapshot_vocab_size != vocab_size
        or snapshot_max_new_tokens != max_new_tokens
        or snapshot_token_history_capacity != token_history_capacity
        or snapshot_token_history_count != token_history_count
        or snapshot_top_p_q16 != top_p_q16
        or snapshot_random_q16_capacity != random_q16_capacity
        or snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity
        or snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity
        or snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity
        or snapshot_generated_capacity != generated_capacity
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if commit_required_step_logits_cells[0] != diagnostics["required_step_logits_cells"]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if commit_required_generated_capacity[0] != diagnostics["required_generated_capacity"]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if commit_required_random_capacity[0] != diagnostics["required_random_capacity"]:
        return SAMPLING_Q16_ERR_BAD_PARAM

    out_required_step_logits_cells[0] = commit_required_step_logits_cells[0]
    out_required_generated_capacity[0] = commit_required_generated_capacity[0]
    out_required_random_capacity[0] = commit_required_random_capacity[0]
    return SAMPLING_Q16_OK


def explicit_parity_composition(
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
    out_required_step_logits_cells: list[int],
    out_required_generated_capacity: list[int],
    out_required_random_capacity: list[int],
) -> int:
    commit_required_step_logits_cells = [0]
    commit_required_generated_capacity = [0]
    commit_required_random_capacity = [0]

    status = inference_generate_tokens_checked_topk_default_preflight_only_commit_only_reference(
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
        out_required_step_logits_cells=commit_required_step_logits_cells,
        out_required_generated_capacity=commit_required_generated_capacity,
        out_required_random_capacity=commit_required_random_capacity,
    )
    if status != SAMPLING_Q16_OK:
        return status

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

    if commit_required_step_logits_cells[0] != diagnostics["required_step_logits_cells"]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if commit_required_generated_capacity[0] != diagnostics["required_generated_capacity"]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if commit_required_random_capacity[0] != diagnostics["required_random_capacity"]:
        return SAMPLING_Q16_ERR_BAD_PARAM

    out_required_step_logits_cells[0] = commit_required_step_logits_cells[0]
    out_required_generated_capacity[0] = commit_required_generated_capacity[0]
    out_required_random_capacity[0] = commit_required_random_capacity[0]
    return SAMPLING_Q16_OK


def test_source_contains_commit_only_parity_helper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParity(" in source
    assert "InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnly(" in source
    assert "InferenceGenerateTokensCheckedTopKDefaultPreflightOnly(" in source
    assert "if (commit_required_step_logits_cells != canonical_required_step_logits_cells)" in source


def test_known_vector() -> None:
    out_step = [7]
    out_gen = [8]
    out_rand = [9]

    status = inference_generate_tokens_checked_topk_default_preflight_only_commit_only_parity_reference(
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

    status = inference_generate_tokens_checked_topk_default_preflight_only_commit_only_parity_reference(
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


def test_randomized_parity_vs_explicit() -> None:
    rng = random.Random(20260421_843)

    for _ in range(700):
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

        out_step_new = [11]
        out_gen_new = [22]
        out_rand_new = [33]

        out_step_ref = [44]
        out_gen_ref = [55]
        out_rand_ref = [66]

        status_new = inference_generate_tokens_checked_topk_default_preflight_only_commit_only_parity_reference(
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
            out_required_step_logits_cells=out_step_new,
            out_required_generated_capacity=out_gen_new,
            out_required_random_capacity=out_rand_new,
        )

        status_ref = explicit_parity_composition(
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
            out_required_step_logits_cells=out_step_ref,
            out_required_generated_capacity=out_gen_ref,
            out_required_random_capacity=out_rand_ref,
        )

        assert status_new == status_ref
        if status_new == SAMPLING_Q16_OK:
            assert out_step_new == out_step_ref
            assert out_gen_new == out_gen_ref
            assert out_rand_new == out_rand_ref
        else:
            assert out_step_new == [11]
            assert out_gen_new == [22]
            assert out_rand_new == [33]


if __name__ == "__main__":
    test_source_contains_commit_only_parity_helper()
    test_known_vector()
    test_error_no_publish()
    test_randomized_parity_vs_explicit()
    print("inference_generate_tokens_checked_topk_default_preflight_only_commit_only_parity=ok")
