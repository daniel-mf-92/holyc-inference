#!/usr/bin/env python3
"""Parity harness for IQ-1229 strict one-step preflight parity gate."""

from __future__ import annotations

from pathlib import Path
import random

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4
SAMPLING_Q16_ONE = 1 << 16
I64_MAX = (1 << 63) - 1


def _one_step_requirements(
    *,
    logits_capacity: int,
    vocab_size: int,
    token_history_capacity: int,
    token_history_count: int,
    workspace_stage_logits_capacity: int,
    workspace_topk_logits_capacity: int,
    workspace_topk_index_capacity: int,
    workspace_history_stage_capacity: int,
) -> tuple[int, dict[str, int] | None]:
    if (
        logits_capacity < 0
        or vocab_size < 0
        or token_history_capacity < 0
        or token_history_count < 0
        or workspace_stage_logits_capacity < 0
        or workspace_topk_logits_capacity < 0
        or workspace_topk_index_capacity < 0
        or workspace_history_stage_capacity < 0
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    if vocab_size <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if token_history_count > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if token_history_count == I64_MAX:
        return SAMPLING_Q16_ERR_OVERFLOW, None

    required_step_logits_cells = vocab_size
    required_history_capacity = token_history_count + 1
    required_stage_logits_capacity = vocab_size
    required_topk_capacity = vocab_size
    required_random_capacity = 1
    required_generated_capacity = 1

    if required_step_logits_cells > logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_history_capacity > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_history_capacity > workspace_history_stage_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_stage_logits_capacity > workspace_stage_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_topk_capacity > workspace_topk_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_topk_capacity > workspace_topk_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    return SAMPLING_Q16_OK, {
        "required_step_logits_cells": required_step_logits_cells,
        "required_history_capacity": required_history_capacity,
        "required_stage_logits_capacity": required_stage_logits_capacity,
        "required_topk_capacity": required_topk_capacity,
        "required_random_capacity": required_random_capacity,
        "required_generated_capacity": required_generated_capacity,
    }


def inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_reference(
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
) -> tuple[int, dict[str, int] | None]:
    if (
        out_required_step_logits_cells is None
        or out_required_history_capacity is None
        or out_required_stage_logits_capacity is None
        or out_required_topk_capacity is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR, None

    if (
        out_required_step_logits_cells is out_required_history_capacity
        or out_required_step_logits_cells is out_required_stage_logits_capacity
        or out_required_step_logits_cells is out_required_topk_capacity
        or out_required_history_capacity is out_required_stage_logits_capacity
        or out_required_history_capacity is out_required_topk_capacity
        or out_required_stage_logits_capacity is out_required_topk_capacity
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    if (
        logits_q16 is None
        or token_history is None
        or workspace_stage_logits_q16 is None
        or workspace_topk_logits_q16 is None
        or workspace_topk_indices is None
        or workspace_history_stage is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR, None

    if temperature_q16 <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if repetition_penalty_q16 < SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if top_k <= 0 or top_k > vocab_size:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if top_p_q16 <= 0 or top_p_q16 > SAMPLING_Q16_ONE:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    snap_scalars = (
        logits_capacity,
        vocab_size,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        workspace_history_stage_capacity,
    )
    snap_ptrs = (
        logits_q16,
        token_history,
        workspace_stage_logits_q16,
        workspace_topk_logits_q16,
        workspace_topk_indices,
        workspace_history_stage,
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
    )

    status, diagnostics = _one_step_requirements(
        logits_capacity=logits_capacity,
        vocab_size=vocab_size,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        workspace_stage_logits_capacity=workspace_stage_logits_capacity,
        workspace_topk_logits_capacity=workspace_topk_logits_capacity,
        workspace_topk_index_capacity=workspace_topk_index_capacity,
        workspace_history_stage_capacity=workspace_history_stage_capacity,
    )
    if status != SAMPLING_Q16_OK:
        return status, None

    assert diagnostics is not None
    if diagnostics["required_random_capacity"] != 1:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if diagnostics["required_generated_capacity"] != 1:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    if snap_scalars != (
        logits_capacity,
        vocab_size,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        workspace_history_stage_capacity,
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    if snap_ptrs != (
        logits_q16,
        token_history,
        workspace_stage_logits_q16,
        workspace_topk_logits_q16,
        workspace_topk_indices,
        workspace_history_stage,
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    out_required_step_logits_cells[0] = diagnostics["required_step_logits_cells"]
    out_required_history_capacity[0] = diagnostics["required_history_capacity"]
    out_required_stage_logits_capacity[0] = diagnostics["required_stage_logits_capacity"]
    out_required_topk_capacity[0] = diagnostics["required_topk_capacity"]
    return SAMPLING_Q16_OK, diagnostics


def inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_reference(
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

    staged_step = [0]
    staged_hist = [0]
    staged_stage = [0]
    staged_topk = [0]

    snap_scalars = (
        logits_capacity,
        vocab_size,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        workspace_history_stage_capacity,
    )
    snap_ptrs = (
        logits_q16,
        token_history,
        workspace_stage_logits_q16,
        workspace_topk_logits_q16,
        workspace_topk_indices,
        workspace_history_stage,
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
    )

    status, _ = inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_reference(
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
        out_required_step_logits_cells=staged_step,
        out_required_history_capacity=staged_hist,
        out_required_stage_logits_capacity=staged_stage,
        out_required_topk_capacity=staged_topk,
    )
    if status != SAMPLING_Q16_OK:
        return status

    status, canonical = _one_step_requirements(
        logits_capacity=logits_capacity,
        vocab_size=vocab_size,
        token_history_capacity=token_history_capacity,
        token_history_count=token_history_count,
        workspace_stage_logits_capacity=workspace_stage_logits_capacity,
        workspace_topk_logits_capacity=workspace_topk_logits_capacity,
        workspace_topk_index_capacity=workspace_topk_index_capacity,
        workspace_history_stage_capacity=workspace_history_stage_capacity,
    )
    if status != SAMPLING_Q16_OK:
        return status

    assert canonical is not None
    if canonical["required_random_capacity"] != 1:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if canonical["required_generated_capacity"] != 1:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if snap_scalars != (
        logits_capacity,
        vocab_size,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        workspace_history_stage_capacity,
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if snap_ptrs != (
        logits_q16,
        token_history,
        workspace_stage_logits_q16,
        workspace_topk_logits_q16,
        workspace_topk_indices,
        workspace_history_stage,
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged_step[0] != canonical["required_step_logits_cells"]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_hist[0] != canonical["required_history_capacity"]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_stage[0] != canonical["required_stage_logits_capacity"]:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if staged_topk[0] != canonical["required_topk_capacity"]:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if out_required_step_logits_cells is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_required_history_capacity is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_required_stage_logits_capacity is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if out_required_topk_capacity is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    out_required_step_logits_cells[0] = staged_step[0]
    out_required_history_capacity[0] = staged_hist[0]
    out_required_stage_logits_capacity[0] = staged_stage[0]
    out_required_topk_capacity[0] = staged_topk[0]
    return SAMPLING_Q16_OK


def _extract_function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    idx = brace + 1
    while depth and idx < len(source):
        ch = source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        idx += 1
    return source[brace:idx]


def test_source_contains_iq_1229_helper() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParity("
    assert signature in source
    body = _extract_function_body(source, signature)
    assert "InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnly(" in body
    assert "InferenceGenerateTokensPreflightChecked(" in body
    assert "if (canonical_required_random_capacity != 1 ||" in body
    assert "if (staged_required_step_logits_cells != canonical_required_step_logits_cells)" in body
    assert "snapshot_out_required_step_logits_cells" in body
    assert "*out_required_topk_capacity = staged_required_topk_capacity;" in body


def test_known_vector_success() -> None:
    out_step = [-1]
    out_hist = [-2]
    out_stage = [-3]
    out_topk = [-4]

    status = inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_reference(
        logits_q16=[0] * 64,
        logits_capacity=64,
        vocab_size=64,
        token_history=[1, 2, 3, 4, 5, 6],
        token_history_capacity=6,
        token_history_count=5,
        temperature_q16=SAMPLING_Q16_ONE,
        top_k=32,
        top_p_q16=SAMPLING_Q16_ONE,
        repetition_penalty_q16=SAMPLING_Q16_ONE,
        random_q16=123,
        workspace_stage_logits_q16=[0] * 64,
        workspace_stage_logits_capacity=64,
        workspace_topk_logits_q16=[0] * 64,
        workspace_topk_logits_capacity=64,
        workspace_topk_indices=[0] * 64,
        workspace_topk_index_capacity=64,
        workspace_history_stage=[0] * 6,
        workspace_history_stage_capacity=6,
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


def test_error_path_no_publish() -> None:
    out_step = [101]
    out_hist = [102]
    out_stage = [103]
    out_topk = [104]

    status = inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_reference(
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
        random_q16=77,
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
    assert out_step == [101]
    assert out_hist == [102]
    assert out_stage == [103]
    assert out_topk == [104]


def test_alias_rejected() -> None:
    shared = [0]
    status = inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_reference(
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
        random_q16=7,
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


def test_fuzz_vectors() -> None:
    rng = random.Random(1229)
    for _ in range(300):
        vocab_size = rng.randint(1, 128)
        token_history_capacity = rng.randint(1, 128)
        token_history_count = rng.randint(0, token_history_capacity)
        logits_capacity = rng.randint(vocab_size, vocab_size + 64)

        status, diagnostics = _one_step_requirements(
            logits_capacity=logits_capacity,
            vocab_size=vocab_size,
            token_history_capacity=token_history_capacity,
            token_history_count=token_history_count,
            workspace_stage_logits_capacity=rng.randint(vocab_size, vocab_size + 64),
            workspace_topk_logits_capacity=rng.randint(vocab_size, vocab_size + 64),
            workspace_topk_index_capacity=rng.randint(vocab_size, vocab_size + 64),
            workspace_history_stage_capacity=rng.randint(token_history_count + 1, token_history_count + 65),
        )
        if status != SAMPLING_Q16_OK:
            continue

        assert diagnostics is not None
        out_step = [-11]
        out_hist = [-12]
        out_stage = [-13]
        out_topk = [-14]

        gate_status = inference_generate_step_checked_topk_topp_nopartial_commit_only_preflight_only_parity_reference(
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
            workspace_stage_logits_q16=[0] * (diagnostics["required_stage_logits_capacity"] + 2),
            workspace_stage_logits_capacity=diagnostics["required_stage_logits_capacity"] + 2,
            workspace_topk_logits_q16=[0] * (diagnostics["required_topk_capacity"] + 2),
            workspace_topk_logits_capacity=diagnostics["required_topk_capacity"] + 2,
            workspace_topk_indices=[0] * (diagnostics["required_topk_capacity"] + 2),
            workspace_topk_index_capacity=diagnostics["required_topk_capacity"] + 2,
            workspace_history_stage=[0] * (diagnostics["required_history_capacity"] + 2),
            workspace_history_stage_capacity=diagnostics["required_history_capacity"] + 2,
            out_required_step_logits_cells=out_step,
            out_required_history_capacity=out_hist,
            out_required_stage_logits_capacity=out_stage,
            out_required_topk_capacity=out_topk,
        )

        assert gate_status == SAMPLING_Q16_OK
        assert out_step[0] == diagnostics["required_step_logits_cells"]
        assert out_hist[0] == diagnostics["required_history_capacity"]
        assert out_stage[0] == diagnostics["required_stage_logits_capacity"]
        assert out_topk[0] == diagnostics["required_topk_capacity"]


if __name__ == "__main__":
    test_source_contains_iq_1229_helper()
    test_known_vector_success()
    test_error_path_no_publish()
    test_alias_rejected()
    test_fuzz_vectors()
    print("ok")
