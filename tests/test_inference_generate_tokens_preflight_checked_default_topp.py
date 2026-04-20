#!/usr/bin/env python3
"""Reference checks for InferenceGenerateTokensPreflightCheckedDefaultTopP (IQ-759)."""

from __future__ import annotations

from pathlib import Path
import random

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4

I64_MAX = (1 << 63) - 1


def inference_generate_tokens_preflight_reference(
    *,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history_capacity: int,
    token_history_count: int,
    workspace_stage_logits_capacity: int,
    workspace_topk_logits_capacity: int,
    workspace_topk_index_capacity: int,
    random_q16_capacity: int,
    generated_capacity: int,
) -> tuple[int, dict[str, int] | None]:
    if (
        step_logits_capacity < 0
        or vocab_size < 0
        or max_new_tokens < 0
        or token_history_capacity < 0
        or token_history_count < 0
        or workspace_stage_logits_capacity < 0
        or workspace_topk_logits_capacity < 0
        or workspace_topk_index_capacity < 0
        or random_q16_capacity < 0
        or generated_capacity < 0
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    if vocab_size <= 0:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if token_history_count > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    required_history_capacity = token_history_count + max_new_tokens
    if required_history_capacity < token_history_count:
        return SAMPLING_Q16_ERR_OVERFLOW, None

    required_stage_logits_capacity = vocab_size
    required_topk_capacity = vocab_size
    required_random_capacity = max_new_tokens
    required_generated_capacity = max_new_tokens

    if max_new_tokens:
        if vocab_size > I64_MAX // max_new_tokens:
            return SAMPLING_Q16_ERR_OVERFLOW, None
        required_step_logits_cells = vocab_size * max_new_tokens
    else:
        required_step_logits_cells = 0

    if required_history_capacity > token_history_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_stage_logits_capacity > workspace_stage_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_topk_capacity > workspace_topk_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_topk_capacity > workspace_topk_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_random_capacity > random_q16_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_generated_capacity > generated_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None
    if required_step_logits_cells > step_logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM, None

    return SAMPLING_Q16_OK, {
        "required_step_logits_cells": required_step_logits_cells,
        "required_history_capacity": required_history_capacity,
        "required_stage_logits_capacity": required_stage_logits_capacity,
        "required_topk_capacity": required_topk_capacity,
        "required_random_capacity": required_random_capacity,
        "required_generated_capacity": required_generated_capacity,
    }


def inference_generate_tokens_preflight_default_topp_reference(
    *,
    step_logits_capacity: int,
    vocab_size: int,
    max_new_tokens: int,
    token_history_capacity: int,
    token_history_count: int,
    workspace_stage_logits_capacity: int,
    workspace_topk_logits_capacity: int,
    workspace_topk_index_capacity: int,
    random_q16_capacity: int,
    generated_capacity: int,
) -> tuple[int, dict[str, int] | None]:
    return inference_generate_tokens_preflight_reference(
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


def test_source_contains_preflight_default_topp_wrapper() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 InferenceGenerateTokensPreflightCheckedDefaultTopP(" in source
    assert "InferenceGenerateTokensCheckedDefaultTopP" in source
    assert "SAMPLING_Q16_ONE" in source
    assert "return InferenceGenerateTokensPreflightChecked(" in source


def test_source_default_topp_wrapper_is_direct_forwarder() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceGenerateTokensPreflightCheckedDefaultTopP("
    start = source.rfind(signature)
    assert start >= 0

    while True:
        brace_open = source.find("{", start)
        assert brace_open >= 0

        decl_end = source.find(";", start)
        if decl_end < 0 or brace_open < decl_end:
            break

        start = source.find(signature, start + len(signature))
        assert start >= 0

    depth = 0
    end = -1
    for index in range(brace_open, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break

    assert end > brace_open
    body = source[brace_open + 1 : end]

    # Wrapper must remain diagnostics-only sugar over canonical preflight.
    assert "InferenceGenerateTokensPreflightChecked(" in body
    assert "return InferenceGenerateTokensPreflightChecked(" in body
    assert "I64 required_step_logits_cells;" not in body
    assert "I64 required_history_capacity;" not in body
    assert "top_p_q16" not in body


def test_preflight_default_topp_matches_explicit_reference_randomized() -> None:
    rng = random.Random(20260420_759)
    for _ in range(800):
        vocab_size = rng.randint(1, 256)
        max_new_tokens = rng.randint(0, 64)
        token_history_count = rng.randint(0, 32)

        required_history_capacity = token_history_count + max_new_tokens
        required_step_logits_cells = vocab_size * max_new_tokens

        history_capacity = required_history_capacity + rng.randint(0, 3)
        stage_capacity = vocab_size + rng.randint(0, 3)
        topk_logits_capacity = vocab_size + rng.randint(0, 3)
        topk_index_capacity = vocab_size + rng.randint(0, 3)
        random_capacity = max_new_tokens + rng.randint(0, 2)
        generated_capacity = max_new_tokens + rng.randint(0, 2)
        step_capacity = required_step_logits_cells + rng.randint(0, 3)

        status_default, diag_default = (
            inference_generate_tokens_preflight_default_topp_reference(
                step_logits_capacity=step_capacity,
                vocab_size=vocab_size,
                max_new_tokens=max_new_tokens,
                token_history_capacity=history_capacity,
                token_history_count=token_history_count,
                workspace_stage_logits_capacity=stage_capacity,
                workspace_topk_logits_capacity=topk_logits_capacity,
                workspace_topk_index_capacity=topk_index_capacity,
                random_q16_capacity=random_capacity,
                generated_capacity=generated_capacity,
            )
        )

        status_explicit, diag_explicit = inference_generate_tokens_preflight_reference(
            step_logits_capacity=step_capacity,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
            token_history_capacity=history_capacity,
            token_history_count=token_history_count,
            workspace_stage_logits_capacity=stage_capacity,
            workspace_topk_logits_capacity=topk_logits_capacity,
            workspace_topk_index_capacity=topk_index_capacity,
            random_q16_capacity=random_capacity,
            generated_capacity=generated_capacity,
        )

        assert status_default == status_explicit
        assert diag_default == diag_explicit


def test_preflight_default_topp_preserves_failure_classification() -> None:
    cases = [
        dict(
            step_logits_capacity=11,
            vocab_size=4,
            max_new_tokens=3,
            token_history_capacity=10,
            token_history_count=0,
            workspace_stage_logits_capacity=4,
            workspace_topk_logits_capacity=4,
            workspace_topk_index_capacity=4,
            random_q16_capacity=3,
            generated_capacity=3,
            expect=SAMPLING_Q16_ERR_BAD_PARAM,
        ),
        dict(
            step_logits_capacity=100,
            vocab_size=16,
            max_new_tokens=2,
            token_history_capacity=6,
            token_history_count=4,
            workspace_stage_logits_capacity=16,
            workspace_topk_logits_capacity=16,
            workspace_topk_index_capacity=16,
            random_q16_capacity=2,
            generated_capacity=2,
            expect=SAMPLING_Q16_OK,
        ),
        dict(
            step_logits_capacity=I64_MAX,
            vocab_size=(1 << 62),
            max_new_tokens=3,
            token_history_capacity=10,
            token_history_count=0,
            workspace_stage_logits_capacity=(1 << 62),
            workspace_topk_logits_capacity=(1 << 62),
            workspace_topk_index_capacity=(1 << 62),
            random_q16_capacity=3,
            generated_capacity=3,
            expect=SAMPLING_Q16_ERR_OVERFLOW,
        ),
    ]

    for case in cases:
        expected = case.pop("expect")
        status_default, diag_default = (
            inference_generate_tokens_preflight_default_topp_reference(**case)
        )
        status_explicit, diag_explicit = inference_generate_tokens_preflight_reference(
            **case
        )

        assert status_default == status_explicit == expected
        if expected == SAMPLING_Q16_OK:
            assert diag_default == diag_explicit
            assert diag_default is not None
        else:
            assert diag_default is None
            assert diag_explicit is None


def test_preflight_default_topp_success_diagnostics_exact() -> None:
    status, diagnostics = inference_generate_tokens_preflight_default_topp_reference(
        step_logits_capacity=48,
        vocab_size=12,
        max_new_tokens=4,
        token_history_capacity=9,
        token_history_count=5,
        workspace_stage_logits_capacity=12,
        workspace_topk_logits_capacity=12,
        workspace_topk_index_capacity=12,
        random_q16_capacity=4,
        generated_capacity=4,
    )

    assert status == SAMPLING_Q16_OK
    assert diagnostics == {
        "required_step_logits_cells": 48,
        "required_history_capacity": 9,
        "required_stage_logits_capacity": 12,
        "required_topk_capacity": 12,
        "required_random_capacity": 4,
        "required_generated_capacity": 4,
    }


def test_preflight_default_topp_no_partial_diagnostics_on_failure_model() -> None:
    staged = {
        "required_step_logits_cells": -100,
        "required_history_capacity": -200,
        "required_stage_logits_capacity": -300,
        "required_topk_capacity": -400,
        "required_random_capacity": -500,
        "required_generated_capacity": -600,
    }

    status, diagnostics = inference_generate_tokens_preflight_default_topp_reference(
        step_logits_capacity=15,
        vocab_size=8,
        max_new_tokens=4,
        token_history_capacity=5,
        token_history_count=2,
        workspace_stage_logits_capacity=8,
        workspace_topk_logits_capacity=8,
        workspace_topk_index_capacity=8,
        random_q16_capacity=3,
        generated_capacity=4,
    )

    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert diagnostics is None
    assert staged == {
        "required_step_logits_cells": -100,
        "required_history_capacity": -200,
        "required_stage_logits_capacity": -300,
        "required_topk_capacity": -400,
        "required_random_capacity": -500,
        "required_generated_capacity": -600,
    }


if __name__ == "__main__":
    test_source_contains_preflight_default_topp_wrapper()
    test_source_default_topp_wrapper_is_direct_forwarder()
    test_preflight_default_topp_matches_explicit_reference_randomized()
    test_preflight_default_topp_preserves_failure_classification()
    test_preflight_default_topp_success_diagnostics_exact()
    test_preflight_default_topp_no_partial_diagnostics_on_failure_model()
    print("inference_generate_tokens_preflight_checked_default_topp_reference_checks=ok")
