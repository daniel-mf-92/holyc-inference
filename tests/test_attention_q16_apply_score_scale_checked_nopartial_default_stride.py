#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleCheckedNoPartialDefaultStride."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked_nopartial import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
    attention_q16_apply_score_scale_checked,
    attention_q16_apply_score_scale_checked_nopartial,
)
from test_attention_q16_apply_score_scale_checked_default_stride_nopartial_preflight_only import (
    attention_q16_apply_score_scale_checked_default_stride_nopartial_preflight_only,
)


def attention_q16_apply_score_scale_checked_nopartial_default_stride(
    in_scores_q32,
    in_scores_capacity: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count == 0:
        return ATTN_Q16_OK

    default_score_stride = token_count

    preflight_last_in = [0]
    preflight_last_out = [0]
    preflight_required_in = [0]
    preflight_required_out = [0]
    err = attention_q16_apply_score_scale_checked_default_stride_nopartial_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        preflight_last_in,
        preflight_last_out,
        preflight_required_in,
        preflight_required_out,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_scores = [0] * token_count
    err = attention_q16_apply_score_scale_checked(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        default_score_stride,
        score_scale_q16,
        staged_scores,
        token_count,
        1,
    )
    if err != ATTN_Q16_OK:
        return err

    for token_index in range(token_count):
        out_scores_q32[token_index * default_score_stride] = staged_scores[token_index]

    return ATTN_Q16_OK


def explicit_staged_default_stride_composition(
    in_scores_q32,
    in_scores_capacity: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    return attention_q16_apply_score_scale_checked_nopartial(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        token_count,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        token_count,
    )


def test_source_contains_default_stride_nopartial_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleCheckedNoPartialDefaultStride("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "default_score_stride = token_count;" in body
    assert "status = AttentionQ16ApplyScoreScaleChecked(" in body
    assert "staged_scores_q32 = MAlloc(stage_bytes);" in body
    assert "out_scores_q32[out_base] = staged_scores_q32[token_index];" in body


def test_known_vector_matches_explicit_composition() -> None:
    token_count = 4
    stride = token_count
    scale_q16 = 23170

    in_capacity = 1 + (token_count - 1) * stride
    out_capacity = 1 + (token_count - 1) * stride

    in_scores = [0] * in_capacity
    seeds = [123456789, -222222222, 333333333, -444444444]
    for index, value in enumerate(seeds):
        in_scores[index * stride] = value

    out_new = [777] * out_capacity
    out_ref = out_new.copy()

    err_new = attention_q16_apply_score_scale_checked_nopartial_default_stride(
        in_scores,
        in_capacity,
        token_count,
        scale_q16,
        out_new,
        out_capacity,
    )
    err_ref = explicit_staged_default_stride_composition(
        in_scores,
        in_capacity,
        token_count,
        scale_q16,
        out_ref,
        out_capacity,
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref


def test_adversarial_contracts() -> None:
    assert (
        attention_q16_apply_score_scale_checked_nopartial_default_stride(
            None,
            0,
            0,
            1 << 16,
            [0],
            1,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_apply_score_scale_checked_nopartial_default_stride(
            [0],
            1,
            0,
            1 << 16,
            None,
            0,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_apply_score_scale_checked_nopartial_default_stride(
            [0],
            -1,
            0,
            1 << 16,
            [0],
            1,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_checked_nopartial_default_stride(
            [0],
            1,
            -1,
            1 << 16,
            [0],
            1,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    token_count = 3
    stride = token_count
    in_scores = [11, 0, 0, -22, 0, 0, 33]
    out_scores = [0] * 7

    assert (
        attention_q16_apply_score_scale_checked_nopartial_default_stride(
            in_scores,
            6,
            token_count,
            1 << 16,
            out_scores,
            len(out_scores),
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_checked_nopartial_default_stride(
            in_scores,
            len(in_scores),
            token_count,
            1 << 16,
            out_scores,
            6,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260419_551)

    for _ in range(4000):
        token_count = rng.randint(0, 64)
        score_scale_q16 = rng.randint(-(1 << 18), (1 << 18))

        stride = token_count
        if token_count == 0:
            in_capacity = rng.randint(0, 16)
            out_capacity = rng.randint(0, 16)
        else:
            need = 1 + (token_count - 1) * stride
            in_capacity = max(0, need + rng.randint(-4, 4))
            out_capacity = max(0, need + rng.randint(-4, 4))

        if rng.random() < 0.05:
            in_capacity = -rng.randint(1, 8)
        if rng.random() < 0.05:
            out_capacity = -rng.randint(1, 8)

        in_scores = [rng.randint(-(1 << 40), (1 << 40)) for _ in range(max(in_capacity, 1))]
        out_seed = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(out_capacity, 1))]

        out_new = out_seed.copy()
        out_ref = out_seed.copy()

        err_new = attention_q16_apply_score_scale_checked_nopartial_default_stride(
            in_scores,
            in_capacity,
            token_count,
            score_scale_q16,
            out_new,
            out_capacity,
        )
        err_ref = explicit_staged_default_stride_composition(
            in_scores,
            in_capacity,
            token_count,
            score_scale_q16,
            out_ref,
            out_capacity,
        )

        assert err_new == err_ref
        assert out_new == out_ref


if __name__ == "__main__":
    test_source_contains_default_stride_nopartial_wrapper()
    test_known_vector_matches_explicit_composition()
    test_adversarial_contracts()
    test_randomized_parity_against_explicit_composition()
    print("ok")
