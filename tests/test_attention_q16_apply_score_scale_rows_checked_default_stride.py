#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleRowsCheckedDefaultStride."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
)
from test_attention_q16_apply_score_scale_rows_checked import (
    attention_q16_apply_score_scale_rows_checked,
)


def attention_q16_apply_score_scale_rows_checked_default_stride(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_score_stride = token_count
    return attention_q16_apply_score_scale_rows_checked(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        default_score_stride,
        default_score_stride,
        default_score_stride,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
    )


def explicit_rows_default_stride_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    stride = token_count
    return attention_q16_apply_score_scale_rows_checked(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        stride,
        stride,
        stride,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
    )


def test_source_contains_default_stride_rows_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedDefaultStride("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "default_score_stride = token_count;" in body
    assert "return AttentionQ16ApplyScoreScaleRowsChecked(" in body


def test_known_vector_matches_explicit_composition() -> None:
    row_count = 3
    token_count = 4
    score_scale_q16 = 23170

    stride = token_count
    row_cells = 1 + (token_count - 1) * stride
    row_span = stride
    capacity = (row_count - 1) * row_span + row_cells

    in_scores = [0] * capacity
    seeds = [
        [1000000, -2000000, 3000000, -4000000],
        [1111111, -2222222, 3333333, -4444444],
        [5555555, -6666666, 7777777, -8888888],
    ]
    for row_index in range(row_count):
        row_base = row_index * row_span
        for token_index in range(token_count):
            in_scores[row_base + token_index * stride] = seeds[row_index][token_index]

    out_new = [123] * capacity
    out_ref = out_new.copy()

    err_new = attention_q16_apply_score_scale_rows_checked_default_stride(
        in_scores,
        capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_new,
        capacity,
    )
    err_ref = explicit_rows_default_stride_composition(
        in_scores,
        capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_ref,
        capacity,
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref


def test_adversarial_contracts() -> None:
    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride(
            None,
            0,
            0,
            0,
            1 << 16,
            [0],
            1,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride(
            [0],
            1,
            0,
            0,
            1 << 16,
            None,
            0,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride(
            [0],
            -1,
            0,
            0,
            1 << 16,
            [0],
            1,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride(
            [0],
            1,
            -1,
            0,
            1 << 16,
            [0],
            1,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride(
            [0],
            1,
            1,
            -1,
            1 << 16,
            [0],
            1,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260419_588)

    for _ in range(5000):
        row_count = rng.randint(0, 24)
        token_count = rng.randint(0, 48)
        score_scale_q16 = rng.randint(-(1 << 18), (1 << 18))

        stride = token_count
        if row_count == 0 or token_count == 0:
            required = 0
        else:
            row_cells = 1 + (token_count - 1) * stride
            required = (row_count - 1) * stride + row_cells

        in_capacity = max(0, required + rng.randint(-8, 8))
        out_capacity = max(0, required + rng.randint(-8, 8))

        if rng.random() < 0.04:
            in_capacity = -rng.randint(1, 16)
        if rng.random() < 0.04:
            out_capacity = -rng.randint(1, 16)

        in_scores = [rng.randint(-(1 << 42), (1 << 42)) for _ in range(max(in_capacity, 1))]
        out_seed = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(out_capacity, 1))]

        out_new = out_seed.copy()
        out_ref = out_seed.copy()

        err_new = attention_q16_apply_score_scale_rows_checked_default_stride(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_new,
            out_capacity,
        )
        err_ref = explicit_rows_default_stride_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_ref,
            out_capacity,
        )

        assert err_new == err_ref
        assert out_new == out_ref


if __name__ == "__main__":
    test_source_contains_default_stride_rows_wrapper()
    test_known_vector_matches_explicit_composition()
    test_adversarial_contracts()
    test_randomized_parity_against_explicit_composition()
    print("ok")
