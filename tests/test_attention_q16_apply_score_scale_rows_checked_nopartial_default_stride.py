#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStride."""

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
from test_attention_q16_apply_score_scale_rows_checked_nopartial import (
    attention_q16_apply_score_scale_rows_checked_nopartial,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride(
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
    default_row_stride = token_count

    return attention_q16_apply_score_scale_rows_checked_nopartial(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        default_score_stride,
        default_score_stride,
        default_row_stride,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
    )


def explicit_default_stride_nopartial_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    return attention_q16_apply_score_scale_rows_checked_nopartial(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        token_count,
        token_count,
        token_count,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
    )


def test_source_contains_default_stride_nopartial_rows_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStride("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "default_score_stride = token_count;" in body
    assert "default_row_stride = token_count;" in body
    assert "return AttentionQ16ApplyScoreScaleRowsCheckedNoPartial(" in body


def test_known_vector_matches_explicit_composition() -> None:
    row_count = 4
    token_count = 1
    score_scale_q16 = 23170

    capacity = row_count * token_count
    in_scores = [
        101000,
        505000,
        909000,
        -111000,
    ]
    out_new = [0x4141] * capacity
    out_ref = out_new.copy()

    err_new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride(
        in_scores,
        capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_new,
        capacity,
    )
    err_ref = explicit_default_stride_nopartial_composition(
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


def test_error_parity_and_no_partial_contract() -> None:
    in_scores = [100, 200, 300, 400]
    out_seed = [777] * 4
    out_new = out_seed.copy()
    out_ref = out_seed.copy()

    err_new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride(
        in_scores,
        len(in_scores),
        2,
        2,
        1 << 16,
        out_new,
        3,
    )
    err_ref = explicit_default_stride_nopartial_composition(
        in_scores,
        len(in_scores),
        2,
        2,
        1 << 16,
        out_ref,
        3,
    )
    assert err_new == err_ref == ATTN_Q16_ERR_BAD_PARAM
    assert out_new == out_ref == out_seed

    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride(
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


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_593)

    for _ in range(6000):
        row_count = rng.randint(0, 18)
        token_count = rng.randint(0, 24)
        score_scale_q16 = rng.randint(-(1 << 18), (1 << 18))

        required = row_count * token_count if row_count > 0 and token_count > 0 else 0

        in_capacity = max(0, required + rng.randint(-6, 6))
        out_capacity = max(0, required + rng.randint(-6, 6))

        if rng.random() < 0.03:
            in_capacity = -rng.randint(1, 16)
        if rng.random() < 0.03:
            out_capacity = -rng.randint(1, 16)

        in_scores = [
            rng.randint(-(1 << 42), (1 << 42)) for _ in range(max(in_capacity, 1))
        ]
        out_seed = [
            rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(out_capacity, 1))
        ]

        out_new = out_seed.copy()
        out_ref = out_seed.copy()

        err_new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_new,
            out_capacity,
        )
        err_ref = explicit_default_stride_nopartial_composition(
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
    test_source_contains_default_stride_nopartial_rows_wrapper()
    test_known_vector_matches_explicit_composition()
    test_error_parity_and_no_partial_contract()
    test_randomized_parity_against_explicit_composition()
    print("ok")
