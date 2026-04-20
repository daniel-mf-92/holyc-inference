#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    I64_MAX,
    try_mul_i64_checked,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity import (
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0 or staged_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, default_commit_stage_cell_capacity = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, default_commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_commit_stage_cell_capacity,
        default_commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
    )


def explicit_checked_default_capacity_commit_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0 or staged_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, default_commit_stage_cell_capacity = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, default_commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_commit_stage_cell_capacity,
        default_commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
    )


def test_source_contains_default_capacity_commit_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionTryMulI64Checked(row_count," in body
    assert "AttentionTryMulI64Checked(staged_scores_capacity," in body
    assert "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(" in body


def test_known_vectors_match_explicit_checked_composition() -> None:
    row_count = 4
    token_count = 6
    score_scale_q16 = 23170

    required = row_count * token_count
    in_scores = [((i * 17) - 91) << 14 for i in range(required)]

    out_a = [0x5151] * required
    out_b = out_a.copy()
    stage_a = [0x3A3A] * required
    stage_b = stage_a.copy()

    err_a = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_a,
        required,
        stage_a,
        len(stage_a),
    )
    err_b = explicit_checked_default_capacity_commit_composition(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_b,
        required,
        stage_b,
        len(stage_b),
    )

    assert err_a == err_b
    if err_a == ATTN_Q16_OK:
        assert out_a == out_b


def test_staging_shortfall_rejects_without_partial_writes() -> None:
    row_count = 4
    token_count = 6
    score_scale_q16 = 32768

    required = row_count * token_count
    in_scores = [11] * required
    out_scores = [0x5555] * required
    out_before = out_scores.copy()
    stage = [0x3333] * (required - 1)
    stage_before = stage.copy()

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_scores,
        required,
        stage,
        len(stage),
    )

    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_scores == out_before
    assert stage == stage_before


def test_error_paths_and_overflow() -> None:
    in_scores = [0, 0, 0, 0]
    out_scores = [0, 0, 0, 0]
    stage = [0, 0, 0, 0]

    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
            None, 4, 2, 2, 123, out_scores, 4, stage, 4
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
            in_scores, 4, 2, 2, 123, out_scores, 4, stage, -1
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        [0],
        I64_MAX,
        huge,
        huge,
        123,
        [0],
        I64_MAX,
        [0],
        I64_MAX,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260420_625)

    for _ in range(2200):
        row_count = rng.randint(0, 12)
        token_count = rng.randint(0, 12)
        required = row_count * token_count

        in_capacity = required + rng.randint(0, 5)
        out_capacity = required + rng.randint(0, 6)
        stage_capacity = max(0, required + rng.randint(-3, 9))
        score_scale_q16 = rng.randint(-3 << 16, 3 << 16)

        in_scores = [rng.randint(-(5 << 16), (5 << 16)) for _ in range(max(in_capacity, 1))]
        out_a = [rng.randint(-(2 << 16), (2 << 16)) for _ in range(max(out_capacity, 1))]
        out_b = out_a.copy()
        stage_a = [rng.randint(-(2 << 16), (2 << 16)) for _ in range(max(stage_capacity, 1))]
        stage_b = stage_a.copy()

        err_a = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_a,
            out_capacity,
            stage_a,
            stage_capacity,
        )
        err_b = explicit_checked_default_capacity_commit_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_b,
            out_capacity,
            stage_b,
            stage_capacity,
        )

        assert err_a == err_b
        assert out_a == out_b
        assert stage_a == stage_b


if __name__ == "__main__":
    test_source_contains_default_capacity_commit_wrapper()
    test_known_vectors_match_explicit_checked_composition()
    test_staging_shortfall_rejects_without_partial_writes()
    test_error_paths_and_overflow()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
