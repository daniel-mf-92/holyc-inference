#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_q16_apply_score_scale_rows_checked import (
    attention_q16_apply_score_scale_rows_checked,
)
from test_attention_q16_apply_score_scale_rows_checked_default_stride import (
    attention_q16_apply_score_scale_rows_checked_default_stride,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc(
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

    if (
        in_scores_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_score_stride = token_count
    default_row_stride = token_count

    last_row_base_index = [0]
    required_in_cells = [0]
    required_out_cells = [0]
    required_stage_cells = [0]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        default_score_stride,
        default_score_stride,
        default_row_stride,
        out_scores_q32,
        out_scores_capacity,
        last_row_base_index,
        required_in_cells,
        required_out_cells,
        required_stage_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if staged_scores_capacity < required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is in_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_apply_score_scale_rows_checked(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        default_score_stride,
        default_score_stride,
        default_row_stride,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

    row_base = 0
    for row_index in range(row_count):
        for token_index in range(token_count):
            err, out_index = try_add_i64_checked(row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

            err, stage_index = try_mul_i64_checked(row_index, token_count)
            if err != ATTN_Q16_OK:
                return err
            err, stage_index = try_add_i64_checked(stage_index, token_index)
            if err != ATTN_Q16_OK:
                return err

            out_scores_q32[out_index] = staged_scores_q32[stage_index]

        err, row_base = try_add_i64_checked(row_base, default_row_stride)
        if err != ATTN_Q16_OK:
            return err

    return ATTN_Q16_OK


def explicit_staged_default_stride_composition(
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

    if (
        in_scores_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_score_stride = token_count
    default_row_stride = token_count

    last_row_base_index = [0]
    required_in_cells = [0]
    required_out_cells = [0]
    required_stage_cells = [0]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        default_score_stride,
        default_score_stride,
        default_row_stride,
        out_scores_q32,
        out_scores_capacity,
        last_row_base_index,
        required_in_cells,
        required_out_cells,
        required_stage_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if staged_scores_capacity < required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is in_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_apply_score_scale_rows_checked_default_stride(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

    for row_index in range(row_count):
        row_base = row_index * token_count
        for token_index in range(token_count):
            out_scores_q32[row_base + token_index] = staged_scores_q32[
                row_base + token_index
            ]

    return ATTN_Q16_OK


def test_source_contains_noalloc_default_stride_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc("
    assert signature in source

    body = source.rsplit(signature, 1)[1]
    assert "default_score_stride = token_count;" in body
    assert "default_row_stride = token_count;" in body
    assert "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly(" in body
    assert "AttentionQ16ApplyScoreScaleRowsChecked(in_scores_q32," in body
    assert "staged_begin < in_end && in_begin < staged_end" in body
    assert "staged_begin < out_end && out_begin < staged_end" in body


def test_known_vectors_match_explicit_staged_composition() -> None:
    row_count = 5
    token_count = 1
    score_scale_q16 = 23170

    required = row_count * token_count
    in_scores = [0] * required
    out_new = [0x66] * required
    out_ref = [0x66] * required
    stage_new = [0x55] * required
    stage_ref = stage_new.copy()

    seeds = [[1234567], [-2345678], [3456789], [-4567890], [5678901]]
    for r in range(row_count):
        for t in range(token_count):
            in_scores[r * token_count + t] = seeds[r][t]

    err_new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_new,
        required,
        stage_new,
        required,
    )
    err_ref = explicit_staged_default_stride_composition(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_ref,
        required,
        stage_ref,
        required,
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref


def test_no_partial_output_preserved_on_scale_failure() -> None:
    row_count = 3
    token_count = 1
    score_scale_q16 = (1 << 63) - 1

    required = row_count * token_count
    in_scores = [10, 20, 30]
    out_scores = [0x7171] * required
    stage = [0x2626] * required

    before = out_scores.copy()
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_scores,
        required,
        stage,
        required,
    )

    assert err != ATTN_Q16_OK
    assert out_scores == before


def test_staging_guards() -> None:
    in_scores = [0, 0, 0, 0]
    out_scores = [0, 0, 0, 0]
    stage = [0, 0, 0, 0]

    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc(
            in_scores, 2, 2, 1, 1 << 16, out_scores, 2, None, 4
        )
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc(
            in_scores, 2, 2, 1, 1 << 16, out_scores, 2, stage, -1
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc(
            in_scores, 2, 2, 1, 1 << 16, out_scores, 2, stage, 1
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc(
            in_scores, 2, 2, 1, 1 << 16, out_scores, 2, in_scores, 4
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc(
            in_scores, 2, 2, 1, 1 << 16, out_scores, 2, out_scores, 4
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )


def test_randomized_parity_against_explicit_staged_composition() -> None:
    rng = random.Random(20260419_589)

    for _ in range(4000):
        row_count = rng.randint(0, 24)
        token_count = rng.randint(0, 24)
        score_scale_q16 = rng.randint(-(1 << 18), (1 << 18))

        required = row_count * token_count
        in_capacity = max(0, required + rng.randint(-8, 8))
        out_capacity = max(0, required + rng.randint(-8, 8))
        stage_capacity = max(0, required + rng.randint(-8, 8))

        if rng.random() < 0.05:
            in_capacity = -rng.randint(1, 16)
        if rng.random() < 0.05:
            out_capacity = -rng.randint(1, 16)
        if rng.random() < 0.05:
            stage_capacity = -rng.randint(1, 16)

        in_scores = [rng.randint(-(1 << 44), (1 << 44)) for _ in range(max(in_capacity, 1))]
        out_seed = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(out_capacity, 1))]

        stage_new = [rng.randint(-(1 << 16), (1 << 16)) for _ in range(max(stage_capacity, 1))]
        stage_ref = stage_new.copy()

        out_new = out_seed.copy()
        out_ref = out_seed.copy()

        err_new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_new,
            out_capacity,
            stage_new,
            stage_capacity,
        )
        err_ref = explicit_staged_default_stride_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_ref,
            out_capacity,
            stage_ref,
            stage_capacity,
        )

        assert err_new == err_ref
        assert out_new == out_ref


def run() -> None:
    test_source_contains_noalloc_default_stride_wrapper()
    test_known_vectors_match_explicit_staged_composition()
    test_no_partial_output_preserved_on_scale_failure()
    test_staging_guards()
    test_randomized_parity_against_explicit_staged_composition()
    print("attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc=ok")


if __name__ == "__main__":
    run()
