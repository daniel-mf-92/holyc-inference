#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleRowsCheckedNoPartial."""

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
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_q16_apply_score_scale_checked_nopartial import (
    attention_q16_apply_score_scale_checked_nopartial,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only,
)


def attention_q16_apply_score_scale_rows_checked_nopartial(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    in_score_stride: int,
    out_score_stride: int,
    row_stride: int,
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
    if in_score_stride < 0 or out_score_stride < 0 or row_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    last_in_index = [0]
    last_out_index = [0]
    required_in_cells = [0]
    required_out_cells = [0]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        in_score_stride,
        out_score_stride,
        row_stride,
        row_stride,
        out_scores_q32,
        out_scores_capacity,
        last_in_index,
        last_out_index,
        required_in_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    err, in_row_cells = try_mul_i64_checked(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, in_row_cells = try_add_i64_checked(in_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_cells = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    stage = [0] * required_stage_cells

    row_base = 0
    for row_index in range(row_count):
        err, stage_row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        err = attention_q16_apply_score_scale_checked_nopartial(
            in_scores_q32[row_base:],
            in_row_cells,
            token_count,
            in_score_stride,
            score_scale_q16,
            stage[stage_row_base:],
            token_count,
            1,
        )
        if err != ATTN_Q16_OK:
            return err

        err, row_base = try_add_i64_checked(row_base, row_stride)
        if err != ATTN_Q16_OK:
            return err

    row_base = 0
    for row_index in range(row_count):
        err, stage_row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(token_count):
            err, out_index = try_add_i64_checked(row_base, token_index)
            if err != ATTN_Q16_OK:
                return err
            err, stage_index = try_add_i64_checked(stage_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err
            out_scores_q32[out_index] = stage[stage_index]

        err, row_base = try_add_i64_checked(row_base, row_stride)
        if err != ATTN_Q16_OK:
            return err

    return ATTN_Q16_OK


def explicit_staged_row_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    in_score_stride: int,
    out_score_stride: int,
    row_stride: int,
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
    if in_score_stride < 0 or out_score_stride < 0 or row_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    last_in_index = [0]
    last_out_index = [0]
    required_in_cells = [0]
    required_out_cells = [0]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        in_score_stride,
        out_score_stride,
        row_stride,
        row_stride,
        out_scores_q32,
        out_scores_capacity,
        last_in_index,
        last_out_index,
        required_in_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    err, in_row_cells = try_mul_i64_checked(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, in_row_cells = try_add_i64_checked(in_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_cells = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    stage = [0] * required_stage_cells

    for row_index in range(row_count):
        row_base = row_index * row_stride
        stage_row_base = row_index * token_count

        err = attention_q16_apply_score_scale_checked_nopartial(
            in_scores_q32[row_base:],
            in_row_cells,
            token_count,
            in_score_stride,
            score_scale_q16,
            stage[stage_row_base:],
            token_count,
            1,
        )
        if err != ATTN_Q16_OK:
            return err

    for row_index in range(row_count):
        row_base = row_index * row_stride
        stage_row_base = row_index * token_count
        for token_index in range(token_count):
            out_scores_q32[row_base + token_index] = stage[stage_row_base + token_index]

    return ATTN_Q16_OK


def test_source_contains_rows_nopartial_symbol_and_staged_row_calls() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartial("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly(" in body
    assert "status = AttentionQ16ApplyScoreScaleCheckedNoPartial(" in body
    assert "staged_scores_q32 = MAlloc(stage_bytes);" in body


def test_known_vector_matches_explicit_staged_row_composition() -> None:
    row_count = 3
    token_count = 4
    in_score_stride = 2
    out_score_stride = 3
    row_stride = 11
    score_scale_q16 = 23170

    in_capacity = (row_count - 1) * row_stride + (token_count - 1) * in_score_stride + 1
    out_capacity = (row_count - 1) * row_stride + (token_count - 1) * out_score_stride + 1

    in_scores = [0] * in_capacity
    out_new = [0x4141] * out_capacity
    out_ref = out_new.copy()

    values = [
        [101000, -202000, 303000, -404000],
        [505000, -606000, 707000, -808000],
        [909000, -111000, 222000, -333000],
    ]
    for row in range(row_count):
        for token in range(token_count):
            in_scores[row * row_stride + token * in_score_stride] = values[row][token]

    err_new = attention_q16_apply_score_scale_rows_checked_nopartial(
        in_scores,
        len(in_scores),
        row_count,
        token_count,
        in_score_stride,
        out_score_stride,
        row_stride,
        score_scale_q16,
        out_new,
        len(out_new),
    )
    err_ref = explicit_staged_row_composition(
        in_scores,
        len(in_scores),
        row_count,
        token_count,
        in_score_stride,
        out_score_stride,
        row_stride,
        score_scale_q16,
        out_ref,
        len(out_ref),
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref


def test_no_partial_output_preserved_on_scaling_error() -> None:
    row_count = 2
    token_count = 1
    in_score_stride = 1
    out_score_stride = 1
    row_stride = 2
    score_scale_q16 = I64_MAX

    in_scores = [I64_MAX, 11, I64_MAX, 22]
    out_scores = [0x5A5A] * 4
    before = out_scores.copy()

    err = attention_q16_apply_score_scale_rows_checked_nopartial(
        in_scores,
        len(in_scores),
        row_count,
        token_count,
        in_score_stride,
        out_score_stride,
        row_stride,
        score_scale_q16,
        out_scores,
        len(out_scores),
    )

    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out_scores == before


def test_randomized_parity_and_no_partial_contract() -> None:
    rng = random.Random(20260419_574)

    for _ in range(5000):
        row_count = rng.randint(0, 16)
        token_count = rng.randint(0, 24)
        in_score_stride = rng.randint(0, 8)
        out_score_stride = rng.randint(0, 8)
        row_stride = rng.randint(0, 64)
        score_scale_q16 = rng.randint(-(1 << 16), (1 << 16))

        in_row_cells = 0
        out_row_cells = 0
        if token_count > 0 and in_score_stride > 0:
            in_row_cells = 1 + (token_count - 1) * in_score_stride
        if token_count > 0 and out_score_stride > 0:
            out_row_cells = 1 + (token_count - 1) * out_score_stride

        in_capacity = 0
        out_capacity = 0
        if row_count > 0:
            in_capacity = (row_count - 1) * max(row_stride, 0) + in_row_cells
            out_capacity = (row_count - 1) * max(row_stride, 0) + out_row_cells

        in_capacity = max(in_capacity, 0)
        out_capacity = max(out_capacity, 0)

        in_scores = [0] * max(in_capacity, 1)
        out_new = [0x33] * max(out_capacity, 1)
        out_ref = out_new.copy()

        if (
            row_count > 0
            and token_count > 0
            and in_score_stride > 0
            and row_stride > 0
            and row_stride >= in_row_cells
            and row_stride >= out_row_cells
            and in_capacity > 0
        ):
            for row in range(row_count):
                row_base = row * row_stride
                for token in range(token_count):
                    in_scores[row_base + token * in_score_stride] = rng.randint(
                        -(1 << 30), (1 << 30)
                    )

        err_new = attention_q16_apply_score_scale_rows_checked_nopartial(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            in_score_stride,
            out_score_stride,
            row_stride,
            score_scale_q16,
            out_new,
            out_capacity,
        )
        err_ref = explicit_staged_row_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            in_score_stride,
            out_score_stride,
            row_stride,
            score_scale_q16,
            out_ref,
            out_capacity,
        )

        assert err_new == err_ref
        if err_new == ATTN_Q16_OK:
            assert out_new == out_ref
        else:
            assert out_new == [0x33] * max(out_capacity, 1)
            assert out_ref == [0x33] * max(out_capacity, 1)


def test_error_surfaces_match_reference() -> None:
    out_a = [7, 8, 9]
    out_b = [7, 8, 9]

    err_a = attention_q16_apply_score_scale_rows_checked_nopartial(
        None,
        0,
        1,
        1,
        1,
        1,
        1,
        1 << 16,
        out_a,
        len(out_a),
    )
    err_b = explicit_staged_row_composition(
        None,
        0,
        1,
        1,
        1,
        1,
        1,
        1 << 16,
        out_b,
        len(out_b),
    )

    assert err_a == err_b == ATTN_Q16_ERR_NULL_PTR
    assert out_a == [7, 8, 9]
    assert out_b == [7, 8, 9]

    err_a = attention_q16_apply_score_scale_rows_checked_nopartial(
        [1],
        1,
        1,
        2,
        1,
        1,
        1,
        1 << 16,
        out_a,
        len(out_a),
    )
    err_b = explicit_staged_row_composition(
        [1],
        1,
        1,
        2,
        1,
        1,
        1,
        1 << 16,
        out_b,
        len(out_b),
    )

    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM
