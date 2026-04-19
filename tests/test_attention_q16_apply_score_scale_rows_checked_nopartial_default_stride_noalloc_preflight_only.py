#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly."""

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
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only,
)

I64_MAX = (1 << 63) - 1


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    out_scores_q32,
    out_scores_capacity: int,
    stage_cell_capacity: int,
) -> tuple[int, int, int, int]:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR, 0, 0, 0

    if in_scores_capacity < 0 or out_scores_capacity < 0 or stage_cell_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0, 0
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0, 0

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
        return err, 0, 0, 0

    if required_stage_cells[0] > stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0, 0

    return (
        ATTN_Q16_OK,
        required_in_cells[0],
        required_out_cells[0],
        required_stage_cells[0],
    )


def explicit_checked_default_stride_noalloc_preflight(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    out_scores_q32,
    out_scores_capacity: int,
    stage_cell_capacity: int,
) -> tuple[int, int, int, int]:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR, 0, 0, 0

    if in_scores_capacity < 0 or out_scores_capacity < 0 or stage_cell_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0, 0
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0, 0

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK, 0, 0, 0

    default_score_stride = token_count
    default_row_stride = token_count

    err, in_row_cells = try_mul_i64_checked(token_count - 1, default_score_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0, 0
    err, in_row_cells = try_add_i64_checked(in_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err, 0, 0, 0

    err, out_row_cells = try_mul_i64_checked(token_count - 1, default_score_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0, 0
    err, out_row_cells = try_add_i64_checked(out_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err, 0, 0, 0

    if default_row_stride < in_row_cells or default_row_stride < out_row_cells:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0, 0

    err, last_row_base = try_mul_i64_checked(row_count - 1, default_row_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0, 0

    err, required_in_cells = try_add_i64_checked(last_row_base, in_row_cells)
    if err != ATTN_Q16_OK:
        return err, 0, 0, 0
    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0, 0

    err, required_out_cells = try_add_i64_checked(last_row_base, out_row_cells)
    if err != ATTN_Q16_OK:
        return err, 0, 0, 0
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0, 0

    err, required_stage_cells = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err, 0, 0, 0
    if required_stage_cells > stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0, 0

    return ATTN_Q16_OK, required_in_cells, required_out_cells, required_stage_cells


def test_source_contains_noalloc_default_stride_preflight_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "default_score_stride = token_count;" in body
    assert "default_row_stride = token_count;" in body
    assert "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly(" in body
    assert "if (*out_required_stage_cells > stage_cell_capacity)" in body


def test_noalloc_wrapper_routes_through_new_preflight_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc("
    assert signature in source
    body = source.rsplit(signature, 1)[1]
    assert "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(" in body


def test_known_vectors_and_stage_capacity_gate() -> None:
    row_count = 3
    token_count = 1
    required = row_count * token_count
    in_scores = [11] * required
    out_scores = [0x55] * required

    err_new, need_in_new, need_out_new, need_stage_new = (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
            in_scores,
            required,
            row_count,
            token_count,
            out_scores,
            required,
            required,
        )
    )
    err_ref, need_in_ref, need_out_ref, need_stage_ref = (
        explicit_checked_default_stride_noalloc_preflight(
            in_scores,
            required,
            row_count,
            token_count,
            out_scores,
            required,
            required,
        )
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert (need_in_new, need_out_new, need_stage_new) == (
        need_in_ref,
        need_out_ref,
        need_stage_ref,
    )

    err_new, _, _, _ = (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
            in_scores,
            required,
            row_count,
            token_count,
            out_scores,
            required,
            required - 1,
        )
    )
    assert err_new == ATTN_Q16_ERR_BAD_PARAM


def test_adversarial_error_vectors() -> None:
    sample = [1, 2, 3, 4]
    out = [0, 0, 0, 0]

    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
            None, 4, 1, 1, out, 4, 1
        )[0]
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
            sample, 4, 1, 1, None, 4, 1
        )[0]
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
            sample, -1, 1, 1, out, 4, 1
        )[0]
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
            sample, 4, -1, 1, out, 4, 1
        )[0]
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
            sample, 4, 1, 1, out, 4, -1
        )[0]
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err_new, _, _, _ = (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
            [7],
            I64_MAX,
            huge,
            huge,
            [0],
            I64_MAX,
            I64_MAX,
        )
    )
    assert err_new == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_checked_math() -> None:
    rng = random.Random(20260420_594)

    for _ in range(5000):
        row_count = rng.randint(0, 30)
        token_count = rng.randint(0, 30)
        required = row_count * token_count

        in_capacity = max(0, required + rng.randint(-10, 10))
        out_capacity = max(0, required + rng.randint(-10, 10))
        stage_capacity = max(0, required + rng.randint(-10, 10))

        if rng.random() < 0.05:
            in_capacity = -rng.randint(1, 20)
        if rng.random() < 0.05:
            out_capacity = -rng.randint(1, 20)
        if rng.random() < 0.05:
            stage_capacity = -rng.randint(1, 20)

        if rng.random() < 0.05:
            row_count = (1 << 62) + rng.randint(0, 8)
            token_count = (1 << 62) + rng.randint(0, 8)

        in_scores = None if rng.random() < 0.03 else [0] * max(in_capacity, 1)
        out_scores = None if rng.random() < 0.03 else [0] * max(out_capacity, 1)

        new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            out_scores,
            out_capacity,
            stage_capacity,
        )
        ref = explicit_checked_default_stride_noalloc_preflight(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            out_scores,
            out_capacity,
            stage_capacity,
        )

        assert new == ref


def run() -> None:
    test_source_contains_noalloc_default_stride_preflight_only_helper()
    test_noalloc_wrapper_routes_through_new_preflight_helper()
    test_known_vectors_and_stage_capacity_gate()
    test_adversarial_error_vectors()
    test_randomized_parity_against_explicit_checked_math()
    print("attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only=ok")


if __name__ == "__main__":
    run()
