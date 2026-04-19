#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleRowsCheckedDefaultStrideNoPartialPreflightOnly."""

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


def attention_q16_apply_score_scale_rows_checked_default_stride_nopartial_preflight_only(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> tuple[int, int, int]:
    if (
        in_scores_q32 is None
        or out_scores_q32 is None
    ):
        return ATTN_Q16_ERR_NULL_PTR, 0, 0

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0

    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK, 0, 0

    default_score_stride = token_count
    default_row_stride = token_count

    if default_score_stride < 1 or default_row_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0

    err, in_row_cells = try_mul_i64_checked(token_count - 1, default_score_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0
    err, in_row_cells = try_add_i64_checked(in_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err, 0, 0

    err, out_row_cells = try_mul_i64_checked(token_count - 1, default_score_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0
    err, out_row_cells = try_add_i64_checked(out_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err, 0, 0

    if default_row_stride < in_row_cells or default_row_stride < out_row_cells:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0

    err, required_in_cells = try_mul_i64_checked(row_count - 1, default_row_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0
    err, required_in_cells = try_add_i64_checked(required_in_cells, in_row_cells)
    if err != ATTN_Q16_OK:
        return err, 0, 0

    err, required_out_cells = try_mul_i64_checked(row_count - 1, default_row_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0
    err, required_out_cells = try_add_i64_checked(required_out_cells, out_row_cells)
    if err != ATTN_Q16_OK:
        return err, 0, 0

    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0

    return ATTN_Q16_OK, required_in_cells, required_out_cells


def explicit_checked_guard_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> tuple[int, int, int]:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR, 0, 0

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0

    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK, 0, 0

    default_score_stride = token_count
    default_row_stride = token_count

    err, in_row_cells = try_mul_i64_checked(token_count - 1, default_score_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0
    err, in_row_cells = try_add_i64_checked(in_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err, 0, 0

    err, out_row_cells = try_mul_i64_checked(token_count - 1, default_score_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0
    err, out_row_cells = try_add_i64_checked(out_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err, 0, 0

    if default_row_stride < in_row_cells or default_row_stride < out_row_cells:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0

    err, required_in_cells = try_mul_i64_checked(row_count - 1, default_row_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0
    err, required_in_cells = try_add_i64_checked(required_in_cells, in_row_cells)
    if err != ATTN_Q16_OK:
        return err, 0, 0

    err, required_out_cells = try_mul_i64_checked(row_count - 1, default_row_stride)
    if err != ATTN_Q16_OK:
        return err, 0, 0
    err, required_out_cells = try_add_i64_checked(required_out_cells, out_row_cells)
    if err != ATTN_Q16_OK:
        return err, 0, 0

    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM, 0, 0

    return ATTN_Q16_OK, required_in_cells, required_out_cells


def test_source_contains_default_stride_nopartial_preflight_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedDefaultStrideNoPartialPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "default_score_stride = token_count;" in body
    assert "default_row_stride = token_count;" in body
    assert "*out_required_in_cells = required_in_cells;" in body
    assert "*out_required_out_cells = required_out_cells;" in body


def test_known_vectors_and_diagnostics() -> None:
    row_count = 4
    token_count = 1
    cap = row_count
    in_scores = [11, -22, 33, -44]
    out_scores = [0] * cap

    err_new, need_in_new, need_out_new = (
        attention_q16_apply_score_scale_rows_checked_default_stride_nopartial_preflight_only(
            in_scores,
            cap,
            row_count,
            token_count,
            out_scores,
            cap,
        )
    )
    err_ref, need_in_ref, need_out_ref = explicit_checked_guard_composition(
        in_scores,
        cap,
        row_count,
        token_count,
        out_scores,
        cap,
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert (need_in_new, need_out_new) == (need_in_ref, need_out_ref) == (4, 4)


def test_adversarial_contracts() -> None:
    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride_nopartial_preflight_only(
            None, 0, 0, 0, [0], 1
        )[0]
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride_nopartial_preflight_only(
            [0], 1, 0, 0, None, 0
        )[0]
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride_nopartial_preflight_only(
            [0], -1, 0, 0, [0], 1
        )[0]
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride_nopartial_preflight_only(
            [0], 1, -1, 0, [0], 1
        )[0]
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride_nopartial_preflight_only(
            [0], 1, 1, -1, [0], 1
        )[0]
        == ATTN_Q16_ERR_BAD_PARAM
    )

    # For token_count > 1, default row stride (=token_count) is shorter than
    # active row span (=1 + (token_count-1)*token_count), so preflight fails.
    assert (
        attention_q16_apply_score_scale_rows_checked_default_stride_nopartial_preflight_only(
            [0] * 64,
            64,
            2,
            3,
            [0] * 64,
            64,
        )[0]
        == ATTN_Q16_ERR_BAD_PARAM
    )


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260419_572)

    for _ in range(5000):
        row_count = rng.randint(0, 80)
        token_count = rng.randint(0, 80)

        if row_count == 0 or token_count == 0:
            required = 0
        else:
            row_cells = 1 + (token_count - 1) * token_count
            required = (row_count - 1) * token_count + row_cells

        in_capacity = max(0, required + rng.randint(-8, 8))
        out_capacity = max(0, required + rng.randint(-8, 8))

        if rng.random() < 0.05:
            in_capacity = -rng.randint(1, 8)
        if rng.random() < 0.05:
            out_capacity = -rng.randint(1, 8)

        in_scores = [rng.randint(-(1 << 30), (1 << 30)) for _ in range(max(in_capacity, 1))]
        out_scores = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(out_capacity, 1))]

        got = attention_q16_apply_score_scale_rows_checked_default_stride_nopartial_preflight_only(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            out_scores,
            out_capacity,
        )
        ref = explicit_checked_guard_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            out_scores,
            out_capacity,
        )

        assert got == ref


if __name__ == "__main__":
    test_source_contains_default_stride_nopartial_preflight_helper()
    test_known_vectors_and_diagnostics()
    test_adversarial_contracts()
    test_randomized_parity_against_explicit_composition()
    print("ok")
