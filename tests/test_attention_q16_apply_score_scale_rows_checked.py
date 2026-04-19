#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleRowsChecked."""

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
    attention_q16_apply_score_scale_checked,
)


def attention_q16_apply_score_scale_rows_checked(
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

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if row_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM
    if in_score_stride < 1 or out_score_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, in_row_cells = try_mul_i64_checked(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, in_row_cells = try_add_i64_checked(in_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err

    err, out_row_cells = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, out_row_cells = try_add_i64_checked(out_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err

    if row_stride < in_row_cells or row_stride < out_row_cells:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_in_cells = try_mul_i64_checked(row_count - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_in_cells = try_add_i64_checked(required_in_cells, in_row_cells)
    if err != ATTN_Q16_OK:
        return err

    err, required_out_cells = try_mul_i64_checked(row_count - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, out_row_cells)
    if err != ATTN_Q16_OK:
        return err

    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    row_base = 0
    for _ in range(row_count):
        status = attention_q16_apply_score_scale_checked(
            in_scores_q32[row_base:],
            in_row_cells,
            token_count,
            in_score_stride,
            score_scale_q16,
            out_scores_q32[row_base:],
            out_row_cells,
            out_score_stride,
        )
        if status != ATTN_Q16_OK:
            return status

        err, row_base = try_add_i64_checked(row_base, row_stride)
        if err != ATTN_Q16_OK:
            return err

    return ATTN_Q16_OK


def explicit_rows_composition(
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

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if row_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM
    if in_score_stride < 1 or out_score_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, in_row_cells = try_mul_i64_checked(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, in_row_cells = try_add_i64_checked(in_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err

    err, out_row_cells = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, out_row_cells = try_add_i64_checked(out_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err

    if row_stride < in_row_cells or row_stride < out_row_cells:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_in_cells = try_mul_i64_checked(row_count - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_in_cells = try_add_i64_checked(required_in_cells, in_row_cells)
    if err != ATTN_Q16_OK:
        return err

    err, required_out_cells = try_mul_i64_checked(row_count - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, out_row_cells)
    if err != ATTN_Q16_OK:
        return err

    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for row_index in range(row_count):
        row_base = row_index * row_stride
        status = attention_q16_apply_score_scale_checked(
            in_scores_q32[row_base:],
            in_row_cells,
            token_count,
            in_score_stride,
            score_scale_q16,
            out_scores_q32[row_base:],
            out_row_cells,
            out_score_stride,
        )
        if status != ATTN_Q16_OK:
            return status

    return ATTN_Q16_OK


def test_source_contains_rows_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsChecked("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "AttentionQ16ApplyScoreScaleChecked(" in body


def test_known_vector_row_major_parity() -> None:
    row_count = 3
    token_count = 4
    in_score_stride = 2
    out_score_stride = 3
    row_stride = 11
    score_scale_q16 = 19395

    in_capacity = (row_count - 1) * row_stride + (token_count - 1) * in_score_stride + 1
    out_capacity = (row_count - 1) * row_stride + (token_count - 1) * out_score_stride + 1

    in_scores = [0] * in_capacity
    seeds = [
        [123456789, -222222222, 333333333, -444444444],
        [111111111, 222222222, -333333333, -444444444],
        [98765432, -87654321, 76543210, -65432109],
    ]

    for r in range(row_count):
        base = r * row_stride
        for t in range(token_count):
            in_scores[base + t * in_score_stride] = seeds[r][t]

    out_new = [77] * out_capacity
    out_ref = out_new.copy()

    err_new = attention_q16_apply_score_scale_rows_checked(
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
    err_ref = explicit_rows_composition(
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

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref


def test_adversarial_error_contracts() -> None:
    assert (
        attention_q16_apply_score_scale_rows_checked(
            None,
            0,
            1,
            1,
            1,
            1,
            1,
            1 << 16,
            [0],
            1,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_apply_score_scale_rows_checked(
            [0],
            1,
            -1,
            1,
            1,
            1,
            1,
            1 << 16,
            [0],
            1,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    assert (
        attention_q16_apply_score_scale_rows_checked(
            [0],
            1,
            1,
            2,
            0,
            1,
            1,
            1 << 16,
            [0, 0],
            2,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    assert (
        attention_q16_apply_score_scale_rows_checked(
            [0, 0, 0, 0],
            4,
            2,
            2,
            1,
            1,
            1,
            1 << 16,
            [0, 0, 0, 0],
            4,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    max_i64 = (1 << 63) - 1
    assert (
        attention_q16_apply_score_scale_rows_checked(
            [max_i64],
            1,
            1,
            1,
            1,
            1,
            1,
            max_i64,
            [0],
            1,
        )
        == ATTN_Q16_ERR_OVERFLOW
    )


def test_randomized_parity_against_explicit_rows_composition() -> None:
    rng = random.Random(20260419_571)

    for _ in range(3000):
        row_count = rng.randint(0, 24)
        token_count = rng.randint(0, 24)
        in_score_stride = rng.randint(0, 8)
        out_score_stride = rng.randint(0, 8)
        row_stride = rng.randint(0, 32)
        score_scale_q16 = rng.randint(-(1 << 18), (1 << 18))

        in_capacity = rng.randint(0, 256)
        out_capacity = rng.randint(0, 256)

        if rng.random() < 0.03:
            in_capacity = -rng.randint(1, 8)
        if rng.random() < 0.03:
            out_capacity = -rng.randint(1, 8)

        in_scores = [rng.randint(-(1 << 40), (1 << 40)) for _ in range(max(in_capacity, 1))]
        out_seed = [rng.randint(-(1 << 24), (1 << 24)) for _ in range(max(out_capacity, 1))]

        out_new = out_seed.copy()
        out_ref = out_seed.copy()

        err_new = attention_q16_apply_score_scale_rows_checked(
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
        err_ref = explicit_rows_composition(
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
        assert out_new == out_ref


if __name__ == "__main__":
    test_source_contains_rows_helper()
    test_known_vector_row_major_parity()
    test_adversarial_error_contracts()
    test_randomized_parity_against_explicit_rows_composition()
    print("ok")
