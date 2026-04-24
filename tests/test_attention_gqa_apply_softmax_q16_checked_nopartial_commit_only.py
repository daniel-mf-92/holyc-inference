#!/usr/bin/env python3
"""Reference checks for GQAAttentionApplySoftmaxQ16CheckedNoPartialCommitOnly (IQ-1370)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_gqa_apply_softmax_q16_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    gqa_attention_apply_softmax_q16_checked,
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_gqa_apply_softmax_q16_checked_nopartial import (
    gqa_attention_apply_softmax_q16_checked_nopartial,
)


def gqa_attention_apply_softmax_q16_checked_nopartial_commit_only(
    scores_q32,
    scores_capacity: int,
    query_rows: int,
    key_rows: int,
    head_groups: int,
    row_stride: int,
    out_probs_q16,
    out_capacity: int,
) -> int:
    if scores_q32 is None or out_probs_q16 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if scores_capacity < 0 or out_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_rows < 0 or key_rows < 0 or head_groups <= 0 or row_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_query_rows = query_rows
    snapshot_key_rows = key_rows
    snapshot_head_groups = head_groups
    snapshot_row_stride = row_stride
    snapshot_scores_capacity = scores_capacity
    snapshot_out_capacity = out_capacity

    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if key_rows > row_stride:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_rows == 0 or key_rows == 0:
        return ATTN_Q16_OK

    err, required_score_cells = try_mul_i64_checked(query_rows - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_score_cells = try_add_i64_checked(required_score_cells, key_rows)
    if err != ATTN_Q16_OK:
        return err

    err, required_out_cells = try_mul_i64_checked(query_rows, key_rows)
    if err != ATTN_Q16_OK:
        return err

    if required_score_cells > scores_capacity or required_out_cells > out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells == 0:
        return ATTN_Q16_OK

    if out_probs_q16 is scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_probs = [0] * required_out_cells
    parity_probs = [0] * required_out_cells

    err = gqa_attention_apply_softmax_q16_checked_nopartial(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        staged_probs,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err = gqa_attention_apply_softmax_q16_checked(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        parity_probs,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_score_cells = try_mul_i64_checked(
        snapshot_query_rows - 1, snapshot_row_stride
    )
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_required_score_cells = try_add_i64_checked(
        recomputed_required_score_cells, snapshot_key_rows
    )
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_required_out_cells = try_mul_i64_checked(
        snapshot_query_rows, snapshot_key_rows
    )
    if err != ATTN_Q16_OK:
        return err

    if (
        snapshot_query_rows != query_rows
        or snapshot_key_rows != key_rows
        or snapshot_head_groups != head_groups
        or snapshot_row_stride != row_stride
        or snapshot_scores_capacity != scores_capacity
        or snapshot_out_capacity != out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        required_score_cells != recomputed_required_score_cells
        or required_out_cells != recomputed_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if required_score_cells > snapshot_scores_capacity or required_out_cells > snapshot_out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for idx in range(required_out_cells):
        if staged_probs[idx] != parity_probs[idx]:
            return ATTN_Q16_ERR_BAD_PARAM

    for idx in range(required_out_cells):
        out_probs_q16[idx] = staged_probs[idx]

    return ATTN_Q16_OK


def explicit_commit_only_composition(*args, **kwargs) -> int:
    return gqa_attention_apply_softmax_q16_checked_nopartial_commit_only(*args, **kwargs)


def test_fixed_vector_reference() -> None:
    query_rows = 4
    key_rows = 3
    head_groups = 2
    row_stride = 5

    scores_q32 = [
        9 << 16,
        7 << 16,
        4 << 16,
        123,
        456,
        12 << 16,
        6 << 16,
        1 << 16,
        789,
        321,
        2 << 16,
        2 << 16,
        2 << 16,
        111,
        222,
        5 << 16,
        6 << 16,
        7 << 16,
        333,
        444,
    ]

    out_a = [777] * (query_rows * key_rows)
    out_b = out_a.copy()

    err_a = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only(
        scores_q32,
        len(scores_q32),
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        out_a,
        len(out_a),
    )
    err_b = explicit_commit_only_composition(
        scores_q32,
        len(scores_q32),
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        out_b,
        len(out_b),
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b


def test_error_contract_null_alias_bounds_overflow() -> None:
    seed = [9191] * 6

    out = seed.copy()
    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only(
        None,
        0,
        0,
        0,
        1,
        0,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out == seed

    scores = [7 << 16, 6 << 16, 5 << 16, 4 << 16]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only(
        scores,
        len(scores),
        1,
        2,
        1,
        2,
        scores,
        len(scores),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    out = seed.copy()
    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only(
        scores,
        1,
        1,
        2,
        1,
        2,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == seed

    out = seed.copy()
    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only(
        scores,
        len(scores),
        1 << 62,
        2,
        1,
        2,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out == seed


def test_randomized_parity() -> None:
    rng = random.Random(1370)

    for _ in range(200):
        head_groups = rng.choice([1, 2, 4])
        group_rows = rng.randint(1, 3)
        query_rows = head_groups * group_rows
        key_rows = rng.randint(1, 6)
        row_stride = key_rows + rng.randint(0, 3)
        required_score_cells = (query_rows - 1) * row_stride + key_rows

        scores = [rng.randint(-(15 << 16), (15 << 16)) for _ in range(required_score_cells)]
        out_a = [0] * (query_rows * key_rows)
        out_b = [0] * (query_rows * key_rows)

        err_a = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            row_stride,
            out_a,
            len(out_a),
        )
        err_b = explicit_commit_only_composition(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            row_stride,
            out_b,
            len(out_b),
        )

        assert err_a == err_b == ATTN_Q16_OK
        assert out_a == out_b
        for row in range(query_rows):
            row_sum = sum(out_a[row * key_rows : (row + 1) * key_rows])
            assert abs(row_sum - (1 << 16)) <= key_rows

