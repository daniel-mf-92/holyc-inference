#!/usr/bin/env python3
"""Reference checks for GQAAttentionApplySoftmaxQ16CheckedNoPartialPreflightDefaultStrideCommitOnlyPreflightOnlyParityCommitOnly (IQ-1426)."""

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
    I64_MAX,
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_gqa_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only import (
    gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only,
)
from test_attention_gqa_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity import (
    gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity,
)


def gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
    scores_q32,
    scores_capacity: int,
    query_rows: int,
    key_rows: int,
    head_groups: int,
    out_probs_q16,
    out_capacity: int,
    out_required_score_cells,
    out_required_out_cells,
) -> int:
    if out_required_score_cells is None or out_required_out_cells is None:
        return ATTN_Q16_ERR_NULL_PTR
    if out_required_score_cells is out_required_out_cells:
        return ATTN_Q16_ERR_BAD_PARAM

    if scores_q32 is None or out_probs_q16 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if scores_capacity < 0 or out_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_rows < 0 or key_rows < 0 or head_groups <= 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_query_rows = query_rows
    snapshot_key_rows = key_rows
    snapshot_head_groups = head_groups
    snapshot_scores_capacity = scores_capacity
    snapshot_out_capacity = out_capacity
    snapshot_scores = scores_q32
    snapshot_out = out_probs_q16

    snapshot_required_score_ptr = out_required_score_cells
    snapshot_required_out_ptr = out_required_out_cells
    snapshot_required_score_slot = out_required_score_cells[0]
    snapshot_required_out_slot = out_required_out_cells[0]

    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM

    recomputed_required_score_cells = 0
    recomputed_required_out_cells = 0
    if not (query_rows == 0 or key_rows == 0):
        err, recomputed_required_score_cells = try_mul_i64_checked(query_rows - 1, key_rows)
        if err != ATTN_Q16_OK:
            return err
        err, recomputed_required_score_cells = try_add_i64_checked(
            recomputed_required_score_cells,
            key_rows,
        )
        if err != ATTN_Q16_OK:
            return err

        err, recomputed_required_out_cells = try_mul_i64_checked(query_rows, key_rows)
        if err != ATTN_Q16_OK:
            return err

        if (
            recomputed_required_score_cells > scores_capacity
            or recomputed_required_out_cells > out_capacity
        ):
            return ATTN_Q16_ERR_BAD_PARAM

    staged_parity_score = [recomputed_required_score_cells]
    staged_parity_out = [recomputed_required_out_cells]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        out_probs_q16,
        out_capacity,
        staged_parity_score,
        staged_parity_out,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_preflight_score = [recomputed_required_score_cells]
    staged_preflight_out = [recomputed_required_out_cells]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        out_probs_q16,
        out_capacity,
        staged_preflight_score,
        staged_preflight_out,
    )
    if err != ATTN_Q16_OK:
        return err

    if (
        snapshot_query_rows != query_rows
        or snapshot_key_rows != key_rows
        or snapshot_head_groups != head_groups
        or snapshot_scores_capacity != scores_capacity
        or snapshot_out_capacity != out_capacity
        or snapshot_scores is not scores_q32
        or snapshot_out is not out_probs_q16
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_score[0] != staged_preflight_score[0]
        or staged_parity_out[0] != staged_preflight_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_score[0] != recomputed_required_score_cells
        or staged_parity_out[0] != recomputed_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_score[0] > snapshot_scores_capacity
        or staged_parity_out[0] > snapshot_out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_score_cells is not snapshot_required_score_ptr
        or out_required_out_cells is not snapshot_required_out_ptr
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_score_cells[0] != snapshot_required_score_slot
        or out_required_out_cells[0] != snapshot_required_out_slot
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_score_cells[0] = staged_parity_score[0]
    out_required_out_cells[0] = staged_parity_out[0]
    return ATTN_Q16_OK


def explicit_default_stride_commit_only_preflight_only_parity_commit_only_composition(*args) -> int:
    scores_q32 = args[0]
    scores_capacity = args[1]
    query_rows = args[2]
    key_rows = args[3]
    head_groups = args[4]
    out_probs_q16 = args[5]
    out_capacity = args[6]
    out_required_score_cells = args[7]
    out_required_out_cells = args[8]

    recomputed_required_score_cells = 0
    recomputed_required_out_cells = 0
    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if not (query_rows == 0 or key_rows == 0):
        err, recomputed_required_score_cells = try_mul_i64_checked(query_rows - 1, key_rows)
        if err != ATTN_Q16_OK:
            return err
        err, recomputed_required_score_cells = try_add_i64_checked(
            recomputed_required_score_cells,
            key_rows,
        )
        if err != ATTN_Q16_OK:
            return err
        err, recomputed_required_out_cells = try_mul_i64_checked(query_rows, key_rows)
        if err != ATTN_Q16_OK:
            return err

    staged_parity_score = [recomputed_required_score_cells]
    staged_parity_out = [recomputed_required_out_cells]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        out_probs_q16,
        out_capacity,
        staged_parity_score,
        staged_parity_out,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_preflight_score = [recomputed_required_score_cells]
    staged_preflight_out = [recomputed_required_out_cells]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        out_probs_q16,
        out_capacity,
        staged_preflight_score,
        staged_preflight_out,
    )
    if err != ATTN_Q16_OK:
        return err

    if (
        staged_parity_score[0] != staged_preflight_score[0]
        or staged_parity_out[0] != staged_preflight_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_score_cells[0] = staged_parity_score[0]
    out_required_out_cells[0] = staged_parity_out[0]
    return ATTN_Q16_OK


def test_fixed_vector_reference_tuple_publish_no_writes() -> None:
    query_rows = 4
    key_rows = 3
    head_groups = 2

    scores = [42] * (query_rows * key_rows)
    out = [777] * (query_rows * key_rows)
    out_before = out.copy()

    required_score = [-1]
    required_out = [-1]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
        scores,
        len(scores),
        query_rows,
        key_rows,
        head_groups,
        out,
        len(out),
        required_score,
        required_out,
    )

    assert err == ATTN_Q16_OK
    assert required_score[0] == 12
    assert required_out[0] == 12
    assert out == out_before


def test_null_alias_capacity_overflow_parity_vectors() -> None:
    out = [99] * 12
    required_score = [1]
    required_out = [2]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
        None,
        0,
        1,
        1,
        1,
        out,
        len(out),
        required_score,
        required_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert required_score[0] == 1
    assert required_out[0] == 2

    alias_slot = [7]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
        [1],
        1,
        1,
        1,
        1,
        out,
        len(out),
        alias_slot,
        alias_slot,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert alias_slot[0] == 7

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
        [1 << 16, 2 << 16, 3 << 16],
        2,
        1,
        3,
        1,
        out,
        len(out),
        required_score,
        required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
        [1 << 16],
        1,
        1,
        2,
        1,
        out,
        1,
        required_score,
        required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
        [1],
        1,
        I64_MAX,
        2,
        1,
        out,
        len(out),
        required_score,
        required_out,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_equivalence_to_explicit_composition() -> None:
    rng = random.Random(20260425_1426)

    for _ in range(240):
        key_rows = rng.randint(1, 8)
        head_groups = rng.randint(1, 4)
        query_rows = head_groups * rng.randint(1, 8)

        required_scores = query_rows * key_rows
        scores = [rng.randint(-(40 << 16), (40 << 16)) for _ in range(required_scores)]
        out = [rng.randint(-999, 999) for _ in range(max(1, required_scores))]

        required_score_a = [rng.randint(-9, 9)]
        required_out_a = [rng.randint(-9, 9)]
        required_score_b = [required_score_a[0]]
        required_out_b = [required_out_a[0]]

        err_a = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            out,
            len(out),
            required_score_a,
            required_out_a,
        )

        err_b = explicit_default_stride_commit_only_preflight_only_parity_commit_only_composition(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            out,
            len(out),
            required_score_b,
            required_out_b,
        )

        assert err_a == ATTN_Q16_OK
        assert err_a == err_b
        assert required_score_a[0] == required_score_b[0]
        assert required_out_a[0] == required_out_b[0]


if __name__ == "__main__":
    test_fixed_vector_reference_tuple_publish_no_writes()
    test_null_alias_capacity_overflow_parity_vectors()
    test_randomized_equivalence_to_explicit_composition()
    print("ok")
