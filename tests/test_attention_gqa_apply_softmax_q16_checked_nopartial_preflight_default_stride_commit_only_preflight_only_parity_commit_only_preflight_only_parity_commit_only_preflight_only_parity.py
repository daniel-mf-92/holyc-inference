#!/usr/bin/env python3
"""Reference checks for GQAAttentionApplySoftmaxQ16CheckedNoPartialPreflightDefaultStrideCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity (IQ-1453)."""

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
from test_attention_gqa_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only import (
    gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only,
)
from test_attention_gqa_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only import (
    gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only,
)


def gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if (
        out_required_score_cells is scores_q32
        or out_required_score_cells is out_probs_q16
        or out_required_out_cells is scores_q32
        or out_required_out_cells is out_probs_q16
    ):
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

    required_score_cells = 0
    required_out_cells = 0
    if not (query_rows == 0 or key_rows == 0):
        err, required_score_cells = try_mul_i64_checked(query_rows - 1, key_rows)
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

    if required_out_cells > 0:
        staged_preflight_out = [0] * required_out_cells
        staged_commit_out = [0] * required_out_cells
        snapshot_out_vals = out_probs_q16[:required_out_cells]
        preflight_out = staged_preflight_out
        preflight_out_capacity = required_out_cells
        commit_out = staged_commit_out
        commit_out_capacity = required_out_cells
    else:
        staged_preflight_out = []
        staged_commit_out = []
        snapshot_out_vals = []
        preflight_out = out_probs_q16
        preflight_out_capacity = out_capacity
        commit_out = out_probs_q16
        commit_out_capacity = out_capacity

    staged_preflight_score = [required_score_cells]
    staged_preflight_required_out = [required_out_cells]
    staged_commit_score = [required_score_cells]
    staged_commit_required_out = [required_out_cells]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        preflight_out,
        preflight_out_capacity,
        staged_preflight_score,
        staged_preflight_required_out,
    )
    if err != ATTN_Q16_OK:
        return err

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        commit_out,
        commit_out_capacity,
        staged_commit_score,
        staged_commit_required_out,
    )
    if err != ATTN_Q16_OK:
        return err

    recomputed_required_score_cells = 0
    recomputed_required_out_cells = 0
    if not (snapshot_query_rows == 0 or snapshot_key_rows == 0):
        err, recomputed_required_score_cells = try_mul_i64_checked(snapshot_query_rows - 1, snapshot_key_rows)
        if err != ATTN_Q16_OK:
            return err
        err, recomputed_required_score_cells = try_add_i64_checked(recomputed_required_score_cells, snapshot_key_rows)
        if err != ATTN_Q16_OK:
            return err
        err, recomputed_required_out_cells = try_mul_i64_checked(snapshot_query_rows, snapshot_key_rows)
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
        required_score_cells != recomputed_required_score_cells
        or required_out_cells != recomputed_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_preflight_score[0] != required_score_cells
        or staged_preflight_required_out[0] != required_out_cells
        or staged_commit_score[0] != required_score_cells
        or staged_commit_required_out[0] != required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_preflight_score[0] != staged_commit_score[0]
        or staged_preflight_required_out[0] != staged_commit_required_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_preflight_out != staged_commit_out:
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

    if required_out_cells > 0 and out_probs_q16[:required_out_cells] != snapshot_out_vals:
        return ATTN_Q16_ERR_BAD_PARAM

    return ATTN_Q16_OK


def explicit_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_composition(*args) -> int:
    scores_q32 = args[0]
    scores_capacity = args[1]
    query_rows = args[2]
    key_rows = args[3]
    head_groups = args[4]
    out_probs_q16 = args[5]
    out_capacity = args[6]
    out_required_score_cells = args[7]
    out_required_out_cells = args[8]

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

    if recomputed_required_score_cells > scores_capacity or recomputed_required_out_cells > out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if recomputed_required_out_cells > 0:
        preflight_out = [0] * recomputed_required_out_cells
        commit_out = [0] * recomputed_required_out_cells
    else:
        preflight_out = out_probs_q16
        commit_out = out_probs_q16

    staged_preflight_score = [recomputed_required_score_cells]
    staged_preflight_out = [recomputed_required_out_cells]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        preflight_out,
        len(preflight_out),
        staged_preflight_score,
        staged_preflight_out,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_commit_score = [recomputed_required_score_cells]
    staged_commit_out_required = [recomputed_required_out_cells]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        commit_out,
        len(commit_out),
        staged_commit_score,
        staged_commit_out_required,
    )
    if err != ATTN_Q16_OK:
        return err

    if (
        staged_preflight_score[0] != staged_commit_score[0]
        or staged_preflight_out[0] != staged_commit_out_required[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if preflight_out != commit_out:
        return ATTN_Q16_ERR_BAD_PARAM

    return ATTN_Q16_OK


def test_fixed_vector_zero_write_required_slots_preserved() -> None:
    query_rows = 4
    key_rows = 3
    head_groups = 2

    scores = [42] * (query_rows * key_rows)
    out = [777] * (query_rows * key_rows)
    out_before = out.copy()

    required_score = [7001]
    required_out = [7002]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
    assert out == out_before
    assert required_score[0] == 7001
    assert required_out[0] == 7002


def test_null_alias_capacity_overflow_parity_vectors() -> None:
    out = [99] * 12
    required_score = [1]
    required_out = [2]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    scores = [11, 22, 33]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        scores,
        len(scores),
        1,
        3,
        1,
        out,
        len(out),
        scores,
        required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        scores,
        len(scores),
        1,
        3,
        1,
        out,
        len(out),
        required_score,
        out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        [1],
        1,
        -1,
        1,
        1,
        out,
        len(out),
        required_score,
        required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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


def test_random_vectors_match_explicit_composition_and_zero_write() -> None:
    rng = random.Random(20260425_1453)

    for _ in range(120):
        key_rows = rng.randint(1, 8)
        head_groups = rng.randint(1, 4)
        query_rows = head_groups * rng.randint(1, 8)

        required_cells = query_rows * key_rows
        scores = [rng.randint(-(8 << 20), (8 << 20)) for _ in range(required_cells)]
        out = [rng.randint(-2000, 2000) for _ in range(required_cells)]
        out_ref = out.copy()

        score_slot = [rng.randint(-500, 500)]
        out_slot = [rng.randint(-500, 500)]
        score_slot_ref = [score_slot[0]]
        out_slot_ref = [out_slot[0]]

        err_impl = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            out,
            len(out),
            score_slot,
            out_slot,
        )
        err_ref = explicit_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_composition(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            out_ref,
            len(out_ref),
            score_slot_ref,
            out_slot_ref,
        )

        assert err_impl == ATTN_Q16_OK
        assert err_impl == err_ref
        assert out == out_ref
        assert score_slot[0] == score_slot_ref[0]
        assert out_slot[0] == out_slot_ref[0]


def test_zero_geometry_preserves_slots_and_output() -> None:
    scores = [1, 2, 3]
    out = [10, 11, 12]
    out_before = out.copy()
    required_score = [77]
    required_out = [88]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        scores,
        len(scores),
        0,
        7,
        1,
        out,
        len(out),
        required_score,
        required_out,
    )

    assert err == ATTN_Q16_OK
    assert required_score[0] == 77
    assert required_out[0] == 88
    assert out == out_before


if __name__ == "__main__":
    test_fixed_vector_zero_write_required_slots_preserved()
    test_null_alias_capacity_overflow_parity_vectors()
    test_random_vectors_match_explicit_composition_and_zero_write()
    test_zero_geometry_preserves_slots_and_output()
    print("ok")
