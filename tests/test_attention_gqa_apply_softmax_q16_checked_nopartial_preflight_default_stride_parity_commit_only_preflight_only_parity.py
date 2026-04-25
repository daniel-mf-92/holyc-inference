#!/usr/bin/env python3
"""Reference checks for ...PreflightDefaultStrideParityCommitOnlyPreflightOnlyParity (IQ-1427)."""

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
from test_attention_gqa_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only import (
    gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only,
)
from test_attention_gqa_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only import (
    gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only,
)


def gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
            recomputed_required_score_cells, key_rows
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

    out_snapshot = out_probs_q16[:recomputed_required_out_cells]

    staged_preflight_score = [recomputed_required_score_cells]
    staged_preflight_out = [recomputed_required_out_cells]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only(
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

    staged_commit_score = [0]
    staged_commit_out = [0]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        out_probs_q16,
        out_capacity,
        staged_commit_score,
        staged_commit_out,
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
        staged_preflight_score[0] != staged_commit_score[0]
        or staged_preflight_out[0] != staged_commit_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_preflight_score[0] != recomputed_required_score_cells
        or staged_preflight_out[0] != recomputed_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_preflight_score[0] > snapshot_scores_capacity
        or staged_preflight_out[0] > snapshot_out_capacity
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

    if out_probs_q16[:recomputed_required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    return ATTN_Q16_OK


def test_fixed_vector_reference_zero_write_diagnostics() -> None:
    query_rows = 6
    key_rows = 3
    head_groups = 3

    scores = [17] * (query_rows * key_rows)
    out = [9191] * (query_rows * key_rows)
    out_before = out.copy()

    required_score = [4201]
    required_out = [4202]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
    assert required_score[0] == 4201
    assert required_out[0] == 4202
    assert out == out_before


def test_null_alias_capacity_overflow_parity_vectors() -> None:
    out = [55] * 8
    required_score = [11]
    required_out = [22]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
    assert required_score[0] == 11
    assert required_out[0] == 22

    alias_slot = [77]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
    assert alias_slot[0] == 77

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
        [1],
        -1,
        1,
        1,
        1,
        out,
        len(out),
        required_score,
        required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert required_score[0] == 11
    assert required_out[0] == 22

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
        [1],
        1,
        5,
        2,
        4,
        out,
        len(out),
        required_score,
        required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert required_score[0] == 11
    assert required_out[0] == 22

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
    assert required_score[0] == 11
    assert required_out[0] == 22


def test_randomized_zero_write_success() -> None:
    rng = random.Random(20260425_1427)

    for _ in range(240):
        key_rows = rng.randint(1, 6)
        head_groups = rng.randint(1, 4)
        query_rows = head_groups * rng.randint(1, 5)

        required_scores = query_rows * key_rows
        scores = [rng.randint(-(40 << 16), (40 << 16)) for _ in range(required_scores)]
        out = [rng.randint(-9999, 9999) for _ in range(max(1, required_scores))]
        out_snapshot = out.copy()

        required_score = [rng.randint(-100, 100)]
        required_out = [rng.randint(-100, 100)]
        required_score_slot = required_score[0]
        required_out_slot = required_out[0]

        err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
        assert out == out_snapshot
        assert required_score[0] == required_score_slot
        assert required_out[0] == required_out_slot


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert (
        "I32 GQAAttentionApplySoftmaxQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnlyPreflightOnlyParity("
        in source
    )
    body = source.split(
        "I32 GQAAttentionApplySoftmaxQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnlyPreflightOnlyParity(",
        1,
    )[1]
    assert (
        "status = GQAAttentionApplySoftmaxQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnlyPreflightOnly("
        in body
    )
    assert (
        "status = GQAAttentionApplySoftmaxQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnly("
        in body
    )
    assert "snapshot_out_required_score_cells_value = *out_required_score_cells;" in body
    assert "if (out_probs_q16[copy_index] != snapshot_out_probs_q16_values[copy_index])" in body


if __name__ == "__main__":
    test_fixed_vector_reference_zero_write_diagnostics()
    test_null_alias_capacity_overflow_parity_vectors()
    test_randomized_zero_write_success()
    test_source_contract_markers()
    print(
        "gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_reference_checks=ok"
    )
