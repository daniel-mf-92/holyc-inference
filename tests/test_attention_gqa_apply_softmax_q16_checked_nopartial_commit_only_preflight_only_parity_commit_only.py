#!/usr/bin/env python3
"""Reference checks for GQAAttentionApplySoftmaxQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly (IQ-1381)."""

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
from test_attention_gqa_apply_softmax_q16_checked_nopartial_commit_only_preflight_only import (
    gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only,
)
from test_attention_gqa_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity import (
    gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity,
)


def gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
    snapshot_scores = scores_q32
    snapshot_out = out_probs_q16

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

    err, required_score_bytes = try_mul_i64_checked(required_score_cells, 8)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_bytes = try_mul_i64_checked(required_out_cells, 8)
    if err != ATTN_Q16_OK:
        return err
    if required_score_bytes <= 0 or required_out_bytes <= 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if out_probs_q16 is scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_parity = [0] * required_out_cells
    out_snapshot = out_probs_q16[:required_out_cells]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        staged_parity,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        out_probs_q16,
        out_capacity,
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
    err, recomputed_required_out_cells = try_mul_i64_checked(snapshot_query_rows, snapshot_key_rows)
    if err != ATTN_Q16_OK:
        return err

    if (
        snapshot_query_rows != query_rows
        or snapshot_key_rows != key_rows
        or snapshot_head_groups != head_groups
        or snapshot_row_stride != row_stride
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

    if required_score_cells > snapshot_scores_capacity or required_out_cells > snapshot_out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if out_probs_q16[:required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    for idx in range(required_out_cells):
        out_probs_q16[idx] = staged_parity[idx]

    return ATTN_Q16_OK


def test_fixed_vector_reference_atomic_publish() -> None:
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

    out = [5150] * (query_rows * key_rows)
    baseline = [5150] * (query_rows * key_rows)
    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        scores_q32,
        len(scores_q32),
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        out,
        len(out),
    )
    assert err == ATTN_Q16_OK
    assert out != baseline


def test_error_contract_null_alias_capacity_overflow() -> None:
    seed = [4040] * 8

    out = seed.copy()
    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
    assert scores == [7 << 16, 6 << 16, 5 << 16, 4 << 16]

    out = seed.copy()
    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
    err = gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        [1 << 16],
        1,
        I64_MAX,
        2,
        1,
        2,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out == seed


def test_randomized_parity_commit_publish_path() -> None:
    rng = random.Random(20260425_1381)

    for _ in range(220):
        key_rows = rng.randint(1, 6)
        head_groups = rng.randint(1, 4)
        query_rows = head_groups * rng.randint(1, 5)
        row_stride = key_rows + rng.randint(0, 3)

        required_scores = (query_rows - 1) * row_stride + key_rows
        required_out = query_rows * key_rows

        scores = [rng.randint(-(40 << 16), (40 << 16)) for _ in range(required_scores)]
        out = [rng.randint(-9999, 9999) for _ in range(max(1, required_out))]

        err = (
            gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only(
                scores,
                len(scores),
                query_rows,
                key_rows,
                head_groups,
                row_stride,
                out,
                len(out),
            )
        )
        assert err == ATTN_Q16_OK


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert (
        "I32 GQAAttentionApplySoftmaxQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
        in source
    )
    body = source.split(
        "I32 GQAAttentionApplySoftmaxQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(",
        1,
    )[1]
    assert "status = GQAAttentionApplySoftmaxQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "status = GQAAttentionApplySoftmaxQ16CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "if (out_begin < scores_end && scores_begin < out_end)" in body
    assert "if (out_probs_q16[copy_index] != snapshot_out_probs_q16_values[copy_index])" in body


if __name__ == "__main__":
    test_fixed_vector_reference_atomic_publish()
    test_error_contract_null_alias_capacity_overflow()
    test_randomized_parity_commit_publish_path()
    test_source_contract_markers()
    print(
        "gqa_attention_apply_softmax_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference_checks=ok"
    )
