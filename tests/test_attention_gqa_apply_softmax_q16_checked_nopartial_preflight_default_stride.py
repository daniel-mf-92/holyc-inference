#!/usr/bin/env python3
"""Reference checks for GQAAttentionApplySoftmaxQ16CheckedNoPartialPreflightDefaultStride (IQ-1403)."""

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


def gqa_attention_apply_softmax_q16_checked_nopartial_preflight(
    scores_q32,
    scores_capacity: int,
    query_rows: int,
    key_rows: int,
    head_groups: int,
    row_stride: int,
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
    if query_rows < 0 or key_rows < 0 or head_groups <= 0 or row_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_query_rows = query_rows
    snapshot_key_rows = key_rows
    snapshot_head_groups = head_groups
    snapshot_row_stride = row_stride
    snapshot_scores_capacity = scores_capacity
    snapshot_out_capacity = out_capacity

    snapshot_required_score_ptr = out_required_score_cells
    snapshot_required_out_ptr = out_required_out_cells
    snapshot_required_score_slot = out_required_score_cells[0]
    snapshot_required_out_slot = out_required_out_cells[0]

    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if key_rows > row_stride:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_rows == 0 or key_rows == 0:
        if (
            snapshot_query_rows != query_rows
            or snapshot_key_rows != key_rows
            or snapshot_head_groups != head_groups
            or snapshot_row_stride != row_stride
            or snapshot_scores_capacity != scores_capacity
            or snapshot_out_capacity != out_capacity
        ):
            return ATTN_Q16_ERR_BAD_PARAM
        if out_required_score_cells is not snapshot_required_score_ptr or out_required_out_cells is not snapshot_required_out_ptr:
            return ATTN_Q16_ERR_BAD_PARAM
        if (
            out_required_score_cells[0] != snapshot_required_score_slot
            or out_required_out_cells[0] != snapshot_required_out_slot
        ):
            return ATTN_Q16_ERR_BAD_PARAM
        out_required_score_cells[0] = 0
        out_required_out_cells[0] = 0
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

    if (
        out_required_score_cells is scores_q32
        or out_required_score_cells is out_probs_q16
        or out_required_out_cells is scores_q32
        or out_required_out_cells is out_probs_q16
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_snapshot = out_probs_q16[:required_out_cells]

    err, recomputed_required_score_cells = try_mul_i64_checked(snapshot_query_rows - 1, snapshot_row_stride)
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
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        required_score_cells != recomputed_required_score_cells
        or required_out_cells != recomputed_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if required_score_cells > snapshot_scores_capacity or required_out_cells > snapshot_out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if out_required_score_cells is not snapshot_required_score_ptr or out_required_out_cells is not snapshot_required_out_ptr:
        return ATTN_Q16_ERR_BAD_PARAM
    if (
        out_required_score_cells[0] != snapshot_required_score_slot
        or out_required_out_cells[0] != snapshot_required_out_slot
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if out_probs_q16[:required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_score_cells[0] = required_score_cells
    out_required_out_cells[0] = required_out_cells
    return ATTN_Q16_OK


def gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride(
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

    if (
        out_required_score_cells is scores_q32
        or out_required_score_cells is out_probs_q16
        or out_required_out_cells is scores_q32
        or out_required_out_cells is out_probs_q16
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_snapshot = out_probs_q16[:required_out_cells]

    staged_required_score_cells = [0]
    staged_required_out_cells = [0]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        key_rows,
        out_probs_q16,
        out_capacity,
        staged_required_score_cells,
        staged_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_required_score_cells_second = [0]
    staged_required_out_cells_second = [0]
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight(
        scores_q32,
        scores_capacity,
        query_rows,
        key_rows,
        head_groups,
        key_rows,
        out_probs_q16,
        out_capacity,
        staged_required_score_cells_second,
        staged_required_out_cells_second,
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
        staged_required_score_cells[0] != staged_required_score_cells_second[0]
        or staged_required_out_cells[0] != staged_required_out_cells_second[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_required_score_cells[0] != required_score_cells
        or staged_required_out_cells[0] != required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_required_score_cells[0] > snapshot_scores_capacity
        or staged_required_out_cells[0] > snapshot_out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if out_required_score_cells is not snapshot_required_score_ptr or out_required_out_cells is not snapshot_required_out_ptr:
        return ATTN_Q16_ERR_BAD_PARAM
    if (
        out_required_score_cells[0] != snapshot_required_score_slot
        or out_required_out_cells[0] != snapshot_required_out_slot
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if out_probs_q16[:required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_score_cells[0] = staged_required_score_cells[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    return ATTN_Q16_OK


def test_fixed_vector_reference_tuple_publish_no_writes() -> None:
    query_rows = 4
    key_rows = 3
    head_groups = 2

    scores = [42] * 16
    out = [777] * (query_rows * key_rows)
    out_before = out.copy()

    required_score = [-1]
    required_out = [-1]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride(
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


def test_error_contract_null_alias_capacity_overflow() -> None:
    out = [99] * 8
    required_score = [1]
    required_out = [2]

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride(
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

    scores = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    required_alias_score = scores
    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride(
        scores,
        len(scores),
        1,
        2,
        1,
        out,
        len(scores),
        required_alias_score,
        required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride(
        scores,
        1,
        1,
        2,
        1,
        out,
        len(out),
        required_score,
        required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride(
        [1 << 16],
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


def test_randomized_parity_zero_write() -> None:
    rng = random.Random(20260425_1403)

    for _ in range(240):
        key_rows = rng.randint(1, 6)
        head_groups = rng.randint(1, 4)
        query_rows = head_groups * rng.randint(1, 5)

        required_scores = (query_rows - 1) * key_rows + key_rows
        required_out = query_rows * key_rows

        scores = [rng.randint(-(30 << 16), (30 << 16)) for _ in range(required_scores)]
        out = [rng.randint(-9999, 9999) for _ in range(max(1, required_out))]
        baseline = out.copy()

        out_required_score = [rng.randint(-50, 50)]
        out_required_out = [rng.randint(-50, 50)]

        err = gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            out,
            len(out),
            out_required_score,
            out_required_out,
        )
        assert err == ATTN_Q16_OK
        assert out == baseline
        assert out_required_score[0] == required_scores
        assert out_required_out[0] == required_out


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    marker_preflight = "I32 GQAAttentionApplySoftmaxQ16CheckedNoPartialPreflight("
    marker_default = "I32 GQAAttentionApplySoftmaxQ16CheckedNoPartialPreflightDefaultStride("
    assert marker_preflight in source
    assert marker_default in source

    default_body = source.split(marker_default, 1)[1]
    assert "status = GQAAttentionApplySoftmaxQ16CheckedNoPartialPreflight(" in default_body
    assert "key_rows,\n        out_probs_q16," in default_body
    assert "staged_required_score_cells_second" in default_body


if __name__ == "__main__":
    test_fixed_vector_reference_tuple_publish_no_writes()
    test_error_contract_null_alias_capacity_overflow()
    test_randomized_parity_zero_write()
    test_source_contract_markers()
    print("gqa_attention_apply_softmax_q16_checked_nopartial_preflight_default_stride_reference_checks=ok")
