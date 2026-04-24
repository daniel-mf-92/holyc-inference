#!/usr/bin/env python3
"""Reference checks for GQAAttentionApplySoftmaxQ16CheckedNoPartial semantics (IQ-1369)."""

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


def gqa_attention_apply_softmax_q16_checked_nopartial(
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
    if required_score_cells > scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_out_cells = try_mul_i64_checked(query_rows, key_rows)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells == 0:
        return ATTN_Q16_OK

    staged_probs = [0] * required_out_cells
    err = gqa_attention_apply_softmax_q16_checked(
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

    for idx in range(required_out_cells):
        out_probs_q16[idx] = staged_probs[idx]

    return ATTN_Q16_OK


def explicit_staged_composition(*args, **kwargs) -> int:
    return gqa_attention_apply_softmax_q16_checked_nopartial(*args, **kwargs)


def test_fixed_vector_reference() -> None:
    query_rows = 4
    key_rows = 3
    head_groups = 2
    row_stride = 5

    # Two tail cells per row are intentionally present and should be ignored.
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

    out_a = [999] * (query_rows * key_rows)
    out_b = out_a.copy()

    err_a = gqa_attention_apply_softmax_q16_checked_nopartial(
        scores_q32,
        len(scores_q32),
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        out_a,
        len(out_a),
    )
    err_b = explicit_staged_composition(
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


def test_alias_capacity_tailstride_contracts() -> None:
    query_rows = 2
    key_rows = 3
    head_groups = 1
    row_stride = 4
    scores = [
        7 << 16,
        5 << 16,
        3 << 16,
        111,
        6 << 16,
        2 << 16,
        1 << 16,
        222,
    ]

    aliased = scores.copy()
    err = gqa_attention_apply_softmax_q16_checked_nopartial(
        aliased,
        len(aliased),
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        aliased,
        len(aliased),
    )
    assert err == ATTN_Q16_OK
    for row in range(query_rows):
        row_sum = sum(aliased[row * key_rows : (row + 1) * key_rows])
        assert row_sum == (1 << 16)

    out_seed = [7777] * (query_rows * key_rows)
    out = out_seed.copy()
    err = gqa_attention_apply_softmax_q16_checked_nopartial(
        scores,
        len(scores) - 2,
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_seed

    out = out_seed.copy()
    err = gqa_attention_apply_softmax_q16_checked_nopartial(
        scores,
        len(scores),
        query_rows,
        key_rows,
        head_groups,
        row_stride,
        out,
        (query_rows * key_rows) - 1,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_seed


def test_error_and_no_partial() -> None:
    out_seed = [1234] * 6
    out = out_seed.copy()

    err = gqa_attention_apply_softmax_q16_checked_nopartial(
        [1 << 16, 2 << 16, 3 << 16],
        3,
        1,
        2,
        2,
        2,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_seed

    err = gqa_attention_apply_softmax_q16_checked_nopartial(
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

    huge = (1 << 63) - 1
    err = gqa_attention_apply_softmax_q16_checked_nopartial(
        [1 << 16],
        huge,
        huge,
        huge,
        1,
        huge,
        out,
        huge,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out == out_seed


def test_randomized_parity() -> None:
    rng = random.Random(20260425_1369)

    for _ in range(260):
        query_rows = rng.randint(1, 6)
        head_groups = rng.randint(1, query_rows)
        if query_rows % head_groups != 0:
            continue

        key_rows = rng.randint(1, 6)
        row_stride = key_rows + rng.randint(0, 3)

        score_cells = (query_rows - 1) * row_stride + key_rows
        scores = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(score_cells)]

        out_capacity = query_rows * key_rows
        out_a = [rng.randint(-999, 999) for _ in range(out_capacity)]
        out_b = out_a.copy()

        err_a = gqa_attention_apply_softmax_q16_checked_nopartial(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            row_stride,
            out_a,
            out_capacity,
        )
        err_b = explicit_staged_composition(
            scores,
            len(scores),
            query_rows,
            key_rows,
            head_groups,
            row_stride,
            out_b,
            out_capacity,
        )

        assert err_a == err_b == ATTN_Q16_OK
        assert out_a == out_b


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 GQAAttentionApplySoftmaxQ16CheckedNoPartial(" in source
    body = source.split("I32 GQAAttentionApplySoftmaxQ16CheckedNoPartial(", 1)[1]
    assert "snapshot_query_rows = query_rows;" in body
    assert "status = GQAAttentionApplySoftmaxQ16Checked(" in body
    assert "staged_probs_q16 = MAlloc(stage_bytes);" in body


if __name__ == "__main__":
    test_fixed_vector_reference()
    test_alias_capacity_tailstride_contracts()
    test_error_and_no_partial()
    test_randomized_parity()
    test_source_contract_markers()
    print("gqa_attention_apply_softmax_q16_checked_nopartial_reference_checks=ok")
