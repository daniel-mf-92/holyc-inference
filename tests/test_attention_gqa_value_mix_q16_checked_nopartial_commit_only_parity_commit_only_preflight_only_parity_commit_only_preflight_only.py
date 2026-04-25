#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly (IQ-1449)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_gqa_value_mix_q16_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    I64_MAX,
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_gqa_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity import (
    gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity,
)
from test_attention_gqa_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only import (
    gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only,
)


def gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
    scores_q16,
    scores_capacity: int,
    query_rows: int,
    key_rows: int,
    value_dim: int,
    head_groups: int,
    row_stride: int,
    values_q16,
    values_capacity: int,
    out_values_q16,
    out_capacity: int,
    out_required_score_cells,
    out_required_value_cells,
    out_required_out_cells,
) -> int:
    if (
        scores_q16 is None
        or values_q16 is None
        or out_values_q16 is None
        or out_required_score_cells is None
        or out_required_value_cells is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_score_cells is out_required_value_cells
        or out_required_score_cells is out_required_out_cells
        or out_required_value_cells is out_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if scores_capacity < 0 or values_capacity < 0 or out_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_rows < 0 or key_rows < 0 or value_dim < 0 or head_groups <= 0 or row_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_score_cells is scores_q16
        or out_required_score_cells is values_q16
        or out_required_score_cells is out_values_q16
        or out_required_value_cells is scores_q16
        or out_required_value_cells is values_q16
        or out_required_value_cells is out_values_q16
        or out_required_out_cells is scores_q16
        or out_required_out_cells is values_q16
        or out_required_out_cells is out_values_q16
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_query_rows = query_rows
    snapshot_key_rows = key_rows
    snapshot_value_dim = value_dim
    snapshot_head_groups = head_groups
    snapshot_row_stride = row_stride
    snapshot_scores_capacity = scores_capacity
    snapshot_values_capacity = values_capacity
    snapshot_out_capacity = out_capacity
    snapshot_scores_q16 = scores_q16
    snapshot_values_q16 = values_q16
    snapshot_out_values_q16 = out_values_q16
    snapshot_req_score = out_required_score_cells[0]
    snapshot_req_value = out_required_value_cells[0]
    snapshot_req_out = out_required_out_cells[0]

    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if key_rows > row_stride:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_score_cells = try_mul_i64_checked(query_rows - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_score_cells = try_add_i64_checked(required_score_cells, key_rows)
    if err != ATTN_Q16_OK:
        return err

    kv_rows = 0
    if not (query_rows == 0 or key_rows == 0 or value_dim == 0):
        kv_rows = query_rows // head_groups
        if kv_rows <= 0:
            return ATTN_Q16_ERR_BAD_PARAM

    err, required_value_cells = try_mul_i64_checked(kv_rows, key_rows)
    if err != ATTN_Q16_OK:
        return err
    err, required_value_cells = try_mul_i64_checked(required_value_cells, value_dim)
    if err != ATTN_Q16_OK:
        return err

    err, required_out_cells = try_mul_i64_checked(query_rows, value_dim)
    if err != ATTN_Q16_OK:
        return err

    if (
        required_score_cells > scores_capacity
        or required_value_cells > values_capacity
        or required_out_cells > out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    staged_parity_req_score = [required_score_cells]
    staged_parity_req_value = [required_value_cells]
    staged_parity_req_out = [required_out_cells]
    staged_commit_req_score = [required_score_cells]
    staged_commit_req_value = [required_value_cells]
    staged_commit_req_out = [required_out_cells]

    if required_out_cells > 0:
        out_snapshot = out_values_q16[:required_out_cells]
        parity_out = [0] * required_out_cells
        parity_out_capacity = required_out_cells
        commit_out = [0] * required_out_cells
        commit_out_capacity = required_out_cells
    else:
        out_snapshot = []
        parity_out = out_values_q16
        parity_out_capacity = out_capacity
        commit_out = out_values_q16
        commit_out_capacity = out_capacity

    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values_q16,
        values_capacity,
        parity_out,
        parity_out_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values_q16,
        values_capacity,
        commit_out,
        commit_out_capacity,
        staged_commit_req_score,
        staged_commit_req_value,
        staged_commit_req_out,
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

    kv_rows = 0
    if not (snapshot_query_rows == 0 or snapshot_key_rows == 0 or snapshot_value_dim == 0):
        kv_rows = snapshot_query_rows // snapshot_head_groups
        if kv_rows <= 0:
            return ATTN_Q16_ERR_BAD_PARAM

    err, recomputed_required_value_cells = try_mul_i64_checked(kv_rows, snapshot_key_rows)
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_required_value_cells = try_mul_i64_checked(
        recomputed_required_value_cells, snapshot_value_dim
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_out_cells = try_mul_i64_checked(snapshot_query_rows, snapshot_value_dim)
    if err != ATTN_Q16_OK:
        return err

    if (
        snapshot_query_rows != query_rows
        or snapshot_key_rows != key_rows
        or snapshot_value_dim != value_dim
        or snapshot_head_groups != head_groups
        or snapshot_row_stride != row_stride
        or snapshot_scores_capacity != scores_capacity
        or snapshot_values_capacity != values_capacity
        or snapshot_out_capacity != out_capacity
        or snapshot_scores_q16 is not scores_q16
        or snapshot_values_q16 is not values_q16
        or snapshot_out_values_q16 is not out_values_q16
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        required_score_cells != recomputed_required_score_cells
        or required_value_cells != recomputed_required_value_cells
        or required_out_cells != recomputed_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_req_score[0] != required_score_cells
        or staged_parity_req_value[0] != required_value_cells
        or staged_parity_req_out[0] != required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_commit_req_score[0] != required_score_cells
        or staged_commit_req_value[0] != required_value_cells
        or staged_commit_req_out[0] != required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_req_score[0] != staged_commit_req_score[0]
        or staged_parity_req_value[0] != staged_commit_req_value[0]
        or staged_parity_req_out[0] != staged_commit_req_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_score_cells[0] != snapshot_req_score
        or out_required_value_cells[0] != snapshot_req_value
        or out_required_out_cells[0] != snapshot_req_out
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if required_out_cells > 0 and out_values_q16[:required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    return ATTN_Q16_OK


def test_fixed_vector_zero_write_required_slots_preserved() -> None:
    query_rows = 4
    key_rows = 3
    value_dim = 2
    head_groups = 2
    row_stride = 4

    scores_q16 = [
        int(0.5 * (1 << 16)),
        int(0.25 * (1 << 16)),
        int(0.25 * (1 << 16)),
        777,
        int(0.1 * (1 << 16)),
        int(0.7 * (1 << 16)),
        int(0.2 * (1 << 16)),
        888,
        int(0.3 * (1 << 16)),
        int(0.3 * (1 << 16)),
        int(0.4 * (1 << 16)),
        999,
        int(0.8 * (1 << 16)),
        int(0.1 * (1 << 16)),
        int(0.1 * (1 << 16)),
        111,
    ]

    values_q16 = [
        2 << 16,
        1 << 16,
        -(1 << 16),
        3 << 16,
        4 << 16,
        -(2 << 16),
        3 << 16,
        -(1 << 16),
        2 << 16,
        2 << 16,
        -(2 << 16),
        1 << 16,
    ]

    out = [7171] * (query_rows * value_dim)
    req_score = [111]
    req_value = [222]
    req_out = [333]

    err = (
        gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
            scores_q16,
            len(scores_q16),
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            row_stride,
            values_q16,
            len(values_q16),
            out,
            len(out),
            req_score,
            req_value,
            req_out,
        )
    )

    assert err == ATTN_Q16_OK
    assert out == [7171] * (query_rows * value_dim)
    assert req_score == [111]
    assert req_value == [222]
    assert req_out == [333]


def test_null_alias_capacity_overflow_and_parity_vectors() -> None:
    out = [9090] * 8

    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        None,
        0,
        0,
        0,
        0,
        1,
        0,
        [1 << 16],
        1,
        out,
        len(out),
        [0],
        [0],
        [0],
    )
    assert err == ATTN_Q16_ERR_NULL_PTR

    scores = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    values = [1 << 16, 2 << 16, 3 << 16, 4 << 16]

    req = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        scores,
        len(scores),
        2,
        2,
        1,
        1,
        2,
        values,
        len(values),
        out,
        len(out),
        req,
        req,
        [0],
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    out_sentinel = [5050] * 8
    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        scores,
        1,
        2,
        2,
        1,
        1,
        2,
        values,
        len(values),
        out_sentinel,
        len(out_sentinel),
        [9],
        [8],
        [7],
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_sentinel == [5050] * 8

    huge = I64_MAX
    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        [1 << 16],
        huge,
        huge,
        huge,
        huge,
        1,
        huge,
        [1 << 16],
        huge,
        out_sentinel,
        huge,
        [9],
        [8],
        [7],
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_vectors_and_source_signature() -> None:
    rng = random.Random(20260425_1449)

    for _ in range(120):
        query_rows = rng.randint(1, 6)
        head_groups = rng.randint(1, 3)
        query_rows += (head_groups - (query_rows % head_groups)) % head_groups
        key_rows = rng.randint(1, 5)
        value_dim = rng.randint(1, 4)
        row_stride = key_rows + rng.randint(0, 3)

        kv_rows = query_rows // head_groups
        scores_cells = query_rows * row_stride
        values_cells = kv_rows * key_rows * value_dim
        out_cells = query_rows * value_dim

        scores = [rng.randint(-(1 << 16), 1 << 16) for _ in range(scores_cells)]
        values = [rng.randint(-(8 << 16), 8 << 16) for _ in range(values_cells)]

        out = [rng.randint(-3000, 3000) for _ in range(out_cells)]
        req_score = [rng.randint(-100, 100)]
        req_value = [rng.randint(-100, 100)]
        req_out = [rng.randint(-100, 100)]

        err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
            scores,
            len(scores),
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            row_stride,
            values,
            len(values),
            out,
            len(out),
            req_score,
            req_value,
            req_out,
        )

        assert err == ATTN_Q16_OK

    attention_hc = Path(__file__).resolve().parents[1] / "src" / "model" / "attention.HC"
    body = attention_hc.read_text(encoding="utf-8")
    signature = "I32 GQAAttentionValueMixQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    assert signature in body
    assert "status = GQAAttentionValueMixQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "status = GQAAttentionValueMixQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in body


if __name__ == "__main__":
    test_fixed_vector_zero_write_required_slots_preserved()
    test_null_alias_capacity_overflow_and_parity_vectors()
    test_randomized_parity_vectors_and_source_signature()
    print("ok")
