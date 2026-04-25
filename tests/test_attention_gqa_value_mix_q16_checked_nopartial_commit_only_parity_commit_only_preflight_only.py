#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnly (IQ-1404)."""

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
from test_attention_gqa_value_mix_q16_checked_nopartial_commit_only_parity import (
    gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity,
)
from test_attention_gqa_value_mix_q16_checked_nopartial_commit_only_parity_commit_only import (
    gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only,
)


def gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
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
) -> int:
    if scores_q16 is None or values_q16 is None or out_values_q16 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if scores_capacity < 0 or values_capacity < 0 or out_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_rows < 0 or key_rows < 0 or value_dim < 0 or head_groups <= 0 or row_stride < 0:
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

    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if key_rows > row_stride:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_rows == 0 or key_rows == 0 or value_dim == 0:
        return ATTN_Q16_OK

    kv_rows = query_rows // head_groups
    if kv_rows <= 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_score_cells = try_mul_i64_checked(query_rows - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_score_cells = try_add_i64_checked(required_score_cells, key_rows)
    if err != ATTN_Q16_OK:
        return err

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
    if required_out_cells == 0:
        return ATTN_Q16_OK

    err, required_score_bytes = try_mul_i64_checked(required_score_cells, 8)
    if err != ATTN_Q16_OK:
        return err
    err, required_value_bytes = try_mul_i64_checked(required_value_cells, 8)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_bytes = try_mul_i64_checked(required_out_cells, 8)
    if err != ATTN_Q16_OK:
        return err
    if required_score_bytes <= 0 or required_value_bytes <= 0 or required_out_bytes <= 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if out_values_q16 is scores_q16 or out_values_q16 is values_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if scores_q16 is values_q16:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_commit_only = [0] * required_out_cells
    staged_parity = [0] * required_out_cells
    out_snapshot = out_values_q16[:required_out_cells]
    snapshot_scores_digest = tuple(scores_q16[:required_score_cells])
    snapshot_values_digest = tuple(values_q16[:required_value_cells])

    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values_q16,
        values_capacity,
        staged_commit_only,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values_q16,
        values_capacity,
        staged_parity,
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
        required_score_cells > snapshot_scores_capacity
        or required_value_cells > snapshot_values_capacity
        or required_out_cells > snapshot_out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    recomputed_scores_digest = tuple(scores_q16[:required_score_cells])
    recomputed_values_digest = tuple(values_q16[:required_value_cells])
    if (
        snapshot_scores_digest != recomputed_scores_digest
        or snapshot_values_digest != recomputed_values_digest
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_commit_only != staged_parity:
        return ATTN_Q16_ERR_BAD_PARAM

    if out_values_q16[:required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    return ATTN_Q16_OK


def explicit_preflight_only_composition(*args, **kwargs) -> int:
    return gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
        *args, **kwargs
    )


def test_fixed_vector_reference() -> None:
    query_rows = 4
    key_rows = 3
    value_dim = 2
    head_groups = 2
    row_stride = 4

    scores_q16 = [
        int(0.5 * (1 << 16)),
        int(0.25 * (1 << 16)),
        int(0.25 * (1 << 16)),
        111,
        int(0.1 * (1 << 16)),
        int(0.7 * (1 << 16)),
        int(0.2 * (1 << 16)),
        222,
        int(0.3 * (1 << 16)),
        int(0.3 * (1 << 16)),
        int(0.4 * (1 << 16)),
        333,
        int(0.8 * (1 << 16)),
        int(0.1 * (1 << 16)),
        int(0.1 * (1 << 16)),
        444,
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

    seed = [6060] * (query_rows * value_dim)
    out_a = seed.copy()
    out_b = seed.copy()

    err_a = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
        scores_q16,
        len(scores_q16),
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values_q16,
        len(values_q16),
        out_a,
        len(out_a),
    )
    err_b = explicit_preflight_only_composition(
        scores_q16,
        len(scores_q16),
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values_q16,
        len(values_q16),
        out_b,
        len(out_b),
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b == seed


def test_null_alias_capacity_overflow_and_no_write_contract() -> None:
    seed = [8383] * 8
    scores = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    values = [1 << 16, 2 << 16, 3 << 16, 4 << 16]

    out = seed.copy()
    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
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
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out == seed

    out = scores
    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
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
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    out = seed.copy()
    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
        scores,
        1,
        2,
        2,
        1,
        1,
        2,
        values,
        len(values),
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == seed

    huge = I64_MAX
    out = seed.copy()
    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
        [1 << 16],
        huge,
        huge,
        huge,
        huge,
        1,
        huge,
        [1 << 16],
        huge,
        out,
        huge,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out == seed


def test_randomized_preflight_only_parity() -> None:
    rng = random.Random(20260425_1404)

    for _ in range(220):
        query_rows = rng.randint(1, 6)
        head_groups = rng.randint(1, 3)
        query_rows = query_rows + ((head_groups - (query_rows % head_groups)) % head_groups)
        key_rows = rng.randint(1, 5)
        value_dim = rng.randint(1, 4)
        row_stride = key_rows + rng.randint(0, 3)

        kv_rows = query_rows // head_groups
        scores_cells = query_rows * row_stride
        values_cells = kv_rows * key_rows * value_dim
        out_cells = query_rows * value_dim

        scores = [rng.randint(-(1 << 16), 1 << 16) for _ in range(scores_cells)]
        values = [rng.randint(-(8 << 16), 8 << 16) for _ in range(values_cells)]
        seed = [rng.randint(-999, 999) for _ in range(out_cells)]

        out_a = seed.copy()
        out_b = seed.copy()

        err_a = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
            scores,
            len(scores),
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            row_stride,
            values,
            len(values),
            out_a,
            len(out_a),
        )
        err_b = explicit_preflight_only_composition(
            scores,
            len(scores),
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            row_stride,
            values,
            len(values),
            out_b,
            len(out_b),
        )

        assert err_a == err_b == ATTN_Q16_OK
        assert out_a == out_b == seed


if __name__ == "__main__":
    test_fixed_vector_reference()
    test_null_alias_capacity_overflow_and_no_write_contract()
    test_randomized_preflight_only_parity()
    print("ok")
