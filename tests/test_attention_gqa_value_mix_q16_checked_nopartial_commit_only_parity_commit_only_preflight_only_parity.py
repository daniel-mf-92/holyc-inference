#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyParity (IQ-1419)."""

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
from test_attention_gqa_value_mix_q16_checked_nopartial_commit_only_parity_commit_only import (
    gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only,
)
from test_attention_gqa_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only import (
    gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only,
)


def gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
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
    if required_out_cells == 0:
        return ATTN_Q16_OK

    out_snapshot = out_values_q16[:required_out_cells]
    staged_commit = [0] * required_out_cells

    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values_q16,
        values_capacity,
        out_values_q16,
        out_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

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
        staged_commit,
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
        required_score_cells > snapshot_scores_capacity
        or required_value_cells > snapshot_values_capacity
        or required_out_cells > snapshot_out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if out_values_q16[:required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    out_values_q16[:required_out_cells] = staged_commit
    return ATTN_Q16_OK


def explicit_parity_composition(*args, **kwargs) -> int:
    return gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
        *args, **kwargs
    )


def test_fixed_vector_reference_exact_publish() -> None:
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

    out_a = [7373] * (query_rows * value_dim)
    out_b = out_a.copy()

    err_a = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
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
    err_b = explicit_parity_composition(
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
    assert out_a == out_b


def test_null_alias_capacity_overflow_and_zero_write_failures() -> None:
    seed = [8484] * 8

    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
        None,
        0,
        0,
        0,
        0,
        1,
        0,
        [1 << 16],
        1,
        seed,
        len(seed),
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert seed == [8484] * 8

    scores = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    values = [1 << 16, 2 << 16, 3 << 16, 4 << 16]

    aliased = scores
    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
        scores,
        len(scores),
        2,
        2,
        1,
        1,
        2,
        values,
        len(values),
        aliased,
        len(aliased),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    out = [9191] * 8
    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
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
    assert out == [9191] * 8

    huge = I64_MAX
    out = [4545] * 8
    err = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
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
    assert out == [4545] * 8


def test_randomized_parity_and_publish_contract() -> None:
    rng = random.Random(20260425_1419)

    for _ in range(180):
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

        out_a = [rng.randint(-4000, 4000) for _ in range(out_cells)]
        out_b = out_a.copy()

        err_a = gqa_attention_value_mix_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
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
        err_b = explicit_parity_composition(
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
        assert out_a == out_b


if __name__ == "__main__":
    test_fixed_vector_reference_exact_publish()
    test_null_alias_capacity_overflow_and_zero_write_failures()
    test_randomized_parity_and_publish_contract()
    print("ok")
