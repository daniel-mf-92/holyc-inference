#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnlyPreflightOnlyParityCommitOnly (IQ-1443)."""

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
from test_attention_gqa_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only import (
    gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only,
)
from test_attention_gqa_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity import (
    gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity,
)


def gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_commit_only(
    scores_q16,
    scores_capacity: int,
    query_rows: int,
    key_rows: int,
    value_dim: int,
    head_groups: int,
    values_q16,
    values_capacity: int,
    out_values_q16,
    out_capacity: int,
    out_required_score_cells,
    out_required_value_cells,
    out_required_out_cells,
) -> int:
    if out_required_score_cells is None or out_required_value_cells is None or out_required_out_cells is None:
        return ATTN_Q16_ERR_NULL_PTR
    if (
        out_required_score_cells is out_required_value_cells
        or out_required_score_cells is out_required_out_cells
        or out_required_value_cells is out_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if scores_q16 is None or values_q16 is None or out_values_q16 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if scores_capacity < 0 or values_capacity < 0 or out_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_rows < 0 or key_rows < 0 or value_dim < 0 or head_groups <= 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_query_rows = query_rows
    snapshot_key_rows = key_rows
    snapshot_value_dim = value_dim
    snapshot_head_groups = head_groups
    snapshot_scores_capacity = scores_capacity
    snapshot_values_capacity = values_capacity
    snapshot_out_capacity = out_capacity
    snapshot_scores = scores_q16
    snapshot_values = values_q16
    snapshot_out = out_values_q16

    snapshot_required_score_ptr = out_required_score_cells
    snapshot_required_value_ptr = out_required_value_cells
    snapshot_required_out_ptr = out_required_out_cells
    snapshot_required_score_slot = out_required_score_cells[0]
    snapshot_required_value_slot = out_required_value_cells[0]
    snapshot_required_out_slot = out_required_out_cells[0]

    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM

    recomputed_required_score_cells = 0
    recomputed_required_value_cells = 0
    recomputed_required_out_cells = 0
    if not (query_rows == 0 or key_rows == 0 or value_dim == 0):
        err, recomputed_required_score_cells = try_mul_i64_checked(query_rows - 1, key_rows)
        if err != ATTN_Q16_OK:
            return err
        err, recomputed_required_score_cells = try_add_i64_checked(
            recomputed_required_score_cells,
            key_rows,
        )
        if err != ATTN_Q16_OK:
            return err

        kv_rows = query_rows // head_groups
        if kv_rows <= 0:
            return ATTN_Q16_ERR_BAD_PARAM

        err, recomputed_required_value_cells = try_mul_i64_checked(kv_rows, key_rows)
        if err != ATTN_Q16_OK:
            return err
        err, recomputed_required_value_cells = try_mul_i64_checked(
            recomputed_required_value_cells,
            value_dim,
        )
        if err != ATTN_Q16_OK:
            return err

        err, recomputed_required_out_cells = try_mul_i64_checked(query_rows, value_dim)
        if err != ATTN_Q16_OK:
            return err

        if (
            recomputed_required_score_cells > scores_capacity
            or recomputed_required_value_cells > values_capacity
            or recomputed_required_out_cells > out_capacity
        ):
            return ATTN_Q16_ERR_BAD_PARAM

    staged_parity_score = [recomputed_required_score_cells]
    staged_parity_value = [recomputed_required_value_cells]
    staged_parity_out = [recomputed_required_out_cells]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        values_q16,
        values_capacity,
        out_values_q16,
        out_capacity,
        staged_parity_score,
        staged_parity_value,
        staged_parity_out,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_preflight_score = [recomputed_required_score_cells]
    staged_preflight_value = [recomputed_required_value_cells]
    staged_preflight_out = [recomputed_required_out_cells]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        values_q16,
        values_capacity,
        out_values_q16,
        out_capacity,
        staged_preflight_score,
        staged_preflight_value,
        staged_preflight_out,
    )
    if err != ATTN_Q16_OK:
        return err

    if (
        snapshot_query_rows != query_rows
        or snapshot_key_rows != key_rows
        or snapshot_value_dim != value_dim
        or snapshot_head_groups != head_groups
        or snapshot_scores_capacity != scores_capacity
        or snapshot_values_capacity != values_capacity
        or snapshot_out_capacity != out_capacity
        or snapshot_scores is not scores_q16
        or snapshot_values is not values_q16
        or snapshot_out is not out_values_q16
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_score[0] != staged_preflight_score[0]
        or staged_parity_value[0] != staged_preflight_value[0]
        or staged_parity_out[0] != staged_preflight_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_score[0] != recomputed_required_score_cells
        or staged_parity_value[0] != recomputed_required_value_cells
        or staged_parity_out[0] != recomputed_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_score[0] > snapshot_scores_capacity
        or staged_parity_value[0] > snapshot_values_capacity
        or staged_parity_out[0] > snapshot_out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_score_cells is not snapshot_required_score_ptr
        or out_required_value_cells is not snapshot_required_value_ptr
        or out_required_out_cells is not snapshot_required_out_ptr
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_score_cells[0] != snapshot_required_score_slot
        or out_required_value_cells[0] != snapshot_required_value_slot
        or out_required_out_cells[0] != snapshot_required_out_slot
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_score_cells[0] = staged_parity_score[0]
    out_required_value_cells[0] = staged_parity_value[0]
    out_required_out_cells[0] = staged_parity_out[0]
    return ATTN_Q16_OK


def explicit_default_stride_parity_commit_only_preflight_only_parity_commit_only_composition(*args) -> int:
    scores_q16 = args[0]
    scores_capacity = args[1]
    query_rows = args[2]
    key_rows = args[3]
    value_dim = args[4]
    head_groups = args[5]
    values_q16 = args[6]
    values_capacity = args[7]
    out_values_q16 = args[8]
    out_capacity = args[9]
    out_required_score_cells = args[10]
    out_required_value_cells = args[11]
    out_required_out_cells = args[12]

    recomputed_required_score_cells = 0
    recomputed_required_value_cells = 0
    recomputed_required_out_cells = 0
    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if not (query_rows == 0 or key_rows == 0 or value_dim == 0):
        err, recomputed_required_score_cells = try_mul_i64_checked(query_rows - 1, key_rows)
        if err != ATTN_Q16_OK:
            return err
        err, recomputed_required_score_cells = try_add_i64_checked(
            recomputed_required_score_cells,
            key_rows,
        )
        if err != ATTN_Q16_OK:
            return err

        kv_rows = query_rows // head_groups
        if kv_rows <= 0:
            return ATTN_Q16_ERR_BAD_PARAM

        err, recomputed_required_value_cells = try_mul_i64_checked(kv_rows, key_rows)
        if err != ATTN_Q16_OK:
            return err
        err, recomputed_required_value_cells = try_mul_i64_checked(
            recomputed_required_value_cells,
            value_dim,
        )
        if err != ATTN_Q16_OK:
            return err

        err, recomputed_required_out_cells = try_mul_i64_checked(query_rows, value_dim)
        if err != ATTN_Q16_OK:
            return err

    staged_parity_score = [recomputed_required_score_cells]
    staged_parity_value = [recomputed_required_value_cells]
    staged_parity_out = [recomputed_required_out_cells]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        values_q16,
        values_capacity,
        out_values_q16,
        out_capacity,
        staged_parity_score,
        staged_parity_value,
        staged_parity_out,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_preflight_score = [recomputed_required_score_cells]
    staged_preflight_value = [recomputed_required_value_cells]
    staged_preflight_out = [recomputed_required_out_cells]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        values_q16,
        values_capacity,
        out_values_q16,
        out_capacity,
        staged_preflight_score,
        staged_preflight_value,
        staged_preflight_out,
    )
    if err != ATTN_Q16_OK:
        return err

    if (
        staged_parity_score[0] != staged_preflight_score[0]
        or staged_parity_value[0] != staged_preflight_value[0]
        or staged_parity_out[0] != staged_preflight_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_score_cells[0] = staged_parity_score[0]
    out_required_value_cells[0] = staged_parity_value[0]
    out_required_out_cells[0] = staged_parity_out[0]
    return ATTN_Q16_OK


def test_fixed_vector_reference_tuple_publish() -> None:
    query_rows = 6
    key_rows = 4
    value_dim = 3
    head_groups = 3

    scores = [101] * (query_rows * key_rows)
    values = [202] * ((query_rows // head_groups) * key_rows * value_dim)
    out = [303] * (query_rows * value_dim)
    out_before = out.copy()

    required_score = [8101]
    required_value = [8102]
    required_out = [8103]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_commit_only(
        scores,
        len(scores),
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        values,
        len(values),
        out,
        len(out),
        required_score,
        required_value,
        required_out,
    )

    assert err == ATTN_Q16_OK
    assert required_score[0] == query_rows * key_rows
    assert required_value[0] == (query_rows // head_groups) * key_rows * value_dim
    assert required_out[0] == query_rows * value_dim
    assert out == out_before


def test_null_alias_capacity_overflow_parity_vectors() -> None:
    scores = [1] * 24
    values = [2] * 24
    out = [3] * 24

    req_score = [11]
    req_value = [12]
    req_out = [13]

    assert (
        gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_commit_only(
            scores,
            len(scores),
            4,
            3,
            2,
            2,
            values,
            len(values),
            out,
            len(out),
            None,
            req_value,
            req_out,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_commit_only(
            scores,
            len(scores),
            4,
            3,
            2,
            2,
            values,
            len(values),
            out,
            len(out),
            req_score,
            req_score,
            req_out,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    assert (
        gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_commit_only(
            None,
            len(scores),
            4,
            3,
            2,
            2,
            values,
            len(values),
            out,
            len(out),
            req_score,
            req_value,
            req_out,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_commit_only(
            scores,
            -1,
            4,
            3,
            2,
            2,
            values,
            len(values),
            out,
            len(out),
            req_score,
            req_value,
            req_out,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    assert (
        gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_commit_only(
            scores,
            len(scores),
            5,
            3,
            2,
            2,
            values,
            len(values),
            out,
            len(out),
            req_score,
            req_value,
            req_out,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    assert (
        gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_commit_only(
            [7] * 64,
            5,
            4,
            3,
            2,
            2,
            [8] * 64,
            5,
            [9] * 64,
            5,
            [0],
            [0],
            [0],
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    overflow_scores = [1] * 8
    overflow_values = [2] * 8
    overflow_out = [3] * 8
    assert (
        gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_commit_only(
            overflow_scores,
            len(overflow_scores),
            I64_MAX,
            2,
            2,
            1,
            overflow_values,
            len(overflow_values),
            overflow_out,
            len(overflow_out),
            [0],
            [0],
            [0],
        )
        == ATTN_Q16_ERR_OVERFLOW
    )


def test_randomized_composition_parity() -> None:
    random.seed(1443)
    for _ in range(220):
        head_groups = random.randint(1, 4)
        kv_rows = random.randint(0, 5)
        query_rows = kv_rows * head_groups
        key_rows = random.randint(0, 5)
        value_dim = random.randint(0, 6)

        req_score = query_rows * key_rows
        req_value = kv_rows * key_rows * value_dim
        req_out = query_rows * value_dim

        scores = [random.randint(-256, 256) for _ in range(max(1, req_score + 4))]
        values = [random.randint(-256, 256) for _ in range(max(1, req_value + 4))]
        out = [random.randint(-256, 256) for _ in range(max(1, req_out + 4))]
        out_before = out.copy()

        required_score_a = [random.randint(-100, 100)]
        required_value_a = [random.randint(-100, 100)]
        required_out_a = [random.randint(-100, 100)]
        required_score_b = [required_score_a[0]]
        required_value_b = [required_value_a[0]]
        required_out_b = [required_out_a[0]]

        err_a = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_commit_only(
            scores,
            req_score,
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            values,
            req_value,
            out,
            req_out,
            required_score_a,
            required_value_a,
            required_out_a,
        )
        err_b = explicit_default_stride_parity_commit_only_preflight_only_parity_commit_only_composition(
            scores,
            req_score,
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            values,
            req_value,
            out,
            req_out,
            required_score_b,
            required_value_b,
            required_out_b,
        )

        assert err_a == err_b
        assert required_score_a[0] == required_score_b[0]
        assert required_value_a[0] == required_value_b[0]
        assert required_out_a[0] == required_out_b[0]
        assert out == out_before


if __name__ == "__main__":
    test_fixed_vector_reference_tuple_publish()
    test_null_alias_capacity_overflow_parity_vectors()
    test_randomized_composition_parity()
    print("ok")
