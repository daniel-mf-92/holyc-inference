#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStride (IQ-1402)."""

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
from test_attention_gqa_value_mix_q16_checked_nopartial_preflight import (
    gqa_attention_value_mix_q16_checked_nopartial_preflight,
)


def gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
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

    required_score_cells = 0
    required_value_cells = 0
    required_out_cells = 0

    if not (query_rows == 0 or key_rows == 0 or value_dim == 0):
        err, required_score_cells = try_mul_i64_checked(query_rows - 1, key_rows)
        if err != ATTN_Q16_OK:
            return err
        err, required_score_cells = try_add_i64_checked(required_score_cells, key_rows)
        if err != ATTN_Q16_OK:
            return err

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

    out_snapshot = out_values_q16[:required_out_cells]

    staged_required_score_cells = [0]
    staged_required_value_cells = [0]
    staged_required_out_cells = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        key_rows,
        values_q16,
        values_capacity,
        out_values_q16,
        out_capacity,
        staged_required_score_cells,
        staged_required_value_cells,
        staged_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_required_score_cells_second = [0]
    staged_required_value_cells_second = [0]
    staged_required_out_cells_second = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        key_rows,
        values_q16,
        values_capacity,
        out_values_q16,
        out_capacity,
        staged_required_score_cells_second,
        staged_required_value_cells_second,
        staged_required_out_cells_second,
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
        staged_required_score_cells[0] != staged_required_score_cells_second[0]
        or staged_required_value_cells[0] != staged_required_value_cells_second[0]
        or staged_required_out_cells[0] != staged_required_out_cells_second[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_required_score_cells[0] != required_score_cells
        or staged_required_value_cells[0] != required_value_cells
        or staged_required_out_cells[0] != required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_required_score_cells[0] > snapshot_scores_capacity
        or staged_required_value_cells[0] > snapshot_values_capacity
        or staged_required_out_cells[0] > snapshot_out_capacity
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

    if out_values_q16[:required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_score_cells[0] = staged_required_score_cells[0]
    out_required_value_cells[0] = staged_required_value_cells[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    return ATTN_Q16_OK


def test_fixed_vector_reference_tuple_publish_no_writes() -> None:
    query_rows = 4
    key_rows = 3
    value_dim = 2
    head_groups = 2

    scores = [123] * 16
    values = [456] * 12
    out = [777] * (query_rows * value_dim)
    out_before = out.copy()

    required_score = [-1]
    required_value = [-1]
    required_out = [-1]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
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
    assert required_score[0] == 12
    assert required_value[0] == 12
    assert required_out[0] == 8
    assert out == out_before


def test_error_contract_null_capacity_overflow() -> None:
    out = [99] * 8
    required_score = [1]
    required_value = [2]
    required_out = [3]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
        None,
        0,
        1,
        1,
        1,
        1,
        [1],
        1,
        out,
        len(out),
        required_score,
        required_value,
        required_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out == [99] * 8

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
        [1],
        -1,
        1,
        1,
        1,
        1,
        [1],
        1,
        out,
        len(out),
        required_score,
        required_value,
        required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == [99] * 8

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
        [1],
        1,
        I64_MAX,
        2,
        1,
        1,
        [1],
        1,
        out,
        len(out),
        required_score,
        required_value,
        required_out,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out == [99] * 8


def test_alias_and_capacity_rejections() -> None:
    scores = [10] * 10
    values = [20] * 10
    out = [30] * 10

    slot = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
        scores,
        len(scores),
        2,
        2,
        2,
        1,
        values,
        len(values),
        out,
        len(out),
        slot,
        slot,
        [0],
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    out_req_score = [0]
    out_req_value = [0]
    out_req_out = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
        scores,
        3,
        4,
        3,
        2,
        2,
        values,
        len(values),
        out,
        len(out),
        out_req_score,
        out_req_value,
        out_req_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_req_score[0] == 0
    assert out_req_value[0] == 0
    assert out_req_out[0] == 0


def test_parity_with_canonical_preflight_row_stride_equals_key_rows() -> None:
    rng = random.Random(20260425_1402)

    for _ in range(240):
        key_rows = rng.randint(1, 6)
        head_groups = rng.randint(1, 4)
        query_rows = head_groups * rng.randint(1, 5)
        value_dim = rng.randint(1, 8)

        kv_rows = query_rows // head_groups
        required_scores = (query_rows - 1) * key_rows + key_rows
        required_values = kv_rows * key_rows * value_dim
        required_out = query_rows * value_dim

        scores = [rng.randint(-(6 << 16), (6 << 16)) for _ in range(required_scores)]
        values = [rng.randint(-(6 << 16), (6 << 16)) for _ in range(required_values)]
        out = [rng.randint(-7777, 7777) for _ in range(required_out)]

        out_req_score = [rng.randint(-99, 99)]
        out_req_value = [rng.randint(-99, 99)]
        out_req_out = [rng.randint(-99, 99)]

        out_before = out.copy()
        err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
            scores,
            required_scores,
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            values,
            required_values,
            out,
            required_out,
            out_req_score,
            out_req_value,
            out_req_out,
        )
        assert err == ATTN_Q16_OK

        ref_score = [111]
        ref_value = [222]
        ref_out = [333]
        ref_err = gqa_attention_value_mix_q16_checked_nopartial_preflight(
            scores,
            required_scores,
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            key_rows,
            values,
            required_values,
            out,
            required_out,
            ref_score,
            ref_value,
            ref_out,
        )
        assert ref_err == ATTN_Q16_OK
        assert out_req_score[0] == ref_score[0]
        assert out_req_value[0] == ref_value[0]
        assert out_req_out[0] == ref_out[0]
        assert out == out_before


def test_zero_geometry_publishes_zero_tuple() -> None:
    scores = [1, 2, 3]
    values = [4, 5, 6]
    out = [9, 9, 9, 9]

    out_required_score = [333]
    out_required_value = [444]
    out_required_out = [555]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
        scores,
        len(scores),
        0,
        3,
        2,
        1,
        values,
        len(values),
        out,
        len(out),
        out_required_score,
        out_required_value,
        out_required_out,
    )

    assert err == ATTN_Q16_OK
    assert out_required_score[0] == 0
    assert out_required_value[0] == 0
    assert out_required_out[0] == 0
    assert out == [9, 9, 9, 9]


if __name__ == "__main__":
    test_fixed_vector_reference_tuple_publish_no_writes()
    test_error_contract_null_capacity_overflow()
    test_alias_and_capacity_rejections()
    test_parity_with_canonical_preflight_row_stride_equals_key_rows()
    test_zero_geometry_publishes_zero_tuple()
    print("ok")
