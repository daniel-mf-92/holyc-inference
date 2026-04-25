#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16CheckedNoPartialPreflight (IQ-1387)."""

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


def gqa_attention_value_mix_q16_checked_nopartial_preflight(
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

    snapshot_required_score_ptr = out_required_score_cells
    snapshot_required_value_ptr = out_required_value_cells
    snapshot_required_out_ptr = out_required_out_cells
    snapshot_required_score_slot = out_required_score_cells[0]
    snapshot_required_value_slot = out_required_value_cells[0]
    snapshot_required_out_slot = out_required_out_cells[0]

    if query_rows > 0 and (query_rows % head_groups) != 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if key_rows > row_stride:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_rows == 0 or key_rows == 0 or value_dim == 0:
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

        out_required_score_cells[0] = 0
        out_required_value_cells[0] = 0
        out_required_out_cells[0] = 0
        return ATTN_Q16_OK

    err, required_score_cells = try_mul_i64_checked(query_rows - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_score_cells = try_add_i64_checked(required_score_cells, key_rows)
    if err != ATTN_Q16_OK:
        return err

    kv_rows = query_rows // head_groups
    if query_rows > 0 and kv_rows <= 0:
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

    err, recomputed_required_score_cells = try_mul_i64_checked(snapshot_query_rows - 1, snapshot_row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_required_score_cells = try_add_i64_checked(
        recomputed_required_score_cells, snapshot_key_rows
    )
    if err != ATTN_Q16_OK:
        return err

    kv_rows = snapshot_query_rows // snapshot_head_groups
    if snapshot_query_rows > 0 and kv_rows <= 0:
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

    out_required_score_cells[0] = required_score_cells
    out_required_value_cells[0] = required_value_cells
    out_required_out_cells[0] = required_out_cells
    return ATTN_Q16_OK


def test_fixed_vector_reference_tuple_publish_no_writes() -> None:
    query_rows = 4
    key_rows = 3
    value_dim = 2
    head_groups = 2
    row_stride = 4

    scores = [123] * 16
    values = [456] * 12
    out = [777] * (query_rows * value_dim)
    out_before = out.copy()

    required_score = [-1]
    required_value = [-1]
    required_out = [-1]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight(
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
        required_score,
        required_value,
        required_out,
    )
    assert err == ATTN_Q16_OK
    assert required_score[0] == 15
    assert required_value[0] == 12
    assert required_out[0] == 8
    assert out == out_before


def test_zero_geometry_publishes_zero_tuple() -> None:
    scores = [1, 2, 3]
    values = [4, 5, 6]
    out = [9, 9, 9, 9]

    out_required_score = [333]
    out_required_value = [444]
    out_required_out = [555]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight(
        scores,
        len(scores),
        0,
        7,
        11,
        1,
        7,
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


def test_rejects_invalid_alias_and_preserves_slots() -> None:
    scores = [1] * 8
    values = [2] * 8
    out = [3] * 8
    req_score = [10]
    req_value = [20]
    req_out = [30]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight(
        scores,
        len(scores),
        2,
        2,
        2,
        1,
        2,
        values,
        len(values),
        out,
        len(out),
        req_score,
        req_score,
        req_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert req_score[0] == 10
    assert req_value[0] == 20
    assert req_out[0] == 30


def test_rejects_overflow_geometry() -> None:
    out_required_score = [0]
    out_required_value = [0]
    out_required_out = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight(
        [1, 2, 3],
        I64_MAX,
        I64_MAX,
        2,
        2,
        1,
        2,
        [4, 5, 6],
        I64_MAX,
        [7, 8, 9],
        I64_MAX,
        out_required_score,
        out_required_value,
        out_required_out,
    )
    assert err in (ATTN_Q16_ERR_OVERFLOW, ATTN_Q16_ERR_BAD_PARAM)


def test_randomized_preflight_parity_vectors() -> None:
    rng = random.Random(1387)
    for _ in range(300):
        query_rows = rng.randint(0, 6)
        head_groups = rng.randint(1, 3)
        if query_rows > 0 and query_rows % head_groups != 0:
            head_groups = 1
        key_rows = rng.randint(0, 6)
        value_dim = rng.randint(0, 6)
        row_stride = rng.randint(key_rows, key_rows + 3)

        if query_rows == 0 or key_rows == 0 or value_dim == 0:
            score_cells = 0
            value_cells = 0
            out_cells = 0
        else:
            score_cells = (query_rows - 1) * row_stride + key_rows
            kv_rows = query_rows // head_groups
            value_cells = kv_rows * key_rows * value_dim
            out_cells = query_rows * value_dim

        scores_capacity = max(score_cells + rng.randint(0, 5), 0)
        values_capacity = max(value_cells + rng.randint(0, 5), 0)
        out_capacity = max(out_cells + rng.randint(0, 5), 0)

        scores = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(scores_capacity, 1))]
        values = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(values_capacity, 1))]
        out = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(out_capacity, 1))]
        out_before = out.copy()

        req_score = [rng.randint(-1000, 1000)]
        req_value = [rng.randint(-1000, 1000)]
        req_out = [rng.randint(-1000, 1000)]

        err = gqa_attention_value_mix_q16_checked_nopartial_preflight(
            scores,
            scores_capacity,
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            row_stride,
            values,
            values_capacity,
            out,
            out_capacity,
            req_score,
            req_value,
            req_out,
        )

        if err == ATTN_Q16_OK:
            assert req_score[0] == score_cells
            assert req_value[0] == value_cells
            assert req_out[0] == out_cells
            assert out[:out_cells] == out_before[:out_cells]
        else:
            assert err in (ATTN_Q16_ERR_BAD_PARAM, ATTN_Q16_ERR_OVERFLOW)


if __name__ == "__main__":
    test_fixed_vector_reference_tuple_publish_no_writes()
    test_zero_geometry_publishes_zero_tuple()
    test_rejects_invalid_alias_and_preserves_slots()
    test_rejects_overflow_geometry()
    test_randomized_preflight_parity_vectors()
    print("ok")
