#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16CheckedNoPartial semantics (IQ-1373)."""

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
    gqa_attention_value_mix_q16_checked,
    try_add_i64_checked,
    try_mul_i64_checked,
)


def gqa_attention_value_mix_q16_checked_nopartial(
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
    if required_score_cells > scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_value_cells = try_mul_i64_checked(kv_rows, key_rows)
    if err != ATTN_Q16_OK:
        return err
    err, required_value_cells = try_mul_i64_checked(required_value_cells, value_dim)
    if err != ATTN_Q16_OK:
        return err
    if required_value_cells > values_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_out_cells = try_mul_i64_checked(query_rows, value_dim)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells == 0:
        return ATTN_Q16_OK

    staged = [0] * required_out_cells
    err = gqa_attention_value_mix_q16_checked(
        scores_q16,
        scores_capacity,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values_q16,
        values_capacity,
        staged,
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

    for idx in range(required_out_cells):
        out_values_q16[idx] = staged[idx]

    return ATTN_Q16_OK


def explicit_staged_composition(*args, **kwargs) -> int:
    return gqa_attention_value_mix_q16_checked_nopartial(*args, **kwargs)


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

    out_a = [999] * (query_rows * value_dim)
    out_b = out_a.copy()

    err_a = gqa_attention_value_mix_q16_checked_nopartial(
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
    err_b = explicit_staged_composition(
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


def test_alias_capacity_and_no_partial_failure_paths() -> None:
    query_rows = 2
    key_rows = 3
    value_dim = 2
    head_groups = 1
    row_stride = 4

    scores = [
        int(0.6 * (1 << 16)),
        int(0.3 * (1 << 16)),
        int(0.1 * (1 << 16)),
        77,
        int(0.2 * (1 << 16)),
        int(0.3 * (1 << 16)),
        int(0.5 * (1 << 16)),
        88,
    ]
    values = [
        3 << 16,
        1 << 16,
        2 << 16,
        -(1 << 16),
        -(2 << 16),
        2 << 16,
        4 << 16,
        -(3 << 16),
        1 << 16,
        2 << 16,
        -(1 << 16),
        -(2 << 16),
    ]

    expected = [0] * (query_rows * value_dim)
    err = gqa_attention_value_mix_q16_checked(
        scores,
        len(scores),
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values,
        len(values),
        expected,
        len(expected),
    )
    assert err == ATTN_Q16_OK

    aliased = scores.copy()
    err = gqa_attention_value_mix_q16_checked_nopartial(
        aliased,
        len(aliased),
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values,
        len(values),
        aliased,
        len(aliased),
    )
    assert err == ATTN_Q16_OK
    assert aliased[: query_rows * value_dim] == expected

    out_seed = [5151] * 8
    out = out_seed.copy()
    err = gqa_attention_value_mix_q16_checked_nopartial(
        scores,
        len(scores) - 2,
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values,
        len(values),
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_seed

    out = out_seed.copy()
    err = gqa_attention_value_mix_q16_checked_nopartial(
        scores,
        len(scores),
        query_rows,
        key_rows,
        value_dim,
        head_groups,
        row_stride,
        values,
        len(values) - 1,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_seed

    out = out_seed.copy()
    err = gqa_attention_value_mix_q16_checked_nopartial(
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
        (query_rows * value_dim) - 1,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_seed


def test_error_contract_and_overflow_vectors() -> None:
    out_seed = [1234] * 4
    out = out_seed.copy()

    err = gqa_attention_value_mix_q16_checked_nopartial(
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
    assert out == out_seed

    huge = (1 << 63) - 1
    err = gqa_attention_value_mix_q16_checked_nopartial(
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
    assert out == out_seed


def test_randomized_parity() -> None:
    rng = random.Random(20260425_1373)

    for _ in range(240):
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
        out_a = [rng.randint(-999, 999) for _ in range(out_cells)]
        out_b = out_a.copy()

        err_a = gqa_attention_value_mix_q16_checked_nopartial(
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
        err_b = gqa_attention_value_mix_q16_checked(
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


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 GQAAttentionValueMixQ16CheckedNoPartial(" in source
    body = source.split("I32 GQAAttentionValueMixQ16CheckedNoPartial(", 1)[1]
    assert "snapshot_query_rows = query_rows;" in body
    assert "status = GQAAttentionValueMixQ16Checked(" in body
    assert "staged_out_values_q16 = MAlloc(stage_bytes);" in body


if __name__ == "__main__":
    test_fixed_vector_reference()
    test_alias_capacity_and_no_partial_failure_paths()
    test_error_contract_and_overflow_vectors()
    test_randomized_parity()
    test_source_contract_markers()
    print("gqa_attention_value_mix_q16_checked_nopartial_reference_checks=ok")
