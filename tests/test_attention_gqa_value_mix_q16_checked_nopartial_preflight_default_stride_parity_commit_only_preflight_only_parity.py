#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnlyPreflightOnlyParity (IQ-1430)."""

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
from test_attention_gqa_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only import (
    gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only,
)
from test_attention_gqa_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only import (
    gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only,
)


def gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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

    out_snapshot = out_values_q16[:recomputed_required_out_cells]

    staged_preflight_only_score = [recomputed_required_score_cells]
    staged_preflight_only_value = [recomputed_required_value_cells]
    staged_preflight_only_out = [recomputed_required_out_cells]
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
        staged_preflight_only_score,
        staged_preflight_only_value,
        staged_preflight_only_out,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_commit_only_score = [0]
    staged_commit_only_value = [0]
    staged_commit_only_out = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only(
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
        staged_commit_only_score,
        staged_commit_only_value,
        staged_commit_only_out,
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
        staged_preflight_only_score[0] != staged_commit_only_score[0]
        or staged_preflight_only_value[0] != staged_commit_only_value[0]
        or staged_preflight_only_out[0] != staged_commit_only_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_preflight_only_score[0] != recomputed_required_score_cells
        or staged_preflight_only_value[0] != recomputed_required_value_cells
        or staged_preflight_only_out[0] != recomputed_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_preflight_only_score[0] > snapshot_scores_capacity
        or staged_preflight_only_value[0] > snapshot_values_capacity
        or staged_preflight_only_out[0] > snapshot_out_capacity
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

    if out_values_q16[:recomputed_required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    return ATTN_Q16_OK


def test_fixed_vector_reference_zero_write_required_slots() -> None:
    query_rows = 6
    key_rows = 4
    value_dim = 3
    head_groups = 3

    scores = [111] * (query_rows * key_rows)
    values = [222] * ((query_rows // head_groups) * key_rows * value_dim)
    out = [333] * (query_rows * value_dim)
    out_before = out.copy()

    required_score = [7001]
    required_value = [7002]
    required_out = [7003]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
    assert required_score[0] == 7001
    assert required_value[0] == 7002
    assert required_out[0] == 7003
    assert out == out_before


def test_null_alias_capacity_overflow_parity_vectors() -> None:
    out = [55] * 8
    required_score = [11]
    required_value = [22]
    required_out = [33]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
    assert required_score[0] == 11
    assert required_value[0] == 22
    assert required_out[0] == 33

    alias_slot = [77]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
        [1],
        1,
        1,
        1,
        1,
        1,
        [1],
        1,
        out,
        len(out),
        alias_slot,
        alias_slot,
        [0],
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert alias_slot[0] == 77

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
    assert required_score[0] == 11
    assert required_value[0] == 22
    assert required_out[0] == 33

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
    assert required_score[0] == 11
    assert required_value[0] == 22
    assert required_out[0] == 33


def test_randomized_zero_write_parity_vectors() -> None:
    rng = random.Random(20260425_1430)

    for _ in range(240):
        head_groups = rng.randint(1, 4)
        key_rows = rng.randint(1, 7)
        value_dim = rng.randint(1, 6)
        query_rows = head_groups * rng.randint(1, 5)

        required_scores = (query_rows - 1) * key_rows + key_rows
        kv_rows = query_rows // head_groups
        required_values = kv_rows * key_rows * value_dim
        required_out = query_rows * value_dim

        scores = [rng.randint(-(9 << 16), (9 << 16)) for _ in range(required_scores)]
        values = [rng.randint(-(9 << 16), (9 << 16)) for _ in range(required_values)]
        out = [rng.randint(-9999, 9999) for _ in range(max(1, required_out))]
        out_before = out.copy()

        required_score = [rng.randint(-10, 10)]
        required_value = [rng.randint(-10, 10)]
        required_out_slot = [rng.randint(-10, 10)]
        expected_slots = (required_score[0], required_value[0], required_out_slot[0])

        err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity(
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
            required_out_slot,
        )

        assert err == ATTN_Q16_OK
        assert (required_score[0], required_value[0], required_out_slot[0]) == expected_slots
        assert out == out_before


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert (
        "I32 GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnlyPreflightOnlyParity("
        in source
    )
    body = source.split(
        "I32 GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnlyPreflightOnlyParity(",
        1,
    )[1]
    assert (
        "status = GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnlyPreflightOnly("
        in body
    )
    assert "status = GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnly(" in body
    assert "snapshot_out_required_score_cells_value = *out_required_score_cells;" in body
    assert "if (out_values_q16[copy_index] != snapshot_out_values_q16_values[copy_index])" in body


if __name__ == "__main__":
    test_fixed_vector_reference_zero_write_required_slots()
    test_null_alias_capacity_overflow_parity_vectors()
    test_randomized_zero_write_parity_vectors()
    test_source_contract_markers()
    print(
        "gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_preflight_only_parity_reference_checks=ok"
    )
