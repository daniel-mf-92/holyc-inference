#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnly (IQ-1415)."""

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
from test_attention_gqa_value_mix_q16_checked_nopartial_preflight_default_stride import (
    gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride,
)
from test_attention_gqa_value_mix_q16_checked_nopartial_preflight_default_stride_parity import (
    gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity,
)


def gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only(
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

    staged_parity_required_score_cells = [0]
    staged_parity_required_value_cells = [0]
    staged_parity_required_out_cells = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity(
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
        staged_parity_required_score_cells,
        staged_parity_required_value_cells,
        staged_parity_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    staged_default_required_score_cells = [0]
    staged_default_required_value_cells = [0]
    staged_default_required_out_cells = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
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
        staged_default_required_score_cells,
        staged_default_required_value_cells,
        staged_default_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

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
        staged_parity_required_score_cells[0] != staged_default_required_score_cells[0]
        or staged_parity_required_value_cells[0] != staged_default_required_value_cells[0]
        or staged_parity_required_out_cells[0] != staged_default_required_out_cells[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_required_score_cells[0] != recomputed_required_score_cells
        or staged_parity_required_value_cells[0] != recomputed_required_value_cells
        or staged_parity_required_out_cells[0] != recomputed_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_required_score_cells[0] > snapshot_scores_capacity
        or staged_parity_required_value_cells[0] > snapshot_values_capacity
        or staged_parity_required_out_cells[0] > snapshot_out_capacity
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

    out_required_score_cells[0] = staged_parity_required_score_cells[0]
    out_required_value_cells[0] = staged_parity_required_value_cells[0]
    out_required_out_cells[0] = staged_parity_required_out_cells[0]
    return ATTN_Q16_OK


def explicit_default_stride_parity_commit_only_composition(*args) -> int:
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

    staged_parity_score = [0]
    staged_parity_value = [0]
    staged_parity_out = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity(
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

    staged_default_score = [0]
    staged_default_value = [0]
    staged_default_out = [0]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride(
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
        staged_default_score,
        staged_default_value,
        staged_default_out,
    )
    if err != ATTN_Q16_OK:
        return err

    if (
        staged_parity_score[0] != staged_default_score[0]
        or staged_parity_value[0] != staged_default_value[0]
        or staged_parity_out[0] != staged_default_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_score_cells[0] = staged_parity_score[0]
    out_required_value_cells[0] = staged_parity_value[0]
    out_required_out_cells[0] = staged_parity_out[0]
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

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only(
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


def test_null_alias_capacity_overflow_parity_vectors() -> None:
    out = [99] * 8
    required_score = [1]
    required_value = [2]
    required_out = [3]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only(
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
    assert required_score[0] == 1
    assert required_value[0] == 2
    assert required_out[0] == 3

    alias_slot = [7]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only(
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
    assert alias_slot[0] == 7

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only(
        [1],
        2,
        2,
        2,
        2,
        1,
        [1],
        8,
        out,
        7,
        required_score,
        required_value,
        required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only(
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


def test_randomized_equivalence_with_explicit_composition() -> None:
    rng = random.Random(20260425_1415)

    for _ in range(240):
        key_rows = rng.randint(1, 6)
        head_groups = rng.randint(1, 4)
        query_rows = head_groups * rng.randint(1, 5)
        value_dim = rng.randint(1, 8)

        required_scores = query_rows * key_rows
        required_values = (query_rows // head_groups) * key_rows * value_dim
        required_out = query_rows * value_dim

        scores = [rng.randint(-(8 << 16), (8 << 16)) for _ in range(required_scores + rng.randint(0, 5))]
        values = [rng.randint(-(8 << 16), (8 << 16)) for _ in range(required_values)]
        out = [rng.randint(-9000, 9000) for _ in range(required_out)]
        out_before = out.copy()

        req_score_a = [rng.randint(-99, 99)]
        req_value_a = [rng.randint(-99, 99)]
        req_out_a = [rng.randint(-99, 99)]

        req_score_b = [req_score_a[0]]
        req_value_b = [req_value_a[0]]
        req_out_b = [req_out_a[0]]

        err_a = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only(
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
            req_score_a,
            req_value_a,
            req_out_a,
        )
        err_b = explicit_default_stride_parity_commit_only_composition(
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
            req_score_b,
            req_value_b,
            req_out_b,
        )

        assert err_a == err_b == ATTN_Q16_OK
        assert req_score_a[0] == req_score_b[0]
        assert req_value_a[0] == req_value_b[0]
        assert req_out_a[0] == req_out_b[0]
        assert out == out_before


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    marker_parity = "I32 GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideParity("
    marker_default = "I32 GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStride("
    marker_commit_only = (
        "I32 GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideParityCommitOnly("
    )

    assert marker_parity in source
    assert marker_default in source
    assert marker_commit_only in source

    commit_body = source.split(marker_commit_only, 1)[1]
    assert "status = GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideParity(" in commit_body
    assert "status = GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStride(" in commit_body


if __name__ == "__main__":
    test_fixed_vector_reference_tuple_publish_no_writes()
    test_null_alias_capacity_overflow_parity_vectors()
    test_randomized_equivalence_with_explicit_composition()
    test_source_contract_markers()
    print("gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_parity_commit_only_reference_checks=ok")
