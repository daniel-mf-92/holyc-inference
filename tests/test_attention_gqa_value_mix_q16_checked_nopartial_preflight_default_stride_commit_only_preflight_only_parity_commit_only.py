#!/usr/bin/env python3
"""Reference checks for GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideCommitOnlyPreflightOnlyParityCommitOnly (IQ-1432)."""

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
from test_attention_gqa_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only import (
    gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only,
)
from test_attention_gqa_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity import (
    gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity,
)


def gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
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

    staged_parity_score = [recomputed_required_score_cells]
    staged_parity_value = [recomputed_required_value_cells]
    staged_parity_out = [recomputed_required_out_cells]
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity(
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
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only(
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

    if out_values_q16[:recomputed_required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_score_cells[0] = staged_parity_score[0]
    out_required_value_cells[0] = staged_parity_value[0]
    out_required_out_cells[0] = staged_parity_out[0]
    return ATTN_Q16_OK


def explicit_commit_only_composition(*args) -> int:
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
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity(
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
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only(
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


def test_fixed_vector_reference_tuple_publish_no_writes() -> None:
    query_rows = 6
    key_rows = 4
    value_dim = 3
    head_groups = 3

    scores = [77] * (query_rows * key_rows)
    values = [123] * ((query_rows // head_groups) * key_rows * value_dim)
    out = [456] * (query_rows * value_dim)
    out_before = out.copy()

    required_score = [-1]
    required_value = [-2]
    required_out = [-3]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
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
    assert required_score[0] == 24
    assert required_value[0] == 24
    assert required_out[0] == 18
    assert out == out_before


def test_null_alias_capacity_overflow_parity_vectors() -> None:
    out = [55] * 8
    required_score = [11]
    required_value = [22]
    required_out = [33]

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
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
    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
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

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
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

    err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
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


def test_randomized_tuple_publish_vectors() -> None:
    rng = random.Random(20260425_1432)

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

        required_score_slot = [rng.randint(-10, 10)]
        required_value_slot = [rng.randint(-10, 10)]
        required_out_slot = [rng.randint(-10, 10)]

        err = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
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
            required_score_slot,
            required_value_slot,
            required_out_slot,
        )

        assert err == ATTN_Q16_OK
        assert required_score_slot[0] == required_scores
        assert required_value_slot[0] == required_values
        assert required_out_slot[0] == required_out
        assert out == out_before


def test_explicit_composition_matches_reference() -> None:
    rng = random.Random(1432001)

    for _ in range(120):
        head_groups = rng.randint(1, 4)
        query_rows = head_groups * rng.randint(1, 5)
        key_rows = rng.randint(1, 7)
        value_dim = rng.randint(1, 5)

        score_cells = query_rows * key_rows
        value_cells = (query_rows // head_groups) * key_rows * value_dim
        out_cells = query_rows * value_dim

        scores = [rng.randint(-(3 << 16), (3 << 16)) for _ in range(score_cells)]
        values = [rng.randint(-(3 << 16), (3 << 16)) for _ in range(value_cells)]
        out_a = [rng.randint(-500, 500) for _ in range(max(1, out_cells))]
        out_b = out_a.copy()

        req_a = [rng.randint(-5, 5)]
        req_b = [rng.randint(-5, 5)]
        req_c = [rng.randint(-5, 5)]
        req_d = [req_a[0]]
        req_e = [req_b[0]]
        req_f = [req_c[0]]

        err_ref = gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only(
            scores,
            len(scores),
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            values,
            len(values),
            out_a,
            len(out_a),
            req_a,
            req_b,
            req_c,
        )
        err_comp = explicit_commit_only_composition(
            scores,
            len(scores),
            query_rows,
            key_rows,
            value_dim,
            head_groups,
            values,
            len(values),
            out_b,
            len(out_b),
            req_d,
            req_e,
            req_f,
        )

        assert err_ref == ATTN_Q16_OK
        assert err_comp == ATTN_Q16_OK
        assert req_a == req_d
        assert req_b == req_e
        assert req_c == req_f
        assert out_a == out_b


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert (
        "I32 GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideCommitOnlyPreflightOnlyParityCommitOnly("
        in source
    )
    body = source.split(
        "I32 GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideCommitOnlyPreflightOnlyParityCommitOnly(",
        1,
    )[1]
    assert (
        "status = GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideCommitOnlyPreflightOnlyParity("
        in body
    )
    assert (
        "status = GQAAttentionValueMixQ16CheckedNoPartialPreflightDefaultStrideCommitOnlyPreflightOnly("
        in body
    )
    assert "snapshot_out_required_score_cells_value = *out_required_score_cells;" in body
    assert "*out_required_value_cells = staged_parity_required_value_cells;" in body


if __name__ == "__main__":
    test_fixed_vector_reference_tuple_publish_no_writes()
    test_null_alias_capacity_overflow_parity_vectors()
    test_randomized_tuple_publish_vectors()
    test_explicit_composition_matches_reference()
    test_source_contract_markers()
    print(
        "gqa_attention_value_mix_q16_checked_nopartial_preflight_default_stride_commit_only_preflight_only_parity_commit_only_reference_checks=ok"
    )
